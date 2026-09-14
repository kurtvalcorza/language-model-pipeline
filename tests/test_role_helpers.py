"""Offline tests for the public validation / evaluation stage helpers and the chat-masking and
adapter-bundle contract of lmpipeline.pipeline (DAT24 / EVAL21 / §18). A fake tokenizer stands
in for transformers so no model, weights or network are needed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lmpipeline.pipeline import (
    IGNORE_INDEX,
    INPUT_SCHEMA,
    MAX_SEQUENCE_LENGTH_CEILING,
    MODEL_ID,
    MODEL_REVISION,
    build_masked_example,
    canonical,
    dataset_digest,
    evaluation_report,
    fingerprint,
    manufacture_validation,
    perplexity,
    render_chat,
    show_supervision,
    validate_inputs,
    validate_prompts,
    verify_artifact_bundle,
    write_artifact_manifest,
)


class FakeTokenizer:
    """ChatML-style template over a character-level vocabulary with offsets."""

    chat_template = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"  # noqa: E501
    eos_token = "<|im_end|>"
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **_):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **_):
        ids = [ord(c) for c in text]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out

    def decode(self, ids, **_):
        return "".join(chr(i) for i in ids)


TOK = FakeTokenizer()


def _rec(question: str, answer: str) -> dict:
    return canonical({"instruction": question, "output": answer})


def _splits(n_train: int = 4, n_val: int = 2) -> dict:
    train = [_rec(f"q{i}", f"a{i}") for i in range(n_train)]
    validation = [_rec(f"v{i}", f"b{i}") for i in range(n_val)]
    return {"train": train, "validation": validation}


def _token_length(record: dict) -> int:
    return len(TOK(render_chat(TOK, record["messages"]))["input_ids"])


def test_canonical_normalises_three_schemas_and_rejects_unknown() -> None:
    turns = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    chat = canonical({"messages": turns})
    pc = canonical({"prompt": "q", "completion": "a"})
    alpaca = canonical({"instruction": "q", "input": "ctx", "output": "a"})
    assert chat["messages"][-1] == {"role": "assistant", "content": "a"}
    assert pc["messages"][0] == {"role": "user", "content": "q"}
    assert alpaca["messages"][0]["content"] == "q\n\nctx"
    with pytest.raises(ValueError, match="Unsupported SFT schema"):
        canonical({"text": "free"})
    with pytest.raises(ValueError, match="no non-empty assistant turn"):
        canonical({"prompt": "q", "completion": "   "})
    assert fingerprint(chat) == fingerprint(canonical({"messages": chat["messages"]}))


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    splits = _splits()
    manifest = validate_inputs(splits, 128, token_length=_token_length, names=["tr", "va"])
    assert manifest["verdict"] == "accepted" and manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["sequence_tokens"] == [1, MAX_SEQUENCE_LENGTH_CEILING]
    assert [i["id"] for i in manifest["inputs"]] == ["tr", "va"]
    assert [i["records"] for i in manifest["inputs"]] == [4, 2]
    assert manifest["inputs"][0]["tokens"]["longest"] > 0
    assert manifest["token_lengths_measured"] is True
    assert manifest["dataset_digest"] == dataset_digest(splits)
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_without_tokenizer_records_no_token_counts() -> None:
    manifest = validate_inputs(_splits())
    assert manifest["token_lengths_measured"] is False
    assert manifest["inputs"][0]["tokens"] is None
    assert [i["id"] for i in manifest["inputs"]] == ["train", "validation"]


def test_validate_inputs_rejects_like_prepare_splits() -> None:
    leaked = _splits()
    leaked["validation"].append(dict(leaked["train"][0]))
    with pytest.raises(ValueError, match="Split leakage"):
        validate_inputs(leaked)
    with pytest.raises(ValueError, match="DATASET_SEQUENCE_TOO_LONG"):
        validate_inputs(_splits(), 8, token_length=_token_length)
    with pytest.raises(ValueError, match="MAX_SEQUENCE_LENGTH_CEILING"):
        validate_inputs(_splits(), MAX_SEQUENCE_LENGTH_CEILING + 1)
    with pytest.raises(ValueError, match="MIN_TRAIN_EXAMPLES"):
        validate_inputs({"train": [_rec("q", "a")]})
    with pytest.raises(TypeError, match="'train' split"):
        validate_inputs({"validation": []})
    with pytest.raises(ValueError, match="names must have one entry per split"):
        validate_inputs(_splits(), names=["only-one"])


def test_validate_inputs_reports_in_split_duplicates_as_findings() -> None:
    splits = _splits()
    splits["train"].append(dict(splits["train"][0]))
    manifest = validate_inputs(splits)
    assert manifest["verdict"] == "accepted"
    assert manifest["inputs"][0]["duplicates"] == 1
    assert manifest["findings"][0]["input"] == "train"


def test_manufactured_validation_is_a_fifth_and_leak_free() -> None:
    rows = [_rec(f"q{i}", f"a{i}") for i in range(10)]
    splits = manufacture_validation(rows)
    assert (len(splits["train"]), len(splits["validation"])) == (8, 2)
    validate_inputs(splits)
    with pytest.raises(ValueError, match="MIN_TRAIN_EXAMPLES"):
        manufacture_validation(rows[:1])


def test_masking_supervises_only_the_assistant_turn() -> None:
    record = _rec("hello?", "world")
    input_ids, labels = build_masked_example(TOK, record, 200)
    text = show_supervision(TOK, input_ids, labels)
    assert "⟦world<|im_end|>⟧" in text
    assert "hello?" not in text.split("⟦", 1)[1]  # the user turn is never supervised
    supervised = [i for i, lab in zip(input_ids, labels, strict=True) if lab != IGNORE_INDEX]
    assert TOK.decode(supervised) == "world<|im_end|>"
    with pytest.raises(ValueError, match="DATASET_SEQUENCE_TOO_LONG"):
        build_masked_example(TOK, record, 8)


def test_render_chat_folds_system_turn_when_template_has_no_system_role() -> None:
    messages = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "q"}]
    assert render_chat(TOK, messages).count("<|im_start|>") == 1
    assert "be brief\n\nq" in render_chat(TOK, messages)


def test_evaluation_report_not_measurable_without_validation_loss() -> None:
    report = evaluation_report(None, sample_kind="BYOD")
    assert report["verdict"] == "not-measurable" and report["metrics"] == []
    assert "held-out evaluation set" in report["needs"]
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    assert evaluation_report({"trainLoss": 1.0})["verdict"] == "not-measurable"


def test_evaluation_report_sample_sanity_with_metrics() -> None:
    metrics = {"trainLoss": 3.2, "validationLoss": 3.1, "testLoss": None}
    metrics["validationPerplexity"] = perplexity(3.1)
    probes = [{"prompt": "p", "base": "b", "adapted": "a"}]
    report = evaluation_report(metrics, sample_kind="sample", n_train=96, n_validation=24, probes=probes)  # noqa: E501
    assert report["verdict"] == "sample-sanity"
    ids = {m["id"]: m for m in report["metrics"]}
    assert set(ids) == {"trainLoss", "validationLoss", "validationPerplexity"}
    assert ids["validationLoss"]["units"] == "nats per supervised token"
    assert report["n_validation"] == 24 and report["probes"][0]["adapted"] == "a"
    assert "not task quality" in report["reason"]
    assert perplexity(None) is None and perplexity(25.0) is None


def _bundle(tmp_path: Path, **overrides) -> Path:
    bundle = tmp_path / "bundle"
    (bundle / "tokenizer").mkdir(parents=True)
    (bundle / "adapter_config.json").write_text("{}", encoding="utf-8")
    (bundle / "adapter_model.safetensors").write_bytes(b"\x00" * 16)
    (bundle / "tokenizer" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    provenance = {
        "artifactFormat": "peft_adapter",
        "baseModel": MODEL_ID,
        "baseModelRevision": MODEL_REVISION,
        "trustRemoteCode": False,
        **overrides,
    }
    (bundle / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    write_artifact_manifest(bundle)
    return bundle


def test_artifact_bundle_verifies_before_any_state_is_loaded(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    manifest, provenance = verify_artifact_bundle(bundle)
    assert manifest["format"] == "peft_adapter" and len(manifest["files"]) == 4
    assert provenance["baseModelRevision"] == MODEL_REVISION


def test_artifact_bundle_rejects_tampering_and_foreign_base(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    (bundle / "adapter_model.safetensors").write_bytes(b"\x01" * 16)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_artifact_bundle(bundle)
    bundle = _bundle(tmp_path / "extra")
    (bundle / "smuggled.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="Manifest/file-set mismatch"):
        verify_artifact_bundle(bundle)
    bundle = _bundle(tmp_path / "branch", baseModelRevision="main")
    with pytest.raises(ValueError, match="40-character commit SHA"):
        verify_artifact_bundle(bundle)
    bundle = _bundle(tmp_path / "foreign", baseModel="someone/else")
    with pytest.raises(ValueError, match="refusing to attach"):
        verify_artifact_bundle(bundle)
    bundle = _bundle(tmp_path / "remote", trustRemoteCode=True)
    with pytest.raises(ValueError, match="trustRemoteCode"):
        verify_artifact_bundle(bundle)


def test_validate_prompts_manifest_and_rejections() -> None:
    def prompt_tokens(messages):
        return len(TOK(render_chat(TOK, messages, add_generation_prompt=True))["input_ids"])

    prompts = ["hi", [{"role": "user", "content": "there"}]]
    manifest = validate_prompts(prompts, 32, token_length=prompt_tokens, names=["a", "b"])
    assert manifest["verdict"] == "accepted" and manifest["findings"] == []
    assert [i["id"] for i in manifest["inputs"]] == ["a", "b"]
    assert manifest["inputs"][0]["rendered_tokens"] > 2 and manifest["max_new_tokens"] == 32
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    assert validate_prompts(["x"])["inputs"][0]["id"] == "prompt-0"
    with pytest.raises(ValueError, match="non-empty"):
        validate_prompts(["   "])
    with pytest.raises(ValueError, match="MAX_NEW_TOKENS_CEILING"):
        validate_prompts(["x"], 4096)
    with pytest.raises(ValueError, match="end with a user turn"):
        validate_prompts([[{"role": "assistant", "content": "x"}]])
    with pytest.raises(TypeError):
        validate_prompts("not a list")
    with pytest.raises(ValueError, match="names must have one entry per prompt"):
        validate_prompts(["x"], names=["a", "b"])
