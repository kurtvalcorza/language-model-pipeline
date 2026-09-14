"""Standalone tutorial pipeline for the DIMER language-model capability.

This module is what the standalone tutorials carry verbatim (NOTEBOOK_SPEC 1.1 §3.6): the
pinned base-model identity, manifest-driven snapshot verification and staging, chat rendering,
assistant-only loss masking, greedy generation, the PEFT adapter-bundle contract, and the public
``validate_inputs`` / ``evaluation_report`` stage helpers. The QLoRA training loop itself stays
in the tutorial (it is deliberately plain PyTorch so every step is visible); this module owns
everything the loop consumes and everything the exported artifact must satisfy.

Heavy libraries (``torch``, ``transformers``, ``peft``) are imported lazily inside the functions
that need them so the contract helpers stay importable and testable offline.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import stat
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
MODEL_REVISION = "a10cc1512eabd3dde888204e902eca88bddb4951"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "smollm2-360m"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
WEIGHTS_FILE = "model.safetensors"

# Registry ceilings for the pinned model (src/lmpipeline/data/model-registry.yaml, smollm2-360m).
MAX_SEQUENCE_LENGTH_CEILING = 2048  # tokens; the registry's hard ceiling for this base model
DEFAULT_MAX_SEQUENCE_LENGTH = 512  # tokens; the tutorial default, well inside the ceiling
MAX_TOTAL_TRAIN_TOKENS = 50_000_000  # tokens across the training split per run
MIN_TRAIN_EXAMPLES = 2  # one for training and one for the manufactured validation split
IGNORE_INDEX = -100  # PyTorch cross-entropy ignore_index: unsupervised label positions
ALLOWED_ROLES = frozenset({"system", "user", "assistant"})
ARTIFACT_FORMAT = "peft_adapter"
ARTIFACT_FORMAT_VERSION = 1
ARTIFACT_MANIFEST_NAME = "artifact-manifest.json"
DECODING_RULE = "greedy"  # do_sample=False: deterministic given weights, device and versions
MAX_NEW_TOKENS_CEILING = 1024  # generated tokens per call; the tutorials use 16..96
MAX_PROMPT_CHARS = 20_000  # characters per prompt turn; rendered prompts must fit the ceiling

# The other base models the repository's DIMER worker admits. Informational in the standalone
# tutorial: the notebook pins MODEL_ID/MODEL_REVISION above and carries that snapshot's manifest;
# switching base model means regenerating the notebook from a template that pins another key.
TUTORIAL_REGISTRY: dict[str, dict[str, Any]] = {
    "smollm2-360m": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "license": "apache-2.0", "min_vram_gb": None, "requires_hf_token": False, "dimer_zip": True, "state": "tutorial-default/apache-2.0"},  # noqa: E501
    "qwen3-0.6b": {"model_id": "Qwen/Qwen3-0.6B", "revision": "c1899de289a04d12100db370d81485cdf75e47ca", "license": "apache-2.0", "min_vram_gb": None, "requires_hf_token": False, "dimer_zip": True, "state": "smoke-ci/apache-2.0"},  # noqa: E501
    "smollm3-3b": {"model_id": "HuggingFaceTB/SmolLM3-3B", "revision": "a07cc9a04f16550a088caea529712d1d335b0ac1", "license": "apache-2.0", "min_vram_gb": None, "requires_hf_token": False, "dimer_zip": True, "state": "tutorial-candidate/internal-only"},  # noqa: E501
    "qwen3-1.7b": {"model_id": "Qwen/Qwen3-1.7B", "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", "license": "apache-2.0", "min_vram_gb": 9.3, "requires_hf_token": False, "dimer_zip": True, "state": "user-facing"},  # noqa: E501
    "qwen3-4b": {"model_id": "Qwen/Qwen3-4B", "revision": "1cfa9a7208912126459214e8b04321603b3df60c", "license": "apache-2.0", "min_vram_gb": 11.5, "requires_hf_token": False, "dimer_zip": True, "state": "user-facing"},  # noqa: E501
    "granite-4.1-3b": {"model_id": "ibm-granite/granite-4.1-3b", "revision": "c0650403e44e78ec0262dab1c90914c65b196c4e", "license": "apache-2.0", "min_vram_gb": 8.7, "requires_hf_token": False, "dimer_zip": True, "state": "user-facing"},  # noqa: E501
    "deepseek-r1-distill-qwen-1.5b": {"model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "revision": "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562", "license": "mit", "min_vram_gb": None, "requires_hf_token": False, "dimer_zip": True, "state": "open-reasoning/mit"},  # noqa: E501
    "qwen2.5-coder-1.5b": {"model_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "revision": "2e1fd397ee46e1388853d2af2c993145b0f1098a", "license": "apache-2.0", "min_vram_gb": None, "requires_hf_token": False, "dimer_zip": True, "state": "open-code/apache-2.0"},  # noqa: E501
    "smollm2-1.7b": {"model_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "revision": "31b70e2e869a7173562077fd711b654946d38674", "license": "apache-2.0", "min_vram_gb": None, "requires_hf_token": False, "dimer_zip": True, "state": "open-small/apache-2.0"},  # noqa: E501
    "granite-3.1-2b-instruct": {"model_id": "ibm-granite/granite-3.1-2b-instruct", "revision": "bbc2aed595bd38bd770263dc3ab831db9794441d", "license": "apache-2.0", "min_vram_gb": None, "requires_hf_token": False, "dimer_zip": True, "state": "enterprise-transparent/apache-2.0"},  # noqa: E501
    "h2o-danube3-4b-chat": {"model_id": "h2oai/h2o-danube3-4b-chat", "revision": "1e5c6fa6620f8bf078958069ab4581cd88e0202c", "license": "apache-2.0", "min_vram_gb": None, "requires_hf_token": False, "dimer_zip": True, "state": "mobile-edge/apache-2.0"},  # noqa: E501
    "llama-3.2-3b-instruct": {"model_id": "meta-llama/Llama-3.2-3B-Instruct", "revision": "0cb88a4f764b7a12671c53f0838cd831a0843b95", "license": "llama3.2", "min_vram_gb": None, "requires_hf_token": True, "dimer_zip": False, "state": "credential-test-candidate"},  # noqa: E501
}

# Pinned public samples the tutorial can load (id, immutable dataset revision, license).
SAMPLE_DATASETS: dict[str, dict[str, str]] = {
    "Sample: Filipino SFT": {
        "dataset_id": "jpaulpoliquit/ph-sft-ai-authored-v1",
        "revision": "8333699c6cc7296cc69cefc09def010851ded919",
        "license": "apache-2.0",
    },
    "Sample: Dolly": {
        "dataset_id": "databricks/databricks-dolly-15k",
        "revision": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
        "license": "cc-by-sa-3.0",
    },
}


# ---------------------------------------------------------------------------
# Snapshot verification and staging (MOD6-MOD8, ST3/ST4)
# ---------------------------------------------------------------------------


def sha256_of_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local base-model snapshot against its manifest; raise naming the first mismatch."""
    root = Path(path or DEFAULT_WEIGHTS_DIR)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("format") != "dimer_hf_snapshot" or manifest.get("formatVersion") != 1:
        raise ValueError("unsupported snapshot manifest format (expected dimer_hf_snapshot/1)")
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    if not any(entry["path"].endswith(".safetensors") for entry in manifest.get("files", [])):
        raise ValueError("manifest lists no .safetensors weights")
    for entry in manifest.get("files", []):
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = sha256_of_file(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {"path": str(root), **manifest}


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; ``verify_snapshot`` still runs
    afterwards."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    files = manifest["files"]
    missing = [entry["path"] for entry in files if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


# ---------------------------------------------------------------------------
# Dataset normalisation and hygiene (DAT*)
# ---------------------------------------------------------------------------


def canonical(record: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one source record into ``{"messages": [...]}``; raise if the schema is unknown."""
    if "messages" in record:
        messages = [{"role": m["role"], "content": str(m["content"])} for m in record["messages"]]
    elif "prompt" in record and ("completion" in record or "response" in record):
        answer = record.get("completion", record.get("response"))
        messages = [
            {"role": "user", "content": str(record["prompt"])},
            {"role": "assistant", "content": str(answer)},
        ]
    elif "instruction" in record and ("output" in record or "response" in record):
        context = record.get("input") or record.get("context")
        question = str(record["instruction"]) + (f"\n\n{context}" if context else "")
        answer = record.get("output", record.get("response"))
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": str(answer)},
        ]
    else:
        raise ValueError(
            "Unsupported SFT schema: expected messages, prompt/completion, or instruction/output"
        )
    if any(m["role"] not in ALLOWED_ROLES for m in messages):
        raise ValueError("Invalid role in record")
    if not any(m["role"] == "assistant" and m["content"].strip() for m in messages):
        raise ValueError("Record has no non-empty assistant turn to learn from")
    return {"messages": messages}


def fingerprint(record: Mapping[str, Any]) -> str:
    """Stable identity for a record: SHA-256 of its canonical JSON."""
    payload = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dataset_digest(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    """Identity of an exact dataset: SHA-256 over every split's record fingerprints, in order."""
    joined = "".join(fingerprint(r) for name in sorted(splits) for r in splits[name])
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def manufacture_validation(
    records: Sequence[Mapping[str, Any]], fraction: float = 0.2
) -> dict[str, list[dict[str, Any]]]:
    """Hold out the first ``fraction`` of ``records`` (already in a deterministic order) as
    validation. The split is manufactured from the training source, so its loss is an
    optimisation signal, not a task-quality measurement."""
    rows = [dict(r) for r in records]
    if len(rows) < MIN_TRAIN_EXAMPLES:
        raise ValueError(f"at least MIN_TRAIN_EXAMPLES={MIN_TRAIN_EXAMPLES} records are required")
    validation_size = max(1, int(len(rows) * fraction))
    return {"train": rows[validation_size:], "validation": rows[:validation_size]}


INPUT_SCHEMA: dict[str, Any] = {
    "input": "splits: {'train': [...], optional 'validation'/'test': [...]} of canonical chat "
    "records {'messages': [{'role', 'content'}, ...]} (messages, prompt/completion and "
    "instruction/output source schemas are normalised by `canonical`)",
    "roles": sorted(ALLOWED_ROLES),
    "assistant_turns": "every record needs at least one non-empty assistant turn",
    "train_examples": [MIN_TRAIN_EXAMPLES, None],
    "sequence_tokens": [1, MAX_SEQUENCE_LENGTH_CEILING],
    "max_sequence_length_default": DEFAULT_MAX_SEQUENCE_LENGTH,
    "train_tokens_total": [1, MAX_TOTAL_TRAIN_TOKENS],
    "split_leakage": "identical records in two splits are rejected",
    "duplicates": "exact duplicates inside a split are reported, not removed",
    "preprocessing": "chat template rendered by the base tokenizer; loss masked to assistant "
    f"tokens (label {IGNORE_INDEX} elsewhere); no truncation - over-length rows are rejected",
}


def _check_splits(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    max_sequence_length: int,
    token_length: Callable[[Mapping[str, Any]], int] | None,
) -> dict[str, Any]:
    """Raise ValueError/TypeError naming the first violated rule; return the observations."""
    if not isinstance(splits, Mapping) or "train" not in splits:
        raise TypeError("splits must be a mapping with at least a 'train' split")
    if isinstance(max_sequence_length, bool) or not isinstance(max_sequence_length, int):
        raise TypeError("max_sequence_length must be an int")
    if not 1 <= max_sequence_length <= MAX_SEQUENCE_LENGTH_CEILING:
        raise ValueError(
            "max_sequence_length must be between 1 and "
            f"MAX_SEQUENCE_LENGTH_CEILING={MAX_SEQUENCE_LENGTH_CEILING}"
        )
    observed: dict[str, Any] = {"splits": {}, "duplicates": {}, "tokens": {}}
    prints: dict[str, set[str]] = {}
    for name, records in splits.items():
        if not isinstance(records, Sequence) or isinstance(records, str | bytes):
            raise TypeError(f"split {name!r} must be a sequence of records")
        if name == "train" and len(records) < MIN_TRAIN_EXAMPLES:
            raise ValueError(
                f"train split needs at least MIN_TRAIN_EXAMPLES={MIN_TRAIN_EXAMPLES} records"
            )
        fps = []
        for record in records:
            if not isinstance(record, Mapping) or "messages" not in record:
                raise TypeError(f"split {name!r}: every record must be canonical ({{'messages'}})")
            messages = record["messages"]
            if any(m.get("role") not in ALLOWED_ROLES for m in messages):
                raise ValueError(f"split {name!r}: invalid role in record")
            if not any(m["role"] == "assistant" and str(m["content"]).strip() for m in messages):
                raise ValueError(f"split {name!r}: record has no non-empty assistant turn")
            fps.append(fingerprint(record))
        prints[name] = set(fps)
        observed["splits"][name] = len(records)
        observed["duplicates"][name] = len(fps) - len(prints[name])
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if left in prints and right in prints and prints[left] & prints[right]:
            raise ValueError(f"Split leakage: identical records in {left} and {right}")
    if token_length is not None:
        for name, records in splits.items():
            lengths = [int(token_length(r)) for r in records]
            longest = max(lengths, default=0)
            if longest > max_sequence_length:
                raise ValueError(
                    f"DATASET_SEQUENCE_TOO_LONG: split {name!r} has a {longest}-token example; "
                    f"max_sequence_length={max_sequence_length}"
                )
            observed["tokens"][name] = {"total": sum(lengths), "longest": longest}
        if observed["tokens"].get("train", {}).get("total", 0) > MAX_TOTAL_TRAIN_TOKENS:
            raise ValueError(
                f"DATASET_TOKEN_BUDGET_EXCEEDED: MAX_TOTAL_TRAIN_TOKENS={MAX_TOTAL_TRAIN_TOKENS}"
            )
    return observed


def validate_inputs(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    *,
    token_length: Callable[[Mapping[str, Any]], int] | None = None,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, per-split observations, verdict).

    Applies exactly the checks ``LanguageModelPipeline.prepare_splits`` applies before masking
    (schema, roles, assistant turns, leakage, duplicates, and - when ``token_length`` is given,
    normally ``pipe.token_length`` - the sequence and budget ceilings). Rejection is reported by
    raising exactly as the pipeline would; a caller that wants the finding recorded catches the
    exception and stores ``str(exc)`` under ``findings``.
    """
    observed = _check_splits(splits, max_sequence_length, token_length)
    if names is not None and len(names) != len(splits):
        raise ValueError("names must have one entry per split")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [
            {
                "id": names[i] if names else name,
                "split": name,
                "records": observed["splits"][name],
                "duplicates": observed["duplicates"][name],
                "tokens": observed["tokens"].get(name),
            }
            for i, name in enumerate(splits)
        ],
        "max_sequence_length": max_sequence_length,
        "token_lengths_measured": token_length is not None,
        "dataset_digest": dataset_digest(splits),
        "verdict": "accepted",
        "findings": [
            {"input": name, "verdict": "accepted", "message": f"{n} exact duplicate(s) inside the split; none removed"}  # noqa: E501
            for name, n in observed["duplicates"].items()
            if n
        ],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


PROMPT_SCHEMA: dict[str, Any] = {
    "input": "one prompt string per request (rendered as a single user turn) or a list of "
    "{'role', 'content'} turns ending with a user turn",
    "prompt_chars": [1, MAX_PROMPT_CHARS],
    "max_new_tokens": [1, MAX_NEW_TOKENS_CEILING],
    "rendered_prompt_tokens": [1, MAX_SEQUENCE_LENGTH_CEILING],
    "decoding": "greedy (do_sample=False) unless sampling settings are passed explicitly",
}


def _check_prompt(prompt: Any, max_new_tokens: int) -> list[dict[str, str]]:
    """Raise TypeError/ValueError naming the first violated prompt rule; return the turns."""
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise TypeError("max_new_tokens must be an int")
    if not 1 <= max_new_tokens <= MAX_NEW_TOKENS_CEILING:
        raise ValueError(
            f"max_new_tokens must be between 1 and MAX_NEW_TOKENS_CEILING={MAX_NEW_TOKENS_CEILING}"
        )
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, Sequence) and not isinstance(prompt, bytes):
        messages = [dict(m) for m in prompt]
    else:
        raise TypeError("prompt must be a string or a sequence of {'role', 'content'} turns")
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("a prompt must end with a user turn")
    for turn in messages:
        if turn.get("role") not in ALLOWED_ROLES:
            raise ValueError(f"invalid role {turn.get('role')!r}")
        content = turn.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("every turn needs non-empty string content")
        if len(content) > MAX_PROMPT_CHARS:
            raise ValueError(f"prompt turn exceeds MAX_PROMPT_CHARS={MAX_PROMPT_CHARS} characters")
    return messages


def validate_prompts(
    prompts: Sequence[Any],
    max_new_tokens: int = 96,
    *,
    token_length: Callable[[Sequence[Mapping[str, str]]], int] | None = None,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage for inference requests: the input manifest for a list of prompts.

    Applies exactly the checks ``LanguageModelPipeline.generate`` applies (turn shape, roles,
    non-empty content, character and ``max_new_tokens`` ceilings; the rendered-token ceiling when
    ``token_length`` - normally ``pipe.prompt_token_length`` - is given). Raises as ``generate``
    would; a caller records the message under ``findings``."""
    if not isinstance(prompts, Sequence) or isinstance(prompts, str | bytes):
        raise TypeError("prompts must be a sequence")
    if not prompts:
        raise ValueError("prompts must not be empty")
    checked = [_check_prompt(p, max_new_tokens) for p in prompts]
    if names is not None and len(names) != len(checked):
        raise ValueError("names must have one entry per prompt")
    inputs = []
    for i, messages in enumerate(checked):
        entry: dict[str, Any] = {
            "id": names[i] if names else f"prompt-{i}",
            "turns": len(messages),
            "chars": sum(len(m["content"]) for m in messages),
        }
        if token_length is not None:
            tokens = int(token_length(messages))
            if tokens > MAX_SEQUENCE_LENGTH_CEILING:
                raise ValueError(
                    f"rendered prompt has {tokens} tokens; the ceiling is "
                    f"MAX_SEQUENCE_LENGTH_CEILING={MAX_SEQUENCE_LENGTH_CEILING}"
                )
            entry["rendered_tokens"] = tokens
        inputs.append(entry)
    return {
        "schema": dict(PROMPT_SCHEMA),
        "inputs": inputs,
        "max_new_tokens": max_new_tokens,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


# ---------------------------------------------------------------------------
# Chat rendering, assistant-only masking, loss (§20.7)
# ---------------------------------------------------------------------------


def render_chat(tokenizer: Any, messages: Sequence[Mapping[str, str]], add_generation_prompt: bool = False) -> str:  # noqa: E501
    """Render turns with the tokenizer's own chat template. ``enable_thinking=False`` asks
    Qwen3-style templates for a direct answer; templates without that switch ignore it. Templates
    with no ``system`` role (danube3) get the system text folded into the first user turn."""
    messages = [dict(m) for m in messages]
    template = getattr(tokenizer, "chat_template", None) or ""
    if messages and messages[0]["role"] == "system" and "system" not in template:
        system, rest = messages[0], messages[1:]
        if rest and rest[0]["role"] == "user":
            rest[0] = {"role": "user", "content": f"{system['content']}\n\n{rest[0]['content']}"}
            messages = rest
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False
    )


def assistant_char_spans(tokenizer: Any, messages: Sequence[Mapping[str, str]], full_text: str) -> list[tuple[int, int]]:  # noqa: E501
    """Character ranges ``[start, end)`` of each assistant turn inside the rendered conversation."""
    end_of_turn = getattr(tokenizer, "eos_token", None) or "<|im_end|>"
    last_assistant = max(i for i, m in enumerate(messages) if m["role"] == "assistant")
    spans = []
    cursor = 0
    for index, message in enumerate(messages):
        content = message["content"].strip()
        if message["role"] != "assistant":
            found = full_text.find(content, cursor)
            if found != -1:
                cursor = found + len(content)
            continue
        start = -1
        if index == last_assistant:
            generation_prompt = render_chat(tokenizer, messages[:index], add_generation_prompt=True)
            if full_text.startswith(generation_prompt):
                start = len(generation_prompt)
        if start == -1:
            start = full_text.find(content, cursor)
            if start == -1:
                raise ValueError(f"Assistant content not found in rendered text: {content[:60]!r}")
        eos_at = full_text.find(end_of_turn, start)
        end = eos_at + len(end_of_turn) if eos_at != -1 else len(full_text)
        spans.append((start, end))
        cursor = end
    return spans


def build_masked_example(tokenizer: Any, record: Mapping[str, Any], max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH) -> tuple[list[int], list[int]]:  # noqa: E501
    """Return ``(input_ids, labels)``, labels ``IGNORE_INDEX`` everywhere except assistant turns.

    Over-length examples are rejected, never truncated (a truncated answer would teach the model
    to stop mid-sentence)."""
    messages = record["messages"]
    full_text = render_chat(tokenizer, messages)

    def encode(text: str) -> list[int]:
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    try:
        encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
        input_ids, offsets = list(encoded["input_ids"]), list(encoded["offset_mapping"])
    except Exception:  # slow tokenizers cannot report offsets; fall back to prefix rendering
        input_ids, offsets = encode(full_text), None
    if len(input_ids) > max_sequence_length:
        raise ValueError("DATASET_SEQUENCE_TOO_LONG")

    labels = [IGNORE_INDEX] * len(input_ids)
    if offsets is not None:
        for start, end in assistant_char_spans(tokenizer, messages, full_text):
            for position, (char_start, char_end) in enumerate(offsets):
                inside_turn = start <= char_start and char_end <= end
                if inside_turn and char_start < char_end:
                    labels[position] = input_ids[position]
    else:
        for index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            before = encode(render_chat(tokenizer, messages[:index], add_generation_prompt=True))
            through = encode(render_chat(tokenizer, messages[: index + 1]))
            if input_ids[: len(before)] != before or input_ids[: len(through)] != through:
                raise ValueError("Chat template rendering is not prefix-stable for this tokenizer")
            labels[len(before) : len(through)] = input_ids[len(before) : len(through)]

    if all(label == IGNORE_INDEX for label in labels):
        raise ValueError("No supervised tokens in example")
    return input_ids, labels


def show_supervision(tokenizer: Any, input_ids: Sequence[int], labels: Sequence[int]) -> str:
    """Decode an example, marking supervised token runs with ⟦ ⟧."""
    pieces = []
    inside = False
    for token_id, label in zip(input_ids, labels, strict=True):
        supervised = label != IGNORE_INDEX
        if supervised != inside:
            pieces.append("⟦" if supervised else "⟧")
            inside = supervised
        pieces.append(tokenizer.decode([token_id]))
    if inside:
        pieces.append("⟧")
    return "".join(pieces)


def supervised_token_count(labels: Sequence[int]) -> int:
    return sum(1 for label in labels if label != IGNORE_INDEX)


def perplexity(loss: float | None) -> float | None:
    """exp(loss); None when the loss is missing or too large to be meaningful (>= 20 nats)."""
    if loss is None or loss >= 20:
        return None
    return math.exp(loss)


# ---------------------------------------------------------------------------
# Adapter-bundle contract (§16, §18): export, safe extraction, verification
# ---------------------------------------------------------------------------


def safe_member_path(root: str | Path, member_name: str) -> Path:
    """Resolve an archive member name under ``root``, refusing anything that could escape it."""
    if "\\" in member_name:
        raise ValueError(f"Unsafe archive path (backslash): {member_name!r}")
    relative = PurePosixPath(member_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe archive path: {member_name!r}")
    root = Path(root).resolve()
    target = (root / Path(*relative.parts)).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"Archive member escapes the extraction root: {member_name!r}")
    return target


def extract_zip_safely(zip_path: str | Path, root: str | Path, size_limit_bytes: int) -> Path:
    """Extract into ``root``, refusing symlinks, path escapes and archives that expand past the
    limit. Members are copied one by one; ``extractall`` is never used."""
    root = Path(root).resolve()
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    expanded = 0
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF):
                raise ValueError("Symlinks are not allowed in the archive")
            expanded += info.file_size
            if expanded > size_limit_bytes:
                raise ValueError("Archive expands beyond the allowed size")
            target = safe_member_path(root, info.filename)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, open(target, "wb") as destination:
                shutil.copyfileobj(source, destination)
    return root


def write_artifact_manifest(bundle_dir: str | Path) -> dict[str, Any]:
    """List every bundle file with its size and SHA-256 into artifact-manifest.json."""
    bundle = Path(bundle_dir)
    records = [
        {
            "path": path.relative_to(bundle).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_of_file(path),
        }
        for path in sorted(bundle.rglob("*"))
        if path.is_file() and path.name != ARTIFACT_MANIFEST_NAME
    ]
    manifest = {
        "format": ARTIFACT_FORMAT,
        "formatVersion": ARTIFACT_FORMAT_VERSION,
        "files": records,
        "totalBytes": sum(r["bytes"] for r in records),
    }
    (bundle / ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_artifact_bundle(bundle_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate an adapter bundle BEFORE any model state is deserialised (AINF3/AINF4).

    Checks the manifest format, every listed file's size and SHA-256, that no unlisted file is
    present, and that ``provenance.json`` names this module's pinned base model at its 40-hex
    revision with ``trustRemoteCode`` false. Returns ``(manifest, provenance)``."""
    bundle = Path(bundle_dir).resolve()
    manifest_path = bundle / ARTIFACT_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"artifact manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != ARTIFACT_FORMAT or manifest.get("formatVersion") != ARTIFACT_FORMAT_VERSION:  # noqa: E501
        raise ValueError("Unsupported artifact format")
    listed, listed_bytes = set(), 0
    for record in manifest.get("files", []):
        file_path = safe_member_path(bundle, record["path"])
        if not file_path.is_file() or file_path.stat().st_size != record["bytes"] or sha256_of_file(file_path) != record["sha256"]:  # noqa: E501
            raise ValueError(f"SHA-256 mismatch: {record['path']}")
        listed.add(record["path"])
        listed_bytes += record["bytes"]
    on_disk = {
        p.relative_to(bundle).as_posix()
        for p in bundle.rglob("*")
        if p.is_file() and p.name != ARTIFACT_MANIFEST_NAME
    }
    if on_disk != listed or listed_bytes != manifest.get("totalBytes"):
        raise ValueError("Manifest/file-set mismatch: files were added, removed, or the total size differs")  # noqa: E501
    for required in ("adapter_config.json", "adapter_model.safetensors", "provenance.json", "tokenizer/tokenizer_config.json"):  # noqa: E501
        if required not in listed:
            raise ValueError(f"artifact bundle is missing {required}")
    provenance = json.loads((bundle / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("trustRemoteCode") is not False:
        raise ValueError("trustRemoteCode must be false")
    if provenance.get("artifactFormat") != ARTIFACT_FORMAT:
        raise ValueError("provenance artifactFormat must be peft_adapter")
    revision = provenance.get("baseModelRevision") or ""
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Invalid baseModelRevision provenance: expected a 40-character commit SHA, not a branch or tag")  # noqa: E501
    if (provenance.get("baseModel"), revision) != (MODEL_ID, MODEL_REVISION):
        raise ValueError(
            f"artifact was trained on {provenance.get('baseModel')}@{revision[:12]}, "
            f"this pipeline pins {MODEL_ID}@{MODEL_REVISION[:12]}; refusing to attach"
        )
    return manifest, provenance


# ---------------------------------------------------------------------------
# Evaluation report (EVAL21)
# ---------------------------------------------------------------------------


def evaluation_report(
    metrics: Mapping[str, Any] | None,
    *,
    sample_kind: str = "sample",
    n_train: int = 0,
    n_validation: int = 0,
    probes: Sequence[Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    ``metrics`` is the dict the tutorial's training loop produces (``trainLoss``,
    ``validationLoss``, ``validationPerplexity`` - the ids the repository's model card and result
    contract use). They are optimisation evidence on a manufactured validation split, so the best
    verdict is ``sample-sanity``; task quality stays ``not-measurable`` without a held-out task
    set. ``probes`` (base vs adapted answers) are recorded as qualitative evidence only."""
    base = {
        "task": "language-model supervised fine-tuning (QLoRA adapter)",
        "score_semantics": "mean cross-entropy per supervised assistant token; perplexity = exp(loss)",  # noqa: E501
        "decoding_rule": DECODING_RULE,
        "sample_kind": sample_kind,
        "n_train": int(n_train),
        "n_validation": int(n_validation),
        "baselines": [],
        "probes": [dict(p) for p in probes] if probes else [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    needs = (
        "a held-out evaluation set from the target task with a rubric or human ratings (and a "
        "pre-adaptation baseline scored the same way) for any task-quality claim; validation loss "
        "and perplexity only show that the adapter fits the format of the training distribution"
    )
    validation_loss = None if metrics is None else metrics.get("validationLoss")
    if validation_loss is None:
        return {
            **base,
            "metrics": [],
            "verdict": "not-measurable",
            "reason": "no labelled held-out split was scored",
            "needs": needs,
        }
    entries = []
    for key in ("trainLoss", "validationLoss", "testLoss", "validationPerplexity"):
        value = metrics.get(key)
        if value is not None:
            entries.append(
                {
                    "id": key,
                    "value": float(value),
                    "units": "nats per supervised token" if key.endswith("Loss") else "tokens",
                    "estimation": "single seeded run on a manufactured validation split; no dispersion estimate",  # noqa: E501
                }
            )
    return {
        **base,
        "metrics": entries,
        "verdict": "sample-sanity",
        "reason": (
            f"optimisation metrics on {n_validation} held-out row(s) manufactured from the "
            "training source; they measure format fit, not task quality"
        ),
        "needs": needs,
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _preload_nvidia_libs() -> None:
    """Preload bundled pip nvidia runtime libraries so bitsandbytes resolves libnvJitLink etc."""
    import ctypes
    import os
    import sys
    from pathlib import Path

    for site_pkg in sys.path:
        nvidia_dir = Path(site_pkg) / "nvidia"
        if nvidia_dir.is_dir():
            libs = list(nvidia_dir.glob("*/lib"))
            if libs:
                lib_paths = [str(p) for p in libs]
                existing = os.environ.get("LD_LIBRARY_PATH", "")
                os.environ["LD_LIBRARY_PATH"] = ":".join(lib_paths) + (":" + existing if existing else "")
            for so in nvidia_dir.rglob("lib*.so*"):
                try:
                    ctypes.CDLL(str(so), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
                except Exception:
                    pass


@dataclass
class LanguageModelPipeline:
    """Tokenizer + base causal LM loaded from a digest-verified snapshot.

    ``model`` is the base ``AutoModelForCausalLM`` (4-bit ``nf4`` when a CUDA device is present
    and ``quantize_4bit`` is not False); the tutorial wraps it with PEFT for training and reload.
    ``generate`` and ``evaluate_loss`` accept a different model object (the PEFT-wrapped one) so the
    same helpers serve base, adapted and reloaded models."""

    tokenizer: Any
    model: Any
    device: str = "cpu"
    compute_dtype: str = "float32"
    quantized: bool = False
    source: str = "local-snapshot"
    weights_dir: str = ""
    load_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
        *,
        quantize_4bit: bool | None = None,
    ) -> LanguageModelPipeline:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        stage_missing_files(root, allow_download=allow_download)
        verify_snapshot(root)
        tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)  # noqa: E501
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if not tokenizer.chat_template:
            raise ValueError("This tokenizer ships no chat template; the pipeline needs one to render turns")  # noqa: E501
        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        on_cuda = resolved_device.startswith("cuda")
        quantized = on_cuda if quantize_4bit is None else (quantize_4bit and on_cuda)
        compute_dtype = (torch.bfloat16 if on_cuda and torch.cuda.is_bf16_supported() else torch.float16) if on_cuda else torch.float32  # noqa: E501
        kwargs: dict[str, Any] = {"trust_remote_code": False, "local_files_only": True, "dtype": compute_dtype, "attn_implementation": "sdpa"}  # noqa: E501
        if on_cuda:
            kwargs["device_map"] = {"": 0}
        if quantized:
            _preload_nvidia_libs()
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        model = AutoModelForCausalLM.from_pretrained(str(root), **kwargs)
        if not on_cuda:
            model = model.to(resolved_device)
        model.eval()
        return cls(tokenizer, model, resolved_device, str(compute_dtype).replace("torch.", ""), quantized, "local-snapshot", str(root), kwargs)  # noqa: E501

    def reload_base(self) -> Any:
        """Load the base model again from the same verified snapshot (fresh reload proof)."""
        from transformers import AutoModelForCausalLM

        if self.quantized:
            _preload_nvidia_libs()
        model = AutoModelForCausalLM.from_pretrained(self.weights_dir, **self.load_kwargs)
        if not self.device.startswith("cuda"):
            model = model.to(self.device)
        return model.eval()

    # -- data -------------------------------------------------------------------------------

    def render_chat(self, messages: Sequence[Mapping[str, str]], add_generation_prompt: bool = False) -> str:  # noqa: E501
        return render_chat(self.tokenizer, messages, add_generation_prompt)

    def token_length(self, record: Mapping[str, Any]) -> int:
        text = self.render_chat(record["messages"])
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def prompt_token_length(self, messages: Sequence[Mapping[str, str]]) -> int:
        text = self.render_chat(messages, add_generation_prompt=True)
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def prepare_splits(
        self,
        splits: Mapping[str, Sequence[Mapping[str, Any]]],
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    ) -> dict[str, list[tuple[list[int], list[int]]]]:
        """Validate the splits exactly as ``validate_inputs`` does, then mask every example."""
        _check_splits(splits, max_sequence_length, self.token_length)
        return {
            name: [build_masked_example(self.tokenizer, r, max_sequence_length) for r in records]
            for name, records in splits.items()
        }

    # -- inference ----------------------------------------------------------------------------

    def _tensor_device(self, model: Any) -> Any:
        try:
            return next(model.parameters()).device
        except StopIteration:  # pragma: no cover - a model without parameters
            return self.device

    def generate(self, prompt: str | Sequence[Mapping[str, str]], *, model: Any = None, max_new_tokens: int = 96, **decoding: Any) -> str:  # noqa: E501
        """Eval-mode generation (greedy unless ``decoding`` says otherwise); restores the model's
        training state afterwards so it is safe to call mid-training."""
        import torch

        model = self.model if model is None else model
        messages = _check_prompt(prompt, max_new_tokens)
        was_training = getattr(model, "training", False)
        model.eval()
        previous_use_cache = getattr(model.config, "use_cache", True)
        model.config.use_cache = True
        prompt_text = self.render_chat(messages, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(self._tensor_device(model))  # noqa: E501
        settings = {"do_sample": False, **decoding}
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.pad_token_id, **settings)  # noqa: E501
        model.config.use_cache = previous_use_cache
        if was_training:
            model.train()
        return self.tokenizer.decode(output_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()  # noqa: E501

    def to_batch(self, example: tuple[Sequence[int], Sequence[int]], model: Any = None) -> dict[str, Any]:  # noqa: E501
        import torch

        device = self._tensor_device(self.model if model is None else model)
        input_ids, labels = example
        return {
            "input_ids": torch.tensor([list(input_ids)], device=device),
            "labels": torch.tensor([list(labels)], device=device),
            "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long, device=device),
        }

    def evaluate_loss(self, examples: Sequence[tuple[Sequence[int], Sequence[int]]], *, model: Any = None) -> float | None:  # noqa: E501
        """Mean cross-entropy per supervised token over a split, in eval mode with gradients off."""
        import torch

        model = self.model if model is None else model
        if not examples:
            return None
        was_training = getattr(model, "training", False)
        model.eval()
        total_loss, total_tokens = 0.0, 0
        with torch.inference_mode():
            for example in examples:
                batch = self.to_batch(example, model)
                count = supervised_token_count(list(example[1]))
                total_loss += float(model(**batch).loss.item()) * count
                total_tokens += count
        if was_training:
            model.train()
        return total_loss / total_tokens if total_tokens else None

    # -- artifact -------------------------------------------------------------------------

    def export_adapter_bundle(self, peft_model: Any, bundle_dir: str | Path, *, metrics: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, Any]:  # noqa: E501
        """Write the adapter, the tokenizer (with its chat template), metrics, provenance and the
        artifact manifest. The bundle contains no training rows and no base weights."""
        bundle = Path(bundle_dir)
        shutil.rmtree(bundle, ignore_errors=True)
        bundle.mkdir(parents=True)
        peft_model.save_pretrained(str(bundle), safe_serialization=True)
        self.tokenizer.save_pretrained(str(bundle / "tokenizer"))
        full_provenance = {
            "artifactFormat": ARTIFACT_FORMAT,
            "artifactFormatVersion": ARTIFACT_FORMAT_VERSION,
            "baseModel": MODEL_ID,
            "baseModelRevision": MODEL_REVISION,
            "baseModelLicense": MODEL_LICENSE,
            "modelKey": MODEL_KEY,
            "trustRemoteCode": False,
            "quantized": self.quantized,
            **dict(provenance),
        }
        (bundle / "metrics.json").write_text(json.dumps(dict(metrics), indent=2), encoding="utf-8")
        (bundle / "provenance.json").write_text(json.dumps(full_provenance, indent=2), encoding="utf-8")  # noqa: E501
        (bundle / "MODEL_CARD.md").write_text(
            f"# PEFT adapter for {MODEL_ID}\n\nBase revision: `{MODEL_REVISION}`. "
            f"Dataset digest: `{full_provenance.get('datasetDigest', 'unknown')}`.\n"
            "Optimization metrics in metrics.json are not task-quality evidence.\n",
            encoding="utf-8",
        )
        return write_artifact_manifest(bundle)

    def load_adapter(self, bundle_dir: str | Path, *, base_model: Any = None) -> Any:
        """Verify a bundle, then attach its adapter to a base model in eval mode (AINF5: the base
        weights come from the digest-verified snapshot, never from the bundle or the network)."""
        from peft import PeftModel
        from transformers import AutoTokenizer

        bundle = Path(bundle_dir).resolve()
        verify_artifact_bundle(bundle)
        base = self.model if base_model is None else base_model
        self.tokenizer = AutoTokenizer.from_pretrained(str(bundle / "tokenizer"), local_files_only=True, trust_remote_code=False)  # noqa: E501
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return PeftModel.from_pretrained(base, str(bundle), is_trainable=False).eval()
