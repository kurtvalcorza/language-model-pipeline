"""Regression tests for the 2026-10-02 Notebook Review Framework v1 findings (LMF-M2..M4, LMF-m1, m3..m6 on the
fine-tuning notebook; LMA-m4, m6, m7 where they still apply), recorded in docs/reviews/2026-10-02-notebook-review/*_Fixes.md.

The artifact-inference companion was retired under NOTEBOOK_SPEC 3.0 (KND11, §32.2). Its LMA-B1/m2/m3/m5/m6 tests
went with it; Section 11 of the E2E notebook, which replaced it, is covered by tests/test_external_artifact_section.py.

Every test needs only CI's dependencies: the notebooks' own cell sources are executed with stand-ins (no model, no
torch, no network). Stand-in model objects are labelled as such; they are not pretrained-inference evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import contextlib
import functools
import io
import json
import os
import re
import shutil
import sys
import types
from pathlib import Path

import pytest

from lmpipeline.pipeline import (
    BASELINE_METRIC_IDS,
    INFERENCE_SCORE_SEMANTICS,
    INFERENCE_TASK,
    canonical,
    evaluation_report,
    extract_zip_safely,
    fingerprint,
    manufacture_validation,
    perplexity,
)

ROOT = Path(__file__).resolve().parents[1]
E2E = "language_model_finetuning_colab"


@functools.cache
def _nb(name: str) -> dict:
    return json.loads((ROOT / "tutorials" / f"{name}.ipynb").read_text(encoding="utf-8"))


def _code_cells(name: str) -> list[str]:
    return [c["source"] for c in _nb(name)["cells"] if c["cell_type"] == "code"]


def _markdown(name: str) -> str:
    return "\n".join(c["source"] for c in _nb(name)["cells"] if c["cell_type"] == "markdown")


def _cell(name: str, marker: str) -> str:
    found = [c for c in _code_cells(name) if marker in c]
    assert len(found) == 1, f"expected one code cell containing {marker!r}, found {len(found)}"
    return found[0]


def _set_param(source: str, **fields) -> str:
    """Rewrite `# @param` lines the way the Colab form does (the review's probes do the same)."""
    for name, value in fields.items():
        match = re.search(rf"^{name} = .*?(  # @param.*)$", source, re.M)
        assert match, name
        source = source.replace(match.group(0), f"{name} = {value!r}{match.group(1)}")
    return source


def _no_colab(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "google.colab", None)  # the import fails, as on Kaggle / Jupyter


# --- LMA-m4: the external-artifact section's report describes inference, not fine-tuning -----------------------------------------------


def test_lma_m4_evaluation_report_takes_task_and_score_semantics_and_section_11_passes_them():
    report = evaluation_report(None, sample_kind="sample", probes=[], task=INFERENCE_TASK, score_semantics=INFERENCE_SCORE_SEMANTICS)
    assert report["verdict"] == "not-measurable" and report["task"] == INFERENCE_TASK
    assert "cross-entropy" not in report["score_semantics"] and "no loss or score is computed" in report["score_semantics"]
    assert "adapter-attached generation" in report["task"] and report["sample_kind"] == "sample"
    default = evaluation_report(None)
    assert "fine-tuning" in default["task"] and "cross-entropy" in default["score_semantics"]
    source = _cell(E2E, "external_report = evaluation_report(")
    assert "sample_kind='sample+BYOD' if EXTERNAL_CUSTOM_PROMPT.strip() else 'sample'" in source
    assert "task=INFERENCE_TASK, score_semantics=INFERENCE_SCORE_SEMANTICS" in source
    assert "sample_kind='BYOD'" not in source


# --- LMA-m6: infrastructure labelled with what matters; a runnable, gated-off exercise ------------------------------------


@pytest.mark.parametrize("name", [E2E])
def test_lma_m6_carrier_cell_says_which_functions_matter(name):
    assert "**What matters for this notebook.**" in _markdown(name)
    assert "> **Infrastructure.** You may run this section without studying its implementation" in _markdown(name)


# --- LMA-m7 / LMF-m5: one spec version everywhere, release records that agree -------------------------------------------


def test_lma_m7_spec_version_is_3_0_in_template_validator_metadata_and_records():
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import build_notebook
        import validate_release_assets
    finally:
        sys.path.pop(0)
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import notebook_template
    finally:
        sys.path.pop(0)
    assert build_notebook.GENERATOR_VERSION == "build_notebook.py/2.3"
    assert notebook_template.TEMPLATE["notebook_spec"] == validate_release_assets.NOTEBOOK_SPEC == "3.0"
    for name in (E2E,):
        assert _nb(name)["metadata"]["dimer"]["notebook_spec"] == "3.0"
        assert "'notebook_spec': '3.0'" in _cell(name, "NOTEBOOK_SOURCE = {")
        assert "DIMER Notebook Specification 3.0 — **standalone** (§4)" in _markdown(name)
    for rel in ("docs/release-verification.md", "tutorials/RELEASE_VERIFICATION.md", "tutorials/README.md", "README.md"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "Specification 3.0" in text, rel
        assert not re.search(r"Notebook Specification (1\.0|1\.1|2\.0)\b", text), rel
        assert "spec `1.1`" not in text, rel


def test_lmf_m5_release_records_tell_one_story_about_the_kaggle_run_and_the_current_blobs():
    docs = (ROOT / "docs/release-verification.md").read_text(encoding="utf-8")
    record = (ROOT / "tutorials/RELEASE_VERIFICATION.md").read_text(encoding="utf-8")
    registry = (ROOT / "tutorials/README.md").read_text(encoding="utf-8")
    for text in (docs, record):
        assert "469911578d05" in text and "restart" in text and "shim" in text
        # Until 2026-10-07 neither blob had a one-pass record; now both carry their recorded Colab CLI runs.
        assert "No clean-runtime one-pass execution" in text or "no passing clean-run record" in text.lower() or ("8a2c6ad4886d" in text and "f3be29a5d95b" in text)
    assert "GITHUB_TOKEN" not in record and "qwen3-1.7b" not in record
    assert "verified — clean-runtime" not in registry and "needed a kernel restart" in registry
    assert "No clean-runtime execution of either standalone notebook has been recorded yet; clean GPU execution evidence" not in docs


# --- LMF-M2: the base model is scored on the validation split before adaptation -------------------------------------------


def test_lmf_m2_base_validation_loss_is_computed_before_lora_and_recorded_as_a_baseline():
    cells = _code_cells(E2E)
    baseline = next(i for i, c in enumerate(cells) if "BASE_VALIDATION_LOSS = pipe.evaluate_loss(MASKED['validation'])" in c)
    training = next(i for i, c in enumerate(cells) if "get_peft_model(prepare_model_for_kbit_training(pipe.model)" in c)
    assert baseline < training
    assert "'baseValidationLoss': BASE_VALIDATION_LOSS" in cells[training] and "validation loss base → adapted" in cells[training]
    metrics = {"trainLoss": 3.2, "validationLoss": 3.1, "validationPerplexity": perplexity(3.1), "baseValidationLoss": 3.4, "baseValidationPerplexity": perplexity(3.4)}
    report = evaluation_report(metrics, sample_kind="sample", n_train=96, n_validation=24)
    assert [b["id"] for b in report["baselines"]] == list(BASELINE_METRIC_IDS)
    base = report["baselines"][0]
    assert base["value"] == 3.4 and base["units"] == "nats per supervised token" and "before any adapter" in base["estimation"]
    assert {m["id"] for m in report["metrics"]} == {"trainLoss", "validationLoss", "validationPerplexity"}
    assert "pre-adaptation baseline" not in report["needs"]
    assert "pre-adaptation baseline" in evaluation_report({"validationLoss": 3.1})["needs"]
    assert "evaluates loss/perplexity base vs adapted" in _markdown(E2E) and "scores the unadapted base model on the validation split" in _markdown(E2E)


# --- LMF-M3: the reload check says what the artifact reproduces -------------------------------------------------------------


def _reload_check(in_memory: float, reloaded: float, same_opening: bool) -> tuple[dict, str]:
    source = _cell(E2E, "reloaded_validation_loss = pipe.evaluate_loss(MASKED['validation'], model=reloaded)")
    start = source.index("reloaded_validation_loss = pipe.evaluate_loss")
    end = source.index("print({'in_memory_adapted_opening'")
    block = source[start:end]
    pipe = types.SimpleNamespace(evaluate_loss=lambda examples, model=None: reloaded, compute_dtype="bfloat16")
    ns = {
        "pipe": pipe, "MASKED": {"validation": [1]}, "reloaded": object(), "validation_loss": in_memory, "RELOAD_LOSS_TOLERANCE": 0.05,
        "reloaded_opening": "same" if same_opening else "other", "expected_opening": "same", "perplexity": perplexity, "METRICS": {}, "REPLAY_TOKENS": 16,
    }
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(block, "<reload check>", "exec"), ns)
    return ns, out.getvalue()


def test_lmf_m3_reload_check_reports_reproduced_loss_and_identical_openings():
    ns, out = _reload_check(3.2611, 3.2700, True)
    check = ns["RELOAD_CHECK"]
    assert check["loss_reproduced_within_tolerance"] and check["greedy_openings_identical"]
    assert check["validation_loss_abs_delta"] == pytest.approx(0.0089)
    assert "PASS:" in out and ": reproduced." in out and "openings: identical" in out
    assert ns["METRICS"]["reloadedValidationLoss"] == 3.27


def test_lmf_m3_reload_check_says_not_reproduced_when_the_loss_moves_past_the_tolerance():
    ns, out = _reload_check(3.2611, 3.40, False)
    assert not ns["RELOAD_CHECK"]["loss_reproduced_within_tolerance"] and not ns["RELOAD_CHECK"]["greedy_openings_identical"]
    assert "NOT reproduced" in out and "openings: differ" in out
    source = _cell(E2E, "reloaded_validation_loss = pipe.evaluate_loss(MASKED['validation'], model=reloaded)")
    assert "'reload': {**RELOAD_CHECK" in source and "'nonQuantizedLayersTrainedIn'" in source
    assert "PASS: fresh base + adapter reload from the verified snapshot; adapter weights present and active')" not in source, "the old unqualified PASS line is gone"


# --- LMF-M4: a full run leaves the notebook re-runnable -----------------------------------------------------------------------


def test_lmf_m4_section_10_keeps_the_trained_model_and_reloads_into_a_separate_object():
    export = _cell(E2E, "bundle_manifest = pipe.export_adapter_bundle(")
    assert "del model" not in export and "pipe.model = None" not in export
    assert "reloaded = pipe.load_adapter(ADAPTER_DIR, base_model=pipe.reload_base())" in export
    new_prompts = _cell(E2E, "RUN_NEW_PROMPT_INFERENCE = True  # @param")
    assert "reloaded" not in new_prompts and "with model.disable_adapter():" in new_prompts
    # The frozen-base guard of the sweep still precedes the baseline and the training cell (a re-run from Section 4 down
    # never takes its baseline from, or trains on top of, an adapted model).
    assert _cell(E2E, "BASELINE_OUTPUTS = [pipe.generate(prompt) for prompt in PROMPTS]").index("frozen_base()") < _cell(E2E, "BASELINE_OUTPUTS = [pipe.generate(prompt) for prompt in PROMPTS]").index("BASELINE_OUTPUTS")
    assert "can be re-run on their own" in _markdown(E2E)


# --- LMF-m1: expectations match the recorded run ------------------------------------------------------------------------------


def test_lmf_m1_expectation_notes_match_the_recorded_run():
    markdown = _markdown(E2E)
    for stale in ("non-thinking", "under 1 %", "Well under 1 %", "float16 on a T4", "Qwen3-0.6B reference", "a fraction of one percent", "expect English or mixed-language replies"):
        assert stale not in markdown, stale
    assert "1.186 %" in markdown and "echo the Filipino prompt" in markdown and "`torch.cuda.is_bf16_supported()`" in markdown
    assert "train loss 3.266, validation loss 3.261, perplexity 26.1" in markdown and "after a kernel\nrestart" in markdown.replace("kernel restart", "kernel\nrestart")


# --- LMF-m3: every epoch is recorded; nothing promises selection ----------------------------------------------------------------


def test_lmf_m3_epoch_losses_are_recorded_and_selection_is_not_promised():
    training = _cell(E2E, "EPOCH_LOSSES = []")
    assert "EPOCH_LOSSES.append({'epoch': epoch, 'trainLoss': train_loss, 'validationLoss': validation_loss" in training
    assert "'epochs': EPOCH_LOSSES" in training
    export = _cell(E2E, "bundle_manifest = pipe.export_adapter_bundle(")
    assert "'exportedEpoch': EPOCHS, 'epochSelection': 'last epoch; no selection by validation loss'" in export
    markdown = _markdown(E2E)
    assert "lowest-validation-loss epoch is the one to keep" not in markdown and "keep the lowest epoch" not in markdown
    assert "exports the **last** epoch's adapter and selects nothing" in markdown


# --- LMF-m4: BYOD split order and the over-length policy -------------------------------------------------------------------------


def _byod(monkeypatch, tmp_path, path: str) -> dict:
    source = _cell(E2E, "BYOD_PATH = ''")
    block = source[source.index("BYOD_FILES = (") : source.index("else:\n    from datasets import load_dataset")]
    work = tmp_path / "work"
    work.mkdir()
    ns = {
        "USE_BYOD": True, "BYOD_PATH": path, "WORK_DIR": work, "Path": Path, "shutil": shutil, "json": json,
        "canonical": canonical, "extract_zip_safely": extract_zip_safely, "manufacture_validation": manufacture_validation, "fingerprint": fingerprint,
    }
    exec(compile(block, "<byod>", "exec"), ns)
    return ns


def test_lmf_m4_a_topic_sorted_byod_file_yields_a_mixed_validation_set(monkeypatch, tmp_path):
    _no_colab(monkeypatch)
    folder = tmp_path / "sorted"
    folder.mkdir()
    rows = [{"prompt": f"topic-A {i}", "completion": "ok"} for i in range(10)] + [{"prompt": f"topic-B {i}", "completion": "ok"} for i in range(10)]
    (folder / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    ns = _byod(monkeypatch, tmp_path, str(folder))
    held = {r["messages"][0]["content"].split()[0] for r in ns["SPLITS"]["validation"]}
    assert held == {"topic-A", "topic-B"}, "the file head (one topic) must not become the validation set"
    assert len(ns["SPLITS"]["validation"]) == 4 and len(ns["SPLITS"]["train"]) == 16
    markdown = _markdown(E2E)
    assert "same deterministic hash order as the sample" in markdown and "A BYOD row longer than `MAX_SEQUENCE_LENGTH` tokens is **not** set aside" in markdown


def test_lmf_m4_a_malformed_line_is_named_with_its_file_and_line(monkeypatch, tmp_path):
    _no_colab(monkeypatch)
    folder = tmp_path / "bad"
    folder.mkdir()
    good = '{"prompt": "a", "completion": "b"}'
    (folder / "train.jsonl").write_text("\n".join([good] * 4 + ['{"prompt": "a", completion: "b"}']) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"train\.jsonl line 5: not a messages"):
        _byod(monkeypatch, tmp_path, str(folder))


# --- LMF-m6: a runnable, gated-off decoding activity ----------------------------------------------------------------------------


def test_lmf_m6_decoding_activity_is_off_by_default_and_runs_with_a_stand_in():
    source = _cell(E2E, "RUN_DECODING_EXPERIMENT = False  # @param")
    seeds: list[int] = []

    def generate(prompt, *, model=None, **decoding):
        return "a b c d a b c d a b c d" if not decoding else f"draw {seeds[-1]} x y z"

    ns = {"torch": types.SimpleNamespace(manual_seed=seeds.append), "pipe": types.SimpleNamespace(generate=generate), "NEW_PROMPTS": ["q"], "model": None, "SEED": 7}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(source, "<activity off>", "exec"), ns)
    assert "RUN_DECODING_EXPERIMENT is off" in out.getvalue() and seeds == []
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(_set_param(source, RUN_DECODING_EXPERIMENT=True), "<activity on>", "exec"), ns)
    text = out.getvalue()
    assert seeds == [7, 8] and "greedy: longest repeated 4-word phrase occurs 3x" in text and "sampled (seed 8)" in text
    assert "'sampled_answers_identical': False" in text and ns["longest_repeat"]("no repeats here at all") == 1
    assert "## 12. Optional activity: greedy versus sampled decoding" in _markdown(E2E)


@pytest.mark.parametrize("name", [E2E])
@pytest.mark.parametrize("real_google", [False, True])
def test_worker_colab_stubs_have_specs(name: str, real_google: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Colab CLI T4 run of f2053aa: accelerate's find_spec("google.colab") raised `google.colab.__spec__ is None` on the
    isolated worker's spec-less stub (reference fix: chronos-2-forecasting-pipeline 33a2f53, CHR-M1)."""
    import importlib.util

    router = _cell(name, '_WORKER_SOURCE = r"""')
    worker = router[router.index('_WORKER_SOURCE = r"""') + len('_WORKER_SOURCE = r"""') :]
    worker = worker[: worker.index('"""')]
    start = worker.index('if os.environ.get("DIMER_KERNEL_IS_COLAB") == "1":')
    shim = worker[start : worker.index('_main = types.ModuleType("__main__")', start)]
    names = ("google", "google.colab", "google.colab.files")
    saved = {n: sys.modules[n] for n in names if n in sys.modules}
    fake_google = types.ModuleType("google")
    fake_google.__path__ = []
    try:
        for n in names:
            sys.modules.pop(n, None)
        # Both branches: no importable `google` (stub created) and an existing namespace package.
        sys.modules["google"] = fake_google if real_google else None
        monkeypatch.setenv("DIMER_KERNEL_IS_COLAB", "1")
        exec(compile(shim, "worker-colab-shim", "exec"), {"os": os, "sys": sys, "types": types, "_send": None, "_recv": None})
        for n in ("google.colab", "google.colab.files"):
            spec = importlib.util.find_spec(n)  # raised ValueError before the fix
            assert spec is not None and spec.name == n
        assert sys.modules["google.colab"].__path__ == [] and callable(sys.modules["google.colab.files"].upload)
        if not real_google:
            assert importlib.util.find_spec("google") is not None
    finally:
        for n in names:
            sys.modules.pop(n, None)
        sys.modules.update(saved)
