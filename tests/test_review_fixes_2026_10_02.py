"""Regression tests for the 2026-10-02 Notebook Review Framework v1 findings (LMA-B1, LMA-m2..m7 on the
artifact-inference companion; LMF-M2..M4, LMF-m1, m3..m6 on the fine-tuning notebook), recorded in
docs/reviews/2026-10-02-notebook-review/*_Fixes.md.

Every test needs only CI's dependencies: the notebooks' own cell sources are executed with stand-ins (no model, no
torch, no network). Stand-in model objects are labelled as such; they are not pretrained-inference evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import ast
import contextlib
import functools
import io
import json
import os
import re
import shutil
import sys
import types
import zipfile
from pathlib import Path

import pytest

from lmpipeline.pipeline import (
    BASELINE_METRIC_IDS,
    INFERENCE_SCORE_SEMANTICS,
    INFERENCE_TASK,
    MODEL_ID,
    MODEL_REVISION,
    canonical,
    evaluation_report,
    extract_zip_safely,
    fingerprint,
    manufacture_validation,
    perplexity,
    sha256_of_file,
    verify_artifact_bundle,
    write_artifact_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
E2E, ART = "language_model_finetuning_colab", "language_model_artifact_inference_colab"


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


def _standin_bundle(root: Path) -> Path:
    """A stand-in adapter bundle with the module's manifest format (NOT an E2E-trained bundle)."""
    bundle = root / "bundle"
    (bundle / "tokenizer").mkdir(parents=True)
    (bundle / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 16}), encoding="utf-8")
    (bundle / "adapter_model.safetensors").write_bytes(bytes(range(64)))
    (bundle / "tokenizer" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (bundle / "provenance.json").write_text(
        json.dumps({"artifactFormat": "peft_adapter", "baseModel": MODEL_ID, "baseModelRevision": MODEL_REVISION, "trustRemoteCode": False}),
        encoding="utf-8",
    )
    write_artifact_manifest(bundle)
    return bundle


def _zip_of(bundle: Path, archive: Path, prefix: str = "adapter-bundle/") -> Path:
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as handle:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                handle.write(path, prefix + path.relative_to(bundle).as_posix())
    return archive


def _run_section_4(monkeypatch, tmp_path, **fields) -> tuple[dict, str]:
    """Execute the companion's Section 4 cell up to (not including) verify_artifact_bundle; returns (namespace, stdout)."""
    source = _set_param(_cell(ART, "SAMPLE_ARTIFACT = {"), **{k: v for k, v in fields.items() if k != "SAMPLE_ARTIFACT"})
    if "SAMPLE_ARTIFACT" in fields:
        pinned = re.findall(r"^SAMPLE_ARTIFACT = \{.*\}$", source, re.M)
        assert len(pinned) == 1
        source = source.replace(pinned[0], f"SAMPLE_ARTIFACT = {fields['SAMPLE_ARTIFACT']!r}")
    block = source[: source.index("artifact_manifest, provenance = verify_artifact_bundle")]
    monkeypatch.chdir(tmp_path)
    ns = {"os": os, "Path": Path, "ARTIFACT_MANIFEST_NAME": "artifact-manifest.json", "sha256_of_file": sha256_of_file, "extract_zip_safely": extract_zip_safely}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(block, "<section 4>", "exec"), ns)
    return ns, out.getvalue()


# --- LMA-B1: the default path reads the pinned sample artifact; no upload, no google.colab ---------------------------


def test_lma_b1_default_path_downloads_and_digest_checks_the_pinned_sample_artifact(monkeypatch, tmp_path):
    _no_colab(monkeypatch)
    archive = _zip_of(_standin_bundle(tmp_path / "src"), tmp_path / "sample.zip")
    digest = sha256_of_file(archive)
    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=0: open(archive, "rb"))
    sample = {"url": "https://example.invalid/sample.zip", "sha256": digest, "producer": {"notebook_blob": "stand-in"}}
    run = tmp_path / "run"
    run.mkdir()
    ns, out = _run_section_4(monkeypatch, run, SAMPLE_ARTIFACT=sample)
    assert ns["artifact_source"] == "sample artifact: https://example.invalid/sample.zip" and ns["archive_sha"] == digest
    assert ns["bundle_dir"].name == "adapter-bundle" and (ns["bundle_dir"] / "artifact-manifest.json").is_file()
    assert "https://example.invalid/sample.zip" in out and digest in out
    # A second run reuses the downloaded archive (its digest still matches) without downloading again.
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no second download")))
    ns2, _ = _run_section_4(monkeypatch, run, SAMPLE_ARTIFACT=sample)
    assert ns2["archive_sha"] == digest


def test_lma_b1_a_wrong_pinned_digest_fails_before_extraction(monkeypatch, tmp_path):
    _no_colab(monkeypatch)
    archive = _zip_of(_standin_bundle(tmp_path / "src"), tmp_path / "sample.zip")
    wrong = ("0" if sha256_of_file(archive)[0] != "0" else "1") + sha256_of_file(archive)[1:]
    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=0: open(archive, "rb"))
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(ValueError, match="Whole-ZIP SHA-256 mismatch"):
        _run_section_4(monkeypatch, run, SAMPLE_ARTIFACT={"url": "https://example.invalid/s.zip", "sha256": wrong, "producer": {}})
    assert not (run / "work" / "external-artifact").exists(), "nothing may be extracted after a digest mismatch"


def test_lma_b1_empty_slot_falls_back_to_a_same_runtime_e2e_bundle_or_stops_with_a_named_message(monkeypatch, tmp_path):
    _no_colab(monkeypatch)
    run = tmp_path / "run"
    run.mkdir()
    empty = {"url": "", "sha256": "", "producer": {}}
    with pytest.raises(RuntimeError, match="No trusted sample artifact is pinned .*USE_OWN_ARTIFACT = True and ARTIFACT_ZIP_PATH"):
        _run_section_4(monkeypatch, run, SAMPLE_ARTIFACT=empty)
    _zip_of(_standin_bundle(tmp_path / "src"), run / "outputs" / "language_model_finetuning_adapter_bundle.zip")
    ns, out = _run_section_4(monkeypatch, run, SAMPLE_ARTIFACT=empty)
    assert ns["artifact_source"].startswith("same-runtime E2E output:") and "SAMPLE_ARTIFACT is empty" in out


def test_lma_b1_the_slot_is_pinned_to_the_recorded_e2e_bundle_release_asset():
    """The pinned sample bundle: release sample-bundle-v1, produced by the recorded 2026-10-07 Colab T4 E2E run."""
    source = _cell(ART, "SAMPLE_ARTIFACT = {")
    slot = ast.literal_eval(re.search(r"^SAMPLE_ARTIFACT = (\{.*\})$", source, re.M).group(1))
    assert slot["url"] == "https://github.com/kurtvalcorza/language-model-pipeline/releases/download/sample-bundle-v1/language_model_finetuning_adapter_bundle.zip"
    assert re.fullmatch(r"[0-9a-f]{64}", slot["sha256"]) and slot["sha256"] == "f99348aab5537bbccc26a0283f0fd6e714c0e3b58d27beb8c3d2586951ffaa35"
    producer = slot["producer"]
    assert producer["notebook"] == "tutorials/language_model_finetuning_colab.ipynb" and len(producer["commit"]) == 40 and len(producer["notebook_blob"]) == 40
    evidence = ROOT / producer["evidence"]
    assert evidence.is_dir() and any(evidence.glob(f"*_{producer['commit'][:7]}_colab-cli-t4_output.ipynb"))
    assert slot["sha256"] in (evidence / next(evidence.glob("*_output.ipynb")).name).read_text(encoding="utf-8")
    assert "still empty" not in _markdown(ART) and "not pinned in this revision" not in source


def test_lma_b1_the_default_path_never_imports_google_colab_and_the_gate_is_off():
    source = _cell(ART, "SAMPLE_ARTIFACT = {")
    assert "USE_OWN_ARTIFACT = False  # @param" in source
    own_branch = source[source.index("if USE_OWN_ARTIFACT:") : source.index("elif SAMPLE_ARTIFACT['url']:")]
    assert "from google.colab import files" in own_branch
    assert source.count("from google.colab import files") == 1, "google.colab is imported only inside the opt-in branch"
    markdown = _markdown(ART)
    assert "Known NOTEBOOK_SPEC 2.0 gap" not in markdown
    assert "no upload dialog and no `google.colab` import on the default path" in markdown


# --- LMA-m3: a ZIP path runs the archive checks; a digest with a folder never passes silently -----------------------


def test_lma_m3_artifact_zip_path_runs_the_digest_and_archive_checks_without_google_colab(monkeypatch, tmp_path):
    _no_colab(monkeypatch)
    archive = _zip_of(_standin_bundle(tmp_path / "src"), tmp_path / "own.zip")
    digest = sha256_of_file(archive)
    run = tmp_path / "run"
    run.mkdir()
    ns, _ = _run_section_4(monkeypatch, run, USE_OWN_ARTIFACT=True, ARTIFACT_ZIP_PATH=str(archive), EXPECTED_ARTIFACT_ZIP_SHA256=digest.upper())
    assert ns["artifact_source"] == f"zip: {archive}" and ns["archive_sha"] == digest and ns["bundle_dir"].is_dir()
    with pytest.raises(ValueError, match="Whole-ZIP SHA-256 mismatch"):
        _run_section_4(monkeypatch, run, USE_OWN_ARTIFACT=True, ARTIFACT_ZIP_PATH=str(archive), EXPECTED_ARTIFACT_ZIP_SHA256="f" * 64)
    slip = tmp_path / "slip.zip"
    with zipfile.ZipFile(slip, "w") as handle:
        handle.writestr("../escape.txt", "x")
    with pytest.raises(ValueError, match="Unsafe archive path"):
        _run_section_4(monkeypatch, run, USE_OWN_ARTIFACT=True, ARTIFACT_ZIP_PATH=str(slip))
    assert not (tmp_path / "escape.txt").exists() and not (run / "escape.txt").exists()
    with pytest.raises(FileNotFoundError, match="ARTIFACT_ZIP_PATH .* is not a file"):
        _run_section_4(monkeypatch, run, USE_OWN_ARTIFACT=True, ARTIFACT_ZIP_PATH=str(tmp_path / "nope.zip"))


def test_lma_m3_a_digest_paired_with_a_folder_is_refused_and_the_folder_route_says_what_it_skips(monkeypatch, tmp_path):
    _no_colab(monkeypatch)
    bundle = _standin_bundle(tmp_path / "src")
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(ValueError, match="EXPECTED_ARTIFACT_ZIP_SHA256 is set but ARTIFACT_DIR names an unpacked folder"):
        _run_section_4(monkeypatch, run, USE_OWN_ARTIFACT=True, ARTIFACT_DIR=str(bundle), EXPECTED_ARTIFACT_ZIP_SHA256="a" * 64)
    ns, out = _run_section_4(monkeypatch, run, USE_OWN_ARTIFACT=True, ARTIFACT_DIR=str(bundle))
    assert ns["bundle_dir"] == bundle and "archive checks (safe extraction, whole-ZIP digest) do not apply" in out
    assert verify_artifact_bundle(ns["bundle_dir"])[0]["totalBytes"] > 0


# --- LMA-m2: adapter activity is measured and recorded, not assumed ---------------------------------------------------


class _Scores:
    """Stand-in for a next-token logit vector (a list of floats with the few tensor methods the cell uses)."""

    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, _index):  # logits[0, -1]
        return self

    def float(self):
        return self

    def __sub__(self, other):
        return _Scores(a - b for a, b in zip(self.values, other.values, strict=True))

    def abs(self):
        return _Scores(abs(v) for v in self.values)

    def max(self):
        return self

    def item(self):
        return max(self.values)


def _activity_namespace(delta: float, differ: list[bool]) -> tuple[dict, str]:
    """Execute the Section 7 readout with a stand-in model whose adapter shifts scores by `delta`."""
    source = _cell(ART, "'next_token_logit_max_abs_delta'")
    block = source[source.index("def base_and_adapted(prompt):") : source.index("print(ADAPTER_ACTIVITY)") + len("print(ADAPTER_ACTIVITY)")]
    state = {"adapter_on": True, "calls": 0}

    @contextlib.contextmanager
    def disable_adapter():
        state["adapter_on"] = False
        try:
            yield
        finally:
            state["adapter_on"] = True

    def call(**_inputs):
        base = [0.5, -1.0, 2.0]
        return types.SimpleNamespace(logits=_Scores([v + (delta if state["adapter_on"] else 0.0) for v in base]))

    model = types.SimpleNamespace(disable_adapter=disable_adapter, parameters=lambda: iter([types.SimpleNamespace(device="cpu")]))
    model.__call__ = call
    model_obj = type("Model", (), {"__call__": staticmethod(call), "disable_adapter": staticmethod(disable_adapter), "parameters": staticmethod(model.parameters)})()
    prompts = ["p0", "p1"]

    def generate(prompt, *, model=None, max_new_tokens=0, **_):
        i = prompts.index(prompt)
        return f"adapted-{i}" if state["adapter_on"] and differ[i] else f"base-{i}"

    tokenizer = lambda text, **_: types.SimpleNamespace(to=lambda device: {})  # noqa: E731
    pipe = types.SimpleNamespace(render_chat=lambda messages, add_generation_prompt=False: messages[0]["content"], tokenizer=tokenizer, generate=generate)
    torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
    ns = {"pipe": pipe, "model": model_obj, "torch": torch, "PROMPTS": prompts, "MAX_NEW_TOKENS": 8, "LORA_B_MAX_ABS": 0.25}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(block, "<section 7 readout>", "exec"), ns)
    return ns, out.getvalue()


def test_lma_m2_readout_records_the_logit_delta_and_which_answers_differ():
    ns, _ = _activity_namespace(0.75, [True, False])
    activity = ns["ADAPTER_ACTIVITY"]
    assert activity["next_token_logit_max_abs_delta"] == pytest.approx(0.75)
    assert activity["answers_differ"] == [True, False] and [r["answers_differ"] for r in ns["rows"]] == [True, False]
    assert activity["verdict"].startswith("active:") and "differ on 1 of 2 prompts" in activity["verdict"]
    assert activity["lora_b_max_abs"] == 0.25


def test_lma_m2_a_tiny_b_adapter_that_moves_nothing_is_reported_inactive():
    """The review's P11 case: a ~1e-9 LoRA-B passes the non-zero check, yet changes no score and no answer."""
    ns, out = _activity_namespace(0.0, [False, False])
    activity = ns["ADAPTER_ACTIVITY"]
    assert activity["verdict"].startswith("inactive:") and activity["answers_differ"] == [False, False]
    assert activity["next_token_logit_max_abs_delta"] == 0.0 and "inactive" in out


def test_lma_m2_interpretation_claims_only_what_is_checked():
    markdown = _markdown(ART)
    assert "changes the model's answers when switched on" not in markdown
    assert "Whether the adapter *changed* anything is not assumed" in markdown
    source = _cell(ART, "'next_token_logit_max_abs_delta'")
    assert "'adapter_activity': ADAPTER_ACTIVITY" in source and "'answers_differ': base_answer != adapted_answer" in source


# --- LMA-m4: the companion's report describes inference, not fine-tuning -----------------------------------------------


def test_lma_m4_evaluation_report_takes_task_and_score_semantics_and_the_companion_passes_them():
    report = evaluation_report(None, sample_kind="sample", probes=[], task=INFERENCE_TASK, score_semantics=INFERENCE_SCORE_SEMANTICS)
    assert report["verdict"] == "not-measurable" and report["task"] == INFERENCE_TASK
    assert "cross-entropy" not in report["score_semantics"] and "no loss or score is computed" in report["score_semantics"]
    assert "adapter-attached generation" in report["task"] and report["sample_kind"] == "sample"
    default = evaluation_report(None)
    assert "fine-tuning" in default["task"] and "cross-entropy" in default["score_semantics"]
    source = _cell(ART, "'next_token_logit_max_abs_delta'")
    assert "sample_kind='sample+BYOD' if CUSTOM_PROMPT.strip() else 'sample'" in source
    assert "task=INFERENCE_TASK, score_semantics=INFERENCE_SCORE_SEMANTICS" in source
    assert "sample_kind='BYOD'" not in source


# --- LMA-m5: seeded sampling, recorded seeds, no "none should loop" -------------------------------------------------------


def test_lma_m5_sampled_generation_is_seeded_and_the_seeds_are_recorded():
    source = _cell(ART, "'next_token_logit_max_abs_delta'")
    block = source[source.index("SAMPLING = {") : source.index("report = evaluation_report(")]
    seeds: list[int] = []
    torch = types.SimpleNamespace(manual_seed=seeds.append)
    pipe = types.SimpleNamespace(generate=lambda prompt, **kw: f"draw-{seeds[-1]}")
    ns = {"torch": torch, "pipe": pipe, "PROMPTS": ["p"], "model": None, "MAX_NEW_TOKENS": 8, "SEED": 42}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(block, "<sampling>", "exec"), ns)
    assert seeds == [42, 43] and ns["sampling_seeds"] == [42, 43] and ns["sampled"] == ["draw-42", "draw-43"]
    assert "'seeds': sampling_seeds" in source and "SEED = 42  # @param" in source
    markdown = _markdown(ART)
    assert "none should loop" not in markdown and "note any that remain" in markdown


# --- LMA-m6: infrastructure labelled with what matters; a runnable, gated-off exercise ------------------------------------


@pytest.mark.parametrize("name", [E2E, ART])
def test_lma_m6_carrier_cell_says_which_functions_matter(name):
    assert "**What matters for this notebook.**" in _markdown(name)
    assert "> **Infrastructure.** You may run this section without studying its implementation" in _markdown(name)


def test_lma_m6_tamper_exercise_refuses_a_one_bit_change_and_is_off_by_default(tmp_path, monkeypatch):
    source = _cell(ART, "RUN_TAMPER_EXERCISE = False  # @param")
    bundle = _standin_bundle(tmp_path / "src")
    manifest, _ = verify_artifact_bundle(bundle)
    monkeypatch.chdir(tmp_path)
    Path("work").mkdir()
    ns = {"Path": Path, "shutil": shutil, "verify_artifact_bundle": verify_artifact_bundle, "bundle_dir": bundle, "artifact_manifest": manifest}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(source, "<exercise off>", "exec"), ns)
    assert "RUN_TAMPER_EXERCISE is off" in out.getvalue()
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(_set_param(source, RUN_TAMPER_EXERCISE=True), "<exercise on>", "exec"), ns)
    assert "Refused, as it should be: SHA-256 mismatch: adapter_model.safetensors" in out.getvalue()
    assert "'bundle_in_use_still_verifies': True" in out.getvalue()
    assert verify_artifact_bundle(bundle)[0] == manifest, "the bundle in use is untouched"


def test_lma_m6_objectives_are_observable_and_the_default_path_has_no_extra_cells():
    markdown = _markdown(ART)
    assert "**Learning objectives:** by the end you can:" in markdown
    gates = [c for c in _code_cells(ART) if "RUN_TAMPER_EXERCISE = False  # @param" in c]
    assert len(gates) == 1 and gates[0].lstrip().startswith("RUN_TAMPER_EXERCISE = False")


# --- LMA-m7 / LMF-m5: one spec version everywhere, release records that agree -------------------------------------------


def test_lma_m7_spec_version_is_2_2_in_generator_validator_metadata_and_records():
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import build_notebook
        import validate_release_assets
    finally:
        sys.path.pop(0)
    assert build_notebook.NOTEBOOK_SPEC == validate_release_assets.NOTEBOOK_SPEC == "2.2"
    for name in (E2E, ART):
        assert _nb(name)["metadata"]["dimer"]["notebook_spec"] == "2.2"
        assert "'notebook_spec': '2.2'" in _cell(name, "NOTEBOOK_SOURCE = {")
        assert "DIMER Notebook Specification 2.2 — **standalone** (§4)" in _markdown(name)
    for rel in ("docs/release-verification.md", "tutorials/RELEASE_VERIFICATION.md", "tutorials/README.md", "README.md"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "Specification 2.2" in text, rel
        assert not re.search(r"Notebook Specification (1\.0|1\.1|2\.0)\b", text), rel
        assert "spec `1.1`" not in text, rel


def test_lmf_m5_release_records_tell_one_story_about_the_kaggle_run_and_the_current_blobs():
    docs = (ROOT / "docs/release-verification.md").read_text(encoding="utf-8")
    record = (ROOT / "tutorials/RELEASE_VERIFICATION.md").read_text(encoding="utf-8")
    registry = (ROOT / "tutorials/README.md").read_text(encoding="utf-8")
    for text in (docs, record):
        assert "469911578d05" in text and "restart" in text and "shim" in text
        assert "No clean-runtime one-pass execution" in text or "no passing clean-run record" in text.lower()
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
    new_prompts = _cell(E2E, "CUSTOM_PROMPT = ''  # @param")
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
    assert "non-thinking" not in _markdown(ART)


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
    assert "### Activity (optional): greedy versus sampled decoding" in _markdown(E2E)


@pytest.mark.parametrize("name", [E2E, ART])
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


def test_st1_allows_only_the_pinned_sample_bundle_asset_url():
    """The ST1 guards exempt exactly the pinned release-asset download; any other github.com/kurtvalcorza URL still fails."""
    import importlib.util as _ilu

    spec = _ilu.spec_from_file_location("_validator", ROOT / "tools" / "validate_release_assets.py")
    validator = _ilu.module_from_spec(spec)
    spec.loader.exec_module(validator)
    st1 = dict(validator.FORBIDDEN_PATTERNS)["repository clone (ST1)"]
    asset = "'https://github.com/kurtvalcorza/language-model-pipeline/releases/download/sample-bundle-v1/language_model_finetuning_adapter_bundle.zip'"
    assert not st1.search(asset)
    for bad in ("'https://github.com/kurtvalcorza/language-model-pipeline.git'", "'https://github.com/kurtvalcorza/language-model-pipeline/archive/main.zip'", "'https://github.com/kurtvalcorza/language-model-pipeline/releases/download/sample-bundle-v1/x.py'"):
        assert st1.search(bad), bad
