"""NOTEBOOK_SPEC 3.0 §19.1: the E2E notebook's Section 11, the external-artifact section that replaced the retired
artifact-inference companion (KND11, §32.2).

The section's own cell sources are executed against the notebook's carried module cell, with a stand-in pipeline
(no model, no torch, no network) and a locally built stand-in bundle served through a test-only override of
SAMPLE_ARTIFACT_URL / SAMPLE_ARTIFACT_SHA256. The stand-in bundle is NOT the pinned release asset and NOT an
E2E-trained adapter; these tests prove the section's control flow, refusals and outputs, not inference quality.
"""
# ruff: noqa: E501

from __future__ import annotations

import ast
import builtins
import contextlib
import functools
import io
import json
import os
import stat
import sys
import zipfile
from pathlib import Path

import pytest

from lmpipeline.pipeline import (
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_ARTIFACT_SHA256,
    SAMPLE_ARTIFACT_URL,
    SAMPLE_DATASETS,
    deterministic_zip,
    fetch_pinned_artifact,
    sha256_of_file,
    verify_artifact_bundle,
    write_artifact_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "language_model_finetuning_colab.ipynb"
RELEASE_URL = "https://github.com/kurtvalcorza/language-model-pipeline/releases/download/sample-bundle-v1/language_model_finetuning_adapter_bundle.zip"
RELEASE_SHA256 = "f99348aab5537bbccc26a0283f0fd6e714c0e3b58d27beb8c3d2586951ffaa35"
SECTION_HEADING = "## 11. Use an adapter bundle made elsewhere (external artifact)"
NEXT_HEADING = "## 12. Optional activity"
EXPECTED_OUTPUTS = {"input_manifest.json", "evaluation_report.json", "result.json", "generations.csv"}


@functools.cache
def _nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _index_of_heading(heading: str) -> int:
    found = [i for i, c in enumerate(_nb()["cells"]) if c["cell_type"] == "markdown" and _src(c).startswith(heading)]
    assert len(found) == 1, heading
    return found[0]


def _section_code() -> list[str]:
    cells = _nb()["cells"][_index_of_heading(SECTION_HEADING) : _index_of_heading(NEXT_HEADING)]
    code = [_src(c) for c in cells if c["cell_type"] == "code"]
    assert len(code) == 3
    return code


def _setup_code() -> list[str]:
    """Sections 1-3 as the isolated worker sees them: every code cell before Section 4 except the kernel bootstrap."""
    cells = _nb()["cells"][: _index_of_heading("## 4. ")]
    return [_src(c) for c in cells if c["cell_type"] == "code" and "# dimer: kernel cell" not in _src(c)]


def _module_cell() -> str:
    tagged = [_src(c) for c in _nb()["cells"] if c["cell_type"] == "code" and c.get("metadata", {}).get("dimer", {}).get("embedded_module")]
    assert len(tagged) == 1
    return tagged[0]


def _bound(sources: list[str]) -> set[str]:
    names: set[str] = set()
    for source in sources:
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                names.add(node.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                names.add(node.name)
            elif isinstance(node, ast.arg):
                names.add(node.arg)
    return names


# --- static: pin, setup-only names, outputs ------------------------------------------------------------------------------


def test_the_section_reads_the_pinned_release_asset_and_never_this_runs_export():
    assert (SAMPLE_ARTIFACT_URL, SAMPLE_ARTIFACT_SHA256) == (RELEASE_URL, RELEASE_SHA256)
    module = _module_cell()
    assert f'SAMPLE_ARTIFACT_URL = "{RELEASE_URL}"' in module and f'SAMPLE_ARTIFACT_SHA256 = "{RELEASE_SHA256}"' in module
    code = "\n".join(_section_code())
    assert "fetch_pinned_artifact(SAMPLE_ARTIFACT_URL, SAMPLE_ARTIFACT_SHA256, external_zip)" in code
    assert "open_artifact_zip(external_zip, SAMPLE_ARTIFACT_SHA256, EXTERNAL_WORK / 'bundle')" in code
    # XAS2 / §25.11: this run's export is never the external artifact.
    for own in ("language_model_finetuning_adapter_bundle", "adapter-bundle", "ARTIFACT_ZIP)", "ADAPTER_DIR", "E2E_BUNDLE"):
        assert own not in code, own
    # SART6: the location field defaults to the pinned asset, as a dropdown (no URL in any notebook cell but the module).
    assert "ARTIFACT_SOURCE = 'Pinned sample artifact'  # @param [\"Pinned sample artifact\"," in code
    assert "github.com" not in code


def test_section_cells_use_only_names_bound_by_the_setup_cells_or_by_the_section():
    section = _section_code()
    allowed = _bound(_setup_code()) | _bound(section) | set(dir(builtins))
    used = {n.id for s in section for n in ast.walk(ast.parse(s)) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    assert used - allowed == set()
    # In particular nothing from the data, training, evaluation or export stages (XAS5).
    for stage_name in ("model", "reloaded", "PROMPTS", "SEED", "METRICS", "SPLITS", "MASKED", "report", "DATA_SOURCE"):
        assert stage_name not in used, stage_name
    setup = _bound(_setup_code())
    for needed in ("pipe", "NOTEBOOK_SOURCE", "SAMPLE_ARTIFACT_URL", "verify_artifact_bundle", "fetch_pinned_artifact"):
        assert needed in setup, needed


def test_outputs_go_under_outputs_external_artifact_only():
    code = "\n".join(_section_code())
    assert "EXTERNAL_OUTPUTS = Path('outputs') / 'external_artifact'" in code
    opens = [line for line in code.splitlines() if "open(" in line and "'w'" in line]
    assert len(opens) == 4 and all("open(EXTERNAL_OUTPUTS / " in line for line in opens)
    assert {name for name in EXPECTED_OUTPUTS if f"EXTERNAL_OUTPUTS / '{name}'" in code} == EXPECTED_OUTPUTS


def test_section_order_follows_notebook_spec_3_0_section_7_1():
    headings = [line for c in _nb()["cells"] if c["cell_type"] == "markdown" for line in _src(c).splitlines() if line.startswith("## ")]
    order = ["## 10. Export", "## Interpretation and limits", "## Conclusion (your notes)", SECTION_HEADING, NEXT_HEADING, "## Troubleshooting", "## References"]
    positions = [next(i for i, h in enumerate(headings) if h.startswith(o)) for o in order]
    assert positions == sorted(positions) and headings[-1] == "## References"
    dimer = _nb()["metadata"]["dimer"]
    assert (dimer["notebook_spec"], dimer["notebook_kind"], dimer["notebook_profile"]) == ("3.0", "TUTORIAL", "E2E")


# --- executed: the section's cells against the carried module and a stand-in pipeline -----------------------------------


class _Param:
    def __init__(self, value: float) -> None:
        self.value = value

    def abs(self):
        return self

    def max(self):
        return self

    def item(self) -> float:
        return self.value


class _StandInModel:
    """Stand-in for the PEFT-wrapped model (NOT a model): answers depend only on whether the adapter is enabled."""

    def __init__(self) -> None:
        self.adapter_on = True

    def named_parameters(self):
        return [("base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight", _Param(0.008))]

    @contextlib.contextmanager
    def disable_adapter(self):
        self.adapter_on = False
        try:
            yield
        finally:
            self.adapter_on = True


class _StandInPipe:
    device, quantized = "cpu", False

    def __init__(self) -> None:
        self.reloads = 0
        self.loaded: list[Path] = []

    def reload_base(self):
        self.reloads += 1
        return "fresh-base"

    def load_adapter(self, bundle_dir, *, base_model=None):
        assert base_model == "fresh-base", "the external adapter must go on a fresh base, never the trained model"
        verify_artifact_bundle(bundle_dir)  # the real check, as LanguageModelPipeline.load_adapter runs it
        self.loaded.append(Path(bundle_dir))
        return _StandInModel()

    def generate(self, prompt, *, model=None, max_new_tokens=96):
        return ("ADAPTED: " if model.adapter_on else "BASE: ") + prompt[:24]

    def prompt_token_length(self, messages):
        return 20


def _standin_zip(tmp: Path, *, dataset: dict | None = None, revision: str = MODEL_REVISION) -> Path:
    bundle = tmp / "standin-bundle"
    (bundle / "tokenizer").mkdir(parents=True)
    (bundle / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 16, "target_modules": ["q_proj"]}), encoding="utf-8")
    (bundle / "adapter_model.safetensors").write_bytes(bytes(range(64)))
    (bundle / "tokenizer" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    default = SAMPLE_DATASETS["Sample: Filipino SFT"]
    provenance = {
        "artifactFormat": "peft_adapter", "baseModel": MODEL_ID, "baseModelRevision": revision, "trustRemoteCode": False,
        "dataset": dataset or {"source": default["dataset_id"], "revision": default["revision"], "license": default["license"]},
    }
    (bundle / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    write_artifact_manifest(bundle)
    deterministic_zip(bundle, tmp / "standin.zip")
    return tmp / "standin.zip"


def _run_section(monkeypatch, workdir: Path, archive: Path, *, pinned_sha: str | None = None, fields: dict | None = None) -> tuple[dict, str]:
    monkeypatch.chdir(workdir)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setitem(sys.modules, "google.colab", None)
    ns: dict = {"__name__": "__main__"}
    exec(compile(_module_cell(), "<carried module>", "exec"), ns)
    # Test-only override: serve the stand-in ZIP instead of the release asset.
    ns["SAMPLE_ARTIFACT_URL"] = archive.resolve().as_uri()
    ns["SAMPLE_ARTIFACT_SHA256"] = pinned_sha or sha256_of_file(archive)
    ns["pipe"] = _StandInPipe()
    ns["NOTEBOOK_SOURCE"] = {"repository": "language-model-pipeline", "repository_revision": "test"}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        for source in _section_code():
            for name, value in (fields or {}).items():
                lines = [line for line in source.splitlines() if line.startswith(f"{name} = ") and "# @param" in line]
                for line in lines:
                    source = source.replace(line, f"{name} = {value!r}  # @param {line.split('# @param', 1)[1]}")
            exec(compile(source, "<section 11>", "exec"), ns)
    return ns, out.getvalue()


def test_default_path_runs_end_to_end_on_a_stand_in_and_writes_the_four_outputs(monkeypatch, tmp_path):
    archive = _standin_zip(tmp_path)
    work = tmp_path / "run"
    work.mkdir()
    (work / "outputs").mkdir()
    (work / "outputs" / "language_model_finetuning_adapter_bundle.zip").write_bytes(b"this run's export: must never be read")
    ns, out = _run_section(monkeypatch, work, archive)
    outputs = work / "outputs" / "external_artifact"
    assert {p.name for p in outputs.iterdir()} == EXPECTED_OUTPUTS
    result = json.loads((outputs / "result.json").read_text(encoding="utf-8"))
    assert result["artifact"]["zip_sha256"] == sha256_of_file(archive) and result["artifact"]["source"] == "Pinned sample artifact"
    assert result["provenance"]["baseModelRevision"] == MODEL_REVISION and result["answers_differ"] == 2
    assert result["evaluation_report"]["verdict"] == "not-measurable"
    assert any(f["input"] == "blank-prompt-probe" for f in result["input_manifest"]["findings"])
    assert ns["pipe"].reloads == 1 and ns["pipe"].loaded == [ns["external_bundle_dir"]]
    assert "pinned sample artifact (sample-bundle-v1)" in out and "'base_identity_matches': True" in out
    assert (outputs / "generations.csv").read_text(encoding="utf-8").startswith("input,prompt,base,adapted,answers_differ")


def test_a_wrong_pinned_digest_is_refused_before_extraction_and_leaves_no_predictions(monkeypatch, tmp_path):
    archive = _standin_zip(tmp_path)
    work = tmp_path / "run"
    work.mkdir()
    _run_section(monkeypatch, work, archive)  # a successful run first: its outputs must not survive a failed re-run
    with pytest.raises(ValueError, match="Whole-ZIP SHA-256 mismatch for the pinned artifact"):
        _run_section(monkeypatch, work, archive, pinned_sha="0" * 64)
    assert list((work / "outputs" / "external_artifact").iterdir()) == []
    # The earlier extraction was cleared by the earlier run's open_artifact_zip; the refused download was deleted.
    assert not (work / "work" / "external_artifact" / "sample_adapter_bundle.zip").exists() or sha256_of_file(work / "work" / "external_artifact" / "sample_adapter_bundle.zip") == sha256_of_file(archive)


def test_a_wrong_digest_is_refused_before_extraction_on_a_fresh_runtime(monkeypatch, tmp_path):
    archive = _standin_zip(tmp_path)
    work = tmp_path / "fresh"
    work.mkdir()
    with pytest.raises(ValueError, match="nothing was extracted"):
        _run_section(monkeypatch, work, archive, pinned_sha="f" * 64)
    assert not (work / "work" / "external_artifact" / "bundle").exists()
    assert not (work / "work" / "external_artifact" / "sample_adapter_bundle.zip").exists()


def test_a_user_zip_with_the_wrong_expected_digest_is_refused_before_extraction(monkeypatch, tmp_path):
    archive = _standin_zip(tmp_path)
    work = tmp_path / "user"
    work.mkdir()
    with pytest.raises(ValueError, match="Whole-ZIP SHA-256 mismatch"):
        _run_section(monkeypatch, work, archive, fields={"ARTIFACT_SOURCE": "Your own bundle ZIP", "ARTIFACT_ZIP_PATH": str(archive), "EXPECTED_ARTIFACT_ZIP_SHA256": "1" * 64})
    assert not (work / "work" / "external_artifact" / "bundle").exists()
    ns, _ = _run_section(monkeypatch, work, archive, fields={"ARTIFACT_SOURCE": "Your own bundle ZIP", "ARTIFACT_ZIP_PATH": str(archive), "EXPECTED_ARTIFACT_ZIP_SHA256": sha256_of_file(archive)})
    assert ns["external_zip_sha256"] == sha256_of_file(archive)


def test_sart10_a_pinned_bundle_from_other_data_is_refused(monkeypatch, tmp_path):
    archive = _standin_zip(tmp_path, dataset={"source": "databricks/databricks-dolly-15k", "revision": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"})
    work = tmp_path / "run"
    work.mkdir()
    with pytest.raises(ValueError, match="does not follow the current default data contract"):
        _run_section(monkeypatch, work, archive)
    assert list((work / "outputs" / "external_artifact").iterdir()) == []


def test_a_bundle_for_another_base_revision_is_refused_before_any_weight_is_loaded(monkeypatch, tmp_path):
    archive = _standin_zip(tmp_path, revision="0" * 40)
    work = tmp_path / "run"
    work.mkdir()
    with pytest.raises(ValueError, match="refusing to attach"):
        _run_section(monkeypatch, work, archive)


def test_user_branches_are_guarded(monkeypatch, tmp_path):
    archive = _standin_zip(tmp_path)
    work = tmp_path / "run"
    work.mkdir()
    with pytest.raises(ValueError, match="a folder has no archive to check"):
        _run_section(monkeypatch, work, archive, fields={"ARTIFACT_SOURCE": "Your own bundle folder", "ARTIFACT_DIR": str(tmp_path / "standin-bundle"), "EXPECTED_ARTIFACT_ZIP_SHA256": "a" * 64})
    with pytest.raises(FileNotFoundError, match="is not a folder"):
        _run_section(monkeypatch, work, archive, fields={"ARTIFACT_SOURCE": "Your own bundle folder", "ARTIFACT_DIR": str(tmp_path / "missing")})
    with pytest.raises(FileNotFoundError, match="is not a file"):
        _run_section(monkeypatch, work, archive, fields={"ARTIFACT_SOURCE": "Your own bundle ZIP"})
    with pytest.raises(RuntimeError, match="no Colab upload dialog"):
        _run_section(monkeypatch, work, archive, fields={"ARTIFACT_SOURCE": "Upload a bundle ZIP (Colab)"})
    ns, out = _run_section(monkeypatch, work, archive, fields={"ARTIFACT_SOURCE": "Your own bundle folder", "ARTIFACT_DIR": str(tmp_path / "standin-bundle")})
    assert ns["external_zip_sha256"] is None and "has no archive" in out


# --- module helpers -------------------------------------------------------------------------------------------------------


def test_deterministic_zip_gives_identical_bytes_for_identical_files(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    for root, mtime in ((first, 1_600_000_000), (second, 1_700_000_000)):
        (root / "tokenizer").mkdir(parents=True)
        for rel, data in (("z.json", b"{}"), ("adapter_model.safetensors", b"\x00\x01"), ("tokenizer/t.json", b"[]")):
            (root / rel).write_bytes(data)
            os.utime(root / rel, (mtime, mtime))
    digest_a = deterministic_zip(first, tmp_path / "a.zip")
    digest_b = deterministic_zip(second, tmp_path / "b.zip")
    assert digest_a == digest_b and (tmp_path / "a.zip").read_bytes() == (tmp_path / "b.zip").read_bytes()
    with zipfile.ZipFile(tmp_path / "a.zip") as archive:
        infos = archive.infolist()
    assert [i.filename for i in infos] == sorted(i.filename for i in infos)
    assert all(i.date_time == (1980, 1, 1, 0, 0, 0) and stat.S_IMODE(i.external_attr >> 16) == 0o644 for i in infos)
    assert not (tmp_path / "a.zip.part").exists()


def test_fetch_pinned_artifact_reuses_a_matching_file_and_deletes_a_mismatching_download(tmp_path):
    source = tmp_path / "asset.zip"
    source.write_bytes(b"pinned bytes")
    digest = sha256_of_file(source)
    target = tmp_path / "dl" / "asset.zip"
    assert fetch_pinned_artifact(source.as_uri(), digest, target) == digest and target.read_bytes() == b"pinned bytes"
    assert fetch_pinned_artifact("file:///nonexistent/never-fetched.zip", digest, target) == digest  # reused, no download
    with pytest.raises(ValueError, match="nothing was extracted"):
        fetch_pinned_artifact(source.as_uri(), "0" * 64, tmp_path / "dl2" / "asset.zip")
    assert list((tmp_path / "dl2").iterdir()) == []


# --- ST1: only the exact pinned asset URL, as a whole token ---------------------------------------------------------------


@functools.cache
def _validator():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_validate_release_assets", ROOT / "tools" / "validate_release_assets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "reference",
    [
        RELEASE_URL + "/../../archive/main.zip",
        "https://github.com/kurtvalcorza/language-model-pipeline/releases/download/sample-bundle-v1/../../../archive/main.zip",
        "https://GITHUB.COM/kurtvalcorza/language-model-pipeline/releases/download/sample-bundle-v1/%2E%2E/x.zip",
        RELEASE_URL.replace("/sample-bundle-v1/", "/sample-bundle-v1/%2e%2e/"),
        RELEASE_URL + "?raw=1",
        RELEASE_URL + "#fragment",
        RELEASE_URL.replace("sample-bundle-v1", "sample-bundle-v2"),
        "https://github.com/kurtvalcorza/tabicl-classifier-pipeline/releases/download/sample-artifact-magic-v1/tabiclv2_classifier_artifact.zip",
        "https://github.com/kurtvalcorza/language-model-pipeline.git",
        RELEASE_URL + "x",
        RELEASE_URL.replace("language_model", "Language_model"),
    ],
)
def test_st1_refuses_every_reference_but_the_exact_pinned_url(reference):
    validator = _validator()
    assert validator.disallowed_own_references(f"URL = '{reference}'", (validator.PINNED_SAMPLE_ARTIFACT_URL,)) == [reference]


def test_st1_accepts_the_exact_pinned_url_and_the_real_notebook():
    validator = _validator()
    assert validator.PINNED_SAMPLE_ARTIFACT_URL == RELEASE_URL
    assert validator.disallowed_own_references(f'SAMPLE_ARTIFACT_URL = "{RELEASE_URL}"', (RELEASE_URL,)) == []
    assert validator.disallowed_own_references(f'SAMPLE_ARTIFACT_URL = "{RELEASE_URL}"') == [RELEASE_URL]  # not outside the module cell
    validator.validate_notebooks()
