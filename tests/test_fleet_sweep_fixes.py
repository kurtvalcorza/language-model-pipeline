"""Regression tests for the 2026-10-05 fleet-sweep fixes (SWP-R restart guard, SWP-G guided layer and the
repository-specific SWP-A / SWP-F / SWP-B fixes recorded in docs/reviews/2026-10-05-fleet-sweep/).

Every test needs only CI's dependencies. The notebooks' own cell sources are executed with stand-ins; no model, no
network and no torch are needed.
"""
# ruff: noqa: E501

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ['language_model_finetuning_colab', 'language_model_artifact_inference_colab']
LOCK = ROOT / 'tutorials/requirements-isolated.lock.txt'
MIN_PREDICT = {'language_model_finetuning_colab': 8, 'language_model_artifact_inference_colab': 5}


@functools.cache
def _nb_text(name: str) -> str:
    return (ROOT / "tutorials" / f"{name}.ipynb").read_text(encoding="utf-8")


def _nb(name: str) -> dict:
    return json.loads(_nb_text(name))


def _code_cells(notebook: dict) -> list[dict]:
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def _cell(notebook: dict, marker: str) -> str:
    found = [c["source"] for c in _code_cells(notebook) if marker in c["source"]]
    assert len(found) == 1, f"expected one code cell containing {marker!r}, found {len(found)}"
    return found[0]


def _build():
    spec = importlib.util.spec_from_file_location("_sweep_build_notebook", ROOT / "tools" / "build_notebook.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    return build


# --- SWP-R: no in-kernel install, no restart, idempotent Section 1 (shared by every notebook) -------------------------


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_r_nothing_is_pip_installed_into_the_kernel_and_no_restart_is_requested(name):
    notebook = _nb(name)
    code = "\n".join(c["source"] for c in _code_cells(notebook))
    assert "pip install" not in code and "'-m', 'pip'" not in code
    assert "restart the runtime" not in json.dumps(notebook).lower()
    kernel = [c for c in _code_cells(notebook) if "# dimer: kernel cell" in c["source"]]
    assert len(kernel) == 1, "exactly one cell may run in the kernel"
    source = kernel[0]["source"]
    for needed in ("'--require-hashes', '--only-binary', ':all:'", "'--managed-python'", "UV_SHA256", "LOCK_SHA256"):
        assert needed in source
    # The worker gets a clean interpreter environment and a non-interactive matplotlib backend.
    for needed in ('MPLBACKEND="Agg"', '"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"'):
        assert needed in source
    assert notebook["metadata"]["dimer"]["environment"].startswith("isolated hash-locked uv environment")


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_r_carried_lock_is_the_committed_lock_and_pins_every_runtime_pin(name):
    source = _cell(_nb(name), "# dimer: kernel cell")
    lock_text = LOCK.read_text(encoding="utf-8")
    digest = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    assert digest == hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    assert f"LOCK_TEXT = r'''{lock_text}'''" in source
    build = _build()
    build.check_lock(build._pins(ROOT), lock_text)  # raises SystemExit on any drift


class _Shell:
    def __init__(self) -> None:
        self.input_transformers_cleanup: list = []


def test_swp_r_section_1_is_idempotent_and_keeps_the_live_worker(tmp_path, monkeypatch, capsys):
    """Re-running the Section 1 cell reuses the matching environment (no download) and keeps the live worker, so the
    variables later cells created survive and the cells after it are not stranded."""
    source = _cell(_nb(NOTEBOOKS[0]), "# dimer: kernel cell")
    lock_sha = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").symlink_to(sys.executable)  # stand-in interpreter for the isolated environment
    (env / ".dimer-lock-sha256").write_text(lock_sha + "\n", encoding="utf-8")
    monkeypatch.setenv("DIMER_ISOLATED_ENV", str(env))
    monkeypatch.delenv("DIMER_NOTEBOOK_CI_PREINSTALLED", raising=False)
    shell = _Shell()
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.display = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)

    def no_download(*args, **kwargs):
        raise AssertionError("a matching environment must be reused, not downloaded again")

    monkeypatch.setattr("urllib.request.urlopen", no_download)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(source, "<section 1>", "exec"), namespace)
    runtime = namespace["_DIMER_ISOLATED_RUNTIME"]
    try:
        assert "'reused': True" in capsys.readouterr().out
        runtime.run("learner_value = 41 + 1\n")
        exec(compile(source, "<section 1 again>", "exec"), namespace)  # the learner re-runs Section 1 on its own
        assert namespace["_DIMER_ISOLATED_RUNTIME"] is runtime and runtime.alive()
        assert [t.__name__ for t in shell.input_transformers_cleanup] == ["_route_to_isolated_runtime"]
        runtime.run("print('value', learner_value)\n")
        assert "value 42" in capsys.readouterr().out
        assert namespace["_route_to_isolated_runtime"](["x = 1\n"]) == ["_DIMER_ISOLATED_RUNTIME.run('x = 1\\n')\n"]
        assert namespace["_route_to_isolated_runtime"]([source]) == [source]  # the kernel cell itself stays in the kernel
        with pytest.raises(RuntimeError, match="ZeroDivisionError"):
            runtime.run("1 / 0\n")
    finally:
        runtime.close()


# --- SWP-G: the guided layer and infrastructure labelling (shared) ----------------------------------------------------


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_g_guided_layer_is_present(name):
    notebook = _nb(name)
    markdown = "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")
    for heading in (
        "**Who this notebook is for.**",
        "**Input → Model → Output.**",
        "**How to use this notebook.**",
        "**Roadmap:**",
        "## Troubleshooting",
        "## Glossary",
        "## Conclusion (your notes)",
        "## Change one thing (next experiments)",
    ):
        assert heading in markdown, heading
    assert markdown.count("**Predict:**") >= MIN_PREDICT[name]
    assert markdown.count("<details><summary>Check your reasoning</summary>") >= MIN_PREDICT[name]
    assert "Run all completes in one pass" in markdown


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_g_infrastructure_cells_are_labelled_and_collapsed(name):
    cells = _code_cells(_nb(name))
    infra = [c for c in cells if c["metadata"].get("cellView") == "form"]
    assert any("# dimer: kernel cell" in c["source"] for c in infra)
    assert any(c["metadata"].get("dimer", {}).get("embedded_module") for c in infra)
    assert any(c["source"].startswith("# @title Infrastructure: stage and digest-verify") for c in infra)
    learner = [c for c in cells if c["metadata"].get("cellView") != "form"]
    assert learner and all("# @title Infrastructure" not in c["source"] for c in learner)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_g_no_template_placeholders_leak(name):
    notebook = _nb(name)
    text = "\n".join(
        c["source"] for c in notebook["cells"] if not c.get("metadata", {}).get("dimer", {}).get("embedded_module")
    )
    for leftover in ("{{", "{MODEL_ID}", "{stem}", "@P:"):
        assert leftover not in text, leftover


def _colab(monkeypatch, upload) -> None:
    google = types.ModuleType("google")
    google.__path__ = []
    colab_mod = types.ModuleType("google.colab")
    files = types.ModuleType("google.colab.files")
    files.upload = upload
    colab_mod.files = files
    google.colab = colab_mod
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab_mod)
    monkeypatch.setitem(sys.modules, "google.colab.files", files)


def _no_colab(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "google.colab", None)  # import fails as it does on Kaggle / Jupyter


# --- SWP-B: BYOD path, guarded upload, named refusals (fine-tuning notebook) ----------------------------------------

E2E, ART = NOTEBOOKS


def _e2e_byod(monkeypatch, tmp_path, path: str = "") -> dict:
    from lmpipeline.pipeline import (
        canonical,
        extract_zip_safely,
        fingerprint,
        manufacture_validation,
    )

    source = _cell(_nb(E2E), "BYOD_PATH = ''")
    block = source[source.index("BYOD_FILES = (") : source.index("else:\n    from datasets import load_dataset")]
    import shutil
    import tempfile

    work = Path(tempfile.mkdtemp(dir=tmp_path)) / "work"
    work.mkdir()

    namespace = {
        "USE_BYOD": True, "BYOD_PATH": path, "WORK_DIR": work, "Path": Path, "shutil": shutil, "json": json,
        "canonical": canonical, "extract_zip_safely": extract_zip_safely, "manufacture_validation": manufacture_validation,
        "fingerprint": fingerprint,
    }
    exec(compile(block, "<section 4 BYOD>", "exec"), namespace)
    return namespace


def _rows(n: int) -> str:
    return "\n".join(json.dumps({"prompt": f"q{i}", "completion": f"a{i}"}) for i in range(n)) + "\n"


def test_swp_b_byod_path_folder_and_zip_work_outside_colab(monkeypatch, tmp_path):
    _no_colab(monkeypatch)
    folder = tmp_path / "mine"
    folder.mkdir()
    (folder / "train.jsonl").write_text(_rows(10), encoding="utf-8")
    (folder / "test.jsonl").write_text(_rows(2), encoding="utf-8")
    ns = _e2e_byod(monkeypatch, tmp_path, str(folder))
    assert sorted(ns["SPLITS"]) == ["test", "train", "validation"] and ns["sample_kind"] == "BYOD"
    assert len(ns["SPLITS"]["train"]) + len(ns["SPLITS"]["validation"]) == 10
    import zipfile

    archive = tmp_path / "mine.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("train.jsonl", _rows(10))
        handle.writestr("validation.jsonl", _rows(3).replace("q", "v"))
    ns = _e2e_byod(monkeypatch, tmp_path, str(archive))
    assert len(ns["SPLITS"]["train"]) == 10 and len(ns["SPLITS"]["validation"]) == 3


def test_swp_b_byod_refusals_name_the_file_line_and_rule(monkeypatch, tmp_path):
    _no_colab(monkeypatch)
    with pytest.raises(RuntimeError, match="BYOD_PATH is empty and this runtime has no Colab upload dialog"):
        _e2e_byod(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError, match="is neither a folder nor a .zip file"):
        _e2e_byod(monkeypatch, tmp_path, str(tmp_path / "nope.txt"))
    folder = tmp_path / "bad"
    folder.mkdir()
    (folder / "train.jsonl").write_text(_rows(3) + '{"foo": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"train\.jsonl line 4: not a messages"):
        _e2e_byod(monkeypatch, tmp_path, str(folder))
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "notes.jsonl").write_text(_rows(2), encoding="utf-8")
    with pytest.raises(ValueError, match="BYOD requires train.jsonl"):
        _e2e_byod(monkeypatch, tmp_path, str(empty))


def test_swp_b_cancelled_upload_is_explained(monkeypatch, tmp_path):
    _colab(monkeypatch, lambda: {})
    with pytest.raises(RuntimeError, match="upload was cancelled or empty"):
        _e2e_byod(monkeypatch, tmp_path)
    _colab(monkeypatch, lambda: {"train.jsonl": _rows(10).encode()})
    ns = _e2e_byod(monkeypatch, tmp_path)
    assert len(ns["SPLITS"]["train"]) + len(ns["SPLITS"]["validation"]) == 10


def test_swp_b_bundle_download_falls_back_to_a_path_outside_colab(monkeypatch, tmp_path, capsys):
    source = _cell(_nb(E2E), "files.download(str(ARTIFACT_ZIP))")
    tail = source[source.index("try:\n    from google.colab import files\n    files.download") :]
    _no_colab(monkeypatch)
    exec(compile(tail, "<section 10 tail>", "exec"), {"ARTIFACT_ZIP": tmp_path / "bundle.zip"})
    assert "No download dialog in this runtime" in capsys.readouterr().out
    _colab(monkeypatch, None)  # the isolated worker's google.colab.files has upload() but no download()
    exec(compile(tail, "<section 10 tail>", "exec"), {"ARTIFACT_ZIP": tmp_path / "bundle.zip"})
    assert str(tmp_path / "bundle.zip") in capsys.readouterr().out


def _artifact_cell(monkeypatch, tmp_path, **fields) -> dict:
    """Execute the companion's Section 4 cell up to (not including) verify_artifact_bundle, with form fields rewritten.
    Since the 2026-10-02 review fixes (LMA-B1/LMA-m3) the user bundle is the opt-in USE_OWN_ARTIFACT branch."""
    from lmpipeline.pipeline import extract_zip_safely, sha256_of_file

    source = _cell(_nb(ART), "ARTIFACT_DIR = ''")
    block = source[: source.index("artifact_manifest, provenance = verify_artifact_bundle")]
    for name, value in fields.items():
        line = re.search(rf"^{name} = .*?(  # @param.*)$", block, re.M)
        assert line, name
        block = block.replace(line.group(0), f"{name} = {value!r}{line.group(1)}")
    monkeypatch.chdir(tmp_path)
    import os

    ns = {"os": os, "Path": Path, "ARTIFACT_MANIFEST_NAME": "artifact-manifest.json", "sha256_of_file": sha256_of_file, "extract_zip_safely": extract_zip_safely}
    exec(compile(block, "<s4>", "exec"), ns)
    return ns


def test_swp_b_artifact_dir_and_upload_are_guarded(monkeypatch, tmp_path):
    _no_colab(monkeypatch)
    with pytest.raises(FileNotFoundError, match="ARTIFACT_DIR .*missing.* is not a folder"):
        _artifact_cell(monkeypatch, tmp_path, USE_OWN_ARTIFACT=True, ARTIFACT_DIR=str(tmp_path / "missing"))
    ns = _artifact_cell(monkeypatch, tmp_path, USE_OWN_ARTIFACT=True, ARTIFACT_DIR=str(tmp_path))
    assert ns["bundle_dir"] == tmp_path and ns["archive_sha"] is None
    with pytest.raises(RuntimeError, match="ARTIFACT_ZIP_PATH and ARTIFACT_DIR are empty and this runtime has no Colab upload dialog"):
        _artifact_cell(monkeypatch, tmp_path, USE_OWN_ARTIFACT=True)
    _colab(monkeypatch, lambda: {})
    with pytest.raises(ValueError, match="got 0 .*upload cancelled or empty"):
        _artifact_cell(monkeypatch, tmp_path, USE_OWN_ARTIFACT=True)


# --- SWP-F: a re-run never takes a baseline from, or trains on top of, an adapted model ------------------------------


class _Pipe:
    def __init__(self, adapted: bool) -> None:
        self.model = types.SimpleNamespace(named_modules=lambda: [("model.layers.0.q_proj.lora_A", None)] if adapted else [("model.layers.0.q_proj", None)])
        self.reloads = 0

    def reload_base(self):
        self.reloads += 1
        return types.SimpleNamespace(named_modules=lambda: [("model.layers.0.q_proj", None)])


def _frozen_base_namespace(pipe) -> dict:
    source = _cell(_nb(E2E), "def frozen_base():")
    helper = source[source.index("import gc") : source.index("\n\n\nfrozen_base()")]
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None))
    namespace = {"pipe": pipe, "torch": torch, "model": "adapted-peft-model", "reloaded": "reloaded-peft-model"}
    exec(compile(helper, "<section 6 helper>", "exec"), namespace)
    return namespace


def test_swp_f_frozen_base_reloads_an_adapted_model_and_keeps_a_frozen_one(capsys):
    pipe = _Pipe(adapted=True)
    ns = _frozen_base_namespace(pipe)
    ns["frozen_base"]()
    assert pipe.reloads == 1 and ns["model"] is None and ns["reloaded"] is None
    assert "Reloaded the frozen base model" in capsys.readouterr().out
    ns["frozen_base"]()  # now frozen: no second reload
    assert pipe.reloads == 1
    frozen = _Pipe(adapted=False)
    _frozen_base_namespace(frozen)["frozen_base"]()
    assert frozen.reloads == 0


def test_swp_f_baseline_and_training_cells_start_from_the_frozen_base():
    baseline = _cell(_nb(E2E), "BASELINE_OUTPUTS = [pipe.generate(prompt) for prompt in PROMPTS]")
    assert baseline.index("\nfrozen_base()\n") < baseline.index("BASELINE_OUTPUTS =")
    training = _cell(_nb(E2E), "model = get_peft_model(prepare_model_for_kbit_training(pipe.model), lora_config)")
    assert training.index("frozen_base()") < training.index("get_peft_model(")


def test_swp_g_checkpoints_invent_no_numbers_for_an_unrecorded_run():
    """No passing clean-run record exists, so the worked answers must say so and quote no loss values."""
    for name in NOTEBOOKS:
        markdown = "\n".join(c["source"] for c in _nb(name)["cells"] if c["cell_type"] == "markdown")
        assert "No passing clean-run record exists yet for this notebook" in markdown
        answers = re.findall(r"<details><summary>Check your reasoning</summary>(.*?)</details>", markdown, re.S)
        assert answers and not any(re.search(r"loss (?:of |≈ ?|= ?)\d", a) for a in answers)
