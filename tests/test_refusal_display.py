"""Refusal display (NOTEBOOK_SPEC 3.0 DAT19/DAT20): a refused user input is one `Refused: <file>: <rule>` line.

Offline, with CI's light dependency set (no torch, transformers or datasets). The Section 4 BYOD branch and the
Section 11 external-artifact cell are executed with stand-ins for the model; a refusal must render as exactly one
`Refused:` line with no traceback, name the refused file, and stop the run (it is raised, so Run all halts at that
cell). IPython renders an exception through its `_render_traceback_` method when it has one; `_render` below applies
that rule. The isolated-environment worker test starts the real worker/kernel pair from the generator, so it needs
`os.pipe` file-descriptor passing (POSIX only, as on Colab).
"""
# ruff: noqa: E501
from __future__ import annotations

import builtins
import importlib.util
import io
import json
import os
import re
import shutil
import sys
import traceback
import types
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from lmpipeline.pipeline import (
    Refused,
    canonical,
    extract_zip_safely,
    fingerprint,
    manufacture_validation,
    validate_inputs,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "language_model_finetuning_colab.ipynb"


def _cell(marker: str) -> str:
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    found = [c["source"] for c in cells if c["cell_type"] == "code" and marker in c["source"]]
    assert len(found) == 1, marker
    return found[0]


def _render(source: str, ns: dict) -> tuple[str, bool]:
    """Run one cell as IPython would show it: printed text, then the error (its _render_traceback_ when defined)."""
    out = io.StringIO()
    halted = False
    with redirect_stdout(out):
        try:
            exec(compile(source, "<cell>", "exec"), ns)  # noqa: S102 - the notebook's own cell
        except Exception as exc:  # noqa: BLE001 - rendered exactly as the kernel would
            halted = True
            render = getattr(exc, "_render_traceback_", None)
            print("\n".join(render()) if render else traceback.format_exc())
    return out.getvalue(), halted


def _assert_one_clean_refusal(out: str, halted: bool, name: str, rule: str) -> None:
    refused = [line for line in out.splitlines() if "Refused:" in line]
    assert len(refused) == 1 and refused[0].startswith(f"Refused: {name}: "), out
    assert re.search(rule, refused[0]) and "run this cell again" in refused[0], out
    assert "Traceback" not in out and halted, out  # no traceback, and Run all stops at this cell


def _kit_rows(bad_line: int | None = None) -> str:
    """The browser kit's train.jsonl: 48 synthetic unit-conversion records; refuse.zip has a {"text": ...} record on line 7."""
    lines = []
    for i in range(1, 49):
        if i == bad_line:
            lines.append(json.dumps({"text": f"Convert {i} kilometres to metres. Answer: {i * 1000} metres."}))
        else:
            lines.append(json.dumps({"prompt": f"Convert {i} kilometres to metres.", "completion": f"{i} kilometres is {i * 1000} metres."}))
    return "\n".join(lines) + "\n"


def _zip(**members: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as handle:
        for name, text in members.items():
            handle.writestr(name, text)
    return buffer.getvalue()


def _section4(tmp_path: Path, path: str = "") -> tuple[str, dict]:
    source = _cell("BYOD_PATH = ''")
    block = source[source.index("BYOD_FILES = (") : source.index("else:\n    from datasets import load_dataset")]
    work = tmp_path / "work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    ns = {
        "__builtins__": builtins, "USE_BYOD": True, "BYOD_PATH": path, "WORK_DIR": work, "Path": Path, "shutil": shutil, "json": json,
        "canonical": canonical, "extract_zip_safely": extract_zip_safely, "manufacture_validation": manufacture_validation,
        "fingerprint": fingerprint, "Refused": Refused, "zipfile": zipfile, "validate_inputs": validate_inputs,
        "MAX_SEQUENCE_LENGTH": 512, "pipe": types.SimpleNamespace(token_length=lambda record: 8),
    }
    return block, ns


def _colab(monkeypatch, uploaded: dict) -> None:
    google = types.ModuleType("google")
    google.__path__ = []
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: uploaded)
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)


def test_the_browser_kit_refusal_is_one_line(monkeypatch, tmp_path) -> None:
    _colab(monkeypatch, {"refuse.zip": _zip(**{"train.jsonl": _kit_rows(bad_line=7)})})
    out, halted = _render(*_section4(tmp_path))
    _assert_one_clean_refusal(out, halted, "refuse.zip", r"train\.jsonl line 7: not a messages, prompt/completion")


def test_the_browser_kit_accept_upload_passes(monkeypatch, tmp_path) -> None:
    _colab(monkeypatch, {"accept.zip": _zip(**{"train.jsonl": _kit_rows()})})
    block, ns = _section4(tmp_path)
    out, halted = _render(block, ns)
    assert not halted and "Refused" not in out, out
    assert len(ns["SPLITS"]["train"]) == 39 and len(ns["SPLITS"]["validation"]) == 9  # the kit's expected split


@pytest.mark.parametrize(
    "case, name, rule",
    [
        ("cancelled", r"\(no file\)", "The upload was cancelled or empty"),
        ("no-train-file", "notes.jsonl", "BYOD requires train.jsonl"),
        ("not-a-zip", "broken.zip", "the ZIP cannot be used"),
        ("leaked-split", "train.jsonl, validation.jsonl", "leakage"),
        ("missing-path", "absent", "is neither a folder nor a .zip file"),
    ],
)
def test_section_4_refusals_are_one_line(monkeypatch, tmp_path, case: str, name: str, rule: str) -> None:
    path = ""
    if case == "cancelled":
        _colab(monkeypatch, {})
    elif case == "no-train-file":
        _colab(monkeypatch, {"notes.jsonl": _kit_rows().encode()})
    elif case == "not-a-zip":
        _colab(monkeypatch, {"broken.zip": b"this is not a zip archive"})
    elif case == "leaked-split":
        rows = _kit_rows()
        _colab(monkeypatch, {"train.jsonl": rows.encode(), "validation.jsonl": rows.splitlines()[0].encode()})
    else:
        monkeypatch.setitem(sys.modules, "google.colab", None)
        path = str(tmp_path / "absent")
    out, halted = _render(*_section4(tmp_path, path))
    refused = [line for line in out.splitlines() if "Refused:" in line]
    assert len(refused) == 1 and re.match(rf"Refused: {name}: ", refused[0]), out
    assert re.search(rule, refused[0], re.I) and "run this cell again" in refused[0] and "Traceback" not in out and halted, out


# ---- Section 11: your own adapter bundle -----------------------------------------------------------------------------


def _external():
    spec = importlib.util.spec_from_file_location("refusal_external_artifact", ROOT / "tests" / "test_external_artifact_section.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_section_11_refuses_an_own_bundle_with_the_wrong_digest_as_one_line(monkeypatch, tmp_path, capsys) -> None:
    external = _external()
    archive = external._standin_zip(tmp_path)
    work = tmp_path / "user"
    work.mkdir()
    with pytest.raises(Exception) as caught:  # noqa: PT011 - the carried module's own Refused class
        external._run_section(monkeypatch, work, archive, fields={"ARTIFACT_SOURCE": "Your own bundle ZIP", "ARTIFACT_ZIP_PATH": str(archive), "EXPECTED_ARTIFACT_ZIP_SHA256": "1" * 64})
    lines = caught.value._render_traceback_()
    assert len(lines) == 1 and lines[0].startswith("Refused: standin.zip: Whole-ZIP SHA-256 mismatch"), lines
    assert "run this cell again" in lines[0] and "Traceback" not in lines[0]


def test_section_11_a_pinned_bundle_failure_is_not_a_refusal(monkeypatch, tmp_path) -> None:
    # A pinned asset failing its contract is a genuine error: it keeps its ordinary exception and traceback.
    external = _external()
    archive = external._standin_zip(tmp_path, dataset={"source": "databricks/databricks-dolly-15k", "revision": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"})
    work = tmp_path / "run"
    work.mkdir()
    with pytest.raises(ValueError, match="does not follow the current default data contract") as caught:
        external._run_section(monkeypatch, work, archive)
    assert not hasattr(caught.value, "_render_traceback_")


# ---- The isolated-environment worker passes a refusal back as its one line -------------------------------------------


def _router() -> dict:
    spec = importlib.util.spec_from_file_location("refusal_build_notebook", ROOT / "tools" / "build_notebook.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = module._ISOLATED_ROUTER.split("\n\nif SKIP_INSTALL:")[0]
    import signal
    import subprocess
    from multiprocessing.connection import Connection

    ns = {"__builtins__": builtins, "__name__": "router", "os": os, "sys": sys, "signal": signal, "subprocess": subprocess, "Connection": Connection}
    exec(compile(source, "<router>", "exec"), ns)  # noqa: S102 - the generator's own router source
    return ns


@pytest.mark.skipif(os.name != "posix", reason="the worker needs os.pipe file-descriptor passing (POSIX), as on Colab")
def test_the_worker_sends_a_refusal_as_one_line_and_a_genuine_error_with_its_traceback(capfd) -> None:
    router = _router()
    runtime = router["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        refusal = f"import sys\nsys.path.insert(0, {str(ROOT / 'src')!r})\nfrom lmpipeline.pipeline import Refused\nraise Refused('refuse.zip', 'train.jsonl line 7: not a record. Fix that line and run this cell again.')\n"
        with pytest.raises(router["InputRefused"]) as caught:
            runtime.run(refusal)
        assert caught.value._render_traceback_() == ["Refused: refuse.zip: train.jsonl line 7: not a record. Fix that line and run this cell again."]
        streamed = capfd.readouterr()
        assert "Traceback" not in streamed.out + streamed.err
        with pytest.raises(router["IsolatedCellError"]):
            runtime.run("raise ValueError('a genuine bug')\n")
        assert "Traceback" in capfd.readouterr().err  # genuine failures still show their traceback
    finally:
        runtime.close()
