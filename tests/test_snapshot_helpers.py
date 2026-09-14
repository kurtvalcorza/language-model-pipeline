"""Offline tests for the pinned-snapshot helpers of lmpipeline.pipeline (MOD6-MOD8, ST3/ST4)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lmpipeline.pipeline import (
    DEFAULT_WEIGHTS_DIR,
    MANIFEST_NAME,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    TUTORIAL_REGISTRY,
    stage_missing_files,
    verify_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]


def _write(root: Path, name: str, payload: bytes) -> dict:
    (root / name).parent.mkdir(parents=True, exist_ok=True)
    (root / name).write_bytes(payload)
    return {"path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _snapshot(tmp_path: Path, *, model_id: str = MODEL_ID, revision: str = MODEL_REVISION) -> Path:
    root = tmp_path / MODEL_KEY
    root.mkdir(parents=True)
    files = [
        _write(root, "config.json", b'{"architectures": ["LlamaForCausalLM"]}'),
        _write(root, "model.safetensors", b"\x00" * 64),
        _write(root, "tokenizer_config.json", b"{}"),
    ]
    manifest = {
        "format": "dimer_hf_snapshot",
        "formatVersion": 1,
        "modelKey": MODEL_KEY,
        "modelId": model_id,
        "revision": revision,
        "files": files,
        "totalBytes": sum(f["bytes"] for f in files),
    }
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_identity_constants_are_immutable_and_registered() -> None:
    assert len(MODEL_REVISION) == 40 and all(c in "0123456789abcdef" for c in MODEL_REVISION)
    entry = TUTORIAL_REGISTRY[MODEL_KEY]
    assert (entry["model_id"], entry["revision"]) == (MODEL_ID, MODEL_REVISION)
    assert DEFAULT_WEIGHTS_DIR == ROOT / "weights" / MODEL_KEY


def test_committed_manifest_names_the_pinned_identity_and_hub_files_only() -> None:
    manifest = json.loads((DEFAULT_WEIGHTS_DIR / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert (manifest["modelId"], manifest["revision"]) == (MODEL_ID, MODEL_REVISION)
    paths = [f["path"] for f in manifest["files"]]
    assert "model.safetensors" in paths and "tokenizer_config.json" in paths
    # Repository-authored files are not Hub files and cannot be staged from the Hub.
    assert "MODEL_CARD.md" not in paths
    assert manifest["totalBytes"] == sum(f["bytes"] for f in manifest["files"])


def test_verify_snapshot_accepts_matching_files(tmp_path: Path) -> None:
    root = _snapshot(tmp_path)
    result = verify_snapshot(root)
    assert result["modelId"] == MODEL_ID and len(result["files"]) == 3


def test_verify_snapshot_rejects_tampered_digest(tmp_path: Path) -> None:
    root = _snapshot(tmp_path)
    (root / "model.safetensors").write_bytes(b"\x01" * 64)  # same size, different bytes
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(root)


def test_verify_snapshot_rejects_wrong_identity(tmp_path: Path) -> None:
    root = _snapshot(tmp_path, revision="0" * 40)
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(root)
    root = _snapshot(tmp_path / "other", model_id="someone/else")
    with pytest.raises(ValueError, match="modelId"):
        verify_snapshot(root)


def test_stage_missing_files_uses_injected_downloader(tmp_path: Path) -> None:
    root = _snapshot(tmp_path)
    (root / "model.safetensors").unlink()
    fetched: list[str] = []

    def downloader(relative_path: str, target: Path) -> None:
        fetched.append(relative_path)
        (target / relative_path).write_bytes(b"\x00" * 64)

    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(root)
    staged = stage_missing_files(root, allow_download=True, downloader=downloader)
    assert staged == ["model.safetensors"]
    assert fetched == ["model.safetensors"]
    assert stage_missing_files(root, allow_download=True, downloader=downloader) == []
    verify_snapshot(root)


def test_stage_missing_files_refuses_foreign_manifest(tmp_path: Path) -> None:
    root = _snapshot(tmp_path, model_id="someone/else")
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(root, allow_download=True, downloader=lambda *_: None)
