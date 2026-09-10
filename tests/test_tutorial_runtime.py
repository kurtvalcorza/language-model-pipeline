from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from lmpipeline.tutorial_runtime import (
    _git_auth_environment,
    consume_adapter_archive,
)


def _write_artifact(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "tokenizer").mkdir()
    files = {
        "adapter_config.json": "{}",
        "adapter_model.safetensors": "safe-bytes",
        "tokenizer/tokenizer.json": "{}",
        "provenance.json": json.dumps(
            {
                "modelKey": "qwen3-1.7b",
                "baseModel": "Qwen/Qwen3-1.7B",
                "baseModelRevision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
                "trustRemoteCode": False,
            }
        ),
    }
    records = []
    for name, text in files.items():
        path = root / name
        path.write_text(text)
        payload = path.read_bytes()
        records.append(
            {
                "path": name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    (root / "artifact-manifest.json").write_text(
        json.dumps(
            {
                "files": records,
                "totalBytes": sum(item["bytes"] for item in records),
            }
        )
    )


def test_git_auth_uses_environment_not_remote_url():
    env = _git_auth_environment("secret-token")
    assert env["GIT_CONFIG_KEY_0"] == "http.https://github.com/.extraHeader"
    assert env["GIT_CONFIG_VALUE_0"].startswith("Authorization: Basic ")
    assert "secret-token" not in env["GIT_CONFIG_VALUE_0"]


def test_git_auth_rejects_whitespace_secrets():
    with pytest.raises(ValueError, match="single-line"):
        _git_auth_environment("bad token")


def test_strict_archive_accepts_manifest_at_zip_root(tmp_path):
    root = tmp_path / "artifact"
    _write_artifact(root)
    archive = tmp_path / "artifact.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in root.rglob("*"):
            if path.is_file():
                handle.write(path, path.relative_to(root).as_posix())

    extracted, manifest, _ = consume_adapter_archive(
        archive,
        extraction_root=tmp_path / "extracted",
    )
    assert extracted == (tmp_path / "extracted").resolve()
    assert manifest["totalBytes"] > 0


def test_strict_archive_rejects_nested_artifact_with_unmanifested_sibling(tmp_path):
    root = tmp_path / "artifact"
    nested = root / "nested"
    _write_artifact(nested)
    (root / "sibling.txt").write_text("unmanifested")
    archive = tmp_path / "nested.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in root.rglob("*"):
            if path.is_file():
                handle.write(path, path.relative_to(root).as_posix())

    with pytest.raises(ValueError, match="ZIP archive root"):
        consume_adapter_archive(
            archive,
            extraction_root=tmp_path / "extracted",
        )
