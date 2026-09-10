import json
import zipfile
from pathlib import Path

import pytest

from lmpipeline.errors import Code, DatasetError, ModelError
from lmpipeline.tutorial_api import (
    assert_no_split_leakage,
    canonical_dataset_digest,
    consume_adapter_archive,
    derive_validation_split,
    normalize_records,
    resolve_artifact_model,
    resolve_tutorial_model,
    safe_extract_zip,
    verify_manifested_directory,
)


def _records():
    return [
        {"prompt": "one", "completion": "answer one"},
        {"prompt": "two", "completion": "answer two"},
        {"prompt": "three", "completion": "answer three"},
        {"prompt": "four", "completion": "answer four"},
    ]


def test_normalize_records_uses_canonical_schema_contract():
    examples = normalize_records(_records())
    assert len(examples) == 4
    assert examples[0].messages == (
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "answer one"},
    )


def test_normalize_records_rejects_mixed_schema_families():
    with pytest.raises(DatasetError) as caught:
        normalize_records(
            [
                {"prompt": "one", "completion": "answer"},
                {"instruction": "two", "output": "answer"},
            ]
        )
    assert caught.value.code == Code.DATASET_SCHEMA_MIXED


def test_validation_split_is_order_independent():
    examples = normalize_records(_records())
    train_a, val_a = derive_validation_split(examples, fraction=0.25, seed=42)
    train_b, val_b = derive_validation_split(list(reversed(examples)), fraction=0.25, seed=42)
    assert {item.fingerprint() for item in train_a} == {item.fingerprint() for item in train_b}
    assert {item.fingerprint() for item in val_a} == {item.fingerprint() for item in val_b}


def test_split_leakage_uses_stable_pipeline_error_code():
    examples = normalize_records(_records())
    with pytest.raises(DatasetError) as caught:
        assert_no_split_leakage({"train": examples[:2], "validation": [examples[0]]})
    assert caught.value.code == Code.DATASET_SPLIT_LEAKAGE


def test_canonical_dataset_digest_is_stable_across_row_order():
    examples = normalize_records(_records())
    first = canonical_dataset_digest({"train": examples})
    second = canonical_dataset_digest({"train": list(reversed(examples))})
    assert first == second


def test_user_facing_tutorial_model_resolves_from_canonical_registry():
    entry = resolve_tutorial_model(
        "qwen3-1.7b", method="qlora", max_sequence_length=512
    )
    assert entry.model_id == "Qwen/Qwen3-1.7B"
    assert len(entry.revision) == 40


def test_internal_model_is_not_a_user_facing_tutorial_choice():
    with pytest.raises(ModelError):
        resolve_tutorial_model(
            "qwen3-0.6b", method="qlora", max_sequence_length=512
        )


def test_safe_extract_rejects_parent_traversal(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "no")
    with pytest.raises(ValueError, match="Unsafe archive path"):
        safe_extract_zip(archive, tmp_path / "out", size_limit_bytes=1024)


def _write_adapter_artifact(root: Path):
    root.mkdir()
    files = {
        "adapter_config.json": "{}",
        "adapter_model.safetensors": "safe-bytes",
        "provenance.json": json.dumps(
            {
                "modelKey": "qwen3-1.7b",
                "baseModel": "Qwen/Qwen3-1.7B",
                "baseModelRevision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
                "trustRemoteCode": False,
            }
        ),
    }
    import hashlib

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
                "format": "peft_adapter",
                "formatVersion": 1,
                "files": records,
                "totalBytes": sum(item["bytes"] for item in records),
            }
        )
    )


def test_manifest_verification_rejects_unlisted_file(tmp_path):
    root = tmp_path / "artifact"
    _write_adapter_artifact(root)
    verify_manifested_directory(root)
    (root / "surprise.txt").write_text("unexpected")
    with pytest.raises(ValueError, match="Manifest/file-set mismatch"):
        verify_manifested_directory(root)


def test_external_adapter_consumption_and_registry_identity(tmp_path):
    root = tmp_path / "artifact"
    _write_adapter_artifact(root)
    archive = tmp_path / "adapter.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in root.iterdir():
            handle.write(path, path.name)
    extracted, manifest, provenance = consume_adapter_archive(
        archive, extraction_root=tmp_path / "extracted"
    )
    assert extracted.is_dir()
    assert manifest["format"] == "peft_adapter"
    entry = resolve_artifact_model(provenance)
    assert entry.key == "qwen3-1.7b"
