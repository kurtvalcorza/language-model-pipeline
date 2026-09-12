import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from lmpipeline.errors import Code, DatasetError, ModelError
from lmpipeline.tutorial_api import (
    assert_finetuner_checkout,
    assert_no_split_leakage,
    assert_runtime_compatible,
    canonical_dataset_digest,
    consume_adapter_archive,
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


def test_runtime_compatibility_accepts_torch_build_suffix():
    producer = {
        "packageVersions": {
            "torch": "2.8.0+cu128",
            "transformers": "5.16.1",
            "tokenizers": "0.23.2",
            "peft": "0.20.0",
            "bitsandbytes": "0.49.0",
            "safetensors": "0.8.0",
        }
    }
    current = {
        "packages": {
            "torch": "2.8.0",
            "transformers": "5.16.1",
            "tokenizers": "0.23.2",
            "peft": "0.20.0",
            "bitsandbytes": "0.49.0",
            "safetensors": "0.8.0",
        }
    }
    assert_runtime_compatible(producer, current)


def test_runtime_compatibility_rejects_material_mismatch():
    producer = {
        "packageVersions": {
            "torch": "2.8.0",
            "transformers": "5.15.0",
            "tokenizers": "0.23.2",
            "peft": "0.20.0",
            "bitsandbytes": "0.49.0",
            "safetensors": "0.8.0",
        }
    }
    current = {
        "packages": {
            "torch": "2.8.0",
            "transformers": "5.16.1",
            "tokenizers": "0.23.2",
            "peft": "0.20.0",
            "bitsandbytes": "0.49.0",
            "safetensors": "0.8.0",
        }
    }
    with pytest.raises(ValueError, match="compatibility mismatch"):
        assert_runtime_compatible(producer, current)


def test_finetuner_checkout_verifies_exact_git_head(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    (repo / "x.txt").write_text("x")
    subprocess.run(["git", "-C", str(repo), "add", "x.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "x"], check=True)
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    assert assert_finetuner_checkout(repo, revision) == revision


def test_safe_extract_rejects_parent_traversal(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "no")
    with pytest.raises(ValueError, match="Unsafe archive path"):
        safe_extract_zip(archive, tmp_path / "out", size_limit_bytes=1024)


def _write_adapter_artifact(root: Path):
    root.mkdir()
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
        for path in root.rglob("*"):
            if path.is_file():
                handle.write(path, path.relative_to(root).as_posix())
    extracted, manifest, provenance = consume_adapter_archive(
        archive, extraction_root=tmp_path / "extracted"
    )
    assert extracted.is_dir()
    assert manifest["totalBytes"] > 0
    entry = resolve_artifact_model(provenance)
    assert entry.key == "qwen3-1.7b"


def test_manifest_verification_rejects_unlisted_nested_manifest(tmp_path):
    # The file-set check used to exempt every file named artifact-manifest.json at any
    # depth, so an unmanifested nested copy passed the "expected file set" guarantee.
    root = tmp_path / "artifact"
    _write_adapter_artifact(root)
    verify_manifested_directory(root)
    (root / "nested").mkdir()
    (root / "nested" / "artifact-manifest.json").write_text('{"files": []}')
    with pytest.raises(ValueError, match="Manifest/file-set mismatch"):
        verify_manifested_directory(root)


def test_manifest_verification_rejects_non_object_records(tmp_path):
    root = tmp_path / "artifact"
    _write_adapter_artifact(root)
    (root / "artifact-manifest.json").write_text(
        json.dumps({"files": ["adapter_config.json"], "totalBytes": 0})
    )
    with pytest.raises(ValueError, match="non-object file record"):
        verify_manifested_directory(root)


@pytest.mark.parametrize("supplied", [2, None, ["2.8.0"], ""])
def test_runtime_compatibility_rejects_non_string_provenance_versions(supplied):
    # Provenance arrives inside an externally supplied ZIP, so a hostile type must produce
    # the module's ValueError contract rather than an AttributeError from `.split`.
    producer = {
        "packageVersions": {
            "torch": supplied,
            "transformers": "5.16.1",
            "tokenizers": "0.23.2",
            "peft": "0.20.0",
            "bitsandbytes": "0.49.0",
            "safetensors": "0.8.0",
        }
    }
    current = {"packages": {"torch": "2.8.0"}}
    with pytest.raises(ValueError, match="lacks runtime package versions"):
        assert_runtime_compatible(producer, current)


def test_runtime_compatibility_rejects_non_object_package_versions():
    with pytest.raises(ValueError, match="must be an object"):
        assert_runtime_compatible({"packageVersions": ["torch==2.8.0"]}, {"packages": {}})


def test_safe_extract_rejects_directory_colliding_with_a_file(tmp_path):
    archive = tmp_path / "collide.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("weights", "file-not-a-directory")
        handle.writestr("weights/adapter.safetensors", "payload")
    with pytest.raises(ValueError, match="collides with a file"):
        safe_extract_zip(archive, tmp_path / "out", size_limit_bytes=1024)


def test_safe_extract_rejects_file_colliding_with_a_directory_entry(tmp_path):
    archive = tmp_path / "collide.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("weights", "file-not-a-directory")
        handle.writestr("weights/", "")
    with pytest.raises(ValueError, match="collides with a file"):
        safe_extract_zip(archive, tmp_path / "out", size_limit_bytes=1024)
