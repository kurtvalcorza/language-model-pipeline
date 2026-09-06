import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fetch_weights import (  # noqa: E402
    DEFAULT_MODEL_KEY,
    MANIFEST_FORMAT,
    MANIFEST_FORMAT_VERSION,
    MANIFEST_NAME,
    generate_manifest,
    list_registered_models,
    main,
    matches_patterns,
    package_dimer_zip,
    sha256_file,
    verify_snapshot,
)


def test_list_registered_models_contains_expected_models():
    models = list_registered_models()
    keys = {m["key"] for m in models}

    # Core lineage models must be present
    assert "qwen3-1.7b" in keys
    assert "qwen3-4b" in keys
    assert "granite-4.1-3b" in keys
    assert "smollm3-3b" in keys
    assert "olmo2-7b" in keys
    assert "llama-3.2-3b-instruct" in keys
    assert "phi-4-mini-instruct" in keys

    # Default model must be flagged
    default_model = next(m for m in models if m["key"] == DEFAULT_MODEL_KEY)
    assert default_model["default"] is True
    assert default_model["enabled"] is True

    # Gated model must be detected
    llama = next(m for m in models if m["key"] == "llama-3.2-3b-instruct")
    assert llama["gated"] is True
    assert llama["enabled"] is False


def test_matches_patterns():
    # Allowed files
    assert matches_patterns("model.safetensors")
    assert matches_patterns("model-00001-of-00002.safetensors")
    assert matches_patterns("model.safetensors.index.json")
    assert matches_patterns("config.json")
    assert matches_patterns("generation_config.json")
    assert matches_patterns("tokenizer.json")
    assert matches_patterns("tokenizer_config.json")
    assert matches_patterns("vocab.json")
    assert matches_patterns("merges.txt")
    assert matches_patterns("chat_template.jinja")
    assert matches_patterns("LICENSE")
    assert matches_patterns("LICENSE.txt")
    assert matches_patterns("README.md")

    # Forbidden / ignored files
    assert not matches_patterns("pytorch_model.bin")
    assert not matches_patterns("model.pt")
    assert not matches_patterns("model.pth")
    assert not matches_patterns("model.onnx")
    assert not matches_patterns("tf_model.h5")
    assert not matches_patterns(".gitattributes")
    assert not matches_patterns(".gitignore")
    assert not matches_patterns("notebook.ipynb")


def test_manifest_generation_and_verification(tmp_path: Path):
    model_dir = tmp_path / "test_model"
    model_dir.mkdir()

    # Create dummy model files
    config_file = model_dir / "config.json"
    config_file.write_text('{"architectures": ["Qwen3ForCausalLM"]}\n', encoding="utf-8")

    weights_file = model_dir / "model.safetensors"
    weights_file.write_bytes(b"dummy safetensors weight bytes for test")

    tokenizer_file = model_dir / "tokenizer.json"
    tokenizer_file.write_text('{"version": "1.0"}\n', encoding="utf-8")

    model_key = "qwen3-1.7b"
    model_id = "Qwen/Qwen3-1.7B"
    revision = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

    manifest = generate_manifest(
        model_dir=model_dir,
        model_key=model_key,
        model_id=model_id,
        revision=revision,
    )

    assert manifest["format"] == MANIFEST_FORMAT
    assert manifest["formatVersion"] == MANIFEST_FORMAT_VERSION
    assert manifest["modelKey"] == model_key
    assert manifest["modelId"] == model_id
    assert manifest["revision"] == revision
    assert len(manifest["files"]) == 3
    assert manifest["totalBytes"] == (
        config_file.stat().st_size
        + weights_file.stat().st_size
        + tokenizer_file.stat().st_size
    )

    # Write manifest to disk
    manifest_file = model_dir / MANIFEST_NAME
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Verify snapshot passes
    valid, errors = verify_snapshot(
        model_dir,
        expected_model_key=model_key,
        expected_model_id=model_id,
        expected_revision=revision,
    )
    assert valid is True
    assert errors == []


def test_verify_snapshot_detects_corrupted_file(tmp_path: Path):
    model_dir = tmp_path / "corrupted_model"
    model_dir.mkdir()

    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"original bytes")

    manifest = generate_manifest(
        model_dir=model_dir,
        model_key="qwen3-1.7b",
        model_id="Qwen/Qwen3-1.7B",
        revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    )
    (model_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Tamper with file
    (model_dir / "model.safetensors").write_bytes(b"tampered bytes!")

    valid, errors = verify_snapshot(model_dir)
    assert valid is False
    assert any("SHA-256 mismatch" in e for e in errors)


def test_verify_snapshot_detects_unlisted_files(tmp_path: Path):
    model_dir = tmp_path / "unlisted_file_model"
    model_dir.mkdir()

    (model_dir / "model.safetensors").write_bytes(b"weights")

    manifest = generate_manifest(
        model_dir=model_dir,
        model_key="qwen3-1.7b",
        model_id="Qwen/Qwen3-1.7B",
        revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    )
    (model_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Add extra unlisted file
    (model_dir / "unexpected.bin").write_bytes(b"secret payload")

    valid, errors = verify_snapshot(model_dir)
    assert valid is False
    assert any("Unlisted files found on disk" in e for e in errors)


def test_verify_snapshot_requires_safetensors(tmp_path: Path):
    model_dir = tmp_path / "no_safetensors_model"
    model_dir.mkdir()

    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    manifest = generate_manifest(
        model_dir=model_dir,
        model_key="qwen3-1.7b",
        model_id="Qwen/Qwen3-1.7B",
        revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    )
    (model_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    valid, errors = verify_snapshot(model_dir)
    assert valid is False
    assert any("No .safetensors files listed in manifest" in e for e in errors)


def test_package_dimer_zip(tmp_path: Path):
    model_dir = tmp_path / "zip_test_model"
    model_dir.mkdir()

    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"test weights")

    manifest = generate_manifest(
        model_dir=model_dir,
        model_key="qwen3-1.7b",
        model_id="Qwen/Qwen3-1.7B",
        revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    )
    (model_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    zip_file = tmp_path / "output.zip"
    sha = package_dimer_zip(model_dir, zip_file)

    assert zip_file.is_file()
    assert len(sha) == 64
    assert sha == sha256_file(zip_file)


def test_cli_list(capsys):
    ret = main(["--list"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Registered Base Models:" in captured.out
    assert "qwen3-1.7b" in captured.out
    assert "smollm3-3b" in captured.out


def test_cli_unknown_model(capsys):
    ret = main(["--model", "unknown-model-xyz"])
    assert ret == 1
    captured = capsys.readouterr()
    assert "Unknown model_key 'unknown-model-xyz'" in captured.err


def test_cli_disabled_model_refusal(capsys):
    ret = main(["--model", "llama-3.2-3b-instruct"])
    assert ret == 1
    captured = capsys.readouterr()
    assert "is disabled in the registry" in captured.err


def test_cli_gated_model_requires_token(monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    ret = main(["--model", "llama-3.2-3b-instruct", "--force"])
    assert ret == 1
    captured = capsys.readouterr()
    assert "is gated on Hugging Face" in captured.err


def test_cli_verify_only(tmp_path: Path, capsys):
    model_dir = tmp_path / "verify_dir"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"data")
    manifest = generate_manifest(
        model_dir=model_dir,
        model_key="qwen3-1.7b",
        model_id="Qwen/Qwen3-1.7B",
        revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    )
    (model_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    ret = main(["--verify-only", "--dest", str(model_dir)])
    assert ret == 0
    captured = capsys.readouterr()
    assert "is VALID" in captured.out


def test_network_free_error_paths_without_huggingface_hub(monkeypatch, capsys):
    """Ensure all pre-download CLI validations succeed without importing huggingface_hub."""
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    # 1. Unknown model check
    assert main(["--model", "unknown-xyz-model"]) == 1
    out, err = capsys.readouterr()
    assert "Unknown model_key 'unknown-xyz-model'" in err

    # 2. Disabled model check
    assert main(["--model", "llama-3.2-3b-instruct"]) == 1
    out, err = capsys.readouterr()
    assert "is disabled in the registry" in err

    # 3. Gated model check without token
    assert main(["--model", "llama-3.2-3b-instruct", "--force"]) == 1
    out, err = capsys.readouterr()
    assert "is gated on Hugging Face" in err

    # 4. Attempted dry-run reports friendly import error without crash
    assert main(["--model", "qwen3-0.6b", "--dry-run"]) == 1
    out, err = capsys.readouterr()
    assert "'huggingface_hub' is required for dry-run inspection" in err


def test_default_destination_resolves_to_model_subfolder_with_sibling_content(
    tmp_path: Path, monkeypatch, capsys
):
    """Ensure default destination is weights/<model-key>/ and does not mix with sibling content."""
    weights_root = tmp_path / "weights"
    weights_root.mkdir()

    # Pre-existing sibling content in weights root
    (weights_root / "README.md").write_text("# Root Readme", encoding="utf-8")
    sibling_dir = weights_root / "other_model"
    sibling_dir.mkdir()
    (sibling_dir / "sibling.bin").write_bytes(b"sibling data")

    # Target model directory
    model_dir = weights_root / DEFAULT_MODEL_KEY
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")

    manifest = generate_manifest(
        model_dir=model_dir,
        model_key=DEFAULT_MODEL_KEY,
        model_id="Qwen/Qwen3-0.6B",
        revision="c1899de289a04d12100db370d81485cdf75e47ca",
    )
    (model_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Manifest must not contain sibling files from weights_root
    paths_in_manifest = {f["path"] for f in manifest["files"]}
    assert "README.md" not in paths_in_manifest
    assert "other_model/sibling.bin" not in paths_in_manifest
    assert "config.json" in paths_in_manifest
    assert "model.safetensors" in paths_in_manifest

    # CLI verify-only with no --dest should resolve to weights/<default_model> and pass cleanly
    import scripts.fetch_weights as fw
    monkeypatch.setattr(fw, "DEFAULT_DEST_DIR", weights_root)

    ret = main(["--verify-only"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "is VALID" in captured.out


def test_verify_snapshot_rejects_unsafe_paths(tmp_path: Path):
    """Ensure verify_snapshot rejects backslashes, drive prefixes, traversal, and UNC paths."""
    model_dir = tmp_path / "unsafe_test_model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"data")

    fake_hash = "0" * 64
    valid_hash = sha256_file(model_dir / "model.safetensors")

    unsafe_paths = [
        "../outside.safetensors",
        r"..\outside.safetensors",
        "/etc/passwd",
        r"\server\share\model.safetensors",
        r"\\server\share\model.safetensors",
        r"C:\outside\model.safetensors",
        "C:/outside/model.safetensors",
        "D:foo.safetensors",
        r"sub\file.safetensors",
        ".",
        "",
    ]

    for bad_path in unsafe_paths:
        manifest = {
            "format": MANIFEST_FORMAT,
            "formatVersion": MANIFEST_FORMAT_VERSION,
            "modelKey": "qwen3-0.6b",
            "modelId": "Qwen/Qwen3-0.6B",
            "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
            "files": [
                {"path": "model.safetensors", "bytes": 4, "sha256": valid_hash},
                {"path": bad_path, "bytes": 100, "sha256": fake_hash},
            ],
            "totalBytes": 104,
        }
        (model_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        valid, errors = verify_snapshot(model_dir)
        assert valid is False, f"Expected invalid for path: {bad_path!r}"
        assert any(
            (
                "Unsafe path" in e
                or "Path escape" in e
                or "Invalid or empty path" in e
                or "references snapshot root" in e
            )
            for e in errors
        ), f"Errors for {bad_path!r} did not contain expected safety message: {errors}"


def test_verify_snapshot_rejects_symlinks(tmp_path: Path):
    """Ensure symlinks are rejected during snapshot verification and manifest generation."""
    import os

    import pytest

    model_dir = tmp_path / "symlink_test_model"
    model_dir.mkdir()
    real_file = tmp_path / "real_file.txt"
    real_file.write_bytes(b"real content")
    (model_dir / "model.safetensors").write_bytes(b"data")

    link_path = model_dir / "linked.safetensors"
    try:
        os.symlink(real_file, link_path)
    except OSError:
        pytest.skip("Creating symlinks requires special privileges on Windows")

    # Manifest generation must reject symlink
    with pytest.raises(ValueError, match="Symlinks not allowed"):
        generate_manifest(
            model_dir=model_dir,
            model_key="qwen3-0.6b",
            model_id="Qwen/Qwen3-0.6B",
            revision="c1899de289a04d12100db370d81485cdf75e47ca",
        )

    # Verification must also reject symlink in manifest
    manifest = {
        "format": MANIFEST_FORMAT,
        "formatVersion": MANIFEST_FORMAT_VERSION,
        "modelKey": "qwen3-0.6b",
        "modelId": "Qwen/Qwen3-0.6B",
        "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "files": [
            {
                "path": "model.safetensors",
                "bytes": 4,
                "sha256": sha256_file(model_dir / "model.safetensors"),
            },
            {
                "path": "linked.safetensors",
                "bytes": len(b"real content"),
                "sha256": sha256_file(real_file),
            },
        ],
        "totalBytes": 4 + len(b"real content"),
    }
    (model_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    valid, errors = verify_snapshot(model_dir)
    assert valid is False
    assert any("Symlinks not allowed" in e for e in errors)
