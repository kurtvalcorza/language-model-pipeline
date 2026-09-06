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
