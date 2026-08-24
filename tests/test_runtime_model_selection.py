from __future__ import annotations

import json

import pytest

from lmpipeline.dimer import DimerEnv
from lmpipeline.errors import Code, ConfigError


def _base(monkeypatch, tmp_path):
    monkeypatch.setenv("DIMER_DATASET_DIR", str(tmp_path / "dataset"))
    monkeypatch.setenv("DIMER_RESULT_PATH", str(tmp_path / "result.json"))
    monkeypatch.setenv("DIMER_OUTPUT_DIR", str(tmp_path / "model"))
    monkeypatch.setenv("DIMER_HYPERPARAMETERS_JSON", '{"model_id":"qwen3-1.7b","epochs":1}')
    monkeypatch.setenv("DIMER_MODEL_CONFIG_JSON", '{"id":"qwen3-1.7b","provider":"qwen"}')
    monkeypatch.setenv("DIMER_PREPROCESSING_ARGS_JSON", "{}")


def test_real_dimer_model_id_is_authoritative_without_legacy_model_key(monkeypatch, tmp_path):
    """C-2 regression: the backend sends model_id, not datasetPreprocessing.model_key."""
    _base(monkeypatch, tmp_path)
    env = DimerEnv.from_environ()
    assert env.model_key == "qwen3-1.7b"
    assert env.model_config["id"] == "qwen3-1.7b"


def test_resolved_model_config_can_supply_the_key_when_model_id_is_absent(monkeypatch, tmp_path):
    _base(monkeypatch, tmp_path)
    monkeypatch.setenv("DIMER_HYPERPARAMETERS_JSON", '{"epochs":1}')
    assert DimerEnv.from_environ().model_key == "qwen3-1.7b"


def test_legacy_preprocessing_model_key_remains_a_compatible_fallback(monkeypatch, tmp_path):
    _base(monkeypatch, tmp_path)
    monkeypatch.setenv("DIMER_HYPERPARAMETERS_JSON", '{"epochs":1}')
    monkeypatch.setenv("DIMER_MODEL_CONFIG_JSON", "{}")
    monkeypatch.setenv("DIMER_PREPROCESSING_ARGS_JSON", '{"model_key":"granite-4.1-3b"}')
    assert DimerEnv.from_environ().model_key == "granite-4.1-3b"


def test_disagreeing_model_channels_fail_instead_of_training_the_wrong_model(monkeypatch, tmp_path):
    _base(monkeypatch, tmp_path)
    monkeypatch.setenv("DIMER_MODEL_CONFIG_JSON", '{"id":"qwen3-4b"}')
    with pytest.raises(ConfigError) as exc:
        _ = DimerEnv.from_environ().model_key
    assert exc.value.code == Code.CONFIG_SCHEMA_INVALID
    assert exc.value.details == {
        "selectors": ["hyperparameters.model_id", "modelConfig.id"]
    }


def test_malformed_model_config_is_a_structured_config_failure(monkeypatch, tmp_path):
    _base(monkeypatch, tmp_path)
    monkeypatch.setenv("DIMER_MODEL_CONFIG_JSON", "{bad json}")
    with pytest.raises(ConfigError) as exc:
        DimerEnv.from_environ()
    assert exc.value.code == Code.CONFIG_INVALID_JSON
    assert exc.value.details == {"variable": "DIMER_MODEL_CONFIG_JSON"}


def test_expected_accelerator_is_parsed_for_cheap_resource_preflight(monkeypatch, tmp_path):
    _base(monkeypatch, tmp_path)
    monkeypatch.setenv("DIMER_EXPECTED_ACCELERATOR", "NVIDIA")
    assert DimerEnv.from_environ().expected_accelerator == "nvidia"


def test_diagnostics_expose_model_config_keys_not_values(monkeypatch, tmp_path):
    _base(monkeypatch, tmp_path)
    monkeypatch.setenv(
        "DIMER_MODEL_CONFIG_JSON",
        json.dumps({"id": "qwen3-1.7b", "secretLookingField": "DO-NOT-ECHO"}),
    )
    diagnostics = DimerEnv.from_environ().diagnostics()
    rendered = json.dumps(diagnostics)
    assert diagnostics["modelConfigKeys"] == ["id", "secretLookingField"]
    assert "DO-NOT-ECHO" not in rendered
