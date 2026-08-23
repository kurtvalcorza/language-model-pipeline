from __future__ import annotations

import json

import pytest

from lmpipeline.dimer import REDACTED, DimerValidationEnv
from lmpipeline.errors import Code, ConfigError


def _set_validator_contract(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DIMER_DATASET_DIR", str(tmp_path / "dataset"))
    monkeypatch.setenv("DIMER_RESULT_PATH", str(tmp_path / "out" / "result.json"))
    monkeypatch.setenv("DIMER_DONE_CALLBACK", "https://dimer.example/cb?token=SUPERSECRET")
    monkeypatch.setenv("DIMER_PIPELINE_METADATA_JSON", '{"taskType":"custom"}')


def test_validator_env_reads_only_the_four_source_verified_channels(monkeypatch, tmp_path):
    _set_validator_contract(monkeypatch, tmp_path)

    # These are finetuner-side channels. Make them actively invalid so this test proves the
    # validator parser does not merely ignore their values after parsing them: it never reads
    # them at all.
    monkeypatch.setenv("DIMER_PREPROCESSING_ARGS_JSON", "{not json}")
    monkeypatch.setenv("DIMER_HYPERPARAMETERS_JSON", "{also not json}")
    monkeypatch.setenv("DIMER_OUTPUT_DIR", "/must/not/matter")
    monkeypatch.setenv("DIMER_TRAIN_DEVICE", "not-a-device")

    env = DimerValidationEnv.from_environ()

    assert env.dataset_dir == tmp_path / "dataset"
    assert env.result_path == tmp_path / "out" / "result.json"
    assert env.done_callback.endswith("SUPERSECRET")
    assert env.pipeline_metadata == {"taskType": "custom"}
    assert not hasattr(env, "model_key")
    assert not hasattr(env, "hyperparameters")
    assert not hasattr(env, "train_device")


def test_validator_diagnostics_expose_only_its_contract(monkeypatch, tmp_path):
    _set_validator_contract(monkeypatch, tmp_path)
    monkeypatch.setenv("DIMER_PREPROCESSING_ARGS_JSON", '{"model_key":"should-not-appear"}')
    monkeypatch.setenv("DIMER_HYPERPARAMETERS_JSON", '{"epochs":99}')
    monkeypatch.setenv("UNRELATED_SECRET", "DO-NOT-LEAK")

    diagnostics = DimerValidationEnv.from_environ().diagnostics()
    serialized = json.dumps(diagnostics)

    assert diagnostics["DIMER_DONE_CALLBACK"] == REDACTED
    assert diagnostics["pipelineMetadataKeys"] == ["taskType"]
    assert "SUPERSECRET" not in serialized
    assert "model_key" not in serialized
    assert "epochs" not in serialized
    assert "DO-NOT-LEAK" not in serialized


def test_validator_metadata_is_still_validated(monkeypatch, tmp_path):
    _set_validator_contract(monkeypatch, tmp_path)
    monkeypatch.setenv("DIMER_PIPELINE_METADATA_JSON", "{not json}")

    with pytest.raises(ConfigError) as excinfo:
        DimerValidationEnv.from_environ()

    assert excinfo.value.code == Code.CONFIG_INVALID_JSON
    assert excinfo.value.details["variable"] == "DIMER_PIPELINE_METADATA_JSON"
