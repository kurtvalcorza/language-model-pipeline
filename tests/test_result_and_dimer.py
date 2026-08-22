from __future__ import annotations

import json

import pytest

from lmpipeline.dimer import (
    REDACTED,
    DimerEnv,
    _normalize_device,
    notify_done_callback_url,
)
from lmpipeline.errors import Code, ConfigError, DatasetError, Stage
from lmpipeline.result import Result, write_result


@pytest.fixture
def dimer_environ(monkeypatch, tmp_path):
    monkeypatch.setenv("DIMER_DATASET_DIR", str(tmp_path / "dataset"))
    monkeypatch.setenv("DIMER_RESULT_PATH", str(tmp_path / "out" / "result.json"))
    monkeypatch.setenv("DIMER_OUTPUT_DIR", str(tmp_path / "model"))
    monkeypatch.setenv("DIMER_DONE_CALLBACK", "https://dimer.example/cb?token=SUPERSECRET")
    monkeypatch.setenv("DIMER_TRAIN_DEVICE", "cuda:0")
    monkeypatch.setenv("DIMER_SESSION_ID", "sess-1")
    monkeypatch.setenv("DIMER_RUN_ID", "run-1")
    monkeypatch.setenv("DIMER_PREPROCESSING_ARGS_JSON", '{"model_key":"qwen3-1.7b"}')
    monkeypatch.setenv("DIMER_HYPERPARAMETERS_JSON", '{"epochs":3}')
    monkeypatch.setenv("DIMER_PIPELINE_METADATA_JSON", '{"taskType":"custom"}')
    return tmp_path


# -- env contract -------------------------------------------------------------


def test_model_key_comes_from_preprocessing_args(dimer_environ):
    """The only user-parameter channel proven to reach BOTH containers."""
    env = DimerEnv.from_environ()
    assert env.model_key == "qwen3-1.7b"
    assert env.hyperparameters["epochs"] == 3


def test_task_type_defaults_to_the_baked_fallback(dimer_environ):
    assert DimerEnv.from_environ().task_type == "language_model_sft"


def test_base_model_absent_when_platform_omits_it(dimer_environ):
    assert DimerEnv.from_environ().base_model is None


def test_base_model_read_when_platform_supplies_it(dimer_environ, monkeypatch):
    monkeypatch.setenv(
        "DIMER_PIPELINE_METADATA_JSON", '{"taskType":"custom","baseModel":"Qwen/Qwen3-1.7B"}'
    )
    assert DimerEnv.from_environ().base_model == "Qwen/Qwen3-1.7B"


@pytest.mark.parametrize("raw,expected", [("0", "cuda:0"), ("1", "cuda:1"),
                                          ("cuda:0", "cuda:0"), ("cpu", "cpu"), ("", "cuda:0")])
def test_bare_ordinal_device_is_normalized(raw, expected):
    """DIMER passes "0"; torch.device("0") raises Invalid device string."""
    assert _normalize_device(raw) == expected


def test_malformed_json_env_is_a_structured_failure(dimer_environ, monkeypatch):
    monkeypatch.setenv("DIMER_HYPERPARAMETERS_JSON", "{not json}")
    with pytest.raises(ConfigError) as exc:
        DimerEnv.from_environ()
    assert exc.value.code == Code.CONFIG_INVALID_JSON


def test_diagnostics_redact_the_signed_callback(dimer_environ):
    """The probe/diagnostics path must never leak the signed URL into a result."""
    diagnostics = DimerEnv.from_environ().diagnostics()
    serialized = json.dumps(diagnostics)
    assert "SUPERSECRET" not in serialized
    assert diagnostics["DIMER_DONE_CALLBACK"] == REDACTED
    # Keys are allowlisted, so an unrelated secret in the environment cannot ride along.
    assert "PATH" not in diagnostics


def test_diagnostics_report_key_names_not_values(dimer_environ):
    diagnostics = DimerEnv.from_environ().diagnostics()
    assert diagnostics["preprocessingArgKeys"] == ["model_key"]


# -- result contract ----------------------------------------------------------


def test_result_carries_the_platform_documented_shape():
    result = Result.success(
        stage=Stage.VALIDATION, message="ok", code=Code.VALIDATION_SUCCEEDED
    )
    result.add_check("train_split_present", True, "Found train.jsonl.")
    payload = result.to_dict()

    for key in ("successful", "message", "datasetSummary", "checks", "metadata"):
        assert key in payload
    assert payload["checks"][0]["name"] == "train_split_present"
    # Mandatory per the portal checklist; SFT has no classes so it is empty, not absent.
    assert payload["metadata"]["classNames"] == []
    assert payload["metadata"]["classCount"] == 0


def test_pipeline_extensions_are_nested_not_substituted():
    result = Result.success(
        stage=Stage.TRAINING, message="done", code=Code.TRAINING_SUCCEEDED,
        metrics={"train_loss": 1.5}, provenance={"baseModelRevision": "abc"},
    )
    ext = result.to_dict()["metadata"]["languageModelPipeline"]
    assert ext["schemaVersion"] == "1.0"
    assert ext["code"] == Code.TRAINING_SUCCEEDED
    assert ext["metrics"]["train_loss"] == 1.5
    assert ext["provenance"]["baseModelRevision"] == "abc"


def test_pipeline_error_becomes_a_structured_result():
    exc = DatasetError("Line 4: invalid JSON.", code=Code.DATASET_INVALID_JSON,
                       details={"line": 4})
    payload = Result.from_exception(exc, stage=Stage.VALIDATION).to_dict()
    assert payload["successful"] is False
    assert payload["message"] == "Line 4: invalid JSON."
    assert payload["metadata"]["languageModelPipeline"]["code"] == Code.DATASET_INVALID_JSON


def test_unexpected_exception_does_not_leak_its_message():
    """Arbitrary exception text can embed signed URLs, paths, or dataset rows."""
    exc = RuntimeError("connection to https://host/x?token=LEAKED failed")
    payload = Result.from_exception(exc, stage=Stage.TRAINING).to_dict()
    assert "LEAKED" not in json.dumps(payload)
    assert payload["metadata"]["languageModelPipeline"]["code"] == Code.RUNTIME_UNEXPECTED


def test_result_is_written_atomically(tmp_path):
    target = tmp_path / "nested" / "result.json"
    write_result(
        Result.success(stage=Stage.VALIDATION, message="ok",
                       code=Code.VALIDATION_SUCCEEDED),
        target,
    )
    assert json.loads(target.read_text(encoding="utf-8"))["successful"] is True
    # No partial temp files left behind.
    assert [p.name for p in target.parent.iterdir()] == ["result.json"]

# -- the callback must fire even when env construction failed ---------------------
#
# DIMER names a missing done-callback as the cause of a Workbench session stuck at
# "Validating...". The entrypoints promised "always POST" from `finally` but guarded the
# call with `if env is not None`, so the one failure that leaves env unbound -- a malformed
# DIMER_*_JSON -- was exactly the failure that skipped the callback.


def test_the_url_callback_posts_without_a_constructed_env(monkeypatch):
    posted = []

    class _Response:
        status = 204
        def __enter__(self): return self
        def __exit__(self, *a): return False

    import urllib.request

    def fake_urlopen(request, timeout=None):
        posted.append((request.full_url, request.get_method(), request.data))
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert notify_done_callback_url("https://backend/done?sig=abc") is True
    assert posted == [("https://backend/done?sig=abc", "POST", b"")]


def test_the_url_callback_refuses_non_http_schemes():
    """urllib registers file:// and ftp:// handlers, so an odd scheme would be followed."""
    assert notify_done_callback_url("file:///etc/passwd") is False
    assert notify_done_callback_url("ftp://host/x") is False


def test_the_url_callback_is_a_no_op_without_a_url():
    assert notify_done_callback_url(None) is False
    assert notify_done_callback_url("") is False


def test_the_url_callback_never_raises(monkeypatch):
    """A failed callback must not mask the real result, and must not leak the signed URL."""
    import urllib.request

    def boom(request, timeout=None):
        raise OSError("connection refused to https://backend/done?sig=secret")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert notify_done_callback_url("https://backend/done?sig=secret") is False
