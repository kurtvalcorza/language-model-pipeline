"""The DIMER container runtime contracts.

Every DIMER-specific assumption lives here so the rest of the package stays platform-neutral
and unit-testable. Source verification against the backend established that the validator and
the finetuner receive different environment-variable sets, so they intentionally use
different parsed views:

* ``DimerValidationEnv`` reads only the four ``DIMER_``-namespace variables delivered to
  validator Jobs: ``DIMER_DATASET_DIR``, ``DIMER_RESULT_PATH``, ``DIMER_DONE_CALLBACK`` and
  ``DIMER_PIPELINE_METADATA_JSON``. The last is absent on the backend's ``main`` branch, so
  it defaults to ``{}`` rather than being required. An on-prem deployment additionally
  injects S3 credentials and object keys; those belong to the storage layer, not to this
  parsed view, and are deliberately not read here.
* ``DimerEnv`` is the broader finetuner-facing view. The finetuner receives the registered
  model selection through ``DIMER_HYPERPARAMETERS_JSON.model_id`` and the resolved entry in
  ``DIMER_MODEL_CONFIG_JSON``, plus user parameters through channels the validator does not
  receive.

Keeping the two contracts distinct prevents a validator from accidentally depending on a
finetuner-only variable and recreating COMPLIANCE.md C-1.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .errors import Code, ConfigError

# The callback must be a real HTTP(S) endpoint. urllib would otherwise happily open a
# file:// or ftp:// URL.
ALLOWED_CALLBACK_SCHEMES = frozenset({"http", "https"})

# Env keys that may be echoed into diagnostics. DIMER_DONE_CALLBACK is deliberately absent:
# it is a signed URL and must never reach logs or a result payload (SECURITY.md). The model
# config document is also excluded; diagnostics expose its KEY NAMES separately, never its
# values, because future registration fields may contain deployment-sensitive metadata.
LOGGABLE_ENV_KEYS = (
    "DIMER_DATASET_DIR",
    "DIMER_RESULT_PATH",
    "DIMER_OUTPUT_DIR",
    "DIMER_TRAIN_DEVICE",
    "DIMER_SESSION_ID",
    "DIMER_RUN_ID",
    "DIMER_TASK_TYPE",
    "DIMER_EXPECTED_ACCELERATOR",
)
VALIDATOR_LOGGABLE_ENV_KEYS = (
    "DIMER_DATASET_DIR",
    "DIMER_RESULT_PATH",
)

REDACTED = "<redacted>"


def _load_json_env(name: str) -> dict[str, Any]:
    raw = os.getenv(name, "") or ""
    raw = raw.strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"{name} is not valid JSON: {exc.msg} at position {exc.pos}.",
            code=Code.CONFIG_INVALID_JSON,
            details={"variable": name},
        ) from exc
    if not isinstance(parsed, dict):
        raise ConfigError(
            f"{name} must decode to a JSON object, got {type(parsed).__name__}.",
            code=Code.CONFIG_INVALID_JSON,
            details={"variable": name},
        )
    return parsed


@dataclass(frozen=True)
class DimerValidationEnv:
    """Source-verified environment delivered to a DIMER validator Job.

    Do not add a field here merely because another Job type receives it. The absence of
    preprocessing args and hyperparameters is part of the validator contract: the validator
    therefore cannot know the user's selected model and must remain model-agnostic.
    """

    dataset_dir: Path
    result_path: Path
    done_callback: str
    pipeline_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_environ(cls) -> DimerValidationEnv:
        """Parse only variables the backend actually injects into validator Jobs."""
        return cls(
            dataset_dir=Path(os.getenv("DIMER_DATASET_DIR", "/data/dataset")),
            result_path=Path(os.getenv("DIMER_RESULT_PATH", "/data/output/result/result.json")),
            done_callback=os.getenv("DIMER_DONE_CALLBACK", "").strip(),
            pipeline_metadata=_load_json_env("DIMER_PIPELINE_METADATA_JSON"),
        )

    def diagnostics(self) -> dict[str, Any]:
        """Validator env snapshot safe to embed in a result payload."""
        snapshot = {key: os.getenv(key, "") for key in VALIDATOR_LOGGABLE_ENV_KEYS}
        snapshot["DIMER_DONE_CALLBACK"] = REDACTED if self.done_callback else ""
        snapshot["pipelineMetadataKeys"] = sorted(self.pipeline_metadata)
        return snapshot


@dataclass(frozen=True)
class DimerEnv:
    """Parsed view of the broader DIMER finetuner environment."""

    dataset_dir: Path
    result_path: Path
    output_dir: Path | None
    done_callback: str
    train_device: str
    session_id: str
    run_id: str
    task_type: str
    expected_accelerator: str | None = None
    pipeline_metadata: dict[str, Any] = field(default_factory=dict)
    preprocessing_args: dict[str, Any] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    model_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_environ(cls) -> DimerEnv:
        """Build from os.environ.

        Call this inside the entrypoint's try/except so malformed platform input still
        produces a structured result rather than an import-time crash.
        """
        output_dir = os.getenv("DIMER_OUTPUT_DIR", "").strip()
        expected_accelerator = os.getenv("DIMER_EXPECTED_ACCELERATOR", "").strip().lower()
        return cls(
            dataset_dir=Path(os.getenv("DIMER_DATASET_DIR", "/data/dataset")),
            result_path=Path(os.getenv("DIMER_RESULT_PATH", "/data/output/result/result.json")),
            output_dir=Path(output_dir) if output_dir else None,
            done_callback=os.getenv("DIMER_DONE_CALLBACK", "").strip(),
            # DIMER documents bare "0" as a real value here; torch.device("0") raises
            # "Invalid device string" — a named failure mode in the portal's Common Errors.
            train_device=_normalize_device(os.getenv("DIMER_TRAIN_DEVICE", "cuda:0")),
            session_id=os.getenv("DIMER_SESSION_ID", "").strip(),
            run_id=os.getenv("DIMER_RUN_ID", "").strip(),
            # Custom / Other resolves to generic platform metadata, so the container bakes
            # its own truth. Never trust DIMER's resolved task type for this pipeline.
            task_type=os.getenv("DIMER_TASK_TYPE", "language_model_sft").strip(),
            expected_accelerator=expected_accelerator or None,
            pipeline_metadata=_load_json_env("DIMER_PIPELINE_METADATA_JSON"),
            preprocessing_args=_load_json_env("DIMER_PREPROCESSING_ARGS_JSON"),
            hyperparameters=_load_json_env("DIMER_HYPERPARAMETERS_JSON"),
            model_config=_load_json_env("DIMER_MODEL_CONFIG_JSON"),
        )

    # -- derived selectors -----------------------------------------------------

    @property
    def model_key(self) -> str | None:
        """DIMER's selected registry key, with the legacy image-side key as fallback.

        Source verification for C-2 established that the backend translates the user's
        model selection to ``DIMER_HYPERPARAMETERS_JSON.model_id`` and also injects the
        resolved ``fineTunableModels`` entry as ``DIMER_MODEL_CONFIG_JSON``. The old
        ``datasetPreprocessing.model_key`` path survives for hand-run/legacy jobs only.

        When more than one channel is present they MUST agree. Choosing one silently would
        let the container train a different model from the one DIMER recorded for the run.

        Unlike every other member of this class, reading this property can RAISE. Read it
        inside the entrypoint's structured-result ``try``/``except``, alongside
        ``from_environ`` -- never from inside an exception handler, where the raise would
        replace the failure being reported.
        """
        # Insertion order IS precedence: the platform's real selector, then the resolved
        # registration entry it produced, then the legacy image-side key. Keeping the two in
        # one literal is what stops the precedence list and the channel list from drifting.
        raw = {
            "hyperparameters.model_id": self.hyperparameters.get("model_id"),
            "modelConfig.id": self.model_config.get("id"),
            "preprocessing.model_key": self.preprocessing_args.get("model_key"),
        }

        # A registry key is a string. `str()` would coerce a JSON `false` into "False" and
        # report it one layer later as an unknown key, hiding a platform type defect behind
        # MODEL_KEY_MISSING.
        mistyped = {
            name: type(value).__name__
            for name, value in raw.items()
            if value is not None and not isinstance(value, str)
        }
        if mistyped:
            raise ConfigError(
                "DIMER supplied a non-string model selector.",
                code=Code.CONFIG_SCHEMA_INVALID,
                details={"selectorTypes": dict(sorted(mistyped.items()))},
            )

        # Every surviving value is `str | None`: the type gate above rejected anything else.
        selected = {
            name: value.strip()
            for name, value in raw.items()
            if value is not None and value.strip()
        }
        if len(set(selected.values())) > 1:
            # The values are approved registry keys, published in the Builder registration
            # and echoed in the model card -- not secrets. Naming them is what lets an
            # operator see WHICH channel is wrong from the result document alone, without
            # shell access to a pod that has already exited.
            raise ConfigError(
                "DIMER model-selection channels disagree; refusing to choose a model.",
                code=Code.CONFIG_SCHEMA_INVALID,
                details={"selectors": dict(sorted(selected.items()))},
            )
        return next(iter(selected.values()), None)

    @property
    def base_model(self) -> str | None:
        """DIMER's registered Base Model, when the platform happens to expose it.

        Display/provenance only. Used for cross-checking, never for selection.
        """
        for candidate in ("baseModel", "base_model", "baseModelId"):
            value = self.pipeline_metadata.get(candidate)
            if value:
                return str(value).strip()
        return None

    def diagnostics(self) -> dict[str, Any]:
        """Env snapshot safe to embed in a result payload.

        Allowlisted by key. Never dump os.environ: DIMER_DONE_CALLBACK is a signed URL and
        the container may also hold registry or Hub credentials.
        """
        snapshot = {key: os.getenv(key, "") for key in LOGGABLE_ENV_KEYS}
        snapshot["DIMER_DONE_CALLBACK"] = REDACTED if self.done_callback else ""
        snapshot["pipelineMetadataKeys"] = sorted(self.pipeline_metadata)
        snapshot["preprocessingArgKeys"] = sorted(self.preprocessing_args)
        snapshot["hyperparameterKeys"] = sorted(self.hyperparameters)
        snapshot["modelConfigKeys"] = sorted(self.model_config)
        return snapshot


def _normalize_device(value: str) -> str:
    """Accept DIMER's device spellings, including a bare ordinal.

    The portal documents `Invalid device string: '0'` as a common finetuner failure: DIMER
    may pass "0" where torch expects "cuda:0".
    """
    value = (value or "").strip()
    if not value:
        return "cuda:0"
    if value.isdigit():
        return f"cuda:{value}"
    return value


def notify_done_callback_url(url: str | None, *, timeout: float = 10.0) -> bool:
    """POST to a callback URL that may not have come from a constructed env object.

    Exists because the entrypoints must call back even when environment construction is what
    failed. Takes the raw URL rather than an env so the one code path that cannot rely on
    parsing still has a way to signal completion.

    Never raises and never logs the URL: it is a signed token.
    """
    if not url:
        return False

    # urllib registers file:// and ftp:// handlers by default, so an unexpected scheme
    # would be followed silently. The shipped Mitra validator guards this the same way.
    if urlparse(url).scheme not in ALLOWED_CALLBACK_SCHEMES:
        return False

    try:
        import urllib.request

        # Empty body, and deliberately NO Content-Type: declaring application/json with a
        # zero-length body is invalid JSON and a body-parsing endpoint may reject it.
        request = urllib.request.Request(url, data=b"", method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 300
    except Exception:
        # Swallow deliberately: a failed callback must not mask the real result, and the
        # exception text can embed the signed URL.
        return False


def notify_done_callback(
    env: DimerEnv | DimerValidationEnv, *, timeout: float = 10.0
) -> bool:
    """POST to DIMER_DONE_CALLBACK. Returns True when the platform acknowledged.

    MUST be called from a `finally` block. A validator that writes result.json but never
    calls back leaves the Workbench UI stuck at "Validating..." until the sweeper marks the
    run OVERDUE.

    Never raises and never logs the URL: it is a signed token.
    """
    return notify_done_callback_url(env.done_callback, timeout=timeout)
