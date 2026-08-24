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
  model selection through ``DIMER_HYPERPARAMETERS_JSON.model_id`` and
  ``DIMER_MODEL_CONFIG_JSON``, plus training parameters and deployment metadata the validator
  does not receive.

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

ALLOWED_CALLBACK_SCHEMES = frozenset({"http", "https"})

# Env keys that may be echoed into diagnostics. DIMER_DONE_CALLBACK is deliberately absent:
# it is a signed URL and must never reach logs or a result payload. DIMER_MODEL_CONFIG_JSON
# is also absent: diagnostics expose its key NAMES, not the document values.
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
    """Source-verified environment delivered to a DIMER validator Job."""

    dataset_dir: Path
    result_path: Path
    done_callback: str
    pipeline_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_environ(cls) -> "DimerValidationEnv":
        """Parse only variables the backend actually injects into validator Jobs."""
        return cls(
            dataset_dir=Path(os.getenv("DIMER_DATASET_DIR", "/data/dataset")),
            result_path=Path(os.getenv("DIMER_RESULT_PATH", "/data/output/result/result.json")),
            done_callback=os.getenv("DIMER_DONE_CALLBACK", "").strip(),
            pipeline_metadata=_load_json_env("DIMER_PIPELINE_METADATA_JSON"),
        )

    def diagnostics(self) -> dict[str, Any]:
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
    def from_environ(cls) -> "DimerEnv":
        """Build from source-verified finetuner environment channels."""
        output_dir = os.getenv("DIMER_OUTPUT_DIR", "").strip()
        expected_accelerator = os.getenv("DIMER_EXPECTED_ACCELERATOR", "").strip().lower()
        return cls(
            dataset_dir=Path(os.getenv("DIMER_DATASET_DIR", "/data/dataset")),
            result_path=Path(os.getenv("DIMER_RESULT_PATH", "/data/output/result/result.json")),
            output_dir=Path(output_dir) if output_dir else None,
            done_callback=os.getenv("DIMER_DONE_CALLBACK", "").strip(),
            # DIMER documents bare "0" as a real value here; torch.device("0") raises.
            train_device=_normalize_device(os.getenv("DIMER_TRAIN_DEVICE", "cuda:0")),
            session_id=os.getenv("DIMER_SESSION_ID", "").strip(),
            run_id=os.getenv("DIMER_RUN_ID", "").strip(),
            # Custom / Other currently normalizes to generic platform metadata, so the
            # container bakes its own task truth rather than trusting this value.
            task_type=os.getenv("DIMER_TASK_TYPE", "language_model_sft").strip(),
            expected_accelerator=expected_accelerator or None,
            pipeline_metadata=_load_json_env("DIMER_PIPELINE_METADATA_JSON"),
            preprocessing_args=_load_json_env("DIMER_PREPROCESSING_ARGS_JSON"),
            hyperparameters=_load_json_env("DIMER_HYPERPARAMETERS_JSON"),
            model_config=_load_json_env("DIMER_MODEL_CONFIG_JSON"),
        )

    @property
    def model_key(self) -> str | None:
        """Return DIMER's selected registry key, failing closed if selectors disagree.

        Backend source establishes ``DIMER_HYPERPARAMETERS_JSON.model_id`` as the real user
        selection and ``DIMER_MODEL_CONFIG_JSON`` as the resolved registered entry. The old
        image-side ``datasetPreprocessing.model_key`` is retained only as a compatibility
        fallback for hand-run/legacy jobs. If more than one selector is supplied they must
        name the same registry entry; silently choosing one would train a different model
        from the one DIMER recorded for the run.
        """
        selectors = {
            "hyperparameters.model_id": self.hyperparameters.get("model_id"),
            "modelConfig.id": self.model_config.get("id"),
            "preprocessing.model_key": self.preprocessing_args.get("model_key"),
        }
        normalized = {
            name: str(value).strip()
            for name, value in selectors.items()
            if value is not None and str(value).strip()
        }
        values = set(normalized.values())
        if len(values) > 1:
            raise ConfigError(
                "DIMER model-selection channels disagree; refusing to choose a model.",
                code=Code.CONFIG_SCHEMA_INVALID,
                details={"selectors": sorted(normalized)},
            )
        if not values:
            return None
        # Prefer the source-verified user selector, then the resolved model config, then the
        # legacy preprocessing key. Agreement above makes the value identical when multiple
        # channels are present.
        for name in (
            "hyperparameters.model_id",
            "modelConfig.id",
            "preprocessing.model_key",
        ):
            if name in normalized:
                return normalized[name]
        return None  # pragma: no cover - normalized/values already prove one exists

    @property
    def base_model(self) -> str | None:
        """DIMER's registered Base Model, when exposed; display/provenance only."""
        for candidate in ("baseModel", "base_model", "baseModelId"):
            value = self.pipeline_metadata.get(candidate)
            if value:
                return str(value).strip()
        return None

    def diagnostics(self) -> dict[str, Any]:
        """Allowlisted env snapshot safe to embed in a result payload."""
        snapshot = {key: os.getenv(key, "") for key in LOGGABLE_ENV_KEYS}
        snapshot["DIMER_DONE_CALLBACK"] = REDACTED if self.done_callback else ""
        snapshot["pipelineMetadataKeys"] = sorted(self.pipeline_metadata)
        snapshot["preprocessingArgKeys"] = sorted(self.preprocessing_args)
        snapshot["hyperparameterKeys"] = sorted(self.hyperparameters)
        snapshot["modelConfigKeys"] = sorted(self.model_config)
        return snapshot


def _normalize_device(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return "cuda:0"
    if value.isdigit():
        return f"cuda:{value}"
    return value


def notify_done_callback_url(url: str | None, *, timeout: float = 10.0) -> bool:
    """POST to a callback URL that may not have come from a constructed env object."""
    if not url:
        return False
    if urlparse(url).scheme not in ALLOWED_CALLBACK_SCHEMES:
        return False

    try:
        import urllib.request

        request = urllib.request.Request(url, data=b"", method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def notify_done_callback(
    env: DimerEnv | DimerValidationEnv, *, timeout: float = 10.0
) -> bool:
    return notify_done_callback_url(env.done_callback, timeout=timeout)
