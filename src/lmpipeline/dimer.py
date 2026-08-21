"""The DIMER container runtime contract.

Every DIMER-specific assumption lives here so the rest of the package stays platform-neutral
and unit-testable. The variable set is transcribed from the DIMER AI Engineer portal
documentation (see DEPLOYMENT.md), not guessed.

Note the two facts that shape this module:

  * DIMER's Pipeline Builder "Base Model" field is NOT reliably delivered to Jobs. The
    authoritative runtime selector is `model_key`, carried in DIMER_PREPROCESSING_ARGS_JSON,
    which is the only user-parameter channel proven to reach both containers.
  * DIMER_DONE_CALLBACK must be POSTed even on crash, or the Workbench UI hangs at
    "Validating..." until the sweeper times the run out.
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
# it is a signed URL and must never reach logs or a result payload (SECURITY.md).
LOGGABLE_ENV_KEYS = (
    "DIMER_DATASET_DIR",
    "DIMER_RESULT_PATH",
    "DIMER_OUTPUT_DIR",
    "DIMER_TRAIN_DEVICE",
    "DIMER_SESSION_ID",
    "DIMER_RUN_ID",
    "DIMER_TASK_TYPE",
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
class DimerEnv:
    """Parsed view of the DIMER-injected environment."""

    dataset_dir: Path
    result_path: Path
    output_dir: Path | None
    done_callback: str
    train_device: str
    session_id: str
    run_id: str
    task_type: str
    pipeline_metadata: dict[str, Any] = field(default_factory=dict)
    preprocessing_args: dict[str, Any] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_environ(cls) -> DimerEnv:
        """Build from os.environ.

        Call this inside the entrypoint's try/except so malformed platform input still
        produces a structured result rather than an import-time crash.
        """
        output_dir = os.getenv("DIMER_OUTPUT_DIR", "").strip()
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
            pipeline_metadata=_load_json_env("DIMER_PIPELINE_METADATA_JSON"),
            preprocessing_args=_load_json_env("DIMER_PREPROCESSING_ARGS_JSON"),
            hyperparameters=_load_json_env("DIMER_HYPERPARAMETERS_JSON"),
        )

    # -- derived selectors -----------------------------------------------------

    @property
    def model_key(self) -> str | None:
        """The authoritative runtime model selector.

        Declared once in dimer-pipeline.json under `datasetPreprocessing`, which is the
        only user-parameter channel that reaches both the validator and the finetuner.
        """
        value = self.preprocessing_args.get("model_key")
        return str(value).strip() if value else None

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


def notify_done_callback(env: DimerEnv, *, timeout: float = 10.0) -> bool:
    """POST to DIMER_DONE_CALLBACK. Returns True when the platform acknowledged.

    MUST be called from a `finally` block. A validator that writes result.json but never
    calls back leaves the Workbench UI stuck at "Validating..." until the sweeper marks the
    run OVERDUE.

    Never raises and never logs the URL: it is a signed token.
    """
    if not env.done_callback:
        return False

    # urllib registers file:// and ftp:// handlers by default, so an unexpected scheme
    # would be followed silently. The shipped Mitra validator guards this the same way.
    if urlparse(env.done_callback).scheme not in ALLOWED_CALLBACK_SCHEMES:
        return False

    try:
        import urllib.request

        # Empty body, and deliberately NO Content-Type: declaring application/json with a
        # zero-length body is invalid JSON and a body-parsing endpoint may reject it.
        request = urllib.request.Request(env.done_callback, data=b"", method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 300
    except Exception:
        # Swallow deliberately: a failed callback must not mask the real result, and the
        # exception text can embed the signed URL.
        return False
