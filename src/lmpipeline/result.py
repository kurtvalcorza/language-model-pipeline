"""The DIMER result contract.

DIMER's documented `result.json` shape is the OUTER object — Workbench reads and renders
`successful`, `message`, `datasetSummary`, `checks[]` and `metadata`. This pipeline's richer
fields (schema_version, code, provenance, metrics, timing, resources) are nested rather than
substituted, so nothing the platform consumes goes missing and nothing this project needs is
lost.

`metadata.classNames` is mandatory per the portal checklist and has its own named failure
mode. A language-model SFT run has no classes, so it emits `[]`.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import Code, PipelineError

SCHEMA_VERSION = "1.0"


@dataclass
class Check:
    """One named check, rendered as a row in the Workbench UI."""

    name: str
    successful: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "successful": self.successful, "message": self.message}


@dataclass
class Result:
    successful: bool
    message: str
    code: str
    stage: str
    checks: list[Check] = field(default_factory=list)
    dataset_summary: dict[str, Any] = field(default_factory=dict)
    task_type: str = "language_model_sft"
    supported_dataset_format: str = "jsonl_messages"
    metrics: dict[str, Any] = field(default_factory=dict)
    artifact: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def add_check(self, name: str, successful: bool, message: str) -> None:
        self.checks.append(Check(name, successful, message))

    def to_dict(self) -> dict[str, Any]:
        return {
            # -- shape DIMER documents and renders ----------------------------
            "successful": self.successful,
            "message": self.message,
            "datasetSummary": self.dataset_summary,
            "checks": [c.to_dict() for c in self.checks],
            "metadata": {
                "template": "dimer-language-model-sft",
                "taskType": self.task_type,
                "supportedDatasetFormat": self.supported_dataset_format,
                # Mandatory even though SFT has no classes; the portal explicitly allows [].
                "classNames": [],
                "classCount": 0,
                # -- this pipeline's extensions, nested so nothing is displaced --
                "languageModelPipeline": {
                    "schemaVersion": SCHEMA_VERSION,
                    "stage": self.stage,
                    "code": self.code,
                    "metrics": self.metrics,
                    "artifact": self.artifact,
                    "provenance": self.provenance,
                    "timing": self.timing,
                    "resources": self.resources,
                    "diagnostics": self.diagnostics,
                    "details": self.details,
                },
            },
        }

    # -- constructors ----------------------------------------------------------

    @classmethod
    def success(cls, *, stage: str, message: str, code: str, **kwargs: Any) -> Result:
        return cls(successful=True, message=message, code=code, stage=stage, **kwargs)

    @classmethod
    def failure(cls, *, stage: str, message: str, code: str, **kwargs: Any) -> Result:
        return cls(successful=False, message=message, code=code, stage=stage, **kwargs)

    @classmethod
    def from_exception(cls, exc: BaseException, *, stage: str, **kwargs: Any) -> Result:
        """Build a structured failure from any exception.

        PipelineError carries a stable code and safe message. Anything else is reported as
        RUNTIME_UNEXPECTED with the exception type but WITHOUT its string, which can embed
        signed URLs, file paths, or raw dataset content.

        Caller-supplied `details` are merged rather than passed alongside the exception's
        own, which would collide. The exception's details win on conflict: they describe
        what actually failed, while the caller is adding surrounding context.
        """
        caller_details = dict(kwargs.pop("details", None) or {})

        if isinstance(exc, PipelineError):
            return cls.failure(
                stage=stage, message=exc.message, code=exc.code,
                details={**caller_details, **exc.details}, **kwargs,
            )
        return cls.failure(
            stage=stage,
            message=(
                f"An unexpected {type(exc).__name__} ended this run. "
                "See the Job logs for the traceback."
            ),
            code=Code.RUNTIME_UNEXPECTED,
            details={**caller_details, "exceptionType": type(exc).__name__},
            **kwargs,
        )


def write_result(result: Result, path: Path | str) -> Path:
    """Write result.json atomically.

    A partially written result read by the platform is worse than none: it can present a
    failed run as a successful one. Write to a temp file in the same directory, then
    os.replace, which is atomic on the same filesystem.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)

    handle, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".result-", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return path
