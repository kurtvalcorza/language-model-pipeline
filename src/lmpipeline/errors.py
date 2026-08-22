"""Stable error-code namespace shared by the validator and the finetuner.

Codes are part of the published contract: they appear in `result.json` and are read by
operators and by cross-repo tests. Renaming one is a breaking change. Add new codes rather
than repurposing existing ones.
"""

from __future__ import annotations


class Stage:
    """Which container produced a result."""

    VALIDATION = "validation"
    TRAINING = "training"


class Code:
    """Stable failure/success codes.

    Grouped by prefix so operators can filter: DATASET_*, MODEL_*, CONFIG_*, RESOURCE_*,
    ARTIFACT_*, RUNTIME_*.
    """

    # -- success ---------------------------------------------------------------
    VALIDATION_SUCCEEDED = "VALIDATION_SUCCEEDED"
    TRAINING_SUCCEEDED = "TRAINING_SUCCEEDED"

    # -- dataset transport / archive safety ------------------------------------
    DATASET_MISSING = "DATASET_MISSING"
    DATASET_ARCHIVE_UNREADABLE = "DATASET_ARCHIVE_UNREADABLE"
    DATASET_ARCHIVE_UNSAFE = "DATASET_ARCHIVE_UNSAFE"
    DATASET_ARCHIVE_TOO_LARGE = "DATASET_ARCHIVE_TOO_LARGE"
    DATASET_ARCHIVE_NESTED = "DATASET_ARCHIVE_NESTED"
    DATASET_ARCHIVE_DUPLICATE_MEMBER = "DATASET_ARCHIVE_DUPLICATE_MEMBER"

    # -- split resolution ------------------------------------------------------
    DATASET_SPLIT_MISSING = "DATASET_SPLIT_MISSING"
    DATASET_SPLIT_AMBIGUOUS = "DATASET_SPLIT_AMBIGUOUS"
    # A split that is transport-valid but too large to ingest. Distinct from
    # DATASET_ARCHIVE_TOO_LARGE: this one also fires for mounted directories, which carry
    # no archive metadata to bound.
    DATASET_SPLIT_TOO_LARGE = "DATASET_SPLIT_TOO_LARGE"

    # -- record-level schema ---------------------------------------------------
    DATASET_INVALID_UTF8 = "DATASET_INVALID_UTF8"
    DATASET_INVALID_JSON = "DATASET_INVALID_JSON"
    DATASET_SCHEMA_MIXED = "DATASET_SCHEMA_MIXED"
    DATASET_SCHEMA_UNKNOWN = "DATASET_SCHEMA_UNKNOWN"
    DATASET_ROLE_INVALID = "DATASET_ROLE_INVALID"
    DATASET_TARGET_EMPTY = "DATASET_TARGET_EMPTY"
    DATASET_LINE_TOO_LARGE = "DATASET_LINE_TOO_LARGE"
    DATASET_TOO_FEW_EXAMPLES = "DATASET_TOO_FEW_EXAMPLES"
    DATASET_TOO_MANY_EXAMPLES = "DATASET_TOO_MANY_EXAMPLES"

    # -- tokenizer-aware checks ------------------------------------------------
    DATASET_SEQUENCE_TOO_LONG = "DATASET_SEQUENCE_TOO_LONG"
    DATASET_TOKEN_BUDGET_EXCEEDED = "DATASET_TOKEN_BUDGET_EXCEEDED"
    DATASET_CHAT_TEMPLATE_MISSING = "DATASET_CHAT_TEMPLATE_MISSING"

    # -- duplicates / leakage --------------------------------------------------
    DATASET_SPLIT_LEAKAGE = "DATASET_SPLIT_LEAKAGE"

    # -- model registry --------------------------------------------------------
    MODEL_KEY_MISSING = "MODEL_KEY_MISSING"
    MODEL_NOT_APPROVED = "MODEL_NOT_APPROVED"
    MODEL_DISABLED = "MODEL_DISABLED"
    MODEL_REMOTE_CODE_BLOCKED = "MODEL_REMOTE_CODE_BLOCKED"
    MODEL_REVISION_UNPINNED = "MODEL_REVISION_UNPINNED"
    MODEL_REVISION_MISMATCH = "MODEL_REVISION_MISMATCH"
    MODEL_BASE_MODEL_CONFLICT = "MODEL_BASE_MODEL_CONFLICT"
    MODEL_TOKENIZER_UNAVAILABLE = "MODEL_TOKENIZER_UNAVAILABLE"

    # -- configuration ---------------------------------------------------------
    CONFIG_INVALID_JSON = "CONFIG_INVALID_JSON"
    CONFIG_SCHEMA_INVALID = "CONFIG_SCHEMA_INVALID"
    CONFIG_OUT_OF_BOUNDS = "CONFIG_OUT_OF_BOUNDS"
    CONFIG_METHOD_UNSUPPORTED = "CONFIG_METHOD_UNSUPPORTED"

    # -- resources / runtime ---------------------------------------------------
    RESOURCE_GPU_UNAVAILABLE = "RESOURCE_GPU_UNAVAILABLE"
    RESOURCE_INSUFFICIENT_VRAM = "RESOURCE_INSUFFICIENT_VRAM"
    RESOURCE_QUANTIZATION_UNSUPPORTED = "RESOURCE_QUANTIZATION_UNSUPPORTED"
    RESOURCE_OOM = "RESOURCE_OOM"
    RESOURCE_WALL_TIME_EXCEEDED = "RESOURCE_WALL_TIME_EXCEEDED"
    RUNTIME_CANCELLED = "RUNTIME_CANCELLED"
    RUNTIME_UNEXPECTED = "RUNTIME_UNEXPECTED"

    # -- artifact --------------------------------------------------------------
    ARTIFACT_PACKAGING_FAILED = "ARTIFACT_PACKAGING_FAILED"
    ARTIFACT_VERIFICATION_FAILED = "ARTIFACT_VERIFICATION_FAILED"
    ARTIFACT_TOO_LARGE = "ARTIFACT_TOO_LARGE"
    ARTIFACT_TOKENIZER_MUTATED = "ARTIFACT_TOKENIZER_MUTATED"


class PipelineError(Exception):
    """Base for every failure that should surface as a structured result.

    Anything raised as a PipelineError produces a `result.json` with `successful: false`
    and a stable `code`. Bare exceptions are caught at the entrypoint boundary and reported
    as RUNTIME_UNEXPECTED, which is deliberately the only opaque code.
    """

    code = Code.RUNTIME_UNEXPECTED

    def __init__(self, message: str, *, code: str | None = None, details: dict | None = None):
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        self.details = details or {}


class DatasetError(PipelineError):
    code = Code.DATASET_SCHEMA_UNKNOWN


class ModelError(PipelineError):
    code = Code.MODEL_NOT_APPROVED


class ConfigError(PipelineError):
    code = Code.CONFIG_SCHEMA_INVALID


class ResourceError(PipelineError):
    code = Code.RESOURCE_GPU_UNAVAILABLE


class ArtifactError(PipelineError):
    code = Code.ARTIFACT_PACKAGING_FAILED
