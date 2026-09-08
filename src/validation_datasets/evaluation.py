"""Evaluation-only dataset adapters and policy.

Evaluation data has a different contract from SFT data.  In particular, Kalahi is held-out
Filipino evaluation material: making it fetchable must never make it trainable, and its
upstream labels must be preserved rather than reinterpreted by this dataset-preparation
layer.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .registry import DatasetSource, RegistryError

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
FINGERPRINT_CHARS = 16

EVALUATION_KEY_FIELDS: dict[str, tuple[str, ...]] = {
    "kalahi": ("id", "label", "prompts", "metadata"),
}


class EvaluationConversionError(ValueError):
    """A source row that cannot be represented faithfully as evaluation data."""


def _required_string(
    mapping: dict[str, Any], field: str, *, where: str, allow_empty: bool = False
) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise EvaluationConversionError(
            f"{where}: field {field!r} must be a"
            + (" string" if allow_empty else " non-empty string")
        )
    return value


def canonicalize_kalahi(row: dict[str, Any], *, index: int) -> dict[str, Any]:
    """Preserve one pinned Kalahi MCQ-compatible row without inventing score semantics.

    The upstream dataset card defines ``id``, opaque ``label``, one or more prompt variants
    carrying ``question``/``mcq_options``/``mcq``, and language/category/topic metadata.
    This adapter validates that shape and keeps those values verbatim.  It deliberately does
    not parse ``mcq_options`` or assign meaning to ``label``; scoring belongs to a separately
    versioned evaluation harness.
    """
    where = f"row {index}"
    row_id = _required_string(row, "id", where=where)
    label = _required_string(row, "label", where=where)

    raw_prompts = row.get("prompts")
    if not isinstance(raw_prompts, list) or not raw_prompts:
        raise EvaluationConversionError(
            f"{where}: field 'prompts' must be a non-empty list"
        )

    prompts: list[dict[str, str]] = []
    for prompt_index, raw_prompt in enumerate(raw_prompts):
        prompt_where = f"{where} prompt {prompt_index}"
        if not isinstance(raw_prompt, dict):
            raise EvaluationConversionError(f"{prompt_where}: prompt must be an object")
        prompts.append(
            {
                "question": _required_string(raw_prompt, "question", where=prompt_where),
                "mcq_options": _required_string(
                    raw_prompt, "mcq_options", where=prompt_where
                ),
                "mcq": _required_string(raw_prompt, "mcq", where=prompt_where),
            }
        )

    raw_metadata = row.get("metadata")
    if not isinstance(raw_metadata, dict):
        raise EvaluationConversionError(f"{where}: field 'metadata' must be an object")

    metadata = {
        "language": _required_string(raw_metadata, "language", where=f"{where} metadata"),
        # Category/topic are typed as strings upstream. Preserve even an empty string rather
        # than making a stronger content assumption than the pinned schema documents.
        "category": _required_string(
            raw_metadata, "category", where=f"{where} metadata", allow_empty=True
        ),
        "topic": _required_string(
            raw_metadata, "topic", where=f"{where} metadata", allow_empty=True
        ),
    }

    return {
        "id": row_id,
        "label": label,
        "prompts": prompts,
        "metadata": metadata,
    }


def to_evaluation(source_id: str, row: dict[str, Any], *, index: int) -> dict[str, Any]:
    """Convert one source row to its evaluation representation, failing closed."""
    if source_id == "kalahi":
        return canonicalize_kalahi(row, index=index)
    raise EvaluationConversionError(
        f"no evaluation adapter for {source_id!r}. Inspect the pinned source schema and "
        "scoring contract before adding one; do not coerce SFT rows into evaluation data."
    )


def evaluation_fingerprints(records: list[dict[str, Any]]) -> list[str]:
    """Stable privacy-preserving fingerprints for arbitrary evaluation records."""
    fingerprints = set()
    for record in records:
        payload = json.dumps(
            record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        fingerprints.add(hashlib.sha256(payload).hexdigest()[:FINGERPRINT_CHARS])
    return sorted(fingerprints)


def require_usable_for_evaluation(source: DatasetSource) -> None:
    """Require an enabled, explicitly evaluation-only, immutably pinned source."""
    if not source.enabled:
        raise RegistryError(
            f"{source.id!r} is disabled (approval_state={source.approval_state!r})."
        )
    if not source.is_evaluation_only or source.pipeline_usage != "evaluation":
        raise RegistryError(
            f"{source.id!r} is not registered as evaluation-only data. Refusing to blur "
            "the training/evaluation boundary."
        )
    if not _SHA_RE.match(source.revision or ""):
        raise RegistryError(
            f"{source.id!r} has no immutable pinned revision (got {source.revision!r})."
        )
    if not source.config or not source.split:
        raise RegistryError(
            f"{source.id!r} has no explicit config/split. Inspect and pin the source "
            "contract before evaluation data is fetched."
        )
    if source.canonical_schema != "evaluation":
        raise RegistryError(
            f"{source.id!r} declares canonical_schema={source.canonical_schema!r}, not "
            "'evaluation'."
        )
