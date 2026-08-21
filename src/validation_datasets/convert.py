"""Source-specific adapters into the canonical `messages` form.

Each converter is a narrow, deterministic mapping. It never invents semantic content, never
paraphrases, and never drops a row silently — a row it cannot map raises.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

CANONICAL_KEY_FIELDS: dict[str, tuple[str, ...]] = {
    "dolly-15k": ("instruction", "context", "response"),
    "uner-tagalog": ("inputs", "targets"),
    "uner-multilingual": ("inputs", "targets"),
}


class ConversionError(ValueError):
    """A source row that cannot be mapped. Never swallowed."""


def _require(row: dict[str, Any], field: str, *, index: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ConversionError(
            f"row {index}: field {field!r} is missing or empty"
        )
    return value


def convert_dolly(row: dict[str, Any], *, index: int) -> dict[str, Any]:
    """Dolly: instruction (+ optional context) -> user, response -> assistant.

    The second field is `context`, not `input`. Context is appended to the user turn rather
    than becoming a system message: it is passage material the answer depends on, not a
    behavioural instruction.
    """
    instruction = _require(row, "instruction", index=index)
    response = _require(row, "response", index=index)
    context = row.get("context") or ""

    user = f"{instruction}\n\n{context}".strip() if str(context).strip() else instruction
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": response},
    ]}


def convert_uner(row: dict[str, Any], *, index: int) -> dict[str, Any]:
    """Universal NER in Aya instruction format: inputs -> user, targets -> assistant.

    Language metadata stays in the manifest rather than being injected into the training
    text as hidden control tokens.
    """
    return {"messages": [
        {"role": "user", "content": _require(row, "inputs", index=index)},
        {"role": "assistant", "content": _require(row, "targets", index=index)},
    ]}


CONVERTERS: dict[str, Callable[..., dict[str, Any]]] = {
    "dolly-15k": convert_dolly,
    "uner-tagalog": convert_uner,
    "uner-multilingual": convert_uner,
}


def to_canonical(source_id: str, row: dict[str, Any], *, index: int) -> dict[str, Any]:
    if source_id not in CONVERTERS:
        raise ConversionError(
            f"no converter for {source_id!r}. Inspect the pinned source schema before "
            "writing one; do not guess field names from the dataset description."
        )
    return CONVERTERS[source_id](row, index=index)
