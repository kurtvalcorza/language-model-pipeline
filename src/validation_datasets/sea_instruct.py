"""Fail-closed SEA-Instruct conversation parsing.

The pinned source stores ``conversations`` as a string representation of a list of message
objects. The source remains disabled until a real gated fetch proves that representation is
strict JSON accepted here. We do not add Python-literal fallbacks: a parser that quietly
accepts a second grammar makes the source contract ambiguous and expands the attack surface.
"""

from __future__ import annotations

import json
from typing import Any

APPROVED_ROLES = frozenset({"system", "user", "assistant"})
SEA_IDENTITY_FIELDS = ("conversations_id", "conversations")


class SeaInstructError(ValueError):
    """Raised when a source row cannot be mapped safely to canonical SFT messages."""


def parse_conversations(value: Any, *, index: int) -> list[dict[str, str]]:
    """Parse the stringified message list and retain semantic role/content only."""
    if not isinstance(value, str) or not value.strip():
        raise SeaInstructError(f"row {index}: 'conversations' must be a non-empty string")
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SeaInstructError(
            f"row {index}: 'conversations' is not strict JSON. The pinned gated source "
            "must be inspected before another grammar is supported."
        ) from exc

    if not isinstance(raw, list) or not raw:
        raise SeaInstructError(
            f"row {index}: parsed 'conversations' must be a non-empty list"
        )

    messages: list[dict[str, str]] = []
    for message_index, message in enumerate(raw):
        where = f"row {index} message {message_index}"
        if not isinstance(message, dict):
            raise SeaInstructError(f"{where}: message must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in APPROVED_ROLES:
            raise SeaInstructError(
                f"{where}: role {role!r} is not one of {sorted(APPROVED_ROLES)}"
            )
        if not isinstance(content, str) or not content.strip():
            raise SeaInstructError(f"{where}: content must be a non-empty string")
        messages.append({"role": role, "content": content})

    if not any(message["role"] == "assistant" for message in messages):
        raise SeaInstructError(
            f"row {index}: conversation has no assistant target and cannot be SFT data"
        )
    return messages


def canonicalize_sea_instruct(row: dict[str, Any], *, index: int) -> dict[str, Any]:
    """Map one source row to the existing canonical messages contract.

    Source/tagging metadata is intentionally not injected into the training text. It remains
    available to the builder for manifest/evidence reporting and profile policy.
    """
    conversation_id = row.get("conversations_id")
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        raise SeaInstructError(
            f"row {index}: 'conversations_id' must be a non-empty string"
        )
    return {"messages": parse_conversations(row.get("conversations"), index=index)}
