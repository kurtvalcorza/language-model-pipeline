"""Deterministic subset selection.

Never "first N": ordering is an accident of how upstream happens to serialize a dataset,
and a subset that depends on it silently changes when upstream reorders. Selection is by
stable hash of a canonical row identity, so the same profile against the same pinned
revision always yields the same rows, in the same order.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any


def content_hash(row: dict[str, Any], key_fields: tuple[str, ...]) -> str:
    """Hash of the row's meaningful content, independent of its position."""
    payload = {k: row.get(k) for k in key_fields}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def row_identity(row: dict[str, Any], *, occurrence: int,
                 key_fields: tuple[str, ...]) -> str:
    """Stable identity for a source row.

    Content plus an **occurrence counter**, not an absolute index. Two identical rows need
    distinct identities so a duplicate does not collapse into one selection slot, but using
    the absolute position would make the whole selection depend on upstream ordering —
    precisely the property this module exists to avoid. Occurrence numbering is
    order-independent in aggregate, because the rows it distinguishes are identical.
    """
    return f"{content_hash(row, key_fields)}:{occurrence}"


def _rank(identity: str, *, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{identity}".encode()).hexdigest()


def select(
    rows: Iterable[dict[str, Any]],
    *,
    count: int | None,
    salt: str,
    key_fields: tuple[str, ...],
    selection: str = "stable_hash",
) -> list[tuple[int, dict[str, Any]]]:
    """Return (source_index, row) pairs, deterministically.

    `salt` should identify the profile, so smoke_100 and smoke_500 are not simply nested
    prefixes of one ordering — that would make the larger profile's extra rows
    systematically different in character from its first hundred.
    """
    indexed = list(enumerate(rows))

    if selection == "all" or count is None or count >= len(indexed):
        return indexed

    if selection != "stable_hash":
        raise ValueError(f"unknown selection strategy {selection!r}")

    seen: dict[str, int] = {}
    ranked = []
    for index, row in indexed:
        digest = content_hash(row, key_fields)
        occurrence = seen.get(digest, 0)
        seen[digest] = occurrence + 1
        identity = f"{digest}:{occurrence}"
        ranked.append((_rank(identity, salt=salt), index, row))

    ranked.sort(key=lambda item: item[0])
    chosen = ranked[:count]
    # Emit in source order so the output is stable and diffable against the source.
    return [(index, row) for _, index, row in sorted(chosen, key=lambda item: item[1])]
