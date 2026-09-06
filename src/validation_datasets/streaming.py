"""Bounded deterministic selection for very large acceptance sources.

The existing selector intentionally materializes its input so it can preserve duplicate-row
occurrence semantics and emit rows in source order. That is appropriate for small Dolly/UNER
sources, but not for a 908k-row / multi-gigabyte SEA-Instruct split.

Large-source selection uses an explicit stable source identity (SEA-Instruct documents
``conversations_id`` as unique) plus a digest of the canonical selection fields. That removes
the need to remember every fingerprint seen during the scan: only the best ``count`` source
rows are retained, so memory is O(profile size).
"""

from __future__ import annotations

import hashlib
import heapq
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .subset import content_hash


class StreamingSelectionError(ValueError):
    """The large source lacks the stable identity required for bounded selection."""


@dataclass(frozen=True)
class BoundedSelection:
    """One selected source row plus deterministic identity/rank evidence."""

    source_index: int
    identity: str
    rank: str
    row: dict[str, Any]


def _rank(identity: str, *, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{identity}".encode()).hexdigest()


def _stable_identity(
    row: dict[str, Any],
    *,
    identity_field: str,
    key_fields: tuple[str, ...],
    source_index: int,
) -> str:
    source_identity = row.get(identity_field)
    if not isinstance(source_identity, str) or not source_identity.strip():
        raise StreamingSelectionError(
            f"row {source_index}: stable identity field {identity_field!r} must be a "
            "non-empty string"
        )
    # Including meaningful content makes an upstream ID collision deterministic without a
    # global duplicate table: same-ID/different-content rows rank independently; exact
    # duplicate rows remain byte-identical even if their source positions are swapped.
    digest = content_hash(row, key_fields)
    return f"{source_identity}:{digest}"


def select_bounded(
    rows: Iterable[dict[str, Any]],
    *,
    count: int,
    salt: str,
    identity_field: str,
    key_fields: tuple[str, ...],
) -> list[BoundedSelection]:
    """Select the ``count`` smallest stable ranks in one pass with O(count) memory.

    Output is rank-ordered rather than source-index-ordered. Source indices can change when
    upstream serialization order changes, while the identity/rank are derived only from a
    pinned source ID and meaningful row content. Reordering therefore cannot change either
    the selected content or generated output order.
    """
    if count <= 0:
        raise ValueError("count must be a positive integer")

    # Python provides a min-heap. Store negative rank integers so heap[0] is the currently
    # WORST (largest) retained rank, which can be replaced whenever a better row arrives.
    heap: list[tuple[int, str, int, dict[str, Any]]] = []

    for source_index, row in enumerate(rows):
        identity = _stable_identity(
            row,
            identity_field=identity_field,
            key_fields=key_fields,
            source_index=source_index,
        )
        rank = _rank(identity, salt=salt)
        rank_int = int(rank, 16)
        candidate = (-rank_int, identity, source_index, row)

        if len(heap) < count:
            heapq.heappush(heap, candidate)
            continue

        worst_rank_int = -heap[0][0]
        if rank_int < worst_rank_int:
            heapq.heapreplace(heap, candidate)

    selected = [
        BoundedSelection(
            source_index=source_index,
            identity=identity,
            rank=f"{-negative_rank:064x}",
            row=row,
        )
        for negative_rank, identity, source_index, row in heap
    ]
    selected.sort(key=lambda item: (item.rank, item.identity))
    return selected


def identity_digest(selected: Iterable[BoundedSelection]) -> str:
    """Order-independent digest proving which stable source identities were selected."""
    identities = sorted(item.identity for item in selected)
    return hashlib.sha256("\n".join(identities).encode()).hexdigest()
