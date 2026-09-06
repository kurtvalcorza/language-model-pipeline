"""Bounded deterministic selection for very large acceptance sources.

The existing selector intentionally materializes its input so it can emit rows in source
order. That is appropriate for the small Dolly/UNER acceptance sources, but not for a
908k-row / multi-gigabyte SEA-Instruct split. This module keeps only the best ``count`` rows
by the same stable content-hash ranking while scanning an arbitrary iterable once.
"""

from __future__ import annotations

import hashlib
import heapq
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .subset import content_hash


@dataclass(frozen=True)
class BoundedSelection:
    """One selected source row plus deterministic identity/rank evidence."""

    source_index: int
    identity: str
    rank: str
    row: dict[str, Any]


def _identity(content_digest: str, occurrence: int) -> str:
    return f"{content_digest}:{occurrence}"


def _rank(identity: str, *, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{identity}".encode("utf-8")).hexdigest()


def select_bounded(
    rows: Iterable[dict[str, Any]],
    *,
    count: int,
    salt: str,
    key_fields: tuple[str, ...],
) -> list[BoundedSelection]:
    """Select the ``count`` smallest stable ranks in one pass with O(count) row memory.

    Duplicate source rows remain distinct through an occurrence counter, matching the
    existing acceptance-suite policy. Occurrence numbers for identical content are stable as
    a multiset regardless of where those identical rows appear, so reordering the source does
    not change which *content occurrences* win.

    Output is rank-ordered rather than source-index-ordered. That is deliberate: source
    indices can change when upstream serialization order changes, while rank/identity are
    content-derived and therefore give a byte-stable order for the generated profile.
    """
    if count <= 0:
        raise ValueError("count must be a positive integer")

    # Python provides a min-heap. Store negative rank integers so heap[0] is the currently
    # WORST (largest) retained rank, which can be replaced whenever a better row arrives.
    heap: list[tuple[int, str, int, dict[str, Any]]] = []
    seen: dict[str, int] = {}

    for source_index, row in enumerate(rows):
        digest = content_hash(row, key_fields)
        occurrence = seen.get(digest, 0)
        seen[digest] = occurrence + 1
        identity = _identity(digest, occurrence)
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
    return hashlib.sha256("\n".join(identities).encode("utf-8")).hexdigest()
