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


class LanguageFilterError(RuntimeError):
    """Raised when a language exclusion cannot be applied with certainty."""


# Minimum length for a derived prompt prefix. A short common prefix would be shared by every
# language in the corpus and would exclude the whole dataset.
MIN_PROMPT_PREFIX_CHARS = 100


def derive_prompt_prefix(
    rows: list[dict[str, Any]], *, field: str, dataset_id: str
) -> str:
    """Longest common prefix of `field` across every row of a single-language corpus.

    Universal NER in the Aya format carries **no language column** — the rows are just
    `inputs` and `targets`. What it does carry is a per-language instruction preamble,
    byte-identical across every row in that language: Danish rows all begin "Angiv venligst
    alle navngivne enheder...", Tagalog rows all begin "Sa aktibidad na ito, kailangan mong
    hanapin...". So the prefix of a known single-language corpus identifies that language's
    rows inside the multilingual one.

    This is why the exclusion is not `language_field` based for this source: there is no
    field to filter on, and inventing one would have meant a filter that silently matched
    nothing.
    """
    if not rows:
        raise LanguageFilterError(f"{dataset_id!r} returned no rows to derive a prefix from.")

    values = []
    for index, row in enumerate(rows):
        value = row.get(field)
        if not isinstance(value, str):
            raise LanguageFilterError(
                f"{dataset_id!r}: row {index} has no string {field!r} to derive a prefix "
                "from."
            )
        values.append(value)

    prefix = values[0]
    for value in values[1:]:
        limit = min(len(prefix), len(value))
        cut = limit
        for position in range(limit):
            if prefix[position] != value[position]:
                cut = position
                break
        prefix = prefix[:cut]
        if len(prefix) < MIN_PROMPT_PREFIX_CHARS:
            break

    if len(prefix) < MIN_PROMPT_PREFIX_CHARS:
        raise LanguageFilterError(
            f"{dataset_id!r}: the rows share only {len(prefix)} leading characters, below "
            f"the {MIN_PROMPT_PREFIX_CHARS} needed to identify a language template. This "
            "corpus is not single-language, or its prompt format changed."
        )
    return prefix


def drop_by_prompt_prefix(
    rows: list[dict[str, Any]], *, field: str, prefix: str, dataset_id: str,
    expected_removals: int | None = None,
) -> list[dict[str, Any]]:
    """Remove every row whose prompt begins with `prefix`. Fails closed.

    Stronger than subtracting the tier-2 rows by fingerprint: that would only remove rows
    that literally appear in the 220-row Tagalog corpus, leaving any Tagalog row living
    elsewhere in the multilingual splits. Matching the template catches them all.

    `expected_removals` is an EXACT count when the registry records one, and it is exact in
    both directions on purpose. Measured against the pinned revision, the multilingual
    `train` split contains zero Tagalog rows — they all live in `test` — so the honest
    expectation for a train-split profile is 0, and "at least one" would fail a build that
    is behaving correctly. Removing more than expected is equally a reason to stop: it means
    the upstream corpus changed shape, and a human should look before that silently becomes
    the new acceptance data.
    """
    kept = [row for row in rows if not str(row.get(field, "")).startswith(prefix)]
    removed = len(rows) - len(kept)

    if expected_removals is None:
        satisfied = removed >= 1
        wanted = "at least 1"
    else:
        satisfied = removed == expected_removals
        wanted = f"exactly {expected_removals}"

    if not satisfied:
        raise LanguageFilterError(
            f"{dataset_id!r}: excluding by prompt template removed {removed} rows, but "
            f"{wanted} was expected. Either the templates no longer match, or the upstream "
            "corpus changed which splits carry that language. Re-verify against the pinned "
            "revision and update `expected_removals` deliberately."
        )
    if not kept:
        raise LanguageFilterError(
            f"{dataset_id!r}: excluding by prompt template removed every row."
        )
    return kept


def drop_languages(
    rows: list[dict[str, Any]], *, field: str | None, exclude: tuple[str, ...],
    dataset_id: str,
) -> list[dict[str, Any]]:
    """Remove rows whose language is in `exclude`. Fails closed, never silently.

    Tier 3 exists to be independent of Tier 2, and that independence is worthless if the
    exclusion quietly no-ops. Three ways it could, all of which raise here instead:

      * the registry does not say which field carries the language;
      * the field is absent from the fetched rows, e.g. upstream renamed it;
      * the exclusion matches nothing, which means the filter is not doing what its
        presence claims.

    The cross-profile fingerprint gate is the second line of defence, not the first: a
    filter that failed open would leave the gate to discover the leakage after both
    profiles were built.
    """
    if not exclude:
        return rows
    if not field:
        raise LanguageFilterError(
            f"{dataset_id!r} declares exclude_languages={list(exclude)} but no "
            "`language_field`. Inspect the pinned source schema and record which column "
            "carries the language code; do not guess a field name."
        )

    missing = [i for i, row in enumerate(rows) if field not in row]
    if missing:
        raise LanguageFilterError(
            f"{dataset_id!r}: language_field {field!r} is absent from "
            f"{len(missing)} of {len(rows)} rows (first at index {missing[0]}). The "
            "pinned revision does not have the schema the registry describes."
        )

    excluded = {code.lower() for code in exclude}
    kept = [row for row in rows if str(row[field]).lower() not in excluded]
    removed = len(rows) - len(kept)
    if removed == 0:
        raise LanguageFilterError(
            f"{dataset_id!r}: excluding {sorted(excluded)} removed no rows. Either the "
            "language codes do not match this source's vocabulary, or the source no longer "
            "contains that language — either way the exclusion is not doing what the "
            "registry claims."
        )
    if not kept:
        raise LanguageFilterError(
            f"{dataset_id!r}: excluding {sorted(excluded)} removed every row."
        )
    return kept
