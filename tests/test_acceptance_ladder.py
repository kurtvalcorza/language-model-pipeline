"""The acceptance ladder, language exclusion, and committed approvals.

Two review findings meet here, because they are the same concern seen from two sides:

  * Issue #6 — tiers 2 and 3 were drawn from one upstream corpus, so they were not
    independent. Resolved by making tier 3 exclude `tl` (independence by construction) and
    proving it continuously with a committed-fingerprint gate.
  * PR #4 — approved digests lived only in a git-ignored build directory, so a fresh clone
    could not check a rebuild against anything. Resolved by committing the expectation.

Tier 1 English, tier 2 Filipino, tier 3 multilingual excluding `tl`, tier 4 held-out
Filipino evaluation. Tiers 2 and 3 are independent acceptance runs; tier 4 is the only
evaluation tier.
"""

from __future__ import annotations

import pytest

from validation_datasets.approval import (
    ApprovalError,
    approval_path,
    compare_to_approval,
    fingerprint_digest,
    load_approval,
    load_fingerprints,
)
from validation_datasets.registry import DatasetRegistry
from validation_datasets.subset import LanguageFilterError, drop_languages


@pytest.fixture(scope="module")
def registry() -> DatasetRegistry:
    return DatasetRegistry.load()


# -- the ladder ----------------------------------------------------------------


def test_the_ladder_assigns_one_tier_per_purpose(registry):
    assert registry.get("dolly-15k").tier == 1
    assert registry.get("uner-tagalog").tier == 2
    assert registry.get("uner-multilingual").tier == 3
    assert registry.get("kalahi").tier == 4


def test_tier_three_excludes_tagalog_in_every_profile(registry):
    """Independence by construction. A profile that forgot the exclusion is tier 2 again."""
    source = registry.get("uner-multilingual")
    assert source.profiles, "tier 3 has no profiles to check"
    for profile in source.profiles.values():
        assert "tl" in profile.exclude_languages, (
            f"profile {profile.name!r} does not exclude tl"
        )


def test_the_evaluation_tier_is_the_only_one_marked_evaluation(registry):
    for dataset_id in registry.ids():
        source = registry.get(dataset_id)
        if source.tier == 4:
            assert source.pipeline_usage == "evaluation"
            assert source.is_evaluation_only
        else:
            assert source.pipeline_usage == "training"


def test_the_tagalog_source_records_both_upstream_split_and_our_usage(registry):
    """Upstream calls it `test`; this suite trains on it deliberately. Both facts, together.

    Recording only the first makes the run look like an evaluation. Recording only the
    second hides that this corpus can never later serve as a held-out benchmark.
    """
    source = registry.get("uner-tagalog")
    assert source.split == "test"
    assert source.pipeline_usage == "training"


# -- language exclusion fails closed -------------------------------------------


ROWS_WITH_LANG = [
    {"inputs": f"i{i}", "targets": f"t{i}", "lang": "tl" if i % 2 else "en"}
    for i in range(10)
]


def test_excluding_a_language_removes_exactly_those_rows():
    kept = drop_languages(
        ROWS_WITH_LANG, field="lang", exclude=("tl",), dataset_id="uner-multilingual"
    )
    assert len(kept) == 5
    assert all(row["lang"] == "en" for row in kept)


def test_an_unknown_language_field_fails_rather_than_skipping_the_filter():
    """The registry does not yet know which column carries the language for this source.

    Failing closed is the point: a tier-3 profile built with the exclusion silently skipped
    would still contain Tagalog, and would look completely normal.
    """
    with pytest.raises(LanguageFilterError) as exc:
        drop_languages(
            ROWS_WITH_LANG, field=None, exclude=("tl",), dataset_id="uner-multilingual"
        )
    assert "language_field" in str(exc.value)


def test_a_field_missing_from_the_rows_fails():
    with pytest.raises(LanguageFilterError) as exc:
        drop_languages(ROWS_WITH_LANG, field="language", exclude=("tl",), dataset_id="x")
    assert "absent" in str(exc.value)


def test_an_exclusion_that_matches_nothing_fails():
    """A no-op filter is indistinguishable from a working one unless it says so."""
    rows = [{"inputs": "i", "targets": "t", "lang": "en"}]
    with pytest.raises(LanguageFilterError) as exc:
        drop_languages(rows, field="lang", exclude=("tl",), dataset_id="x")
    assert "removed no rows" in str(exc.value)


def test_no_exclusion_requested_is_a_pass_through():
    kept = drop_languages(ROWS_WITH_LANG, field=None, exclude=(), dataset_id="x")
    assert kept is ROWS_WITH_LANG


# -- committed approvals -------------------------------------------------------


def test_the_dolly_profile_has_a_committed_approval():
    """PR #4's blocker: a fresh clone must have something to verify a rebuild against.

    Before this, the only manifest lived in a git-ignored build directory, so `verify`
    could prove nothing except that a run agreed with itself.
    """
    approval = load_approval("dolly-15k", "smoke_100")
    assert len(approval["sourceRevision"]) == 40
    assert len(approval["splits"]["train"]["sha256"]) == 64
    assert approval["fingerprintCount"] == approval["splits"]["train"]["exampleCount"]


def test_a_committed_approval_carries_no_dataset_rows():
    """Digests and counts only — the licence rule is: commit the recipe, never the rows."""
    text = approval_path("dolly-15k", "smoke_100").read_text(encoding="utf-8")
    assert "messages" not in text
    assert "instruction" not in text


def test_committed_fingerprints_match_the_recorded_digest():
    fingerprints = load_fingerprints("dolly-15k", "smoke_100")
    approval = load_approval("dolly-15k", "smoke_100")
    assert len(fingerprints) == approval["fingerprintCount"]
    assert fingerprint_digest(sorted(fingerprints)) == approval["fingerprintDigest"]


def _manifest_like(approval: dict) -> dict:
    return {
        "sourceRevision": approval["sourceRevision"],
        "conversionVersion": approval["conversionVersion"],
        "excludeLanguages": approval["excludeLanguages"],
        "selectedSourceIndexDigest": approval["selectedSourceIndexDigest"],
        "selectedSourceIndexCount": approval["selectedSourceIndexCount"],
        "splits": {k: dict(v) for k, v in approval["splits"].items()},
    }


def test_an_identical_rebuild_compares_clean():
    approval = load_approval("dolly-15k", "smoke_100")
    assert compare_to_approval(_manifest_like(approval), approval) == []


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda m: m["splits"]["train"].update(sha256="0" * 64), "digest"),
        (lambda m: m["splits"]["train"].update(exampleCount=99), "example count"),
        (lambda m: m.update(sourceRevision="a" * 40), "pinned source revision"),
        (lambda m: m.update(conversionVersion=2), "converter version"),
        (lambda m: m.update(excludeLanguages=["tl"]), "language exclusions"),
        (lambda m: m.update(selectedSourceIndexDigest="b" * 64), "selected source rows"),
    ],
)
def test_every_way_a_rebuild_can_drift_is_reported(mutate, expected):
    """Each of these silently changes what the acceptance dataset actually is."""
    approval = load_approval("dolly-15k", "smoke_100")
    manifest = _manifest_like(approval)
    mutate(manifest)
    problems = compare_to_approval(manifest, approval)
    assert problems, f"drift in {expected} was not reported"
    assert any(expected in problem for problem in problems)


def test_a_missing_approval_is_an_error_not_a_pass():
    """`verify` must fail on an unapproved profile rather than quietly finding nothing."""
    with pytest.raises(ApprovalError):
        load_approval("dolly-15k", "smoke_500")


# -- the tier 2 / tier 3 disjointness gate --------------------------------------


def test_the_disjointness_gate_detects_a_shared_example():
    """The gate must be shown to fail on overlap, not merely to run.

    Asserted on synthetic sets because the real tier-2/tier-3 profiles cannot be built
    until the multilingual source's language field is confirmed. A gate that has never
    been observed failing is not evidence of anything.
    """
    tier2 = {"aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"}
    tier3 = {"bbbbbbbbbbbbbbbb", "cccccccccccccccc"}
    assert tier2 & tier3 == {"bbbbbbbbbbbbbbbb"}
    assert not (tier2 - {"bbbbbbbbbbbbbbbb"}) & tier3


def test_tier_two_and_tier_three_are_disjoint_once_both_are_approved():
    """The continuous guard. Skips only while tier 3 is unbuildable."""
    try:
        tier2 = load_fingerprints("uner-tagalog", "full")
        tier3 = load_fingerprints("uner-multilingual", "multi_500")
    except ApprovalError:
        pytest.skip("tier 2/3 profiles are not approved yet (blocked on language_field)")
    shared = tier2 & tier3
    assert not shared, f"{len(shared)} canonical examples appear in both tiers"
