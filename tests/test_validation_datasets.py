"""Registry policy, deterministic selection, conversion, and packaging.

No network: these assert the properties that make an acceptance dataset trustworthy, all of
which are testable offline. Actually fetching a pinned source is a separate, opt-in test.
"""

from __future__ import annotations

import json
import zipfile

import pytest

from lmpipeline.datasets.normalize import iter_examples
from validation_datasets.convert import ConversionError, convert_dolly, to_canonical
from validation_datasets.package import (
    FIXED_TIMESTAMP,
    ZIP_CREATE_SYSTEM,
    build_manifest,
    sha256_file,
    write_canonical_jsonl,
    write_dimer_zip,
)
from validation_datasets.registry import DatasetRegistry, RegistryError
from validation_datasets.subset import select


@pytest.fixture(scope="module")
def registry() -> DatasetRegistry:
    return DatasetRegistry.load()


# -- registry policy -----------------------------------------------------------


def test_every_source_pins_an_immutable_revision(registry):
    for dataset_id in registry.ids():
        revision = registry.get(dataset_id).revision
        assert len(revision) == 40, f"{dataset_id} is not pinned to a commit SHA"
        assert revision != "main"


def test_evaluation_only_data_cannot_be_used_for_training(registry):
    """Kalahi is an evaluation asset. Training on it would invalidate its own results."""
    kalahi = registry.get("kalahi")
    assert kalahi.is_evaluation_only
    with pytest.raises(RegistryError) as exc:
        kalahi.require_usable_for_training()
    assert "evaluation-only" in str(exc.value) or "disabled" in str(exc.value)


def test_gated_sources_can_never_be_used_for_training(registry):
    """Terms were accepted for both on 2026-08-22 (issue #10), which changed one thing only.

    Accepting upstream terms makes a source *fetchable*. It does not make it usable: kalahi
    is the held-out evaluation tier and sea-instruct still has no subsetting policy and no
    inspected schema. Both must still refuse a training request, for their own reasons.
    """
    for dataset_id in ("kalahi", "sea-instruct-2602"):
        source = registry.get(dataset_id)
        assert source.gated
        with pytest.raises(RegistryError):
            source.require_usable_for_training()


def test_accepting_terms_does_not_license_committing_rows(registry):
    """Fetchable and redistributable are different questions, and only one was answered."""
    for dataset_id in ("kalahi", "sea-instruct-2602"):
        assert registry.get(dataset_id).may_commit_rows is False


def test_gated_sources_may_not_have_rows_committed(registry):
    for dataset_id in ("kalahi", "sea-instruct-2602"):
        assert registry.get(dataset_id).may_commit_rows is False


def test_the_tagalog_overlap_is_recorded_in_both_directions(registry):
    """Two tiers drawn from one upstream corpus are not independent.

    uner_llm_inst_tagalog converts the Tagalog portion of Universal NER v1, and
    uner_llm_instructions converts the same v1 corpus including `tl`.
    """
    assert registry.overlap_partners("uner-tagalog") == ["uner-multilingual"]
    assert registry.overlap_partners("uner-multilingual") == ["uner-tagalog"]


def test_the_tagalog_source_split_is_recorded_as_test(registry):
    """Upstream publishes no train split; training on it must be visible, not implicit."""
    assert registry.get("uner-tagalog").split == "test"


def test_unknown_dataset_fails_closed(registry):
    with pytest.raises(RegistryError):
        registry.get("some-dataset-we-never-approved")


def test_unknown_profile_fails_closed(registry):
    with pytest.raises(RegistryError) as exc:
        registry.get("dolly-15k").profile("smoke_999")
    assert "smoke_100" in str(exc.value)


# -- deterministic selection ---------------------------------------------------


ROWS = [{"instruction": f"i{i}", "context": "", "response": f"r{i}"} for i in range(500)]
KEYS = ("instruction", "context", "response")


def test_selection_is_reproducible():
    a = select(ROWS, count=50, salt="dolly-15k:smoke_100", key_fields=KEYS)
    b = select(ROWS, count=50, salt="dolly-15k:smoke_100", key_fields=KEYS)
    assert [i for i, _ in a] == [i for i, _ in b]
    assert len(a) == 50


def test_selection_is_independent_of_source_order():
    """A subset that depends on upstream ordering changes when upstream reorders."""
    forward = select(ROWS, count=50, salt="s", key_fields=KEYS)
    shuffled = list(reversed(ROWS))
    backward = select(shuffled, count=50, salt="s", key_fields=KEYS)

    def content(pairs):
        return {json.dumps(r, sort_keys=True) for _, r in pairs}

    assert content(forward) == content(backward)


def test_selection_is_not_the_first_n():
    chosen = [i for i, _ in select(ROWS, count=50, salt="s", key_fields=KEYS)]
    assert chosen != list(range(50))


def test_different_profiles_are_not_nested_prefixes():
    """smoke_500's extra rows must not be systematically unlike its first hundred."""
    small = {i for i, _ in select(ROWS, count=100, salt="dolly-15k:smoke_100",
                                  key_fields=KEYS)}
    large = {i for i, _ in select(ROWS, count=200, salt="dolly-15k:smoke_500",
                                  key_fields=KEYS)}
    assert not small.issubset(large)


def test_selection_emits_rows_in_source_order():
    chosen = [i for i, _ in select(ROWS, count=30, salt="s", key_fields=KEYS)]
    assert chosen == sorted(chosen)


def test_selecting_all_returns_everything():
    assert len(select(ROWS, count=None, salt="s", key_fields=KEYS, selection="all")) == 500


# -- conversion ----------------------------------------------------------------


def test_dolly_context_is_appended_to_the_user_turn():
    record = convert_dolly(
        {"instruction": "Summarize.", "context": "Body text.", "response": "A summary."},
        index=0,
    )
    assert record["messages"][0]["role"] == "user"
    assert "Summarize." in record["messages"][0]["content"]
    assert "Body text." in record["messages"][0]["content"]
    assert record["messages"][1] == {"role": "assistant", "content": "A summary."}


def test_dolly_without_context_does_not_leave_stray_whitespace():
    record = convert_dolly(
        {"instruction": "Name a fruit.", "context": "", "response": "Mango."}, index=0
    )
    assert record["messages"][0]["content"] == "Name a fruit."


def test_a_row_that_cannot_be_mapped_raises_rather_than_being_dropped():
    with pytest.raises(ConversionError):
        convert_dolly({"instruction": "x", "context": "", "response": ""}, index=7)


def test_converted_rows_are_valid_against_the_dataset_contract(tmp_path):
    """The whole point: converter output must satisfy the pipeline's own normalizer."""
    records = [
        to_canonical("dolly-15k",
                     {"instruction": f"Q{i}", "context": "", "response": f"A{i}"},
                     index=i)
        for i in range(12)
    ]
    path = tmp_path / "train.jsonl"
    write_canonical_jsonl(path, records)
    examples = list(iter_examples(path))
    assert len(examples) == 12
    assert all(e.assistant_text.strip() for e in examples)


def test_no_converter_means_no_guessing():
    with pytest.raises(ConversionError) as exc:
        to_canonical("sea-instruct-2602", {"anything": "x"}, index=0)
    assert "do not guess" in str(exc.value)


# -- packaging -----------------------------------------------------------------


def test_canonical_serialization_is_stable(tmp_path):
    records = [{"messages": [{"role": "user", "content": "Kumusta? 🇵🇭"},
                             {"role": "assistant", "content": "Mabuti!"}]}]
    a = write_canonical_jsonl(tmp_path / "a.jsonl", records)
    b = write_canonical_jsonl(tmp_path / "b.jsonl", records)
    assert a == b
    # Unicode is preserved, not escaped.
    assert "🇵🇭" in (tmp_path / "a.jsonl").read_text(encoding="utf-8")


def test_dimer_zip_is_byte_identical_across_builds(tmp_path):
    """A package whose digest changes every build cannot be referenced by an acceptance report."""
    records = [{"messages": [{"role": "user", "content": f"Q{i}"},
                             {"role": "assistant", "content": f"A{i}"}]}
               for i in range(5)]
    write_canonical_jsonl(tmp_path / "train.jsonl", records)
    splits = {"train": tmp_path / "train.jsonl"}

    first = write_dimer_zip(splits, tmp_path / "one.zip")
    second = write_dimer_zip(splits, tmp_path / "two.zip")
    assert first == second
    assert sha256_file(tmp_path / "one.zip") == sha256_file(tmp_path / "two.zip")


def test_dimer_zip_has_canonical_member_names_and_order(tmp_path):
    for name in ("train", "validation", "test"):
        write_canonical_jsonl(
            tmp_path / f"{name}.jsonl",
            [{"messages": [{"role": "user", "content": name},
                           {"role": "assistant", "content": "ok"}]}],
        )
    splits = {n: tmp_path / f"{n}.jsonl" for n in ("test", "train", "validation")}
    write_dimer_zip(splits, tmp_path / "pkg.zip")

    with zipfile.ZipFile(tmp_path / "pkg.zip") as zf:
        names = zf.namelist()
    # Canonical order regardless of dict insertion order, and no directory entries.
    assert names == ["train.jsonl", "validation.jsonl", "test.jsonl"]
    assert not any(n.endswith("/") for n in names)
    assert not any("__MACOSX" in n or n.startswith("/") for n in names)


def test_manifest_identifies_the_data_well_enough_to_reference(tmp_path, registry):
    records = [{"messages": [{"role": "user", "content": "q"},
                             {"role": "assistant", "content": "a"}]}]
    path = tmp_path / "train.jsonl"
    digest = write_canonical_jsonl(path, records)
    manifest = build_manifest(
        source=registry.get("dolly-15k"), profile_name="smoke_100",
        splits={"train": path}, digests={"train": digest}, counts={"train": 1},
        selected_indices=[3, 1, 2],
    )
    assert manifest["sourceRevision"] == registry.get("dolly-15k").revision
    assert manifest["license"] == "CC-BY-SA-3.0"
    assert manifest["splits"]["train"]["sha256"] == digest
    assert manifest["selectedSourceIndexCount"] == 3
    assert len(manifest["selectedSourceIndexDigest"]) == 64


def test_selected_index_digest_is_order_independent(tmp_path, registry):
    """Two builds that chose the same rows must agree, whatever order they recorded them."""
    def digest_for(indices):
        return build_manifest(
            source=registry.get("dolly-15k"), profile_name="p",
            splits={}, digests={}, counts={}, selected_indices=indices,
        )["selectedSourceIndexDigest"]

    assert digest_for([3, 1, 2]) == digest_for([1, 2, 3])
    assert digest_for([3, 1, 2]) != digest_for([1, 2, 4])


def test_archive_metadata_does_not_depend_on_the_building_platform(tmp_path):
    """A zip member records the OS that created it, and Python fills it in from the host.

    0 on Windows, 3 (Unix) elsewhere, written into the central directory — so identical
    inputs produced different archive bytes on a laptop and on CI. Every digest-based claim
    in this suite was therefore only true per-platform, which CI caught and local runs never
    could. Pinned, and asserted here so it cannot drift back.
    """
    records = [{"messages": [{"role": "user", "content": "q"},
                             {"role": "assistant", "content": "a"}]}]
    write_canonical_jsonl(tmp_path / "train.jsonl", records)
    write_dimer_zip({"train": tmp_path / "train.jsonl"}, tmp_path / "pkg.zip")

    with zipfile.ZipFile(tmp_path / "pkg.zip") as zf:
        for info in zf.infolist():
            assert info.create_system == ZIP_CREATE_SYSTEM
            assert info.date_time == FIXED_TIMESTAMP
