from __future__ import annotations

import copy

import pytest

from validation_datasets.evaluation import (
    EvaluationConversionError,
    canonicalize_kalahi,
    evaluation_fingerprints,
    require_usable_for_evaluation,
    to_evaluation,
)
from validation_datasets.registry import DatasetRegistry, RegistryError


@pytest.fixture(scope="module")
def registry() -> DatasetRegistry:
    return DatasetRegistry.load()


@pytest.fixture
def kalahi_row() -> dict:
    return {
        "id": "kalahi-001",
        # Opaque on purpose: this layer preserves the upstream label and assigns no meaning.
        "label": "opaque-upstream-label",
        "prompts": [
            {
                "question": "Alin ang pinakaangkop na tugon?",
                "mcq_options": "A. Isa\nB. Dalawa\nC. Tatlo",
                "mcq": "Tanong\nA. Isa\nB. Dalawa\nC. Tatlo",
            },
            {
                "question": "Pumili ng sagot.",
                "mcq_options": "A. X\nB. Y",
                "mcq": "Pumili ng sagot.\nA. X\nB. Y",
            },
        ],
        "metadata": {
            "language": "tl",
            "category": "shared_knowledge",
            "topic": "synthetic-test-topic",
        },
    }


def test_kalahi_registry_contract_is_exact_and_evaluation_only(registry):
    source = registry.get("kalahi")
    assert source.config == "tl"
    assert source.split == "eval"
    assert source.pipeline_usage == "evaluation"
    assert source.canonical_schema == "evaluation"
    assert source.conversion_version == 1
    assert source.profile("full").count == 150
    assert source.profile("full").selection == "all"
    assert source.is_evaluation_only
    assert source.gated
    assert source.may_commit_rows is False


def test_kalahi_can_be_evaluation_source_but_never_training(registry):
    source = registry.get("kalahi")
    require_usable_for_evaluation(source)
    with pytest.raises(RegistryError, match="evaluation-only"):
        source.require_usable_for_training()


def test_training_source_cannot_be_silently_repurposed_as_evaluation(registry):
    with pytest.raises(RegistryError, match="not registered as evaluation-only"):
        require_usable_for_evaluation(registry.get("dolly-15k"))


def test_canonicalizer_preserves_opaque_label_and_prompt_variants(kalahi_row):
    record = canonicalize_kalahi(kalahi_row, index=3)
    assert record["id"] == "kalahi-001"
    assert record["label"] == "opaque-upstream-label"
    assert record["prompts"] == kalahi_row["prompts"]
    assert record["metadata"] == kalahi_row["metadata"]


def test_evaluation_dispatch_is_source_specific(kalahi_row):
    assert to_evaluation("kalahi", kalahi_row, index=0)["id"] == "kalahi-001"
    with pytest.raises(EvaluationConversionError, match="do not coerce"):
        to_evaluation("dolly-15k", {"instruction": "x"}, index=0)


@pytest.mark.parametrize("field", ["id", "label"])
def test_missing_or_empty_required_top_level_string_fails(kalahi_row, field):
    row = copy.deepcopy(kalahi_row)
    row[field] = ""
    with pytest.raises(EvaluationConversionError, match=field):
        canonicalize_kalahi(row, index=7)


def test_empty_prompt_list_fails(kalahi_row):
    row = copy.deepcopy(kalahi_row)
    row["prompts"] = []
    with pytest.raises(EvaluationConversionError, match="non-empty list"):
        canonicalize_kalahi(row, index=1)


@pytest.mark.parametrize("field", ["question", "mcq_options", "mcq"])
def test_prompt_shape_is_fail_closed(kalahi_row, field):
    row = copy.deepcopy(kalahi_row)
    row["prompts"][0][field] = ""
    with pytest.raises(EvaluationConversionError, match=field):
        canonicalize_kalahi(row, index=2)


def test_metadata_types_are_checked_without_inventing_content_policy(kalahi_row):
    row = copy.deepcopy(kalahi_row)
    row["metadata"]["category"] = 12
    with pytest.raises(EvaluationConversionError, match="category"):
        canonicalize_kalahi(row, index=4)

    # Empty category/topic are still valid strings. The pinned schema establishes type, not
    # a stronger non-empty semantic requirement.
    row = copy.deepcopy(kalahi_row)
    row["metadata"]["category"] = ""
    row["metadata"]["topic"] = ""
    canonicalize_kalahi(row, index=5)


def test_evaluation_fingerprints_are_deterministic_and_order_independent(kalahi_row):
    first = canonicalize_kalahi(kalahi_row, index=0)
    other_row = copy.deepcopy(kalahi_row)
    other_row["id"] = "kalahi-002"
    second = canonicalize_kalahi(other_row, index=1)

    a = evaluation_fingerprints([first, second])
    b = evaluation_fingerprints([second, first])
    assert a == b
    assert len(a) == 2
    assert all(len(value) == 16 for value in a)


def test_duplicate_evaluation_records_collapse_only_in_fingerprint_evidence(kalahi_row):
    record = canonicalize_kalahi(kalahi_row, index=0)
    # Fingerprints are evidence about unique canonical content; the builder still writes all
    # source rows and does not silently deduplicate the evaluation corpus.
    assert len(evaluation_fingerprints([record, record])) == 1
