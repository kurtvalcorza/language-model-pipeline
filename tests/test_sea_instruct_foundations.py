from __future__ import annotations

import json

import pytest

from validation_datasets.sea_instruct import (
    SEA_IDENTITY_FIELDS,
    SeaInstructError,
    canonicalize_sea_instruct,
    parse_conversations,
)
from validation_datasets.streaming import identity_digest, select_bounded
from validation_datasets.subset import select


def _row(index: int) -> dict:
    messages = [
        {"role": "user", "content": f"Tanong {index}"},
        {"role": "assistant", "content": f"Sagot {index}"},
    ]
    return {
        "conversations_id": f"id-{index:05d}",
        "conversations": json.dumps(messages, ensure_ascii=False),
    }


def _content_ids(items) -> set[str]:
    return {item.row["conversations_id"] for item in items}


def test_bounded_selector_matches_existing_stable_hash_choice_for_unique_rows():
    rows = [_row(i) for i in range(500)]
    expected = select(
        rows,
        count=37,
        salt="sea:filipino_37",
        key_fields=SEA_IDENTITY_FIELDS,
        selection="stable_hash",
    )
    actual = select_bounded(
        rows,
        count=37,
        salt="sea:filipino_37",
        key_fields=SEA_IDENTITY_FIELDS,
    )
    assert _content_ids(actual) == {row["conversations_id"] for _, row in expected}


def test_bounded_selector_is_independent_of_source_order():
    rows = [_row(i) for i in range(300)]
    forward = select_bounded(
        rows, count=25, salt="profile", key_fields=SEA_IDENTITY_FIELDS
    )
    backward = select_bounded(
        reversed(rows), count=25, salt="profile", key_fields=SEA_IDENTITY_FIELDS
    )
    assert _content_ids(forward) == _content_ids(backward)
    assert [item.rank for item in forward] == [item.rank for item in backward]
    assert identity_digest(forward) == identity_digest(backward)


def test_bounded_selector_consumes_a_one_pass_generator():
    consumed = 0

    def rows():
        nonlocal consumed
        for i in range(1000):
            consumed += 1
            yield _row(i)

    selected = select_bounded(
        rows(), count=10, salt="stream", key_fields=SEA_IDENTITY_FIELDS
    )
    assert consumed == 1000
    assert len(selected) == 10


def test_bounded_selector_rejects_non_positive_profile_size():
    with pytest.raises(ValueError, match="positive"):
        select_bounded([], count=0, salt="x", key_fields=SEA_IDENTITY_FIELDS)


def test_duplicate_occurrences_have_stable_identity_evidence():
    duplicate = _row(1)
    rows = [_row(0), duplicate, _row(2), dict(duplicate), _row(3)]
    first = select_bounded(
        rows, count=5, salt="all-small", key_fields=SEA_IDENTITY_FIELDS
    )
    reordered = select_bounded(
        [dict(duplicate), _row(3), _row(2), duplicate, _row(0)],
        count=5,
        salt="all-small",
        key_fields=SEA_IDENTITY_FIELDS,
    )
    assert sorted(item.identity for item in first) == sorted(
        item.identity for item in reordered
    )
    assert identity_digest(first) == identity_digest(reordered)


def test_strict_json_conversation_becomes_canonical_messages():
    source_messages = [
        {
            "role": "system",
            "content": "Ikaw ay isang kapaki-pakinabang na assistant.",
            "country": None,
        },
        {
            "role": "user",
            "content": "Magbigay ng maikling halimbawa.",
            "hashed_ip": None,
        },
        {
            "role": "assistant",
            "content": "Narito ang isang halimbawa.",
            "toxic": None,
        },
    ]
    parsed = parse_conversations(
        json.dumps(source_messages, ensure_ascii=False), index=4
    )
    assert parsed == [
        {"role": "system", "content": source_messages[0]["content"]},
        {"role": "user", "content": source_messages[1]["content"]},
        {"role": "assistant", "content": source_messages[2]["content"]},
    ]


def test_canonicalizer_keeps_metadata_out_of_training_text():
    row = _row(12)
    row.update(
        {
            "prompt_primary_language": "Filipino_NativeScript",
            "prompt_primary_domain": "Education",
            "source": "upstream/source",
        }
    )
    record = canonicalize_sea_instruct(row, index=0)
    serialized = json.dumps(record, ensure_ascii=False)
    assert "prompt_primary_language" not in serialized
    assert "upstream/source" not in serialized
    assert record["messages"][0]["content"] == "Tanong 12"


def test_python_literal_fallback_is_deliberately_rejected():
    # The original gated field must prove its exact serialization before this source can be
    # enabled. Supporting Python repr here would widen the accepted grammar on speculation.
    value = "[{'role': 'user', 'content': 'x'}, {'role': 'assistant', 'content': 'y'}]"
    with pytest.raises(SeaInstructError, match="strict JSON"):
        parse_conversations(value, index=9)


@pytest.mark.parametrize("value", [None, "", "   ", "{}", "[]"])
def test_missing_or_non_list_conversations_fail(value):
    with pytest.raises(SeaInstructError):
        parse_conversations(value, index=1)


def test_unapproved_role_fails():
    value = json.dumps(
        [
            {"role": "human", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
    )
    with pytest.raises(SeaInstructError, match="role"):
        parse_conversations(value, index=2)


def test_empty_message_content_fails():
    value = json.dumps(
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": ""},
        ]
    )
    with pytest.raises(SeaInstructError, match="content"):
        parse_conversations(value, index=3)


def test_conversation_without_assistant_target_fails():
    value = json.dumps([{"role": "user", "content": "question"}])
    with pytest.raises(SeaInstructError, match="assistant target"):
        parse_conversations(value, index=5)


def test_missing_conversation_id_fails_before_training_conversion():
    row = _row(1)
    row["conversations_id"] = ""
    with pytest.raises(SeaInstructError, match="conversations_id"):
        canonicalize_sea_instruct(row, index=6)
