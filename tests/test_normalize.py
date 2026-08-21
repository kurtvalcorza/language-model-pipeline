from __future__ import annotations

import json
from pathlib import Path

import pytest

from lmpipeline.datasets.normalize import (
    FAMILY_CONVERSATIONAL,
    FAMILY_INSTRUCTION,
    FAMILY_PROMPT_COMPLETION,
    detect_family,
    iter_examples,
)
from lmpipeline.errors import Code, DatasetError


def write_jsonl(tmp_path: Path, name: str, records: list) -> Path:
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def test_detects_three_families():
    assert detect_family({"messages": []}, 1) == FAMILY_CONVERSATIONAL
    assert detect_family({"prompt": "a", "completion": "b"}, 1) == FAMILY_PROMPT_COMPLETION
    assert detect_family({"instruction": "a", "output": "b"}, 1) == FAMILY_INSTRUCTION


def test_all_families_normalize_to_messages(tmp_path):
    conv = write_jsonl(tmp_path, "c.jsonl", [
        {"messages": [{"role": "user", "content": "hi"},
                      {"role": "assistant", "content": "hello"}]}
    ])
    pc = write_jsonl(tmp_path, "p.jsonl", [{"prompt": "hi", "completion": "hello"}])
    inst = write_jsonl(tmp_path, "i.jsonl", [
        {"instruction": "hi", "input": "", "output": "hello"}
    ])

    for path in (conv, pc, inst):
        examples = list(iter_examples(path))
        assert len(examples) == 1
        assert examples[0].messages[-1]["role"] == "assistant"
        assert examples[0].assistant_text == "hello"


def test_equivalent_content_across_families_shares_a_fingerprint(tmp_path):
    """Leakage detection must survive a family change between splits."""
    pc = write_jsonl(tmp_path, "p.jsonl", [{"prompt": "hi", "completion": "hello"}])
    inst = write_jsonl(tmp_path, "i.jsonl", [{"instruction": "hi", "output": "hello"}])
    a = next(iter(iter_examples(pc))).fingerprint()
    b = next(iter(iter_examples(inst))).fingerprint()
    assert a == b


def test_mixed_families_in_one_file_fail(tmp_path):
    path = write_jsonl(tmp_path, "mixed.jsonl", [
        {"prompt": "a", "completion": "b"},
        {"messages": [{"role": "user", "content": "x"},
                      {"role": "assistant", "content": "y"}]},
    ])
    with pytest.raises(DatasetError) as exc:
        list(iter_examples(path))
    assert exc.value.code == Code.DATASET_SCHEMA_MIXED
    assert exc.value.details["line"] == 2


def test_empty_assistant_target_is_rejected(tmp_path):
    path = write_jsonl(tmp_path, "empty.jsonl", [{"prompt": "a", "completion": "   "}])
    with pytest.raises(DatasetError) as exc:
        list(iter_examples(path))
    assert exc.value.code == Code.DATASET_TARGET_EMPTY


def test_invalid_role_is_rejected(tmp_path):
    path = write_jsonl(tmp_path, "role.jsonl", [
        {"messages": [{"role": "tool", "content": "x"},
                      {"role": "assistant", "content": "y"}]}
    ])
    with pytest.raises(DatasetError) as exc:
        list(iter_examples(path))
    assert exc.value.code == Code.DATASET_ROLE_INVALID


def test_malformed_json_reports_its_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"prompt":"a","completion":"b"}\n{"prompt":"c",,}\n', encoding="utf-8"
    )
    with pytest.raises(DatasetError) as exc:
        list(iter_examples(path))
    assert exc.value.code == Code.DATASET_INVALID_JSON
    assert exc.value.details["line"] == 2


def test_invalid_utf8_reports_its_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_bytes(b'{"prompt":"a","completion":"b"}\n{"prompt":"\xff\xfe"}\n')
    with pytest.raises(DatasetError) as exc:
        list(iter_examples(path))
    assert exc.value.code == Code.DATASET_INVALID_UTF8
    assert exc.value.details["line"] == 2


def test_blank_lines_are_skipped_not_dropped_records(tmp_path):
    path = tmp_path / "gaps.jsonl"
    path.write_text(
        '{"prompt":"a","completion":"b"}\n\n\n{"prompt":"c","completion":"d"}\n',
        encoding="utf-8",
    )
    examples = list(iter_examples(path))
    assert [e.line_number for e in examples] == [1, 4]


def test_unicode_is_preserved(tmp_path):
    path = write_jsonl(tmp_path, "u.jsonl", [
        {"prompt": "Kumusta?", "completion": "Mabuti naman — salamat! 🇵🇭"}
    ])
    assert "🇵🇭" in next(iter(iter_examples(path))).assistant_text


def test_error_messages_never_embed_record_content(tmp_path):
    """Datasets are user-private; a failure must not leak the row into logs."""
    secret = "PATIENT-SSN-123456789"
    path = write_jsonl(tmp_path, "s.jsonl", [{"prompt": secret, "completion": ""}])
    with pytest.raises(DatasetError) as exc:
        list(iter_examples(path))
    assert secret not in str(exc.value)
    assert secret not in json.dumps(exc.value.details)
