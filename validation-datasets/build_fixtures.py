#!/usr/bin/env python
"""Generate the synthetic and adversarial fixture corpus.

Fixtures are committed (issue #2 section 8) so CI never depends on network access, but they
are **generated**, not hand-written: a hand-maintained corpus of 25 adversarial archives
drifts from the expected-code map beside it. Re-running this script must reproduce every
file byte-for-byte.

    python validation-datasets/build_fixtures.py

Content is synthetic and non-sensitive throughout. The Tagalog strings exercise Unicode and
tokenizer handling; they are ordinary civics phrasing, not copied from any corpus.
"""

from __future__ import annotations

import io
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SYNTHETIC = ROOT / "synthetic"
ADVERSARIAL = ROOT / "adversarial"

SYSTEM = "You are a helpful assistant for Philippine civics questions."


def jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def conversational(index: int, *, system: bool = True, turns: int = 1) -> dict:
    messages = []
    if system:
        messages.append({"role": "system", "content": SYSTEM})
    for turn in range(turns):
        messages.append({"role": "user", "content": f"Question {index}.{turn}?"})
        messages.append({"role": "assistant", "content": f"Answer {index}.{turn}."})
    return {"messages": messages}


# -- valid corpus --------------------------------------------------------------


def build_synthetic() -> None:
    if SYNTHETIC.exists():
        shutil.rmtree(SYNTHETIC)

    # Conversational: the canonical family, with a full train/validation/test split.
    train = [conversational(i) for i in range(20)]
    train += [conversational(100 + i, system=False) for i in range(4)]      # no system
    train += [conversational(200 + i, turns=3) for i in range(4)]           # multi-turn
    train += [
        {"messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Ano ang kabisera ng Pilipinas? 🇵🇭"},
            {"role": "assistant", "content": "Ang Maynila po ang kabisera."},
        ]},
        {"messages": [
            {"role": "user", "content": "Ilan ang rehiyon sa Pilipinas?"},
            {"role": "assistant", "content": "Mayroong labimpito (17) na rehiyon."},
        ]},
    ]
    jsonl(SYNTHETIC / "conversational" / "train.jsonl", train)
    jsonl(SYNTHETIC / "conversational" / "validation.jsonl",
          [conversational(500 + i) for i in range(6)])
    jsonl(SYNTHETIC / "conversational" / "test.jsonl",
          [conversational(700 + i) for i in range(5)])

    # Prompt/completion.
    jsonl(SYNTHETIC / "prompt-completion" / "train.jsonl", [
        {"prompt": f"Define term {i}.", "completion": f"Term {i} means something specific."}
        for i in range(15)
    ])

    # Instruction, with and without the optional input field.
    instruction = [
        {"instruction": f"Summarize passage {i}.", "input": f"Passage {i} body text.",
         "output": f"Summary of passage {i}."}
        for i in range(10)
    ]
    instruction += [
        {"instruction": f"Name a Philippine province, number {i}.",
         "output": f"Province number {i}."}
        for i in range(5)
    ]
    jsonl(SYNTHETIC / "instruction" / "train.jsonl", instruction)


# -- adversarial corpus --------------------------------------------------------
#
# Each entry: (directory, expected stable code, builder). The code is the contract; the
# tests assert on it rather than on message text.

VALID_LINE = {"prompt": "Question?", "completion": "Answer."}


def _valid_lines(count: int = 20) -> list:
    return [{"prompt": f"Question {i}?", "completion": f"Answer {i}."} for i in range(count)]


def case_malformed_json(root: Path) -> None:
    lines = [json.dumps(r, sort_keys=True) for r in _valid_lines()]
    lines[3] = '{"prompt": "broken",,}'
    (root / "train.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def case_non_object_row(root: Path) -> None:
    lines = [json.dumps(r, sort_keys=True) for r in _valid_lines()]
    lines[2] = '["not", "an", "object"]'
    (root / "train.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def case_invalid_utf8(root: Path) -> None:
    good = "\n".join(json.dumps(r, sort_keys=True) for r in _valid_lines()[:2])
    with open(root / "train.jsonl", "wb") as fh:
        fh.write(good.encode("utf-8") + b"\n")
        fh.write(b'{"prompt": "\xff\xfe", "completion": "x"}\n')


def case_empty_target(root: Path) -> None:
    records = _valid_lines()
    records[5] = {"prompt": "Question 5?", "completion": "   "}
    jsonl(root / "train.jsonl", records)


def case_mixed_schema(root: Path) -> None:
    records = _valid_lines(5)
    records.append({"messages": [
        {"role": "user", "content": "x"}, {"role": "assistant", "content": "y"},
    ]})
    jsonl(root / "train.jsonl", records)


def case_ambiguous_record(root: Path) -> None:
    records = _valid_lines(5)
    # Two families in ONE record: precedence would silently discard half of it.
    records.append({"prompt": "p", "completion": "c", "instruction": "i", "output": "o"})
    jsonl(root / "train.jsonl", records)


def case_duplicate_validation_alias(root: Path) -> None:
    jsonl(root / "train.jsonl", _valid_lines())
    jsonl(root / "validation.jsonl", _valid_lines(3))
    jsonl(root / "val.jsonl", _valid_lines(3))


def case_nested_duplicate_train(root: Path) -> None:
    jsonl(root / "train.jsonl", _valid_lines())
    jsonl(root / "backup" / "train.jsonl", _valid_lines(3))


def case_invalid_role(root: Path) -> None:
    jsonl(root / "train.jsonl", [{"messages": [
        {"role": "tool", "content": "x"}, {"role": "assistant", "content": "y"},
    ]}])


def case_unknown_schema(root: Path) -> None:
    jsonl(root / "train.jsonl", [{"question": "q", "answer": "a"}])


def case_missing_train(root: Path) -> None:
    jsonl(root / "validation.jsonl", _valid_lines(5))


def case_zero_examples(root: Path) -> None:
    (root / "train.jsonl").write_text("", encoding="utf-8", newline="\n")


# Zip members carry a timestamp, and `writestr` with a plain name uses the current time.
# That alone makes regeneration produce different bytes, which breaks the byte-identical
# reproducibility the dataset contract requires. Pin it.
FIXED_TIMESTAMP = (2026, 1, 1, 0, 0, 0)


def _zip_case(root: Path, members: list, *, name: str = "dataset.zip",
              compression: int = zipfile.ZIP_DEFLATED) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(root / name, "w", compression=compression) as zf:
        for member, content in members:
            if isinstance(member, zipfile.ZipInfo):
                info = member
                info.date_time = FIXED_TIMESTAMP
            else:
                info = zipfile.ZipInfo(member, date_time=FIXED_TIMESTAMP)
                info.compress_type = compression
            zf.writestr(info, content)


def _valid_jsonl_text(count: int = 20) -> str:
    return "\n".join(json.dumps(r, sort_keys=True) for r in _valid_lines(count)) + "\n"


def case_zip_traversal(root: Path) -> None:
    _zip_case(root, [("../escape.jsonl", _valid_jsonl_text(2)),
                     ("train.jsonl", _valid_jsonl_text())])


def case_zip_absolute(root: Path) -> None:
    _zip_case(root, [("/etc/train.jsonl", _valid_jsonl_text(2))])


def case_zip_drive_prefix(root: Path) -> None:
    _zip_case(root, [("C:/evil.jsonl", _valid_jsonl_text(2))])


def case_zip_symlink(root: Path) -> None:
    info = zipfile.ZipInfo("train.jsonl")
    info.external_attr = 0o120777 << 16
    _zip_case(root, [(info, "/etc/passwd")])


def case_zip_bomb_ratio(root: Path) -> None:
    _zip_case(root, [("train.jsonl", "0" * (5 * 1024 * 1024))])


def case_zip_duplicate_members(root: Path) -> None:
    _zip_case(root, [("train.jsonl", _valid_jsonl_text()),
                     ("train.jsonl", _valid_jsonl_text(3))])


def case_zip_nested_archive(root: Path) -> None:
    # The inner archive's own members need pinned timestamps too: its bytes are embedded
    # in the outer archive, so any variation there propagates outward.
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as inner:
        info = zipfile.ZipInfo("train.jsonl", date_time=FIXED_TIMESTAMP)
        info.compress_type = zipfile.ZIP_DEFLATED
        inner.writestr(info, _valid_jsonl_text())
    _zip_case(root, [("inner.zip", buffer.getvalue())])


def case_two_archives(root: Path) -> None:
    _zip_case(root, [("train.jsonl", _valid_jsonl_text())], name="a.zip")
    _zip_case(root, [("train.jsonl", _valid_jsonl_text())], name="b.zip")


ADVERSARIAL_CASES: list[tuple[str, str, object]] = [
    ("malformed-json", "DATASET_INVALID_JSON", case_malformed_json),
    ("non-object-row", "DATASET_INVALID_JSON", case_non_object_row),
    ("invalid-utf8", "DATASET_INVALID_UTF8", case_invalid_utf8),
    ("empty-assistant-target", "DATASET_TARGET_EMPTY", case_empty_target),
    ("mixed-schema-families", "DATASET_SCHEMA_MIXED", case_mixed_schema),
    ("ambiguous-record-schema", "DATASET_SCHEMA_MIXED", case_ambiguous_record),
    ("unknown-schema", "DATASET_SCHEMA_UNKNOWN", case_unknown_schema),
    ("invalid-role", "DATASET_ROLE_INVALID", case_invalid_role),
    ("duplicate-validation-alias", "DATASET_SPLIT_AMBIGUOUS", case_duplicate_validation_alias),
    ("nested-duplicate-train", "DATASET_SPLIT_AMBIGUOUS", case_nested_duplicate_train),
    ("two-archives", "DATASET_SPLIT_AMBIGUOUS", case_two_archives),
    ("missing-train-split", "DATASET_SPLIT_MISSING", case_missing_train),
    ("zero-examples", "DATASET_TOO_FEW_EXAMPLES", case_zero_examples),
    ("zip-traversal", "DATASET_ARCHIVE_UNSAFE", case_zip_traversal),
    ("zip-absolute-path", "DATASET_ARCHIVE_UNSAFE", case_zip_absolute),
    ("zip-drive-prefix", "DATASET_ARCHIVE_UNSAFE", case_zip_drive_prefix),
    ("zip-symlink-member", "DATASET_ARCHIVE_UNSAFE", case_zip_symlink),
    ("zip-bomb-ratio", "DATASET_ARCHIVE_UNSAFE", case_zip_bomb_ratio),
    ("zip-duplicate-members", "DATASET_ARCHIVE_DUPLICATE_MEMBER", case_zip_duplicate_members),
    ("zip-nested-archive", "DATASET_ARCHIVE_NESTED", case_zip_nested_archive),
]


def build_adversarial() -> dict:
    if ADVERSARIAL.exists():
        shutil.rmtree(ADVERSARIAL)
    expected = {}
    for name, code, builder in ADVERSARIAL_CASES:
        root = ADVERSARIAL / name
        root.mkdir(parents=True, exist_ok=True)
        builder(root)
        expected[name] = code
    return expected


def main() -> int:
    build_synthetic()
    expected = build_adversarial()

    (ADVERSARIAL / "expected-codes.json").write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(f"synthetic:   {len(list(SYNTHETIC.rglob('*.jsonl')))} files")
    print(f"adversarial: {len(expected)} cases")
    for name, code in sorted(expected.items()):
        print(f"  {name:32s} -> {code}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
