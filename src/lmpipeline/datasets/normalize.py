"""Schema-family detection and normalization to canonical conversational examples.

Three source families are accepted and all are normalized to a single `messages` form. One
family per file: a file mixing families is ambiguous and fails rather than being guessed at.

No record is ever silently dropped or mutated. Every rejection carries a 1-based line number
and a stable code, and no error message embeds record content — datasets are user-private
(SECURITY.md).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from ..errors import Code, DatasetError

VALID_ROLES = ("system", "user", "assistant")
MAX_LINE_BYTES = 4 * 1024 * 1024

FAMILY_CONVERSATIONAL = "conversational"
FAMILY_PROMPT_COMPLETION = "prompt_completion"
FAMILY_INSTRUCTION = "instruction"


@dataclass(frozen=True)
class Example:
    """One normalized training example."""

    messages: tuple[dict[str, str], ...]
    line_number: int

    @property
    def assistant_text(self) -> str:
        return "\n".join(m["content"] for m in self.messages if m["role"] == "assistant")

    def fingerprint(self) -> str:
        """Stable hash of the canonical form, for duplicate and leakage detection.

        Hashes the normalized messages, so the same content expressed as prompt/completion
        and as conversational collides as intended.
        """
        canonical = json.dumps(
            [[m["role"], m["content"].strip()] for m in self.messages],
            ensure_ascii=False, separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def detect_family(record: dict[str, Any], line_number: int) -> str:
    """Identify a record's schema family, or fail if it is not exactly one.

    Every family is tested — the first match is NOT taken. A record carrying keys from two
    families is ambiguous, and picking one by precedence would silently discard the other
    half of the record. DATASET_SPEC.md requires ambiguity to fail.
    """
    matches: list[str] = []
    if "messages" in record:
        matches.append(FAMILY_CONVERSATIONAL)
    if "prompt" in record or "completion" in record:
        matches.append(FAMILY_PROMPT_COMPLETION)
    if "instruction" in record or "output" in record:
        matches.append(FAMILY_INSTRUCTION)

    if not matches:
        raise DatasetError(
            f"Line {line_number}: unrecognized record schema. Expected one of: "
            "`messages`, `prompt`/`completion`, or `instruction`/`input`/`output`.",
            code=Code.DATASET_SCHEMA_UNKNOWN,
            details={"line": line_number},
        )
    if len(matches) > 1:
        raise DatasetError(
            f"Line {line_number}: record matches more than one schema family "
            f"({', '.join(matches)}). Keep one family per record so no content is "
            "silently discarded.",
            code=Code.DATASET_SCHEMA_MIXED,
            details={"line": line_number, "families": matches},
        )
    return matches[0]


def _require_text(value: Any, *, field: str, line_number: int) -> str:
    if not isinstance(value, str):
        raise DatasetError(
            f"Line {line_number}: field {field!r} must be a string, got "
            f"{type(value).__name__}.",
            code=Code.DATASET_SCHEMA_UNKNOWN,
            details={"line": line_number, "field": field},
        )
    return value


def normalize_record(record: dict[str, Any], family: str, line_number: int) -> Example:
    """Normalize one record of a known family into canonical messages."""
    if family == FAMILY_CONVERSATIONAL:
        raw = record.get("messages")
        if not isinstance(raw, list) or not raw:
            raise DatasetError(
                f"Line {line_number}: `messages` must be a non-empty array.",
                code=Code.DATASET_SCHEMA_UNKNOWN, details={"line": line_number},
            )
        messages: list[dict[str, str]] = []
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                raise DatasetError(
                    f"Line {line_number}: message {index} is not an object.",
                    code=Code.DATASET_SCHEMA_UNKNOWN, details={"line": line_number},
                )
            role = item.get("role")
            if role not in VALID_ROLES:
                raise DatasetError(
                    f"Line {line_number}: message {index} has role {role!r}; v1 supports "
                    + ", ".join(VALID_ROLES) + ".",
                    code=Code.DATASET_ROLE_INVALID,
                    details={"line": line_number, "role": str(role)},
                )
            content = _require_text(
                item.get("content"), field="content", line_number=line_number
            )
            messages.append({"role": role, "content": content})

    elif family == FAMILY_PROMPT_COMPLETION:
        prompt = _require_text(record.get("prompt"), field="prompt", line_number=line_number)
        completion = _require_text(
            record.get("completion"), field="completion", line_number=line_number
        )
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]

    else:  # FAMILY_INSTRUCTION
        instruction = _require_text(
            record.get("instruction"), field="instruction", line_number=line_number
        )
        extra = record.get("input") or ""
        extra = _require_text(extra, field="input", line_number=line_number)
        output = _require_text(record.get("output"), field="output", line_number=line_number)
        user_content = f"{instruction}\n\n{extra}".strip() if extra.strip() else instruction
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]

    example = Example(messages=tuple(messages), line_number=line_number)
    if not example.assistant_text.strip():
        raise DatasetError(
            f"Line {line_number}: every training example needs at least one non-empty "
            "assistant target.",
            code=Code.DATASET_TARGET_EMPTY, details={"line": line_number},
        )
    return example


def _iter_raw_lines(
    fh: BinaryIO, *, max_bytes: int = MAX_LINE_BYTES, chunk_size: int = 1 << 20
) -> Iterator[tuple[int, bytes]]:
    """Yield (line_number, line_bytes) without ever materializing an oversized line.

    Python's own file iteration reads an entire line into memory before any length check
    can run, so a single multi-gigabyte line OOM-kills the process instead of producing
    DATASET_LINE_TOO_LARGE. This reads fixed-size chunks and fails as soon as an
    unterminated line passes the cap, bounding memory at max_bytes + chunk_size.

    Line numbering matches physical lines 1:1, including blank lines and a final line with
    no trailing newline, so reported numbers still match what the user sees in an editor.
    """
    buffer = bytearray()
    line_number = 0
    while True:
        chunk = fh.read(chunk_size)
        if not chunk:
            break
        buffer.extend(chunk)
        while (newline := buffer.find(b"\n")) != -1:
            # Check before yielding: a line can arrive already terminated and oversized
            # within a single chunk, in which case the residual-buffer check below never
            # sees it.
            if newline > max_bytes:
                raise DatasetError(
                    f"Line {line_number + 1}: record exceeds the {max_bytes} byte limit.",
                    code=Code.DATASET_LINE_TOO_LARGE,
                    details={"line": line_number + 1},
                )
            line = bytes(buffer[:newline])
            del buffer[: newline + 1]
            line_number += 1
            yield line_number, line
        if len(buffer) > max_bytes:
            raise DatasetError(
                f"Line {line_number + 1}: record exceeds the {max_bytes} byte limit.",
                code=Code.DATASET_LINE_TOO_LARGE, details={"line": line_number + 1},
            )
    if buffer:
        line_number += 1
        if len(buffer) > max_bytes:
            raise DatasetError(
                f"Line {line_number}: record exceeds the {max_bytes} byte limit.",
                code=Code.DATASET_LINE_TOO_LARGE, details={"line": line_number},
            )
        yield line_number, bytes(buffer)


def load_examples(path: Path, *, max_examples: int) -> list[Example]:
    """Materialize a split, refusing to grow past `max_examples`.

    `iter_examples` streams, but every consumer needs the examples more than once — for
    duplicate counting, for leakage comparison, for masking — so each of them wrapped it in
    `list(...)`. That reintroduces exactly the unbounded allocation the streaming reader
    exists to avoid: a file whose rows are individually legal can still be numerous enough
    to exhaust RAM, and the example-count policy could not fire because it ran after the
    list was already built.

    Counting while filling makes the limit structural. The raise happens on the example
    that would exceed the cap, so peak memory is bounded by the cap rather than by the
    file, and the caller gets DATASET_TOO_MANY_EXAMPLES instead of an OOM kill.
    """
    examples: list[Example] = []
    for example in iter_examples(path):
        if len(examples) >= max_examples:
            raise DatasetError(
                f"{path.name} contains more than the {max_examples} examples this "
                "pipeline accepts.",
                code=Code.DATASET_TOO_MANY_EXAMPLES,
                details={"file": path.name, "maximum": max_examples,
                         "atLeast": max_examples + 1},
            )
        examples.append(example)
    return examples


def iter_examples(path: Path) -> Iterator[Example]:
    """Stream normalized examples from a JSONL file.

    Enforces one schema family per file. Reads in binary and decodes explicitly so an
    invalid byte sequence is reported with its line number rather than raising an opaque
    UnicodeDecodeError somewhere upstream.
    """
    family: str | None = None
    with open(path, "rb") as fh:
        for line_number, raw in _iter_raw_lines(fh):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                text = stripped.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise DatasetError(
                    f"Line {line_number}: not valid UTF-8 (byte offset {exc.start}).",
                    code=Code.DATASET_INVALID_UTF8, details={"line": line_number},
                ) from exc
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise DatasetError(
                    f"Line {line_number}: invalid JSON ({exc.msg} at column {exc.colno}).",
                    code=Code.DATASET_INVALID_JSON, details={"line": line_number},
                ) from exc
            if not isinstance(record, dict):
                raise DatasetError(
                    f"Line {line_number}: each line must be a JSON object, got "
                    f"{type(record).__name__}.",
                    code=Code.DATASET_INVALID_JSON, details={"line": line_number},
                )

            detected = detect_family(record, line_number)
            if family is None:
                family = detected
            elif detected != family:
                raise DatasetError(
                    f"Line {line_number}: file mixes schema families ({family!r} then "
                    f"{detected!r}). Use one family per file.",
                    code=Code.DATASET_SCHEMA_MIXED,
                    details={"line": line_number, "expected": family, "found": detected},
                )
            yield normalize_record(record, family, line_number)
