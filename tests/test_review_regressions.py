"""Regressions for the seven defects found reviewing the initial contracts commit.

Each test names the defect it locks down. They live together so the review's conclusions
stay auditable rather than dissolving into the general suite.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from lmpipeline.datasets.normalize import (
    MAX_LINE_BYTES,
    _iter_raw_lines,
    detect_family,
    iter_examples,
)
from lmpipeline.datasets.resolver import safe_extract_zip
from lmpipeline.dimer import DimerEnv, notify_done_callback
from lmpipeline.errors import Code, DatasetError, PipelineError, Stage
from lmpipeline.result import Result

# -- 1. Result.from_exception collided on `details` ---------------------------


def test_from_exception_accepts_caller_details_for_a_plain_exception():
    result = Result.from_exception(
        RuntimeError("boom"), stage=Stage.TRAINING, details={"phase": "model_load"}
    )
    details = result.to_dict()["metadata"]["languageModelPipeline"]["details"]
    assert details["phase"] == "model_load"
    assert details["exceptionType"] == "RuntimeError"


def test_from_exception_merges_caller_details_with_pipeline_error_details():
    exc = DatasetError("bad line", code=Code.DATASET_INVALID_JSON, details={"line": 4})
    result = Result.from_exception(
        exc, stage=Stage.VALIDATION, details={"phase": "parse", "line": 999}
    )
    details = result.to_dict()["metadata"]["languageModelPipeline"]["details"]
    assert details["phase"] == "parse"
    # The exception knows the real line number; caller context must not overwrite it.
    assert details["line"] == 4


# -- 2. Line-size cap was enforced only after the line was in memory ----------


def test_oversized_line_fails_without_being_fully_read():
    """Memory stays bounded even when the line is far larger than the cap."""
    payload = b"x" * (MAX_LINE_BYTES + (8 << 20))  # 8 MB over
    stream = io.BytesIO(payload)
    with pytest.raises(DatasetError) as exc:
        list(_iter_raw_lines(stream, max_bytes=MAX_LINE_BYTES, chunk_size=1 << 20))
    assert exc.value.code == Code.DATASET_LINE_TOO_LARGE
    # Failed before consuming the whole stream: the point of the fix.
    assert stream.tell() < len(payload)


def test_line_numbering_survives_chunked_reads():
    """Chunk boundaries must not shift the line numbers users are told about."""
    data = b'{"a":1}\n\n{"b":2}\n{"c":3}'  # blank line, and no trailing newline
    got = list(_iter_raw_lines(io.BytesIO(data), max_bytes=1024, chunk_size=3))
    assert [n for n, _ in got] == [1, 2, 3, 4]
    assert [line for _, line in got] == [b'{"a":1}', b"", b'{"b":2}', b'{"c":3}']


def test_oversized_line_still_reports_its_line_number(tmp_path):
    path = tmp_path / "big.jsonl"
    with open(path, "wb") as fh:
        fh.write(b'{"prompt":"a","completion":"b"}\n')
        fh.write(b'{"prompt":"' + b"x" * (MAX_LINE_BYTES + 1024) + b'"}\n')
    with pytest.raises(DatasetError) as exc:
        list(iter_examples(path))
    assert exc.value.code == Code.DATASET_LINE_TOO_LARGE
    assert exc.value.details["line"] == 2


# -- 3. Ambiguous records were classified by precedence ----------------------


def test_record_matching_two_families_is_rejected():
    """Precedence would have silently discarded the instruction/output half."""
    record = {"prompt": "p", "completion": "c", "instruction": "i", "output": "o"}
    with pytest.raises(DatasetError) as exc:
        detect_family(record, 1)
    assert exc.value.code == Code.DATASET_SCHEMA_MIXED
    assert set(exc.value.details["families"]) == {"prompt_completion", "instruction"}


def test_conversational_record_carrying_output_is_rejected():
    with pytest.raises(DatasetError) as exc:
        detect_family({"messages": [], "output": "o"}, 7)
    assert exc.value.code == Code.DATASET_SCHEMA_MIXED
    assert exc.value.details["line"] == 7


def test_unambiguous_records_still_resolve():
    assert detect_family({"messages": []}, 1) == "conversational"
    assert detect_family({"prompt": "p", "completion": "c"}, 1) == "prompt_completion"
    assert detect_family({"instruction": "i", "output": "o"}, 1) == "instruction"


# -- 4 & 5. Callback scheme validation and body/content-type ------------------


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "ftp://host/x", "gopher://host/1", "javascript:alert(1)"]
)
def test_non_http_callback_schemes_are_refused(monkeypatch, url):
    """urllib would otherwise open file:// and ftp:// without complaint."""
    monkeypatch.setenv("DIMER_DONE_CALLBACK", url)
    env = DimerEnv.from_environ()

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError(f"urlopen was called for a {url!r} callback")

    monkeypatch.setattr("urllib.request.urlopen", explode)
    assert notify_done_callback(env) is False


def test_callback_posts_empty_body_without_a_json_content_type(monkeypatch):
    """Declaring application/json on a zero-length body is invalid JSON."""
    monkeypatch.setenv("DIMER_DONE_CALLBACK", "https://dimer.example/cb")
    env = DimerEnv.from_environ()
    captured = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=None):
        captured["method"] = request.get_method()
        captured["headers"] = dict(request.header_items())
        captured["data"] = request.data
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert notify_done_callback(env) is True
    assert captured["method"] == "POST"
    assert captured["data"] == b""
    assert not any(k.lower() == "content-type" for k in captured["headers"])


def test_callback_failure_never_raises(monkeypatch):
    monkeypatch.setenv("DIMER_DONE_CALLBACK", "https://dimer.example/cb")
    env = DimerEnv.from_environ()

    def boom(*args, **kwargs):
        raise OSError("network is unreachable https://dimer.example/cb?token=LEAK")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    assert notify_done_callback(env) is False


# -- 6. Extraction containment used a string prefix --------------------------


def test_containment_rejects_a_sibling_sharing_a_name_prefix(tmp_path):
    """`.../out` is a string prefix of `.../outside`, but is not its parent."""
    dest = tmp_path / "out"
    sibling = tmp_path / "outside" / "x"
    assert str(sibling).startswith(str(dest))  # the old, wrong check
    assert not sibling.is_relative_to(dest)  # the check now in use


def test_safe_extraction_still_accepts_a_benign_archive(tmp_path):
    archive = tmp_path / "ok.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("train.jsonl", '{"prompt":"a","completion":"b"}\n')
        zf.writestr("nested/dir/test.jsonl", '{"prompt":"c","completion":"d"}\n')
    dest = tmp_path / "out"
    safe_extract_zip(archive, dest)
    assert (dest / "train.jsonl").is_file()
    assert (dest / "nested" / "dir" / "test.jsonl").is_file()


# -- 7. detect_file_family was dead code -------------------------------------


def test_detect_file_family_is_gone():
    """Deleted rather than kept untested; re-adding it needs a caller and tests."""
    from lmpipeline.datasets import normalize

    assert not hasattr(normalize, "detect_file_family")


# -- the fixes must not have loosened anything -------------------------------


def test_privacy_rule_still_holds_after_the_details_merge(tmp_path):
    secret = "PATIENT-SSN-123456789"
    path = tmp_path / "s.jsonl"
    path.write_text(json.dumps({"prompt": secret, "completion": ""}) + "\n", encoding="utf-8")
    with pytest.raises(PipelineError) as exc:
        list(iter_examples(path))
    payload = json.dumps(
        Result.from_exception(exc.value, stage=Stage.VALIDATION,
                              details={"phase": "parse"}).to_dict()
    )
    assert secret not in payload


def test_fixture_paths_are_untouched(tmp_path: Path):
    """The chunked reader must behave identically on ordinary files."""
    path = tmp_path / "ok.jsonl"
    path.write_text(
        '{"prompt":"a","completion":"b"}\n\n{"prompt":"c","completion":"d"}\n',
        encoding="utf-8",
    )
    examples = list(iter_examples(path))
    assert [e.line_number for e in examples] == [1, 3]
    assert examples[1].assistant_text == "d"
