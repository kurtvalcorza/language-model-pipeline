from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from lmpipeline.datasets import resolver
from lmpipeline.datasets.resolver import resolve_dataset, safe_extract_zip
from lmpipeline.errors import Code, DatasetError

TRAIN_LINE = '{"prompt":"a","completion":"b"}\n'


def make_dataset_dir(tmp_path: Path, names: list[str]) -> Path:
    root = tmp_path / "dataset"
    root.mkdir()
    for name in names:
        (root / name).write_text(TRAIN_LINE, encoding="utf-8")
    return root


def test_resolves_a_plain_directory(tmp_path):
    root = make_dataset_dir(tmp_path, ["train.jsonl", "validation.jsonl", "test.jsonl"])
    resolved = resolve_dataset(root, workdir=tmp_path / "work")
    assert resolved.source == "directory"
    assert set(resolved.splits) == {"train", "validation", "test"}
    assert resolved.has_validation and resolved.has_test


def test_val_alias_is_accepted(tmp_path):
    root = make_dataset_dir(tmp_path, ["train.jsonl", "val.jsonl"])
    resolved = resolve_dataset(root, workdir=tmp_path / "work")
    assert resolved.splits["validation"].name == "val.jsonl"


def test_duplicate_split_candidates_fail(tmp_path):
    """The blueprint's explicit rule: ambiguity fails, it is never guessed."""
    root = make_dataset_dir(tmp_path, ["train.jsonl", "validation.jsonl", "val.jsonl"])
    with pytest.raises(DatasetError) as exc:
        resolve_dataset(root, workdir=tmp_path / "work")
    assert exc.value.code == Code.DATASET_SPLIT_AMBIGUOUS


def test_missing_train_split_fails(tmp_path):
    root = make_dataset_dir(tmp_path, ["validation.jsonl"])
    with pytest.raises(DatasetError) as exc:
        resolve_dataset(root, workdir=tmp_path / "work")
    assert exc.value.code == Code.DATASET_SPLIT_MISSING


def test_zip_input_is_resolved(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    with zipfile.ZipFile(root / "data.zip", "w") as zf:
        zf.writestr("train.jsonl", TRAIN_LINE)
        zf.writestr("validation.jsonl", TRAIN_LINE)
    resolved = resolve_dataset(root, workdir=tmp_path / "work")
    assert resolved.source == "zip"
    assert resolved.archive == "data.zip"
    assert set(resolved.splits) == {"train", "validation"}


def test_single_directory_wrapper_is_unwrapped(tmp_path):
    """Users routinely zip the folder rather than its contents."""
    root = tmp_path / "dataset"
    root.mkdir()
    with zipfile.ZipFile(root / "data.zip", "w") as zf:
        zf.writestr("my-dataset/train.jsonl", TRAIN_LINE)
    resolved = resolve_dataset(root, workdir=tmp_path / "work")
    assert "train" in resolved.splits


def test_nested_zip_gets_its_own_actionable_error(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    inner = tmp_path / "inner.zip"
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("train.jsonl", TRAIN_LINE)
    with zipfile.ZipFile(root / "outer.zip", "w") as zf:
        zf.writestr("inner.zip", inner.read_bytes())
    with pytest.raises(DatasetError) as exc:
        resolve_dataset(root, workdir=tmp_path / "work")
    assert exc.value.code == Code.DATASET_ARCHIVE_NESTED


def test_two_archives_are_ambiguous(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    for name in ("a.zip", "b.zip"):
        with zipfile.ZipFile(root / name, "w") as zf:
            zf.writestr("train.jsonl", TRAIN_LINE)
    with pytest.raises(DatasetError) as exc:
        resolve_dataset(root, workdir=tmp_path / "work")
    assert exc.value.code == Code.DATASET_SPLIT_AMBIGUOUS


# -- archive safety -----------------------------------------------------------


@pytest.mark.parametrize(
    "member",
    ["../escape.jsonl", "/abs/train.jsonl", "a/../../escape.jsonl", "C:/evil.jsonl"],
)
def test_traversal_members_are_rejected(tmp_path, member):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(member, TRAIN_LINE)
    with pytest.raises(DatasetError) as exc:
        safe_extract_zip(archive, tmp_path / "out")
    assert exc.value.code == Code.DATASET_ARCHIVE_UNSAFE


def test_nothing_is_written_when_a_member_is_unsafe(tmp_path):
    """Bounds are enforced against the central directory before any bytes land."""
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("train.jsonl", TRAIN_LINE)
        zf.writestr("../escape.jsonl", TRAIN_LINE)
    out = tmp_path / "out"
    with pytest.raises(DatasetError):
        safe_extract_zip(archive, out)
    assert not (out / "train.jsonl").exists()


def test_compression_bomb_is_rejected(tmp_path):
    archive = tmp_path / "bomb.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("train.jsonl", "0" * (5 * 1024 * 1024))
    with pytest.raises(DatasetError) as exc:
        safe_extract_zip(archive, tmp_path / "out")
    assert exc.value.code == Code.DATASET_ARCHIVE_UNSAFE


def test_symlink_member_is_rejected(tmp_path):
    archive = tmp_path / "link.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        info = zipfile.ZipInfo("train.jsonl")
        info.external_attr = (0o120777 << 16)  # symlink mode bits
        zf.writestr(info, "/etc/passwd")
    with pytest.raises(DatasetError) as exc:
        safe_extract_zip(archive, tmp_path / "out")
    assert exc.value.code == Code.DATASET_ARCHIVE_UNSAFE


def test_corrupt_archive_reports_cleanly(tmp_path):
    archive = tmp_path / "bad.zip"
    archive.write_bytes(b"not a zip at all")
    with pytest.raises(DatasetError) as exc:
        safe_extract_zip(archive, tmp_path / "out")
    assert exc.value.code == Code.DATASET_ARCHIVE_UNREADABLE


# -- gaps found against the dataset-suite spec (issue #2 section 9) ------------


def test_duplicate_zip_members_with_the_same_path_are_rejected(tmp_path):
    """Spec case 22.

    A zip may legally carry two members with the same path. Extraction is last-one-wins,
    so which bytes land on disk depends on member ordering — the upload is ambiguous and
    must not be guessed at.
    """
    archive = tmp_path / "dupe.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("train.jsonl", TRAIN_LINE)
        zf.writestr("train.jsonl", '{"prompt":"DIFFERENT","completion":"x"}\n')
    with pytest.raises(DatasetError) as exc:
        safe_extract_zip(archive, tmp_path / "out")
    assert exc.value.code == Code.DATASET_ARCHIVE_DUPLICATE_MEMBER
    assert "train.jsonl" in exc.value.details["members"]


def test_duplicate_members_are_detected_across_path_spellings(tmp_path):
    """A zip written on Windows may use backslash separators for the same logical path."""
    archive = tmp_path / "dupe.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("data/train.jsonl", TRAIN_LINE)
        zf.writestr(r"data\train.jsonl", TRAIN_LINE)
    with pytest.raises(DatasetError) as exc:
        safe_extract_zip(archive, tmp_path / "out")
    assert exc.value.code == Code.DATASET_ARCHIVE_DUPLICATE_MEMBER


def test_a_second_train_file_in_a_subdirectory_is_ambiguous(tmp_path):
    """Spec case 7.

    The nested copy would previously have been silently ignored, so the user would train
    on a different file than the one they thought they uploaded.
    """
    root = make_dataset_dir(tmp_path, ["train.jsonl"])
    nested = root / "backup"
    nested.mkdir()
    (nested / "train.jsonl").write_text(TRAIN_LINE, encoding="utf-8")

    with pytest.raises(DatasetError) as exc:
        resolve_dataset(root, workdir=tmp_path / "work")
    assert exc.value.code == Code.DATASET_SPLIT_AMBIGUOUS
    assert "backup/train.jsonl" in exc.value.details["strays"]["train.jsonl"]


def test_a_nested_validation_alias_is_also_ambiguous(tmp_path):
    root = make_dataset_dir(tmp_path, ["train.jsonl", "validation.jsonl"])
    nested = root / "old"
    nested.mkdir()
    (nested / "val.jsonl").write_text(TRAIN_LINE, encoding="utf-8")

    with pytest.raises(DatasetError) as exc:
        resolve_dataset(root, workdir=tmp_path / "work")
    assert exc.value.code == Code.DATASET_SPLIT_AMBIGUOUS


def test_macosx_metadata_copies_are_not_treated_as_strays(tmp_path):
    """Archives from macOS routinely carry __MACOSX shadows; those are noise, not data."""
    root = make_dataset_dir(tmp_path, ["train.jsonl"])
    shadow = root / "__MACOSX"
    shadow.mkdir()
    (shadow / "train.jsonl").write_text("", encoding="utf-8")

    resolved = resolve_dataset(root, workdir=tmp_path / "work")
    assert resolved.splits["train"].parent == root


def test_unrelated_nested_files_do_not_trigger_the_stray_check(tmp_path):
    root = make_dataset_dir(tmp_path, ["train.jsonl"])
    nested = root / "notes"
    nested.mkdir()
    (nested / "README.md").write_text("hello", encoding="utf-8")
    (nested / "extra.jsonl").write_text(TRAIN_LINE, encoding="utf-8")

    resolved = resolve_dataset(root, workdir=tmp_path / "work")
    assert set(resolved.splits) == {"train"}


# -- ingestion bounds ----------------------------------------------------------
#
# Sized down via monkeypatch rather than by writing half a gigabyte to tmp_path: the
# behaviour under test is "stat, compare, raise before parsing", which is independent of
# the constant's real value.


def test_an_oversized_split_in_a_mounted_directory_is_rejected(tmp_path, monkeypatch):
    """The gap a zip cannot cover: DIMER mounts a directory with no metadata to bound."""
    monkeypatch.setattr(resolver, "MAX_SPLIT_BYTES", 64)
    root = make_dataset_dir(tmp_path, ["train.jsonl"])
    (root / "train.jsonl").write_text(TRAIN_LINE * 40, encoding="utf-8")

    with pytest.raises(DatasetError) as exc:
        resolve_dataset(root, workdir=tmp_path / "work")
    assert exc.value.code == Code.DATASET_SPLIT_TOO_LARGE
    assert exc.value.details["split"] == "train"


def test_splits_that_are_individually_legal_can_still_be_too_much_together(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(resolver, "MAX_SPLIT_BYTES", 1024)
    monkeypatch.setattr(resolver, "MAX_DATASET_BYTES", 100)
    root = make_dataset_dir(tmp_path, ["train.jsonl", "validation.jsonl", "test.jsonl"])
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        (root / name).write_text(TRAIN_LINE * 2, encoding="utf-8")

    with pytest.raises(DatasetError) as exc:
        resolve_dataset(root, workdir=tmp_path / "work")
    assert exc.value.code == Code.DATASET_SPLIT_TOO_LARGE
    assert "split" not in exc.value.details  # the aggregate failed, not one file


def test_bounds_are_enforced_for_archives_too(tmp_path, monkeypatch):
    monkeypatch.setattr(resolver, "MAX_SPLIT_BYTES", 64)
    root = tmp_path / "dataset"
    root.mkdir()
    with zipfile.ZipFile(root / "data.zip", "w") as zf:
        zf.writestr("train.jsonl", TRAIN_LINE * 40)

    with pytest.raises(DatasetError) as exc:
        resolve_dataset(root, workdir=tmp_path / "work")
    assert exc.value.code in {
        Code.DATASET_SPLIT_TOO_LARGE, Code.DATASET_ARCHIVE_TOO_LARGE
    }


def test_a_dataset_within_bounds_reports_its_size(tmp_path):
    """Aggregate across splits, taken from stat() rather than from the text written.

    Asserting against `len(TRAIN_LINE)` instead looks equivalent and is not: on Windows
    the text-mode write turns each LF into CRLF, so the on-disk size is larger than the
    string. The bound is about bytes on disk, so the test measures bytes on disk.
    """
    root = make_dataset_dir(tmp_path, ["train.jsonl", "validation.jsonl"])
    resolved = resolve_dataset(root, workdir=tmp_path / "work")
    expected = sum(p.stat().st_size for p in resolved.splits.values())
    assert resolved.total_bytes == expected > 0


def test_the_archive_member_bound_matches_the_split_bound():
    """Two limits that disagree mean paying to extract bytes the next step rejects."""
    assert resolver.MAX_MEMBER_BYTES == resolver.MAX_SPLIT_BYTES
    assert resolver.MAX_UNCOMPRESSED_BYTES <= resolver.MAX_DATASET_BYTES
