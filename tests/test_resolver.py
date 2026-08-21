from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

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
