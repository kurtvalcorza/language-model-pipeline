"""Deterministic, safe resolution of dataset splits from a directory or a .zip.

Shared verbatim by the validator and the finetuner. If these two ever resolve a dataset
differently, validation stops meaning anything — which is why this lives in one package
rather than being reimplemented per repo.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ..errors import Code, DatasetError

# Canonical split filenames. `val.jsonl` is the one accepted alias; if both it and
# validation.jsonl are present the input is ambiguous and fails.
SPLIT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "train": ("train.jsonl",),
    "validation": ("validation.jsonl", "val.jsonl"),
    "test": ("test.jsonl",),
}
REQUIRED_SPLITS = ("train",)

# Ingestion bounds, in one place because both consumers must fail at the same size.
#
# These are deliberately far below "what fits in RAM". Both consumers hold parsed examples
# in memory, and the finetuner additionally holds a tokenized copy, so the peak is a
# multiple of the file size. A dataset that is merely transport-valid must fail with a
# stable code, not by having the kernel OOM-kill the Job — an OOM kill produces no result
# document at all, which is the one outcome the contract cannot report on.
MAX_SPLIT_BYTES = 512 * 1024**2
MAX_DATASET_BYTES = 1024**3

# Archive bounds. Aligned to the split bounds above so an archive cannot smuggle in a
# member that split resolution would then have to reject after paying to extract it.
MAX_COMPRESSED_BYTES = MAX_DATASET_BYTES
MAX_UNCOMPRESSED_BYTES = MAX_DATASET_BYTES
MAX_MEMBER_BYTES = MAX_SPLIT_BYTES
MAX_MEMBERS = 10_000
MAX_COMPRESSION_RATIO = 200.0


@dataclass(frozen=True)
class ResolvedDataset:
    root: Path
    splits: dict[str, Path]
    source: str  # "directory" | "zip"
    archive: str | None = None
    total_bytes: int = 0

    @property
    def has_validation(self) -> bool:
        return "validation" in self.splits

    @property
    def has_test(self) -> bool:
        return "test" in self.splits


def _is_unsafe_member(name: str) -> str | None:
    """Return a reason string when an archive member must be rejected."""
    if not name or name.endswith("/"):
        return None  # directory entries are dropped, not rejected
    pure = PurePosixPath(name.replace("\\", "/"))
    if pure.is_absolute():
        return "absolute path"
    if any(part == ".." for part in pure.parts):
        return "parent-directory traversal"
    # Windows drive prefix, e.g. C:/evil
    if len(name) >= 2 and name[1] == ":":
        return "drive-letter prefix"
    return None


def safe_extract_zip(archive_path: Path, dest: Path) -> None:
    """Extract a zip with traversal, bomb and member-count bounds enforced.

    Bounds are checked against the central directory BEFORE any bytes are written, and the
    real written size is checked again per member so a lying header cannot slip past.
    """
    if archive_path.stat().st_size > MAX_COMPRESSED_BYTES:
        raise DatasetError(
            f"Archive is larger than the {MAX_COMPRESSED_BYTES} byte limit.",
            code=Code.DATASET_ARCHIVE_TOO_LARGE,
        )

    try:
        zf = zipfile.ZipFile(archive_path)
    except zipfile.BadZipFile as exc:
        raise DatasetError(
            "The uploaded archive is not a readable .zip file.",
            code=Code.DATASET_ARCHIVE_UNREADABLE,
        ) from exc

    with zf:
        infos = zf.infolist()
        if len(infos) > MAX_MEMBERS:
            raise DatasetError(
                f"Archive contains {len(infos)} members, above the {MAX_MEMBERS} limit.",
                code=Code.DATASET_ARCHIVE_TOO_LARGE,
            )

        # A zip may legally carry two members with the same path. Extraction is
        # last-one-wins, so which bytes land on disk depends on member ordering — the
        # dataset is genuinely ambiguous and must not be guessed at.
        seen_members: dict[str, int] = {}
        for info in infos:
            if info.filename.endswith("/"):
                continue
            canonical = PurePosixPath(info.filename.replace("\\", "/")).as_posix()
            seen_members[canonical] = seen_members.get(canonical, 0) + 1
        duplicated = sorted(name for name, count in seen_members.items() if count > 1)
        if duplicated:
            raise DatasetError(
                "The archive contains more than one member with the same path: "
                + ", ".join(duplicated[:5])
                + ". Which copy would be used depends on archive ordering, so the upload "
                "is ambiguous.",
                code=Code.DATASET_ARCHIVE_DUPLICATE_MEMBER,
                details={"members": duplicated[:50], "count": len(duplicated)},
            )

        total = 0
        for info in infos:
            reason = _is_unsafe_member(info.filename)
            if reason is not None:
                raise DatasetError(
                    f"Archive member {info.filename!r} rejected: {reason}.",
                    code=Code.DATASET_ARCHIVE_UNSAFE,
                    details={"member": info.filename, "reason": reason},
                )
            # Reparse points / symlinks are encoded in the high bits of external_attr.
            mode = info.external_attr >> 16
            if mode and (mode & 0o170000) == 0o120000:
                raise DatasetError(
                    f"Archive member {info.filename!r} is a symlink, which is not allowed.",
                    code=Code.DATASET_ARCHIVE_UNSAFE,
                    details={"member": info.filename, "reason": "symlink"},
                )
            if info.file_size > MAX_MEMBER_BYTES:
                raise DatasetError(
                    f"Archive member {info.filename!r} exceeds the per-member size limit.",
                    code=Code.DATASET_ARCHIVE_TOO_LARGE,
                    details={"member": info.filename},
                )
            if info.compress_size > 0:
                ratio = info.file_size / info.compress_size
                if ratio > MAX_COMPRESSION_RATIO:
                    raise DatasetError(
                        f"Archive member {info.filename!r} has a compression ratio of "
                        f"{ratio:.0f}:1, above the {MAX_COMPRESSION_RATIO:.0f}:1 limit.",
                        code=Code.DATASET_ARCHIVE_UNSAFE,
                        details={"member": info.filename, "ratio": round(ratio, 1)},
                    )
            total += info.file_size
            if total > MAX_UNCOMPRESSED_BYTES:
                raise DatasetError(
                    "Archive expands beyond the total uncompressed size limit.",
                    code=Code.DATASET_ARCHIVE_TOO_LARGE,
                )

        dest.mkdir(parents=True, exist_ok=True)
        written = 0
        for info in infos:
            if info.filename.endswith("/"):
                continue
            target = (dest / info.filename).resolve()
            # Path containment, not string prefix: `.../out` is a string prefix of
            # `.../outside/x`, which would wrongly read as contained.
            if not target.is_relative_to(dest.resolve()):
                raise DatasetError(
                    f"Archive member {info.filename!r} escapes the extraction root.",
                    code=Code.DATASET_ARCHIVE_UNSAFE,
                    details={"member": info.filename},
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                while chunk := src.read(1024 * 1024):
                    written += len(chunk)
                    if written > MAX_UNCOMPRESSED_BYTES:
                        raise DatasetError(
                            "Archive expanded past the uncompressed size limit during read.",
                            code=Code.DATASET_ARCHIVE_TOO_LARGE,
                        )
                    out.write(chunk)


def _find_content_root(root: Path) -> Path:
    """Descend through single-directory wrappers.

    Users commonly zip a folder rather than its contents. One level of wrapping is
    tolerated; a zip inside a zip is not (the portal documents it as a common error).
    """
    current = root
    for _ in range(4):
        entries = [p for p in current.iterdir() if not p.name.startswith("__MACOSX")]
        if len(entries) == 1 and entries[0].is_dir():
            current = entries[0]
            continue
        if len(entries) == 1 and entries[0].suffix.lower() == ".zip":
            raise DatasetError(
                "The upload contains another .zip archive rather than dataset files. "
                "Please upload the inner archive directly.",
                code=Code.DATASET_ARCHIVE_NESTED,
                details={"member": entries[0].name},
            )
        break
    return current


# Directories that are packaging noise rather than dataset content.
_IGNORED_DIRS = ("__MACOSX", ".git", ".ipynb_checkpoints")

ALL_SPLIT_FILENAMES = frozenset(
    name for candidates in SPLIT_CANDIDATES.values() for name in candidates
)


def _reject_stray_split_files(root: Path) -> None:
    """Fail when a canonical split filename appears anywhere below the content root.

    Splits are resolved at the root only. A second `train.jsonl` in a subdirectory would
    otherwise be silently ignored, and the user would train on a different file than the
    one they believe they uploaded — the same ambiguity as duplicate split aliases, just
    hidden one level down.
    """
    strays: dict[str, list[str]] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.name not in ALL_SPLIT_FILENAMES:
            continue
        if path.parent == root:
            continue
        if any(part in _IGNORED_DIRS for part in path.parts):
            continue
        strays.setdefault(path.name, []).append(
            path.relative_to(root).as_posix()
        )

    if strays:
        listed = "; ".join(
            f"{name} also at {', '.join(sorted(paths)[:3])}"
            for name, paths in sorted(strays.items())
        )
        raise DatasetError(
            "Split files were found outside the dataset root, so which copy to use is "
            f"ambiguous: {listed}. Keep exactly one copy of each split at the top level.",
            code=Code.DATASET_SPLIT_AMBIGUOUS,
            details={"strays": {k: sorted(v)[:20] for k, v in sorted(strays.items())}},
        )


def enforce_split_bounds(splits: dict[str, Path]) -> int:
    """Bound resolved splits by byte size, and return the aggregate.

    Runs for directory and archive inputs alike. The archive path already bounds members,
    but a mounted directory carries no metadata to bound — DIMER mounts whatever the user
    uploaded — so without this a multi-gigabyte `train.jsonl` reaches the consumers with
    nothing standing between it and a materializing read.

    Checked with stat() before a single byte is parsed: the point is to fail structurally
    while failing is still cheap.
    """
    total = 0
    for name in sorted(splits):
        path = splits[name]
        size = path.stat().st_size
        if size > MAX_SPLIT_BYTES:
            raise DatasetError(
                f"Split {name!r} ({path.name}) is {size} bytes, above the "
                f"{MAX_SPLIT_BYTES} byte per-split limit.",
                code=Code.DATASET_SPLIT_TOO_LARGE,
                details={"split": name, "bytes": size, "maximum": MAX_SPLIT_BYTES},
            )
        total += size
    if total > MAX_DATASET_BYTES:
        raise DatasetError(
            f"The dataset totals {total} bytes across its splits, above the "
            f"{MAX_DATASET_BYTES} byte limit.",
            code=Code.DATASET_SPLIT_TOO_LARGE,
            details={"bytes": total, "maximum": MAX_DATASET_BYTES},
        )
    return total


def resolve_dataset(dataset_dir: Path, *, workdir: Path) -> ResolvedDataset:
    """Resolve canonical split files from DIMER's mounted dataset directory."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise DatasetError(
            f"Dataset directory {dataset_dir} does not exist.",
            code=Code.DATASET_MISSING,
        )

    entries = [p for p in dataset_dir.iterdir() if not p.name.startswith("__MACOSX")]
    archives = [p for p in entries if p.is_file() and p.suffix.lower() == ".zip"]

    if len(archives) > 1:
        raise DatasetError(
            f"Found {len(archives)} .zip archives; exactly one is expected.",
            code=Code.DATASET_SPLIT_AMBIGUOUS,
            details={"archives": sorted(p.name for p in archives)},
        )

    if archives:
        extract_root = workdir / "extracted"
        safe_extract_zip(archives[0], extract_root)
        root = _find_content_root(extract_root)
        source, archive = "zip", archives[0].name
    else:
        root = _find_content_root(dataset_dir)
        source, archive = "directory", None

    _reject_stray_split_files(root)

    splits: dict[str, Path] = {}
    for split, candidates in SPLIT_CANDIDATES.items():
        present = [root / name for name in candidates if (root / name).is_file()]
        if len(present) > 1:
            raise DatasetError(
                f"Split {split!r} is ambiguous: found "
                + ", ".join(sorted(p.name for p in present))
                + ". Supply exactly one.",
                code=Code.DATASET_SPLIT_AMBIGUOUS,
                details={"split": split, "candidates": sorted(p.name for p in present)},
            )
        if present:
            splits[split] = present[0]

    for split in REQUIRED_SPLITS:
        if split not in splits:
            raise DatasetError(
                f"Required split {split!r} not found. Expected one of: "
                + ", ".join(SPLIT_CANDIDATES[split])
                + f" at the root of the dataset (looked in {root.name}/).",
                code=Code.DATASET_SPLIT_MISSING,
                details={"split": split, "expected": list(SPLIT_CANDIDATES[split])},
            )

    total_bytes = enforce_split_bounds(splits)

    return ResolvedDataset(
        root=root, splits=splits, source=source, archive=archive, total_bytes=total_bytes
    )
