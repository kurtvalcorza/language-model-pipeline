"""Tree hashing for the vendored copy of this package.

Lives inside the package, not beside the sync script, so it travels with the vendored tree.
That is what lets a consumer verify its own copy offline — in CI, in a container build, in a
fresh clone — without needing access to the private upstream repo or a copy of its scripts.

The alternative was a second implementation of the hash in each consumer, which is the exact
failure this whole vendoring arrangement exists to prevent: two copies of a rule that are
supposed to agree and eventually do not.

Self-referential by design and sound in the direction that matters: this file is part of the
tree it hashes, so editing it changes the recorded hash and the gate fails. It detects
drift, not tampering — a deliberate edit that also rewrites VENDOR_SHA passes here and is
caught by the upstream cross-check instead.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

SHA_FILENAME = "VENDOR_SHA"
SKIP_DIRS = {"__pycache__", ".pytest_cache"}


def iter_package_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        yield path


def tree_hash(root: Path) -> str:
    """Hash of relative paths plus contents.

    Path-sensitive, so a deleted or renamed file changes the hash rather than only an
    edited one.
    """
    digest = hashlib.sha256()
    for path in iter_package_files(root):
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def verify_vendored(vendor_root: Path) -> tuple[bool, str]:
    """Check a vendored tree against its recorded VENDOR_SHA.

    Returns (ok, message). Callers decide whether a mismatch is fatal; in CI it always is.
    """
    package = vendor_root / "lmpipeline"
    sha_file = vendor_root / SHA_FILENAME
    if not package.is_dir():
        return False, f"no vendored package at {package}"
    if not sha_file.is_file():
        return False, f"missing {sha_file}"

    recorded = sha_file.read_text(encoding="utf-8").strip()
    actual = tree_hash(package)
    if recorded != actual:
        return False, (
            "vendored lmpipeline has drifted.\n"
            f"  recorded: {recorded}\n"
            f"  actual:   {actual}\n"
            "Re-run `vendor_sync.py sync` against the pinned upstream revision. Never edit "
            "the vendored tree directly."
        )
    return True, f"vendored lmpipeline matches {recorded}"
