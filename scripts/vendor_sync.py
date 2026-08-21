#!/usr/bin/env python
"""Vendor `lmpipeline` into a consumer repo, or verify a vendored copy is current.

Why this exists: DIMER builds images with `docker build` and no git credentials in the build
context (DEPLOYMENT.md §4), so a private `pip install git+https://...` fails. Until
`language-model-pipeline` is public, consumers carry a vendored copy of the package.

A copy that can drift is exactly what this project exists to prevent, so the copy is
machine-generated and machine-checked:

    # in language-model-dataset-validator / language-model-finetuner
    python scripts/vendor_sync.py sync   --into src/_vendor
    python scripts/vendor_sync.py verify --into src/_vendor   # CI gate, non-zero on drift

`verify` recomputes the tree hash and compares it against VENDOR_SHA. Any edit to a vendored
file — from either side — fails the build.

When the repo goes public, delete the vendored tree and add
`lmpipeline @ git+https://github.com/kurtvalcorza/language-model-pipeline@<tag>` to
requirements.txt. Import paths do not change.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "src" / "lmpipeline"
SHA_FILENAME = "VENDOR_SHA"
SKIP_DIRS = {"__pycache__", ".pytest_cache"}


def iter_package_files(root: Path):
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

    Path-sensitive so a deleted or renamed file changes the hash, not just edited bytes.
    """
    digest = hashlib.sha256()
    for path in iter_package_files(root):
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def do_sync(into: Path) -> int:
    target = into / "lmpipeline"
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        PACKAGE_ROOT, target, ignore=shutil.ignore_patterns(*SKIP_DIRS, "*.pyc", "*.pyo")
    )
    digest = tree_hash(target)
    (into / SHA_FILENAME).write_text(digest + "\n", encoding="utf-8")
    print(f"vendored lmpipeline -> {target}")
    print(f"{SHA_FILENAME} = {digest}")
    return 0


def do_verify(into: Path) -> int:
    target = into / "lmpipeline"
    sha_file = into / SHA_FILENAME
    if not target.is_dir():
        print(f"FAIL: no vendored package at {target}", file=sys.stderr)
        return 1
    if not sha_file.is_file():
        print(f"FAIL: missing {sha_file}", file=sys.stderr)
        return 1

    recorded = sha_file.read_text(encoding="utf-8").strip()
    actual = tree_hash(target)
    if recorded != actual:
        print(
            "FAIL: vendored lmpipeline has drifted.\n"
            f"  recorded: {recorded}\n"
            f"  actual:   {actual}\n"
            "Re-run `vendor_sync.py sync` against the pinned upstream revision. Never edit "
            "the vendored tree directly.",
            file=sys.stderr,
        )
        return 1
    print(f"OK: vendored lmpipeline matches {recorded}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["sync", "verify", "hash"])
    parser.add_argument(
        "--into", type=Path, default=Path("src/_vendor"),
        help="Directory holding the vendored package (default: src/_vendor)",
    )
    args = parser.parse_args()

    if args.command == "hash":
        print(tree_hash(PACKAGE_ROOT))
        return 0
    if args.command == "sync":
        return do_sync(args.into)
    return do_verify(args.into)


if __name__ == "__main__":
    raise SystemExit(main())
