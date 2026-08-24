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
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lmpipeline.vendoring import (  # noqa: E402
    SHA_FILENAME,
    SKIP_DIRS,
    tree_hash,
    verify_vendored,
)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "src" / "lmpipeline"


def do_sync(into: Path) -> int:
    target = into / "lmpipeline"
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        PACKAGE_ROOT, target, ignore=shutil.ignore_patterns(*SKIP_DIRS, "*.pyc", "*.pyo")
    )
    digest = tree_hash(target)
    # newline="" suppresses the platform translation write_text would otherwise apply. On
    # Windows this file was being written with CRLF while `.gitattributes` normalizes it to
    # LF on commit, so `git status` showed VENDOR_SHA permanently modified straight after a
    # clean sync -- noise in exactly the gate whose whole job is to make drift visible.
    (into / SHA_FILENAME).write_text(digest + "\n", encoding="utf-8", newline="")
    print(f"vendored lmpipeline -> {target}")
    print(f"{SHA_FILENAME} = {digest}")
    return 0


def do_verify(into: Path) -> int:
    ok, message = verify_vendored(into)
    if not ok:
        print(f"FAIL: {message}", file=sys.stderr)
        return 1
    print(f"OK: {message}")
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
