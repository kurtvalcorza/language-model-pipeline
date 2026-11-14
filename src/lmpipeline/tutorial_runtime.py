"""Strict adapter-archive intake for release-grade tutorial notebooks.

These helpers are notebook orchestration only. They do not implement training, inference,
model loading, masking, or artifact publication.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .tutorial_api import consume_adapter_archive as _consume_adapter_archive


def consume_adapter_archive(
    archive_path: str | Path,
    *,
    extraction_root: str | Path,
    expected_archive_sha256: str = "",
    size_limit_bytes: int = 512 * 1024**2,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Consume an adapter ZIP and require the manifest at the archive root.

    The underlying support helper validates traversal, links, duplicates, sizes, hashes and
    the manifested subtree. Requiring the manifest at the extraction root closes the remaining
    ambiguity where an attacker could place an otherwise valid artifact in a subdirectory and
    add an unmanifested sibling outside that subtree.
    """
    extraction_root = Path(extraction_root).resolve()
    root, manifest, provenance = _consume_adapter_archive(
        archive_path,
        extraction_root=extraction_root,
        expected_archive_sha256=expected_archive_sha256,
        size_limit_bytes=size_limit_bytes,
    )
    if root.resolve() != extraction_root:
        raise ValueError("artifact-manifest.json must be at the ZIP archive root")
    return root, manifest, provenance
