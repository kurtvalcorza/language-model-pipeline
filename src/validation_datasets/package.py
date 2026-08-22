"""Manifests, digests, and deterministic DIMER ZIP packaging.

Two reproducibility rules drive everything here:

  * The same profile against the same pinned revision must produce byte-identical output.
    Zip members carry timestamps and arbitrary ordering, both of which defeat that unless
    pinned explicitly.
  * A manifest must identify the data well enough that an acceptance report can reference
    the manifest rather than describing a dataset by human-readable name.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

from .registry import DatasetSource

# Fixed member timestamp. The alternative is a package whose digest changes every build.
FIXED_TIMESTAMP = (2026, 1, 1, 0, 0, 0)

# Fixed creating-OS byte. Python derives it from the host (0 Windows, 3 Unix), which made
# the same inputs produce different archive bytes on a laptop and on CI.
ZIP_CREATE_SYSTEM = 3

CANONICAL_SPLIT_ORDER = ("train", "validation", "test")


def write_canonical_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    """Write canonical JSONL and return its SHA-256.

    Serialization is pinned: sorted keys, no ASCII escaping, LF endings. Any of those left
    to defaults makes the digest platform- or version-dependent.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            fh.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    *,
    source: DatasetSource,
    profile_name: str,
    splits: dict[str, Path],
    digests: dict[str, str],
    counts: dict[str, int],
    selected_indices: list[int],
    exclude_languages: tuple[str, ...] = (),
    tool_version: str = "1.0",
) -> dict[str, Any]:
    """Provenance for one built profile.

    Selected source indices are recorded by digest rather than in full: for a 2,000-row
    profile the list itself is noise, but its digest still proves two builds selected the
    same rows.
    """
    index_digest = hashlib.sha256(
        json.dumps(sorted(selected_indices), separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return {
        "schemaVersion": "1.0",
        "datasetId": source.id,
        "sourceId": source.source_id,
        "sourceRevision": source.revision,
        "sourceConfig": source.config,
        # Upstream's own name for the split, and what THIS suite does with it. They differ
        # for uner-tagalog, whose only published split is `test` and which we deliberately
        # repurpose as SFT acceptance training material. Recording both is what stops a
        # later reader treating that run as an independent Tagalog benchmark.
        "sourceSplit": source.split,
        "pipelineUsage": source.pipeline_usage,
        "tier": source.tier,
        "excludeLanguages": sorted(exclude_languages),
        "license": source.license,
        "redistribution": source.redistribution,
        "gated": source.gated,
        "languages": list(source.languages),
        "intendedUse": list(source.intended_use),
        "overlapsWith": list(source.overlaps_with),
        "profile": profile_name,
        "conversionVersion": source.conversion_version,
        "toolVersion": tool_version,
        "selectedSourceIndexCount": len(selected_indices),
        "selectedSourceIndexDigest": index_digest,
        "splits": {
            name: {
                "file": splits[name].name,
                "exampleCount": counts[name],
                "sha256": digests[name],
                "bytes": splits[name].stat().st_size,
            }
            for name in sorted(splits)
        },
    }


def write_dimer_zip(splits: dict[str, Path], destination: Path) -> str:
    """Package canonical splits into a DIMER-uploadable zip, and return its SHA-256.

    Deterministic by construction: fixed member order, fixed timestamps, no directory
    entries, no OS metadata, and stored (uncompressed) members so the digest does not depend
    on the zlib version doing the compressing.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    ordered = [s for s in CANONICAL_SPLIT_ORDER if s in splits]

    with zipfile.ZipFile(destination, "w", zipfile.ZIP_STORED) as zf:
        for split in ordered:
            info = zipfile.ZipInfo(f"{split}.jsonl", date_time=FIXED_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            # Normal file, 0644. Left unset, the mode bits vary by platform.
            info.external_attr = (0o100644 << 16)
            # Creating OS, which Python otherwise fills in from the host: 0 on Windows,
            # 3 (Unix) elsewhere. It lands in the central directory, so leaving it alone
            # makes the package digest depend on which machine built it — and an
            # acceptance package referenced by digest must not.
            info.create_system = ZIP_CREATE_SYSTEM
            zf.writestr(info, splits[split].read_bytes())

    return sha256_file(destination)
