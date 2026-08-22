"""Committed expectations for approved profiles.

The reproducibility problem this solves: `build/` is git-ignored, so a manifest written
there proves only that a run agreed with itself. A fresh clone had nothing to compare a
rebuild against, which means "the same pinned source and converter still produce the
dataset we approved" was not actually checkable — and that is the whole claim an acceptance
dataset makes.

So every approved profile commits two small files, neither of which contains dataset rows:

    approved/<dataset>/<profile>.json           digests, counts, and the pinned revision
    approved/<dataset>/<profile>.fingerprints   sorted canonical fingerprints, truncated

`verify` compares a build against the committed .json. `disjoint` compares two profiles'
.fingerprints files, which is what makes the Tier 2 / Tier 3 independence gate runnable
offline and therefore runnable in CI.

Fingerprints are truncated to 64 bits. A collision is ~1e-13 at this suite's scale and, if
one ever happened, it would report an overlap that is not there — the gate fails safe.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from lmpipeline.datasets.normalize import Example

APPROVED_ROOT = (
    Path(__file__).resolve().parent.parent.parent / "validation-datasets" / "approved"
)

FINGERPRINT_CHARS = 16


class ApprovalError(RuntimeError):
    """A build that does not match its committed expectation. Never a warning."""


def canonical_fingerprints(records: list[dict[str, Any]]) -> list[str]:
    """Sorted, truncated fingerprints of canonical records.

    Uses `lmpipeline`'s own Example.fingerprint rather than a second implementation: the
    disjointness gate is only meaningful if it hashes content the same way the validator's
    leakage detection does.
    """
    seen = {
        Example(messages=tuple(record["messages"]), line_number=0)
        .fingerprint()[:FINGERPRINT_CHARS]
        for record in records
    }
    return sorted(seen)


def fingerprint_digest(fingerprints: list[str]) -> str:
    """Order-independent digest of a fingerprint set."""
    return hashlib.sha256("\n".join(sorted(fingerprints)).encode("utf-8")).hexdigest()


def approval_path(dataset_id: str, profile: str) -> Path:
    return APPROVED_ROOT / dataset_id / f"{profile}.json"


def fingerprints_path(dataset_id: str, profile: str) -> Path:
    return APPROVED_ROOT / dataset_id / f"{profile}.fingerprints"


def load_approval(dataset_id: str, profile: str) -> dict[str, Any]:
    path = approval_path(dataset_id, profile)
    if not path.is_file():
        raise ApprovalError(
            f"{dataset_id}:{profile} has no committed approval at "
            f"{path.relative_to(APPROVED_ROOT.parent.parent)}. Build it, review the "
            "digests, then run `approve` to record them."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_fingerprints(dataset_id: str, profile: str) -> set[str]:
    path = fingerprints_path(dataset_id, profile)
    if not path.is_file():
        raise ApprovalError(f"{dataset_id}:{profile} has no committed fingerprints.")
    return {
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def write_approval(
    *, manifest: dict[str, Any], fingerprints: list[str], package_sha256: str | None,
    approved_on: str,
) -> Path:
    """Record a reviewed build as the expectation every later build is checked against."""
    dataset_id = manifest["datasetId"]
    profile = manifest["profile"]
    path = approval_path(dataset_id, profile)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schemaVersion": "1.0",
        "datasetId": dataset_id,
        "profile": profile,
        "approvedOn": approved_on,
        "sourceId": manifest["sourceId"],
        "sourceRevision": manifest["sourceRevision"],
        "sourceSplit": manifest["sourceSplit"],
        "pipelineUsage": manifest["pipelineUsage"],
        "conversionVersion": manifest["conversionVersion"],
        "toolVersion": manifest["toolVersion"],
        "excludeLanguages": manifest.get("excludeLanguages", []),
        "selectedSourceIndexCount": manifest["selectedSourceIndexCount"],
        "selectedSourceIndexDigest": manifest["selectedSourceIndexDigest"],
        "splits": {
            name: {"exampleCount": record["exampleCount"], "sha256": record["sha256"]}
            for name, record in sorted(manifest["splits"].items())
        },
        "packageSha256": package_sha256,
        "fingerprintCount": len(fingerprints),
        "fingerprintDigest": fingerprint_digest(fingerprints),
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    fingerprints_path(dataset_id, profile).write_text(
        "\n".join(sorted(fingerprints)) + "\n", encoding="utf-8", newline="\n"
    )
    return path


# Fields whose disagreement means the rebuild is not the approved dataset. Ordered so the
# most explanatory difference is reported first: a moved revision explains every digest
# difference downstream of it, and reporting the digest alone would send a reader hunting.
_COMPARED = (
    ("sourceRevision", "pinned source revision"),
    ("conversionVersion", "converter version"),
    ("excludeLanguages", "language exclusions"),
    ("selectedSourceIndexDigest", "set of selected source rows"),
    ("selectedSourceIndexCount", "number of selected source rows"),
)


def compare_to_approval(manifest: dict[str, Any], approval: dict[str, Any]) -> list[str]:
    """Differences between a fresh build and its committed expectation."""
    problems: list[str] = []
    for key, label in _COMPARED:
        expected = approval.get(key)
        actual = manifest.get(key)
        if isinstance(expected, list) or isinstance(actual, list):
            expected, actual = list(expected or []), list(actual or [])
        if expected != actual:
            problems.append(f"{label}: approved {expected!r}, built {actual!r}")

    for name in sorted(set(approval["splits"]) | set(manifest["splits"])):
        expected = approval["splits"].get(name)
        actual = manifest["splits"].get(name)
        if expected is None:
            problems.append(f"split {name!r} was built but is not in the approval")
            continue
        if actual is None:
            problems.append(f"split {name!r} is approved but was not built")
            continue
        if expected["sha256"] != actual["sha256"]:
            problems.append(
                f"split {name!r} digest: approved {expected['sha256'][:16]}..., "
                f"built {actual['sha256'][:16]}..."
            )
        if expected["exampleCount"] != actual["exampleCount"]:
            problems.append(
                f"split {name!r} example count: approved {expected['exampleCount']}, "
                f"built {actual['exampleCount']}"
            )
    return problems
