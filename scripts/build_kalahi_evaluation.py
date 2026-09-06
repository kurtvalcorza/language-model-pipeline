#!/usr/bin/env python
"""Build an evaluation-only Kalahi profile from its pinned gated source.

This is deliberately separate from ``python -m validation_datasets build``.  That command
is the SFT training-data path and calls ``require_usable_for_training``; Kalahi must remain
impossible to enter it.  This script writes ``evaluation.jsonl`` plus provenance/fingerprint
evidence only.  It never creates a DIMER training ZIP.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from validation_datasets.approval import ApprovalError, compare_to_approval, load_approval
from validation_datasets.evaluation import (
    EVALUATION_KEY_FIELDS,
    EvaluationConversionError,
    evaluation_fingerprints,
    require_usable_for_evaluation,
    to_evaluation,
)
from validation_datasets.package import build_manifest, write_canonical_jsonl
from validation_datasets.registry import DatasetRegistry, RegistryError
from validation_datasets.subset import select

OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "validation-datasets" / "build"


def _fetch_rows(source) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional network dependency
        raise RegistryError(
            "The `datasets` package is required to fetch Kalahi: pip install datasets"
        ) from exc

    # Never print or persist the credential. An explicit HF_TOKEN wins; otherwise datasets
    # may use the account token from a prior `huggingface-cli login`.
    token: str | bool = os.environ.get("HF_TOKEN") or True
    dataset = load_dataset(
        source.source_id,
        source.config,
        split=source.split,
        revision=source.revision,
        token=token,
    )
    return [dict(row) for row in dataset]


def build(dataset_id: str, profile_name: str) -> int:
    registry = DatasetRegistry.load()
    source = registry.get(dataset_id)
    require_usable_for_evaluation(source)
    profile = source.profile(profile_name)

    rows = _fetch_rows(source)
    print(f"fetched {len(rows)} evaluation rows from {source.source_id}@{source.revision[:12]}")

    # `full` means the exact pinned source population, not merely "whatever exists today".
    # If the card/source moves from the recorded 150 rows, stop and review before creating a
    # new evaluation corpus.
    if profile.selection == "all" and profile.count is not None and len(rows) != profile.count:
        raise EvaluationConversionError(
            f"{source.id}:{profile.name} expects exactly {profile.count} rows but the "
            f"pinned source returned {len(rows)}. Re-inspect the source before proceeding."
        )

    key_fields = EVALUATION_KEY_FIELDS.get(source.id)
    if key_fields is None:
        raise EvaluationConversionError(
            f"no deterministic identity fields registered for {source.id!r}"
        )
    chosen = select(
        rows,
        count=profile.count,
        salt=f"{source.id}:{profile.name}",
        key_fields=key_fields,
        selection=profile.selection,
    )
    records = [to_evaluation(source.id, row, index=index) for index, row in chosen]

    out = OUTPUT_ROOT / source.id / profile.name
    evaluation_path = out / "evaluation.jsonl"
    digest = write_canonical_jsonl(evaluation_path, records)
    manifest = build_manifest(
        source=source,
        profile_name=profile.name,
        splits={"evaluation": evaluation_path},
        digests={"evaluation": digest},
        counts={"evaluation": len(records)},
        selected_indices=[index for index, _ in chosen],
    )
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    fingerprints = evaluation_fingerprints(records)
    (out / "fingerprints").write_text(
        "\n".join(fingerprints) + "\n", encoding="utf-8", newline="\n"
    )

    print(f"built {len(records)} held-out evaluation records -> {evaluation_path}")
    print(f"sha256 {digest}")
    print("no training ZIP was created (evaluation-only by contract)")

    try:
        approval = load_approval(source.id, profile.name)
    except ApprovalError:
        print(
            "NOTE: no committed approval exists yet. Review this gated-source build, then "
            "run `python -m validation_datasets approve kalahi --profile full --date YYYY-MM-DD`.",
            file=sys.stderr,
        )
    else:
        problems = compare_to_approval(manifest, approval)
        if problems:
            print("FAIL: build differs from the committed approval:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        print(f"OK: matches the approval recorded {approval['approvedOn']}")

    if not source.may_commit_rows:
        print(
            "NOTE: Kalahi is gated. Commit only the recipe, manifest/digests and approved "
            "fingerprints; never commit evaluation rows.",
            file=sys.stderr,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", default="kalahi")
    parser.add_argument("--profile", default="full")
    args = parser.parse_args(argv)
    try:
        return build(args.dataset, args.profile)
    except (RegistryError, EvaluationConversionError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
