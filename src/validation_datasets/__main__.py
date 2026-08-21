"""CLI for the validation dataset suite.

    python -m validation_datasets list
    python -m validation_datasets build dolly-15k --profile smoke_100
    python -m validation_datasets verify dolly-15k --profile smoke_100
    python -m validation_datasets package dolly-15k --profile smoke_100

Every command fails closed. A moved revision, a disabled or gated source, an
evaluation-only dataset requested for training, or a digest that does not match stops the
run rather than quietly producing different data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .convert import CANONICAL_KEY_FIELDS, to_canonical
from .package import build_manifest, sha256_file, write_canonical_jsonl, write_dimer_zip
from .registry import DatasetRegistry, RegistryError
from .subset import select

OUTPUT_ROOT = (
    Path(__file__).resolve().parent.parent.parent / "validation-datasets" / "build"
)


def _fetch_rows(source) -> list[dict]:
    """Load the pinned revision. Network is required and deliberately not cached here."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RegistryError(
            "The `datasets` package is required to fetch sources: pip install datasets"
        ) from exc

    dataset = load_dataset(
        source.source_id,
        source.config if source.config not in (None, "default") else None,
        split=source.split,
        revision=source.revision,
    )
    return [dict(row) for row in dataset]


def cmd_list(args) -> int:
    registry = DatasetRegistry.load()
    for dataset_id in registry.ids():
        source = registry.get(dataset_id)
        flags = []
        if not source.enabled:
            flags.append("DISABLED")
        if source.gated:
            flags.append("gated")
        if source.is_evaluation_only:
            flags.append("EVAL-ONLY")
        if source.overlaps_with:
            flags.append("overlaps:" + ",".join(source.overlaps_with))
        print(f"{dataset_id:22s} {source.license:14s} rev={source.revision[:12]} "
              f"split={source.split or '-':10s} profiles={','.join(sorted(source.profiles)) or '-'}"
              + (f"  [{' '.join(flags)}]" if flags else ""))
    return 0


def cmd_build(args) -> int:
    registry = DatasetRegistry.load()
    source = registry.get(args.dataset)
    source.require_usable_for_training()
    profile = source.profile(args.profile)

    if source.overlaps_with:
        print(
            f"WARNING: {source.id} is declared to overlap with "
            f"{', '.join(source.overlaps_with)}. Do not use them as independent "
            "acceptance tiers without a disjointness check.",
            file=sys.stderr,
        )

    rows = _fetch_rows(source)
    print(f"fetched {len(rows)} rows from {source.source_id}@{source.revision[:12]}")

    key_fields = CANONICAL_KEY_FIELDS[source.id]
    chosen = select(
        rows, count=profile.count, salt=f"{source.id}:{profile.name}",
        key_fields=key_fields, selection=profile.selection,
    )
    records = [to_canonical(source.id, row, index=index) for index, row in chosen]

    out = OUTPUT_ROOT / source.id / profile.name
    train_path = out / "train.jsonl"
    digest = write_canonical_jsonl(train_path, records)

    manifest = build_manifest(
        source=source, profile_name=profile.name,
        splits={"train": train_path}, digests={"train": digest},
        counts={"train": len(records)},
        selected_indices=[index for index, _ in chosen],
    )
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        newline="\n",
    )

    print(f"built {len(records)} examples -> {train_path}")
    print(f"sha256 {digest}")
    if not source.may_commit_rows:
        print("NOTE: this source's redistribution terms do not clearly permit committing "
              "converted rows. Commit the manifest and digest, not the data.",
              file=sys.stderr)
    return 0


def cmd_verify(args) -> int:
    registry = DatasetRegistry.load()
    source = registry.get(args.dataset)
    out = OUTPUT_ROOT / source.id / args.profile
    manifest_path = out / "manifest.json"
    if not manifest_path.is_file():
        print(f"FAIL: nothing built at {out}", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["sourceRevision"] != source.revision:
        print(
            f"FAIL: built against {manifest['sourceRevision'][:12]} but the registry now "
            f"pins {source.revision[:12]}. Rebuild deliberately; never silently accept "
            "drifted acceptance data.",
            file=sys.stderr,
        )
        return 1

    for name, record in sorted(manifest["splits"].items()):
        path = out / record["file"]
        if not path.is_file():
            print(f"FAIL: {name} missing at {path}", file=sys.stderr)
            return 1
        # The canonical writer digests exactly the bytes it writes, so a file digest and
        # the recorded content digest are the same value.
        if sha256_file(path) != record["sha256"]:
            print(f"FAIL: {name} digest mismatch — the built data is not what the "
                  "manifest describes", file=sys.stderr)
            return 1
        print(f"OK: {name} {record['exampleCount']} examples, digest matches")
    return 0


def cmd_package(args) -> int:
    registry = DatasetRegistry.load()
    source = registry.get(args.dataset)
    out = OUTPUT_ROOT / source.id / args.profile
    splits = {
        name: out / f"{name}.jsonl"
        for name in ("train", "validation", "test")
        if (out / f"{name}.jsonl").is_file()
    }
    if not splits:
        print(f"FAIL: nothing built at {out}", file=sys.stderr)
        return 1

    destination = out / f"{source.id}-{args.profile}.zip"
    digest = write_dimer_zip(splits, destination)
    print(f"packaged {', '.join(sorted(splits))} -> {destination}")
    print(f"sha256 {digest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="validation_datasets", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="show registered sources and profiles")

    for name, handler in (("build", cmd_build), ("verify", cmd_verify),
                          ("package", cmd_package)):
        p = sub.add_parser(name)
        p.add_argument("dataset")
        p.add_argument("--profile", required=True)
        p.set_defaults(func=handler)

    args = parser.parse_args(argv)
    handler = getattr(args, "func", cmd_list)

    try:
        return handler(args)
    except RegistryError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
