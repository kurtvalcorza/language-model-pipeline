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

from .approval import (
    ApprovalError,
    canonical_fingerprints,
    compare_to_approval,
    load_approval,
    load_fingerprints,
    write_approval,
)
from .convert import CANONICAL_KEY_FIELDS, to_canonical
from .package import build_manifest, sha256_file, write_canonical_jsonl, write_dimer_zip
from .registry import DatasetRegistry, RegistryError
from .subset import (
    LanguageFilterError,
    derive_prompt_prefix,
    drop_by_prompt_prefix,
    drop_languages,
    select,
)

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


def _apply_language_exclusion(registry, source, profile, rows: list[dict]) -> list[dict]:
    """Remove excluded languages, by whichever mechanism the source actually supports.

    Two mechanisms, tried in order of directness, and NO third fallback: if neither applies
    the build fails rather than producing a profile whose declared exclusion did nothing.

      1. A language column, when the source has one.
      2. The instruction template of a known single-language source. Universal NER in the
         Aya format has no language column at all — verified against the pinned revision,
         issue #12 — but every row in a given language shares a byte-identical prompt
         preamble, so a single-language corpus identifies its own rows inside the
         multilingual one.
    """
    excluded = ", ".join(sorted(profile.exclude_languages))
    before = len(rows)

    if source.language_field:
        kept = drop_languages(
            rows, field=source.language_field,
            exclude=profile.exclude_languages, dataset_id=source.id,
        )
        mechanism = f"language field {source.language_field!r}"
    elif source.language_prefix_source:
        donor = registry.get(source.language_prefix_source)
        donor_rows = _fetch_rows(donor)
        prefix = derive_prompt_prefix(
            donor_rows, field=donor.prompt_field, dataset_id=donor.id
        )
        print(f"derived a {len(prefix)}-character prompt template from {donor.id} "
              f"({len(donor_rows)} rows)")
        kept = drop_by_prompt_prefix(
            rows, field=source.prompt_field, prefix=prefix, dataset_id=source.id,
            expected_removals=profile.expected_removals,
        )
        mechanism = f"prompt template from {donor.id}"
    else:
        raise LanguageFilterError(
            f"{source.id!r} declares exclude_languages=[{excluded}] but supports no "
            "exclusion mechanism: no `language_field`, no `language_prefix_source`. "
            "Inspect the pinned source schema and record one; do not guess a field name."
        )

    print(f"excluded {excluded} via {mechanism}: {before} -> {len(kept)} rows")
    return kept


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

    # Language exclusion runs BEFORE selection, so the profile's count is a count of rows
    # that survived the filter rather than a count taken before it.
    if profile.excludes_languages:
        rows = _apply_language_exclusion(registry, source, profile, rows)

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
        exclude_languages=profile.exclude_languages,
    )
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        newline="\n",
    )
    (out / "fingerprints").write_text(
        "\n".join(canonical_fingerprints(records)) + "\n", encoding="utf-8", newline="\n"
    )

    print(f"built {len(records)} examples -> {train_path}")
    print(f"sha256 {digest}")

    # Compare against the committed expectation immediately when there is one, so a drifted
    # rebuild is reported at the moment it is produced rather than at some later verify.
    try:
        problems = compare_to_approval(manifest, load_approval(source.id, profile.name))
    except ApprovalError:
        print("NOTE: no committed approval for this profile yet. Review the digests above, "
              "then run `approve` to record them.", file=sys.stderr)
    else:
        if problems:
            print("FAIL: this build does not match the committed approval:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        print("OK: matches the committed approval")
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

    # The check that makes this more than self-consistency: compare the build against the
    # COMMITTED approval, which a fresh clone has and the git-ignored build directory does
    # not. Without this, `verify` only proved a run agreed with itself.
    try:
        approval = load_approval(source.id, args.profile)
    except ApprovalError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    problems = compare_to_approval(manifest, approval)
    if problems:
        print("FAIL: the build does not match the committed approval:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"OK: matches the approval recorded {approval['approvedOn']}")

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


def cmd_approve(args) -> int:
    """Record a reviewed build as the expectation every later build is checked against.

    Deliberately a separate, explicit command. If `build` wrote the approval itself, the
    expectation would follow the code around instead of pinning it, and a converter change
    would re-approve its own output.
    """
    registry = DatasetRegistry.load()
    source = registry.get(args.dataset)
    out = OUTPUT_ROOT / source.id / args.profile
    manifest_path = out / "manifest.json"
    fingerprints_file = out / "fingerprints"
    if not manifest_path.is_file() or not fingerprints_file.is_file():
        print(f"FAIL: nothing built at {out}. Build first, then approve.", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["sourceRevision"] != source.revision:
        print("FAIL: refusing to approve a build made against a revision the registry no "
              "longer pins.", file=sys.stderr)
        return 1

    package = out / f"{source.id}-{args.profile}.zip"
    fingerprints = [
        line.strip()
        for line in fingerprints_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    path = write_approval(
        manifest=manifest, fingerprints=fingerprints,
        package_sha256=sha256_file(package) if package.is_file() else None,
        approved_on=args.date,
    )
    print(f"approved {source.id}:{args.profile} -> {path}")
    print("Commit this file. It is the expectation a fresh clone verifies against.")
    return 0


def cmd_disjoint(args) -> int:
    """Prove two approved profiles share no canonical example.

    Tier 2 (Filipino) and Tier 3 (multilingual excluding `tl`) are independent acceptance
    runs by construction — Tier 3 filters `tl` out. This is the continuous proof of that,
    so a later source bump or converter change cannot quietly reintroduce the overlap the
    exclusion was added to remove.

    Runs entirely on committed fingerprint files: no network, no build, so CI can enforce it.
    """
    pairs = []
    for spec in args.profiles:
        if ":" not in spec:
            print(f"FAIL: expected <dataset>:<profile>, got {spec!r}", file=sys.stderr)
            return 1
        pairs.append(tuple(spec.split(":", 1)))

    sets = {}
    for dataset_id, profile in pairs:
        try:
            sets[f"{dataset_id}:{profile}"] = load_fingerprints(dataset_id, profile)
        except ApprovalError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1

    failed = False
    names = sorted(sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            shared = sets[left] & sets[right]
            if shared:
                failed = True
                print(
                    f"FAIL: {left} and {right} share {len(shared)} canonical examples "
                    f"(of {len(sets[left])} and {len(sets[right])}). They are not "
                    "independent acceptance tiers.",
                    file=sys.stderr,
                )
            else:
                print(f"OK: {left} ({len(sets[left])}) and {right} "
                      f"({len(sets[right])}) are disjoint")
    return 1 if failed else 0


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
                          ("package", cmd_package), ("approve", cmd_approve)):
        p = sub.add_parser(name)
        p.add_argument("dataset")
        p.add_argument("--profile", required=True)
        if name == "approve":
            p.add_argument(
                "--date", required=True,
                help="approval date, YYYY-MM-DD (explicit so the record is not "
                     "clock-dependent and a rebuild is reproducible)",
            )
        p.set_defaults(func=handler)

    disjoint = sub.add_parser(
        "disjoint", help="prove approved profiles share no canonical example"
    )
    disjoint.add_argument("profiles", nargs="+", metavar="dataset:profile")
    disjoint.set_defaults(func=cmd_disjoint)

    args = parser.parse_args(argv)
    handler = getattr(args, "func", cmd_list)

    try:
        return handler(args)
    except (RegistryError, ApprovalError, LanguageFilterError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
