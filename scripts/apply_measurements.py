#!/usr/bin/env python
"""Apply a measurement run's output to model-registry.yaml.

Hand-transcribing measured numbers into the registry is exactly where a digit gets dropped
and an authoritative-looking `min_vram_gb` ends up wrong. This applies them mechanically.

    python scripts/apply_measurements.py matrix.json --hardware "RTX 5070 Ti Laptop, 11.94GB"

Rules:
  * Only `status: ok` rows contribute. An OOM row records that the combination does not fit
    on this hardware; it does not produce a number.
  * A model measured under several methods takes the **highest** requirement, because the
    registry's `min_vram_gb` gates every method the entry allows.
  * `measured_on` is required. A profile without provenance is indistinguishable from a
    guess, which is the thing this whole mechanism exists to prevent.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REGISTRY = (
    Path(__file__).resolve().parent.parent
    / "src" / "lmpipeline" / "data" / "model-registry.yaml"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("--hardware", required=True,
                        help="Description recorded as measured_on, e.g. 'RTX 5070 Ti, 11.94GB'")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records = json.loads(args.measurements.read_text(encoding="utf-8"))

    required: dict[str, float] = {}
    skipped: list[str] = []
    for record in records:
        key = record["modelKey"]
        if record.get("status") != "ok" or not record.get("recommendedMinVramGb"):
            skipped.append(f"{key}/{record['method']} ({record.get('code', record.get('status'))})")
            continue
        # Highest across methods: min_vram_gb gates every method the entry permits.
        required[key] = max(required.get(key, 0.0), float(record["recommendedMinVramGb"]))

    text = REGISTRY.read_text(encoding="utf-8")
    updated = 0

    for key, vram in sorted(required.items()):
        # Match the resource_profile block belonging to this model key only.
        pattern = re.compile(
            rf"(^  {re.escape(key)}:\n(?:.*\n)*?    resource_profile:\n)"
            rf"      min_vram_gb:.*\n      measured_on:.*\n",
            re.MULTILINE,
        )
        replacement = (
            rf"\g<1>      min_vram_gb: {vram}\n"
            f"      measured_on: {args.hardware}\n"
        )
        text, count = pattern.subn(replacement, text)
        if count:
            updated += 1
            print(f"  {key}: min_vram_gb = {vram}")
        else:
            print(f"  {key}: NO MATCH in registry (unchanged)", file=sys.stderr)

    if skipped:
        print("\nnot measured (no number recorded):", file=sys.stderr)
        for item in skipped:
            print(f"  {item}", file=sys.stderr)

    if args.dry_run:
        print("\n(dry run; registry not written)")
        return 0

    REGISTRY.write_text(text, encoding="utf-8")
    print(f"\nupdated {updated} entr{'y' if updated == 1 else 'ies'} in {REGISTRY.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
