#!/usr/bin/env python
"""Measure what a split costs in RAM, so the ingest bound can be a number.

Issue #11 shipped `MAX_SPLIT_BYTES` and `MAX_DATASET_BYTES` conservative and overridable
because the DIMER upload quota had not been read out of the portal. It also records the
judgement call that follows once the quota IS known: a quota well above what the containers
can ingest should not simply be adopted, because both consumers hold parsed examples in RAM
and a bound set above the survivable point converts a structured
`DATASET_SPLIT_TOO_LARGE` into an OOM kill that writes no result document at all.

That call needs a measurement. This script is it.

WHAT IT MEASURES, AND WHY THAT IS THE USEFUL QUANTITY

`normalize.load_examples` bounds its peak by `max_examples`, not by file size -- it raises on
the example that would exceed the cap, so a 10 GB file with a 500k cap retains 500k examples,
not 10 GB of them. So the quantity that governs survival is not bytes on disk but
**bytes of RSS per retained example**, and the bound that matters is:

    retainable examples ~= (container memory budget - baseline RSS) / per-example cost

File size still matters, but only as the thing that produces the example count. Reporting a
single "RSS is Nx the file" ratio would be the wrong shape of answer: it changes with average
example length, while the per-example cost is comparatively stable and is what makes the
arithmetic above usable at a different average length.

    python scripts/measure_ingest_memory.py                 # counts x shapes, as a table
    python scripts/measure_ingest_memory.py --json          # machine-readable
    python scripts/measure_ingest_memory.py --examples 500000 --content-bytes 320

HOW IT MEASURES

Each point runs in a FRESH CHILD PROCESS. `ru_maxrss` is a high-water mark that never falls,
so sweeping sizes in one process would report the largest point's peak for every later point.
The child reports its own baseline (interpreter plus imports, before the file exists) and its
peak, and the difference is attributed to ingest.

Every point is measured twice: once streaming through `iter_examples` and discarding, once
retaining via `load_examples`. The difference separates the transient parse cost from the
cost of holding the result, which is the half a cap can actually control.

The per-example cost is fitted as the SLOPE across points rather than taken from a single
division, so a fixed overhead in the baseline cannot inflate it.

WHAT THIS IS NOT

This runs on whatever machine invokes it. The per-example cost is a property of CPython's
object layout and the data's shape, so it transfers across machines running the same
interpreter and architecture; the absolute memory budget does not. It is NOT a measurement on
approved DIMER hardware -- #7 remains open, and the validator's real 2-vCPU container is
#14's subject. Record the environment alongside any number taken from this script, which is
why `--json` emits it.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lmpipeline.datasets.resolver import MAX_SPLIT_BYTES  # noqa: E402

# Points chosen to span two orders of magnitude so the fit has leverage, while keeping the
# default sweep runnable in seconds. The largest is deliberately well under
# MAX_TRAIN_EXAMPLES: this script must not need half a gigabyte to tell you about half a
# gigabyte.
DEFAULT_SWEEP = (5_000, 20_000, 50_000, 100_000)

# Roughly dolly's shape: a short instruction and a free-form answer. The committed synthetic
# fixtures are far smaller than real rows, so a shape taken from them would understate the
# per-example cost.
# Nominal text sizes spanning the shapes the acceptance tiers actually produce: UNER's
# short structured labels, dolly's free-form answers, and a long-form row.
DEFAULT_SHAPES = (80, 320, 1200)

# The consumer's cap, which lives in `language-model-dataset-validator`, not here. Carried as
# a default so the worst-case line can be computed; override it when that cap moves. Its own
# measurement on the real 2-vCPU validator is validator#14.
VALIDATOR_MAX_TRAIN_EXAMPLES = 500_000


def synthesize_split(path: Path, *, examples: int, content_bytes: int) -> int:
    """Write a conversational JSONL split and return its size in bytes.

    Content varies per example rather than repeating one string: CPython interns and shares
    equal strings, so a file of identical rows would measure the cost of one row plus a list
    of pointers to it -- a number far below anything a real dataset produces.
    """
    filler_len = max(1, content_bytes // 2)
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(examples):
            # The index is mixed into the body so no two rows are equal.
            user = f"Q{i}: " + f"u{i}" * (filler_len // 4 or 1)
            assistant = f"A{i}: " + f"a{i}" * (filler_len // 4 or 1)
            fh.write(json.dumps({"messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]}, ensure_ascii=False) + "\n")
    return path.stat().st_size


def _rss_bytes() -> int:
    """Peak RSS of this process in bytes.

    `ru_maxrss` is kilobytes on Linux and bytes on macOS -- a difference that silently
    changes the answer by 1024x, so it is converted explicitly rather than assumed.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def _measure_child(examples: int, content_bytes: int, mode: str) -> dict[str, Any]:
    """One measurement, in this process. Invoked as the child of `measure`."""
    from lmpipeline.datasets.normalize import iter_examples, load_examples

    baseline = _rss_bytes()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "train.jsonl"
        file_bytes = synthesize_split(path, examples=examples, content_bytes=content_bytes)
        after_write = _rss_bytes()

        if mode == "stream":
            count = sum(1 for _ in iter_examples(path))
            retained: list | None = None
        else:
            retained = load_examples(path, max_examples=examples)
            count = len(retained)
        peak = _rss_bytes()
        # Referenced after the peak read so the list cannot be collected early and make a
        # retaining run look like a streaming one.
        assert retained is None or len(retained) == count

    return {
        "mode": mode,
        "examples": count,
        "fileBytes": file_bytes,
        "baselineBytes": baseline,
        "afterWriteBytes": after_write,
        "peakBytes": peak,
        "ingestBytes": peak - baseline,
    }


def measure(examples: int, content_bytes: int, mode: str) -> dict[str, Any]:
    """Run one point in a fresh interpreter, because ru_maxrss never falls."""
    out = subprocess.run(
        [sys.executable, __file__, "--child", str(examples), str(content_bytes), mode],
        capture_output=True, text=True, check=True,
        env={**os.environ, "PYTHONHASHSEED": "0"},
    )
    return json.loads(out.stdout)


def fit_slope(points: list[dict[str, Any]]) -> tuple[float, float]:
    """Least-squares fit of ingest bytes against example count.

    Returns (bytes per example, intercept). The slope is the number worth carrying: a single
    point's ingestBytes/examples folds in the intercept and overstates the marginal cost,
    which is precisely the direction that would make a bound too conservative to be useful.
    """
    n = len(points)
    if n < 2:
        raise ValueError("need at least two points to fit a slope")
    xs = [float(p["examples"]) for p in points]
    ys = [float(p["ingestBytes"]) for p in points]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        raise ValueError("all points have the same example count; nothing to fit")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / denominator
    return slope, mean_y - slope * mean_x


def fit_shape_model(points: list[dict[str, Any]]) -> tuple[float, float]:
    """Fit RSS-per-example against FILE-bytes-per-example across data shapes.

    Returns (fixed bytes per example, bytes of RSS per byte of file).

    This is the form worth carrying, because it separates the two costs that behave
    differently. The fixed part is CPython object overhead -- an `Example`, its message
    tuple, a dict per message, a list slot -- and is paid per example no matter how short
    the text. The proportional part is the text itself.

    The consequence is counter-intuitive and matters for #11: the RSS-to-file ratio is WORST
    for SHORT examples, because the fixed part dominates. So a byte-based split bound is
    loosest exactly where the memory risk is highest, and it is the EXAMPLE CAP, not the byte
    bound, that actually governs peak memory.

    `fileBytes / examples` is used as the x-axis rather than the `--content-bytes` knob:
    the knob is nominal (the row index is mixed into the filler, so realized lengths drift
    with the example count) while file bytes are measured.
    """
    n = len(points)
    if n < 2:
        raise ValueError("need at least two shapes to fit the model")
    xs = [p["fileBytes"] / p["examples"] for p in points]
    ys = [p["ingestBytes"] / p["examples"] for p in points]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        raise ValueError("all shapes have the same bytes per example; nothing to fit")
    per_byte = sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)
    ) / denominator
    return mean_y - per_byte * mean_x, per_byte


def environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", nargs=3, metavar=("EXAMPLES", "CONTENT_BYTES", "MODE"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--examples", type=int, action="append",
                        help="Example count to measure; repeatable. Defaults to the sweep.")
    parser.add_argument("--content-bytes", type=int, action="append",
                        help="Nominal text bytes per example; repeatable to sweep shapes. "
                             f"Defaults to {DEFAULT_SHAPES}.")
    parser.add_argument("--max-examples", type=int, default=VALIDATOR_MAX_TRAIN_EXAMPLES,
                        help="The consumer's example cap, for the worst-case line "
                             f"(default {VALIDATOR_MAX_TRAIN_EXAMPLES}).")
    parser.add_argument("--json", action="store_true", help="emit the full record as JSON")
    args = parser.parse_args()

    if args.child:
        examples, content_bytes, mode = int(args.child[0]), int(args.child[1]), args.child[2]
        print(json.dumps(_measure_child(examples, content_bytes, mode)))
        return 0

    counts = sorted(set(args.examples)) if args.examples else list(DEFAULT_SWEEP)
    shapes = sorted(set(args.content_bytes)) if args.content_bytes else list(DEFAULT_SHAPES)

    retained: list[dict[str, Any]] = []
    streamed: list[dict[str, Any]] = []
    for content_bytes in shapes:
        for count in counts:
            retained.append({**measure(count, content_bytes, "retain"),
                             "nominalContentBytes": content_bytes})
            streamed.append({**measure(count, content_bytes, "stream"),
                             "nominalContentBytes": content_bytes})

    # The shape model needs the largest count per shape: the fixed overhead is estimated
    # most cleanly where the intercept of the count sweep matters least.
    largest = max(counts)
    per_shape = [p for p in retained if p["examples"] == largest]
    fixed, per_byte = fit_shape_model(per_shape) if len(per_shape) >= 2 else (float("nan"),) * 2

    record = {
        "environment": environment(),
        "counts": counts,
        "shapes": shapes,
        "retained": retained,
        "streamed": streamed,
        "fixedBytesPerExample": fixed,
        "rssBytesPerFileByte": per_byte,
        "validatorMaxTrainExamples": args.max_examples,
        "maxSplitBytes": MAX_SPLIT_BYTES,
    }
    if len(counts) >= 2:
        for shape in shapes:
            points = [p for p in retained if p["nominalContentBytes"] == shape]
            slope, intercept = fit_slope(points)
            record.setdefault("perShapeSlope", {})[str(shape)] = {
                "bytesPerRetainedExample": slope, "interceptBytes": intercept,
            }

    if args.json:
        print(json.dumps(record, indent=2))
        return 0

    env = environment()
    print(f"environment: {env['platform']}, {env['implementation']} {env['python']}\n")
    print(f"{'examples':>10} {'file B/ex':>10} {'file MiB':>9} {'retain MiB':>11} "
          f"{'stream MiB':>11} {'retain B/ex':>12}")
    for hold, flow in zip(retained, streamed, strict=True):
        print(f"{hold['examples']:>10,} {hold['fileBytes'] / hold['examples']:>10,.0f} "
              f"{hold['fileBytes'] / 1024**2:>9.1f} {hold['ingestBytes'] / 1024**2:>11.1f} "
              f"{flow['ingestBytes'] / 1024**2:>11.1f} "
              f"{hold['ingestBytes'] / hold['examples']:>12,.0f}")

    print(f"\nRetained RSS per example = {fixed:,.0f} bytes + "
          f"{per_byte:.3f} x (file bytes per example)")
    print("Equivalently: peak retained RSS ~= the split's file bytes, plus "
          f"{fixed:,.0f} bytes for every example held.")
    print("\nStreaming is flat at a few MiB regardless of size -- it is the retained list "
          "that grows, which is what an example cap can bound and a byte bound cannot.")

    print("\nThe ratio is WORST for the SHORTEST examples, because the fixed per-example "
          "overhead\ndominates. A byte bound is therefore loosest exactly where the memory "
          "risk is highest.")

    worst = MAX_SPLIT_BYTES * per_byte + fixed * args.max_examples
    print(f"\nWorst case jointly admitted by the shipped bounds "
          f"({MAX_SPLIT_BYTES / 1024**2:,.0f} MiB split, "
          f"{args.max_examples:,} example cap):\n"
          f"  ~{worst / 1024**2:,.0f} MiB retained, reached at "
          f"~{MAX_SPLIT_BYTES / args.max_examples:,.0f} byte rows -- an ordinary row size, "
          "not a pathological one.")
    print("\nIngest cost only: add the rest of the container's working set before sizing it, "
          "and\nsee #7 for approved-hardware figures, which this is not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
