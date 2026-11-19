"""The Section 8 comparison spans must cover every recorded training run (#95, follow-up to #92 / LM92-R5).

tests/test_review_fixes_2026_10_02.py pins the quoted spans as literal strings, which cannot notice a new run that falls
outside them: that is how four runs drifted out before #92. This test derives the spans from the evidence instead. It
parses the METRICS block of every ``exec.log`` under ``docs/verification/``, takes the min and max of each metric, and
checks that the spans quoted in ``tools/notebook_template.py`` contain them at the precision the prose uses.

Only runs that print the METRICS block count. The refusal runs stop before training and the retired artifact-inference
companion never trains, so neither prints one; any other log without a METRICS block is a parse failure, not an
exclusion.
"""
# ruff: noqa: E501

from __future__ import annotations

import importlib.util
import json
import re
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFICATION = ROOT / "docs" / "verification"

# The keys that identify the METRICS dict the training cell prints (tools/notebook_template.py, Section 8).
METRICS_KEYS = {"trainLoss", "validationLoss", "validationPerplexity", "wallSeconds", "peakGpuMemoryGiB"}
# Directory-name markers of the runs that legitimately print no METRICS block.
NON_TRAINING_MARKERS = ("refusal", "artifact_inference")

_NUM = r"(\d+\.\d+)"
SPAN_SENTENCE = re.compile(
    rf"train loss {_NUM}–{_NUM}, validation loss {_NUM}–{_NUM}, perplexity {_NUM}–{_NUM}, "
    rf"{_NUM}–{_NUM} s of training \(`wallSeconds`\) and {_NUM} GiB peak"
)
# The METRICS key each span of the sentence quotes, in capture-group order; the last group is the single peak figure.
SPANNED = ("trainLoss", "validationLoss", "validationPerplexity", "wallSeconds")


def _template_strings() -> list[str]:
    spec = importlib.util.spec_from_file_location("notebook_template", ROOT / "tools" / "notebook_template.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    found: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, str):
            found.append(node)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)

    walk(module.TEMPLATE)
    return found


def _quoted_spans() -> dict[str, tuple[str, str]]:
    matches = [m for text in _template_strings() for m in SPAN_SENTENCE.finditer(text)]
    assert len(matches) == 1, f"expected the Section 8 comparison sentence once in the template, found {len(matches)}"
    groups = matches[0].groups()
    spans = {key: (groups[2 * i], groups[2 * i + 1]) for i, key in enumerate(SPANNED)}
    spans["peakGpuMemoryGiB"] = (groups[-1], groups[-1])
    return spans


def _metrics_blocks(log: str) -> list[dict]:
    """Every top-level JSON object in the log that carries all of METRICS_KEYS."""
    decoder = json.JSONDecoder()
    blocks = []
    offset = 0
    for line in log.splitlines(keepends=True):
        if line.rstrip("\r\n") == "{":
            try:
                obj, _ = decoder.raw_decode(log, offset)
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, dict) and METRICS_KEYS <= obj.keys():
                blocks.append(obj)
        offset += len(line)
    return blocks


def _recorded_runs() -> tuple[dict[str, dict], list[str]]:
    counted: dict[str, dict] = {}
    excluded: list[str] = []
    for path in sorted(VERIFICATION.rglob("exec.log")):
        run = path.parent.relative_to(VERIFICATION).as_posix()
        blocks = _metrics_blocks(path.read_text(encoding="utf-8"))
        assert len(blocks) <= 1, f"{run}: {len(blocks)} METRICS blocks; expected at most one"
        if blocks:
            counted[run] = blocks[0]
        else:
            excluded.append(run)
    return counted, excluded


def _at(value: float, precision_of: str) -> Decimal:
    places = len(precision_of.split(".")[1])
    return Decimal(f"{value:.{places}f}")


def test_only_refusal_and_artifact_inference_runs_lack_a_metrics_block():
    counted, excluded = _recorded_runs()
    # Fifteen at #92; the floor catches a parser that silently stops reading the logs.
    assert len(counted) >= 15, sorted(counted)
    unexplained = [run for run in excluded if not any(marker in run for marker in NON_TRAINING_MARKERS)]
    assert not unexplained, f"exec.log without a METRICS block in a training run: {unexplained}"


def test_section_8_quoted_spans_contain_every_recorded_run():
    counted, _ = _recorded_runs()
    quoted = _quoted_spans()
    outside = []
    for key, (lo, hi) in quoted.items():
        values = {run: metrics[key] for run, metrics in counted.items()}
        low_run = min(values, key=values.get)
        high_run = max(values, key=values.get)
        computed_lo, computed_hi = _at(values[low_run], lo), _at(values[high_run], hi)
        if computed_lo < Decimal(lo):
            outside.append(f"{key}: quoted {lo}–{hi}, but {low_run} printed {values[low_run]} ({computed_lo})")
        if computed_hi > Decimal(hi):
            outside.append(f"{key}: quoted {lo}–{hi}, but {high_run} printed {values[high_run]} ({computed_hi})")
    assert not outside, "Section 8 comparison spans no longer cover every recorded run:\n" + "\n".join(outside)
