"""The ingest-memory measurement, tested for the parts that are not machine-dependent.

A measurement script needs its own gate, because the failure mode is not a crash: it is a
number that looks plausible and is wrong. So the fits are checked against inputs whose answer
is known exactly, the synthetic corpus is checked for the property the measurement depends on
(distinct rows), and one end-to-end run confirms the retaining path really does cost more than
the streaming one.

Deliberately NOT asserted: any particular byte figure. Those are properties of the machine and
the interpreter, and a test that pinned them would fail on the next runner rather than on the
next bug. The recorded numbers live in `COMPATIBILITY.md` with the environment they were taken
in, which is where a measurement belongs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from lmpipeline.datasets.normalize import iter_examples

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "measure_ingest_memory", ROOT / "scripts" / "measure_ingest_memory.py"
)
measure_ingest_memory = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(measure_ingest_memory)


# -- the fits, against answers known in advance --------------------------------


def test_the_count_slope_recovers_a_known_line():
    """y = 1000x + 5_000_000, sampled at three points, must come back exactly."""
    points = [
        {"examples": n, "ingestBytes": 1000 * n + 5_000_000}
        for n in (10_000, 50_000, 100_000)
    ]
    slope, intercept = measure_ingest_memory.fit_slope(points)
    assert slope == pytest.approx(1000)
    assert intercept == pytest.approx(5_000_000)


def test_the_slope_is_not_just_the_last_point_divided():
    """The distinction the fit exists to make.

    At 100k examples the naive ingestBytes/examples reads 1050, not 1000, because the fixed
    5 MiB is folded in. Taking that as the marginal cost overstates it by 5% -- and always in
    the direction that makes a derived bound needlessly tight.
    """
    points = [
        {"examples": n, "ingestBytes": 1000 * n + 5_000_000}
        for n in (10_000, 100_000)
    ]
    slope, _ = measure_ingest_memory.fit_slope(points)
    naive = points[-1]["ingestBytes"] / points[-1]["examples"]
    assert naive == pytest.approx(1050)
    assert slope == pytest.approx(1000)
    assert slope < naive


def test_the_shape_model_separates_fixed_overhead_from_text():
    """RSS/ex = 800 + 1.0 x file bytes/ex, recovered from two shapes."""
    points = [
        {"examples": 1000, "fileBytes": 200 * 1000, "ingestBytes": (800 + 200) * 1000},
        {"examples": 1000, "fileBytes": 2000 * 1000, "ingestBytes": (800 + 2000) * 1000},
    ]
    fixed, per_byte = measure_ingest_memory.fit_shape_model(points)
    assert fixed == pytest.approx(800)
    assert per_byte == pytest.approx(1.0)


def test_the_shape_model_shows_the_ratio_is_worst_for_short_rows():
    """The finding that matters for #11, expressed as arithmetic rather than prose.

    With a fixed per-example cost, a split of short rows costs more RSS per byte of file than
    a split of long ones -- so a byte bound is loosest where the risk is highest.
    """
    fixed, per_byte = 800.0, 1.0
    short_ratio = (fixed + per_byte * 200) / 200
    long_ratio = (fixed + per_byte * 2000) / 2000
    assert short_ratio > long_ratio
    assert short_ratio == pytest.approx(5.0)
    assert long_ratio == pytest.approx(1.4)


@pytest.mark.parametrize("points", [
    [],
    [{"examples": 100, "ingestBytes": 1, "fileBytes": 1}],
    [{"examples": 100, "ingestBytes": 1, "fileBytes": 100},
     {"examples": 100, "ingestBytes": 2, "fileBytes": 100}],
])
def test_a_degenerate_fit_fails_rather_than_returning_a_number(points):
    """One point, or points with no spread, cannot support a slope. Returning something
    anyway is how an unfalsifiable figure ends up in a document."""
    with pytest.raises(ValueError):
        measure_ingest_memory.fit_slope(points)
    with pytest.raises(ValueError):
        measure_ingest_memory.fit_shape_model(points)


# -- the synthetic corpus ------------------------------------------------------


def test_the_synthetic_split_is_real_input_to_the_real_reader(tmp_path):
    path = tmp_path / "train.jsonl"
    size = measure_ingest_memory.synthesize_split(path, examples=50, content_bytes=120)
    assert size == path.stat().st_size
    examples = list(iter_examples(path))
    assert len(examples) == 50
    assert all(len(e.messages) == 2 for e in examples)


def test_no_two_synthetic_rows_are_equal(tmp_path):
    """The property the whole measurement rests on.

    CPython shares equal strings, so a corpus of identical rows would measure one row plus a
    list of pointers to it -- a per-example cost far below anything real data produces, and
    the resulting bound would be dangerously generous.
    """
    path = tmp_path / "train.jsonl"
    measure_ingest_memory.synthesize_split(path, examples=200, content_bytes=120)
    fingerprints = {e.fingerprint() for e in iter_examples(path)}
    assert len(fingerprints) == 200


def test_a_longer_shape_produces_a_bigger_file(tmp_path):
    small = measure_ingest_memory.synthesize_split(
        tmp_path / "a.jsonl", examples=100, content_bytes=80)
    large = measure_ingest_memory.synthesize_split(
        tmp_path / "b.jsonl", examples=100, content_bytes=1200)
    assert large > small * 3


# -- one end-to-end measurement ------------------------------------------------


def test_retaining_costs_more_than_streaming():
    """The claim the script is for, checked as an inequality rather than a figure.

    `load_examples` holds the parsed result; `iter_examples` does not. If this ever came back
    equal, the measurement would be reporting the wrong thing -- most likely because the
    retained list was collected before the peak was read.
    """
    retain = measure_ingest_memory.measure(30_000, 320, "retain")
    stream = measure_ingest_memory.measure(30_000, 320, "stream")
    assert retain["examples"] == stream["examples"] == 30_000
    assert retain["ingestBytes"] > stream["ingestBytes"]
    assert retain["peakBytes"] >= retain["baselineBytes"]


def test_each_point_is_measured_in_a_fresh_process():
    """`ru_maxrss` is a high-water mark that never falls, so measuring a small point after a
    large one in the same process would report the large one's peak. Two runs of the same
    size, taken after a bigger one, must not inherit its peak."""
    big = measure_ingest_memory.measure(60_000, 1200, "retain")
    small = measure_ingest_memory.measure(5_000, 80, "retain")
    assert small["ingestBytes"] < big["ingestBytes"]
