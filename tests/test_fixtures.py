"""Every adversarial fixture must produce its declared error code.

Issue #2 section 9 requires machine-readable assertions, not message substrings. The
expected code is the contract: `adversarial/expected-codes.json` is generated alongside the
fixtures, so a fixture cannot drift from what it claims to test.

Codes owned by the validator rather than the shared package — leakage, token budget,
sequence length, example counts — are covered in `language-model-dataset-validator`, which
is where those checks live. This file covers everything `lmpipeline` alone can produce.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from lmpipeline.datasets.normalize import iter_examples
from lmpipeline.datasets.resolver import resolve_dataset
from lmpipeline.errors import Code, PipelineError

FIXTURES = Path(__file__).resolve().parent.parent / "validation-datasets"
ADVERSARIAL = FIXTURES / "adversarial"
SYNTHETIC = FIXTURES / "synthetic"

EXPECTED: dict[str, str] = json.loads(
    (ADVERSARIAL / "expected-codes.json").read_text(encoding="utf-8")
)

# Cases the validator owns, not the shared package: they need a tokenizer, a model entry,
# or cross-split state that lmpipeline does not have.
VALIDATOR_OWNED = {"zero-examples"}


def _drive(directory: Path) -> None:
    """Resolve and fully read a dataset, the way a consumer does."""
    with tempfile.TemporaryDirectory() as tmp:
        resolved = resolve_dataset(directory, workdir=Path(tmp))
        for path in resolved.splits.values():
            list(iter_examples(path))


@pytest.mark.parametrize("case", sorted(EXPECTED))
def test_adversarial_fixture_produces_its_declared_code(case):
    if case in VALIDATOR_OWNED:
        pytest.skip(f"{case} is asserted in the validator, which owns that check")

    with pytest.raises(PipelineError) as exc:
        _drive(ADVERSARIAL / case)
    assert exc.value.code == EXPECTED[case], (
        f"{case}: expected {EXPECTED[case]}, got {exc.value.code}"
    )


def test_expected_codes_are_all_real_codes():
    """A typo in the map would silently weaken every assertion above."""
    known = {v for k, v in vars(Code).items() if not k.startswith("_")}
    for case, code in EXPECTED.items():
        assert code in known, f"{case} declares unknown code {code}"


def test_every_case_directory_has_an_expected_code():
    """A fixture with no declared code would be generated and never asserted."""
    on_disk = {p.name for p in ADVERSARIAL.iterdir() if p.is_dir()}
    assert on_disk == set(EXPECTED)


# -- the valid corpus must actually validate ----------------------------------


@pytest.mark.parametrize("family", ["conversational", "prompt-completion", "instruction"])
def test_synthetic_fixtures_are_accepted(family):
    with tempfile.TemporaryDirectory() as tmp:
        resolved = resolve_dataset(SYNTHETIC / family, workdir=Path(tmp))
        assert "train" in resolved.splits
        examples = list(iter_examples(resolved.splits["train"]))
        assert examples
        for example in examples:
            assert example.assistant_text.strip()


def test_conversational_fixture_has_all_three_splits():
    with tempfile.TemporaryDirectory() as tmp:
        resolved = resolve_dataset(SYNTHETIC / "conversational", workdir=Path(tmp))
        assert set(resolved.splits) == {"train", "validation", "test"}


def test_synthetic_splits_do_not_leak():
    """The shipped good corpus must not itself trip the leakage check."""
    with tempfile.TemporaryDirectory() as tmp:
        resolved = resolve_dataset(SYNTHETIC / "conversational", workdir=Path(tmp))
        prints = {
            name: {e.fingerprint() for e in iter_examples(path)}
            for name, path in resolved.splits.items()
        }
    assert not prints["train"] & prints["validation"]
    assert not prints["train"] & prints["test"]
    assert not prints["validation"] & prints["test"]


def test_unicode_survives_the_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        resolved = resolve_dataset(SYNTHETIC / "conversational", workdir=Path(tmp))
        text = " ".join(
            m["content"]
            for e in iter_examples(resolved.splits["train"])
            for m in e.messages
        )
    assert "🇵🇭" in text
    assert "kabisera" in text


def test_multi_turn_and_system_free_records_are_present():
    """The corpus must exercise the shapes the contract allows, not just the easy one."""
    with tempfile.TemporaryDirectory() as tmp:
        resolved = resolve_dataset(SYNTHETIC / "conversational", workdir=Path(tmp))
        examples = list(iter_examples(resolved.splits["train"]))
    assert any(len(e.messages) > 4 for e in examples), "no multi-turn example"
    assert any(e.messages[0]["role"] == "user" for e in examples), "no system-free example"


def test_fixture_generation_is_deterministic():
    """Re-running the generator must reproduce the corpus byte for byte.

    Otherwise a committed fixture and its generator drift, and the expected-code map
    describes files nobody can rebuild.
    """
    import hashlib
    import subprocess
    import sys

    def digest() -> str:
        h = hashlib.sha256()
        for path in sorted(FIXTURES.rglob("*")):
            if path.is_file() and path.suffix in {".jsonl", ".zip", ".json"}:
                h.update(path.relative_to(FIXTURES).as_posix().encode())
                h.update(path.read_bytes())
        return h.hexdigest()

    before = digest()
    subprocess.run(
        [sys.executable, str(FIXTURES / "build_fixtures.py")],
        check=True, capture_output=True,
    )
    assert digest() == before
