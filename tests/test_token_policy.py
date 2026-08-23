"""The tokenizer-aware limits must behave identically wherever they run.

These checks moved out of the validator because a validator Job never learns which model the
user selected (COMPLIANCE.md C-1, decision on validator #10). Moving a check is the easiest
way to silently weaken it, so the behaviour is pinned here rather than only where it is
called: same codes, same clamp, same privacy rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from lmpipeline.datasets.token_policy import (
    MAX_TOTAL_TRAIN_TOKENS,
    enforce_token_policy,
    resolve_ceiling,
    summarize,
)
from lmpipeline.errors import Code, DatasetError
from lmpipeline.registry import ModelRegistry


@dataclass
class FakeExample:
    messages: list
    line_number: int


class FakeTokenizer:
    """Counts one token per whitespace-separated word, so lengths are predictable."""

    chat_template = "{{ messages }}"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return " ".join(m["content"] for m in messages)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def _example(words: int, line: int = 1) -> FakeExample:
    return FakeExample(
        messages=[{"role": "user", "content": " ".join(["w"] * words)}], line_number=line
    )


@pytest.fixture
def entry():
    return ModelRegistry.load().resolve("qwen3-1.7b")


# -- measurement -------------------------------------------------------------------


def test_it_returns_what_it_measured_rather_than_just_passing(entry):
    report = enforce_token_policy(
        {"train": [_example(10, 1), _example(20, 2)]},
        tokenizer=FakeTokenizer(), entry=entry, requested_max_sequence_length=512,
    )
    assert report.ceiling == 512
    assert report.stats["train"].maximum == 20
    assert report.stats["train"].total == 30
    assert report.to_dict()["budget"] == MAX_TOTAL_TRAIN_TOKENS


def test_percentiles_are_never_interpolated():
    """An interpolated p99 would describe a length no example actually has."""
    stats = summarize([1, 2, 3, 100])
    assert stats.p99 in (1, 2, 3, 100)
    assert stats.maximum == 100


# -- the ceiling -------------------------------------------------------------------


def test_the_ceiling_is_the_trainer_limit_not_the_model_limit(entry):
    """A dataset checked only against the model's ceiling can still fail mid-training."""
    assert resolve_ceiling(entry, 256) == 256
    assert resolve_ceiling(entry, None) == entry.max_sequence_length


def test_a_request_above_the_model_ceiling_is_REFUSED_not_silently_reduced(entry):
    """`clamp_sequence_length` is a misleading name: it raises rather than clamping.

    Worth pinning, because the safer behaviour is the non-obvious one. Silently reducing a
    requested 8192 to 4096 would train something the user did not ask for and report
    success; refusing tells them their configuration is impossible for this model.
    """
    from lmpipeline.errors import Code as C
    from lmpipeline.errors import ModelError

    with pytest.raises(ModelError) as excinfo:
        resolve_ceiling(entry, 10**9)
    assert excinfo.value.code == C.CONFIG_OUT_OF_BOUNDS


def test_an_overlength_example_is_rejected_with_its_line_number(entry):
    with pytest.raises(DatasetError) as excinfo:
        enforce_token_policy(
            {"train": [_example(5, 1), _example(99, 7)]},
            tokenizer=FakeTokenizer(), entry=entry, requested_max_sequence_length=10,
        )
    assert excinfo.value.code == Code.DATASET_SEQUENCE_TOO_LONG
    assert excinfo.value.details["linesBySplit"] == {"train": [7]}
    assert excinfo.value.details["longest"] == 99
    assert "does not truncate silently" in excinfo.value.message


def test_rejection_details_carry_line_numbers_and_never_content(entry):
    secret = "CONFIDENTIAL-TOKEN-STRING"
    example = FakeExample(
        messages=[{"role": "user", "content": " ".join([secret] * 50)}], line_number=3
    )
    with pytest.raises(DatasetError) as excinfo:
        enforce_token_policy({"train": [example]}, tokenizer=FakeTokenizer(), entry=entry,
                             requested_max_sequence_length=10)
    assert secret not in repr(excinfo.value.details) + excinfo.value.message


# -- the total budget --------------------------------------------------------------


def test_the_training_split_total_is_bounded(entry, monkeypatch):
    import lmpipeline.datasets.token_policy as tp

    monkeypatch.setattr(tp, "MAX_TOTAL_TRAIN_TOKENS", 25)
    with pytest.raises(DatasetError) as excinfo:
        tp.enforce_token_policy(
            {"train": [_example(20, 1), _example(20, 2)]},
            tokenizer=FakeTokenizer(), entry=entry, requested_max_sequence_length=512,
        )
    assert excinfo.value.code == Code.DATASET_TOKEN_BUDGET_EXCEEDED
    assert excinfo.value.details["total"] == 40


def test_only_the_training_split_is_budgeted(entry, monkeypatch):
    """Validation and test splits are evaluated, not trained on, so they do not count."""
    import lmpipeline.datasets.token_policy as tp

    monkeypatch.setattr(tp, "MAX_TOTAL_TRAIN_TOKENS", 25)
    report = tp.enforce_token_policy(
        {"train": [_example(10, 1)], "validation": [_example(500, 1)]},
        tokenizer=FakeTokenizer(), entry=entry, requested_max_sequence_length=1000,
    )
    assert report.stats["validation"].total == 500


# -- the chat template -------------------------------------------------------------


def test_a_tokenizer_without_a_chat_template_is_a_named_failure(entry):
    """Otherwise conversational data renders as something nobody intended to train on."""

    class NoTemplate(FakeTokenizer):
        chat_template = None

    with pytest.raises(DatasetError) as excinfo:
        enforce_token_policy({"train": [_example(5, 1)]}, tokenizer=NoTemplate(),
                             entry=entry, requested_max_sequence_length=512)
    assert excinfo.value.code == Code.DATASET_CHAT_TEMPLATE_MISSING


def test_an_empty_split_is_not_an_error(entry):
    report = enforce_token_policy({"train": []}, tokenizer=FakeTokenizer(), entry=entry)
    assert report.stats["train"].count == 0
