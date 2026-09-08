from __future__ import annotations

import pytest

from validation_datasets.kalahi_scoring import (
    SEA_HELM_REPOSITORY,
    SEA_HELM_REVISION,
    SEA_HELM_TASK,
    SEA_HELM_TASK_VERSION,
    KalahiScoringError,
    extract_label,
    render_prompt,
    score_responses,
)


def test_prompt_matches_seahelm_kalahi_mc_v1_1_snapshot():
    rendered = render_prompt(
        question="Ano ang tamang sagot?",
        mcq="A. Una\nB. Ikalawa\nC. Ikatlo\nD. Ikaapat",
    )
    assert rendered == (
        "Piliin ang pinaka-angkop na sagot sa sumusunod na tanong.\n\n"
        "Sumagot gamit ang sumusunod na format.\n"
        '"Sagot: ANSWER"\n'
        "Palitan ang ANSWER gamit ang napiling sagot. "
        "Gumamit lang ng letrang A, B, C, o D sa sagot mo.\n\n"
        "Tanong:\n"
        "```\n"
        "Ano ang tamang sagot?\n\n"
        "A. Una\nB. Ikalawa\nC. Ikatlo\nD. Ikaapat\n"
        "```"
    )


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("Sagot: A", "A"),
        ("sagot: b", "B"),
        ("Sagot:   **C**", "C"),
        ("A", "A"),  # upstream fallback accepts a bare known label
        ("  (d)  ", "D"),
        ("The answer is A", "null"),
        ("Sagot: Z", "null"),
        (None, "null"),
    ],
)
def test_response_extraction_matches_answer_tag_and_fallback(response, expected):
    assert extract_label(response, ["A", "B", "C", "D"]) == expected


def test_balanced_accuracy_is_not_micro_accuracy():
    # 3/4 predictions are micro-correct, but the minority B class has zero recall:
    # balanced accuracy = (1.0 + 0.0) / 2 = 0.5.
    result = score_responses(
        ["A", "A", "A", "B"],
        ["Sagot: A", "A", "a", "Sagot: A"],
    )
    assert result["metrics"]["accuracy"] == pytest.approx(50.0)
    assert result["metrics"]["normalized_accuracy"] == pytest.approx(0.0)
    assert result["individual_normalized_accuracy"] == [1, 1, 1, 0]


def test_normalized_accuracy_above_chance():
    # A recall 1.0, B recall 0.5 -> balanced=.75. With two labels chance=.5,
    # normalized = (.75-.5)/(1-.5) = .5.
    result = score_responses(
        ["A", "A", "B", "B"],
        ["A", "Sagot: A", "Sagot: B", "Sagot: A"],
    )
    assert result["metrics"]["accuracy"] == pytest.approx(75.0)
    assert result["metrics"]["normalized_accuracy"] == pytest.approx(50.0)


def test_below_chance_normalized_accuracy_clamps_to_zero():
    result = score_responses(
        ["A", "A", "B", "B"],
        ["B", "B", "A", "A"],
    )
    assert result["metrics"]["accuracy"] == pytest.approx(0.0)
    assert result["metrics"]["normalized_accuracy"] == pytest.approx(0.0)


def test_null_prediction_affects_macro_and_null_weighted_f1_like_upstream():
    result = score_responses(["A", "B"], [None, "Sagot: B"])

    # Reference-only F1s: A=0, B=1 -> .5; one of two is null -> .25 weighted.
    assert result["metrics"]["null_weighted_f1"] == pytest.approx(25.0)
    # Upstream macro_f1 uses the union A/B/null -> (0 + 1 + 0) / 3.
    assert result["metrics"]["macro_f1"] == pytest.approx(100 / 3)
    assert result["metrics"]["null_count"] == 1
    assert result["predictions"] == ["null", "B"]


def test_perfect_four_way_score_is_100():
    result = score_responses(
        ["A", "B", "C", "D"],
        ["Sagot: A", "Sagot: B", "Sagot: C", "Sagot: D"],
    )
    assert result["metrics"] == {
        "accuracy": pytest.approx(100.0),
        "macro_f1": pytest.approx(100.0),
        "null_weighted_f1": pytest.approx(100.0),
        "normalized_accuracy": pytest.approx(100.0),
        "null_count": 0,
    }


def test_single_label_normalized_score_matches_upstream_zero_guard():
    result = score_responses(["A", "A"], ["A", "A"])
    assert result["metrics"]["accuracy"] == pytest.approx(100.0)
    assert result["metrics"]["normalized_accuracy"] == pytest.approx(0.0)


def test_provenance_pins_the_reviewed_upstream_contract():
    result = score_responses(["A", "B"], ["A", "B"])
    provenance = result["provenance"]
    assert provenance["upstreamRepository"] == SEA_HELM_REPOSITORY
    assert provenance["upstreamRevision"] == SEA_HELM_REVISION
    assert provenance["upstreamTask"] == SEA_HELM_TASK
    assert provenance["upstreamTaskVersion"] == SEA_HELM_TASK_VERSION
    assert SEA_HELM_REVISION == "9be3986d00185dc6d6ec362b71542ed29642a6be"
    assert SEA_HELM_TASK_VERSION == "1.1"


def test_invalid_inputs_fail_closed():
    with pytest.raises(KalahiScoringError, match="question"):
        render_prompt(question="", mcq="A. x")
    with pytest.raises(KalahiScoringError, match="mcq"):
        render_prompt(question="q", mcq="")
    with pytest.raises(KalahiScoringError, match="must not be empty"):
        score_responses([], [])
    with pytest.raises(KalahiScoringError, match="length mismatch"):
        score_responses(["A"], [])
    with pytest.raises(KalahiScoringError, match="non-empty"):
        score_responses([""], ["A"])
