"""Dependency-free Kalahi MC scoring aligned with a pinned SEA-HELM contract.

This module reproduces the prompt, response extraction and aggregate metric semantics used by
SEA-HELM's ``kalahi-mc`` task at one reviewed upstream commit. It is deliberately not the
full SEA-HELM inference runner: model serving, chat templating and leaderboard aggregation
remain separate concerns.
"""

from __future__ import annotations

import re
import string
from collections.abc import Iterable, Sequence
from typing import Any

SEA_HELM_REPOSITORY = "aisingapore/SEA-HELM"
SEA_HELM_REVISION = "9be3986d00185dc6d6ec362b71542ed29642a6be"
SEA_HELM_TASK = "kalahi-mc"
SEA_HELM_TASK_VERSION = "1.1"
ANSWER_TAG = "Sagot:"
NULL_LABEL = "null"

_PREAMBLE = """Piliin ang pinaka-angkop na sagot sa sumusunod na tanong.

Sumagot gamit ang sumusunod na format.
\"{answer_tag} ANSWER\"
Palitan ang ANSWER gamit ang napiling sagot. Gumamit lang ng letrang A, B, C, o D sa sagot mo."""
_TASK_TEMPLATE = """Tanong:
```
{question}

{mcq}
```"""


class KalahiScoringError(ValueError):
    """Input cannot be scored under the pinned Kalahi MC contract."""


def render_prompt(*, question: str, mcq: str) -> str:
    """Render the SEA-HELM v1.1 zero-shot user prompt for one Kalahi prompt variant."""
    if not isinstance(question, str) or not question.strip():
        raise KalahiScoringError("question must be a non-empty string")
    if not isinstance(mcq, str) or not mcq.strip():
        raise KalahiScoringError("mcq must be a non-empty string")
    preamble = _PREAMBLE.format(answer_tag=ANSWER_TAG)
    task = _TASK_TEMPLATE.format(question=question, mcq=mcq)
    return f"{preamble}\n\n{task}".strip()


def _normalize_answer(value: str) -> str:
    """Match SEA-HELM's lowercase + edge-punctuation + whitespace normalization."""
    lowered = value.lower()
    stripped = lowered.strip(string.punctuation + " \n")
    return " ".join(stripped.split())


def extract_label(
    response: str | None,
    labels: Iterable[str],
    *,
    null_label: str = NULL_LABEL,
) -> str:
    """Extract a known label with SEA-HELM's Kalahi answer-tag fallback behavior.

    SEA-HELM first searches for ``Sagot: <label>`` case-insensitively. If that search fails,
    it normalizes the *whole response* and still accepts it when the result is exactly one of
    the known labels. Consequently a bare ``A`` is accepted even though the requested output
    format is ``Sagot: A``; prose such as ``The answer is A`` is not.
    """
    label_values = [str(label) for label in labels]
    if not label_values:
        raise KalahiScoringError("at least one reference label is required")
    if any(not label.strip() for label in label_values):
        raise KalahiScoringError("labels must be non-empty strings")

    label_map = {label.lower(): label for label in label_values}
    # Actual Kalahi labels are simple A/B/C/D strings. Escaping them is semantics-preserving
    # while avoiding a future label accidentally becoming regex syntax.
    label_pattern = "|".join(re.escape(label) for label in label_values)
    pattern = re.escape(ANSWER_TAG) + rf"[\s\*]*({label_pattern})+"

    raw = "" if response is None else str(response)
    match = re.search(pattern, raw, flags=re.IGNORECASE)
    candidate = match.group(1) if match is not None else raw
    normalized = _normalize_answer(candidate)
    return label_map.get(normalized, null_label)


def _f1_for_label(
    references: Sequence[str], predictions: Sequence[str], label: str
) -> float:
    pairs = zip(references, predictions, strict=True)
    tp = sum(ref == label and pred == label for ref, pred in pairs)
    pairs = zip(references, predictions, strict=True)
    fp = sum(ref != label and pred == label for ref, pred in pairs)
    pairs = zip(references, predictions, strict=True)
    fn = sum(ref == label and pred != label for ref, pred in pairs)
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else (2 * tp) / denominator


def _balanced_accuracy(references: Sequence[str], predictions: Sequence[str]) -> float:
    labels = sorted(set(references))
    recalls = []
    for label in labels:
        actual = sum(ref == label for ref in references)
        correct = sum(
            ref == label and pred == label
            for ref, pred in zip(references, predictions, strict=True)
        )
        recalls.append(correct / actual)
    return sum(recalls) / len(recalls)


def _normalized_accuracy(score: float, label_count: int) -> float:
    random_baseline = 1 / label_count
    if random_baseline == 1:
        return 0.0
    return max((score - random_baseline) / (1 - random_baseline), 0.0)


def score_responses(
    references: Sequence[str],
    responses: Sequence[str | None],
) -> dict[str, Any]:
    """Score model responses using the pinned SEA-HELM Kalahi MC metric semantics.

    The upstream field named ``accuracy`` is **balanced accuracy**, not ordinary micro
    accuracy. ``null_weighted_f1`` uses macro F1 over reference labels only and then applies
    the null-response penalty; upstream ``macro_f1`` includes the union of reference and
    predicted labels, so a predicted ``null`` contributes a zero-F1 class there.
    """
    if not references:
        raise KalahiScoringError("references must not be empty")
    if len(references) != len(responses):
        raise KalahiScoringError(
            f"references/responses length mismatch: {len(references)} != {len(responses)}"
        )

    refs = [str(label) for label in references]
    if any(not label.strip() for label in refs):
        raise KalahiScoringError("reference labels must be non-empty strings")
    reference_labels = sorted(set(refs))
    predictions = [extract_label(response, reference_labels) for response in responses]

    balanced_accuracy = _balanced_accuracy(refs, predictions)
    reference_macro_f1 = sum(
        _f1_for_label(refs, predictions, label) for label in reference_labels
    ) / len(reference_labels)

    union_labels = sorted(set(refs) | set(predictions))
    macro_f1 = sum(
        _f1_for_label(refs, predictions, label) for label in union_labels
    ) / len(union_labels)
    null_count = sum(prediction == NULL_LABEL for prediction in predictions)
    null_weighted_f1 = reference_macro_f1 * (1 - null_count / len(predictions))
    normalized_accuracy = _normalized_accuracy(balanced_accuracy, len(reference_labels))

    individual = [
        int(reference == prediction)
        for reference, prediction in zip(refs, predictions, strict=True)
    ]

    return {
        "metrics": {
            # Upstream keeps this historical field name even though the calculation is
            # balanced_accuracy_score(). Preserve the name and document the semantics.
            "accuracy": 100 * balanced_accuracy,
            "macro_f1": 100 * macro_f1,
            "null_weighted_f1": 100 * null_weighted_f1,
            "normalized_accuracy": 100 * normalized_accuracy,
            "null_count": null_count,
        },
        "predictions": predictions,
        "individual_normalized_accuracy": individual,
        "provenance": {
            "implementation": "seahelm-kalahi-mc-contract",
            "upstreamRepository": SEA_HELM_REPOSITORY,
            "upstreamRevision": SEA_HELM_REVISION,
            "upstreamTask": SEA_HELM_TASK,
            "upstreamTaskVersion": SEA_HELM_TASK_VERSION,
            "answerTag": ANSWER_TAG,
            "metric": "normalized_accuracy",
        },
    }
