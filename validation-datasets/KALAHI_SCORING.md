# Kalahi scoring contract

This repository reproduces a **narrow, versioned scoring contract** for the Tier 4 Kalahi
multiple-choice evaluation. It does not vendor or replace SEA-HELM.

Upstream source of truth reviewed for this implementation:

- repository: `aisingapore/SEA-HELM`
- commit: `9be3986d00185dc6d6ec362b71542ed29642a6be`
- license: MIT
- task: `kalahi-mc`
- task-config version: `1.1`
- metric: `normalized_accuracy`

The relevant upstream files are:

- `seahelm_tasks/cultural/kalahi/config.yaml`
- `src/metrics/f1_acc_metric.py`
- `src/metrics/seahelm_metric.py`

Any semantic change in those upstream files requires a deliberate review and version bump here;
this module must not silently track SEA-HELM `main`.

## Prompt contract

The zero-shot instruction-tuned prompt is rendered exactly as the reviewed v1.1 task config:

```text
Piliin ang pinaka-angkop na sagot sa sumusunod na tanong.

Sumagot gamit ang sumusunod na format.
"Sagot: ANSWER"
Palitan ang ANSWER gamit ang napiling sagot. Gumamit lang ng letrang A, B, C, o D sa sagot mo.

Tanong:
```
<question>

<mcq>
```
```

The harness exposes `render_prompt(question=..., mcq=...)`. It does not apply a model-specific
chat template; the serving/evaluation runner owns that outer layer.

## Response extraction

SEA-HELM searches case-insensitively for the answer tag followed by one of the labels. If the
regex does not match, it normalizes the entire response and still accepts it when the result is
exactly a known label. Therefore both `Sagot: A` and a bare `A` score as A; prose such as
`The answer is A` becomes `null`.

Missing/unparseable responses also become `null`. This behavior is reproduced intentionally,
including the bare-label fallback.

## Metrics

`score_responses(references, responses)` returns the upstream metric fields in percentage
units plus response/provenance evidence.

| Field | Reproduced meaning |
|---|---|
| `accuracy` | **balanced accuracy** × 100; this historical upstream name is not ordinary micro accuracy |
| `macro_f1` | macro F1 over the union of reference and predicted labels; `null` therefore contributes a zero-F1 class when predicted |
| `null_weighted_f1` | macro F1 over reference labels only, multiplied by `(1 - null_count / n)` |
| `normalized_accuracy` | balanced accuracy normalized from random baseline `1 / number_of_reference_labels` to 1, clamped below chance to zero, × 100 |
| `null_count` | number of responses that did not resolve to a known label |

The result also records the exact SEA-HELM repository, commit, task name, task version and
answer tag. This makes an evaluation result auditable even if upstream changes later.

## Scope boundary

This code reproduces **prompt + response extraction + aggregate metric semantics**. It does
not claim equivalence with the complete SEA-HELM leaderboard environment. In particular it
does not implement:

- model serving or tokenizer/chat-template application;
- batch/retry behavior;
- SEA-HELM run aggregation outside this task;
- the `kalahi-mc-logprobs` variant;
- a model-quality pass/fail threshold.

Kalahi dataset preparation remains separately gated by `validation-datasets/EVALUATION.md`
and issue #47. Gated source rows are never required for the scoring unit tests.
