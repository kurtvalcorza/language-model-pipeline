"""The SFT training controls, owned here because both consumers must agree on them.

Issue #54. The training runtime lives in `language-model-finetuner`; the rules that decide
what a control *means* live here, in the package the finetuner vendors. Two of those rules
are the kind that get re-derived slightly differently in each repository and then disagree:

  * **"Off" has one spelling in a job document** — an absent `earlyStopping` block. The
    registered parameter form has no way to say absent and says `0` instead. The lowering
    between the two is a rule, not a convention, so it is a function rather than a sentence
    in a spec that each consumer implements from memory.
  * **A warmup ratio has to become an integer number of optimizer steps.** Two readers each
    dividing by a step count they derived themselves is how one run acquires two warmup
    lengths, one in its scheduler and another in its provenance. The division happens once,
    here.

Nothing in this module trains anything or touches torch. It is arithmetic and shape, which
is why it can be tested offline in this repository — the behaviour it describes cannot be.

See `TRAINING_SPEC.md` for the reasoning behind each default and `schemas/job.schema.json`
for the shape this lowers into.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

# The schedules the contract admits. `constant` is first because it is the default, and it is
# the default because every figure in COMPATIBILITY.md was measured at a constant rate.
LR_SCHEDULER_TYPES = ("constant", "linear", "cosine")

# Why a run stopped, as recorded in the training result's provenance. Two states: the epoch
# budget ran out, or patience did. A failure never reaches this branch.
STOPPING_REASONS = ("epochs_completed", "early_stopping")

# The registered defaults, in the platform's snake_case. Every value reproduces the
# behaviour of runs taken BEFORE the controls existed, so registering them changes no run:
#
#   weight_decay              0.01 is not a preference, it is the value already in force.
#                             The finetuner builds `torch.optim.AdamW(trainable, lr=...)`
#                             with no `weight_decay`, and torch's default for that argument
#                             is 1e-2. Stating it is the point of #54: an implicit library
#                             default is regularization policy nobody chose and provenance
#                             cannot show. Should the finetuner ever build the optimizer
#                             through HF `TrainingArguments`, whose default is 0.0, this
#                             value follows it rather than the other way round.
#   lr_scheduler_type         constant, and no warmup, because that is what was measured.
#   early_stopping_patience   0 means off. See `lower_controls`.
#   restore_best_adapter      False, because True publishes a different epoch's adapter and
#                             nothing in the file manifest would show that it had.
CONTROL_DEFAULTS: dict[str, Any] = {
    "weight_decay": 0.01,
    "lr_scheduler_type": "constant",
    "warmup_ratio": 0.0,
    "early_stopping_patience": 0,
    "early_stopping_min_delta": 0.0,
    "restore_best_adapter": False,
}


def optimizer_steps_per_epoch(
    examples: int, *, per_device_batch_size: int, gradient_accumulation_steps: int
) -> int:
    """Optimizer updates in one epoch over `examples` rows.

    Ceiling, not floor: TRAINING_SPEC.md flushes the partial accumulation window at the end
    of an epoch, so the leftover examples produce a real update and the scheduler must count
    it. Flooring here would leave the schedule one step short of the run on every epoch that
    does not divide evenly — a small error that compounds across epochs and silently moves
    where a warmup ends.
    """
    if examples < 1:
        raise ValueError(f"examples must be at least 1, got {examples}")
    if per_device_batch_size < 1 or gradient_accumulation_steps < 1:
        raise ValueError("batch size and accumulation steps must both be at least 1")
    effective = per_device_batch_size * gradient_accumulation_steps
    return math.ceil(examples / effective)


def total_optimizer_steps(
    examples: int,
    *,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
) -> int:
    """Optimizer updates across the whole run — what a schedule is stretched over.

    Per-epoch then multiplied, rather than over the concatenated run, because each epoch
    flushes its own partial window. The two differ whenever the epoch does not divide evenly,
    and the per-epoch count is the one that matches what the trainer does.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {epochs}")
    return epochs * optimizer_steps_per_epoch(
        examples,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )


def resolve_warmup_steps(
    total_steps: int, *, warmup_ratio: float | None = None, warmup_steps: int | None = None
) -> int:
    """The effective warmup in optimizer steps, from whichever spelling the job used.

    Exactly one spelling, because both at once is ambiguous and the ambiguity is silent:
    whichever the runtime happens to read wins and the document still looks well-formed. The
    job schema refuses a document carrying both; this refuses the same thing in code, so a
    caller that never validates cannot slip past it.

    Ceiling again, and capped at the run: a ratio that rounds up past the last step would
    describe a warmup that never finishes.
    """
    if warmup_ratio is not None and warmup_steps is not None:
        raise ValueError("give warmup_ratio or warmup_steps, never both")
    if total_steps < 1:
        raise ValueError(f"total_steps must be at least 1, got {total_steps}")

    if warmup_steps is not None:
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must not be negative, got {warmup_steps}")
        return min(warmup_steps, total_steps)

    ratio = 0.0 if warmup_ratio is None else warmup_ratio
    if not 0 <= ratio < 1:
        raise ValueError(f"warmup_ratio must be in [0, 1), got {ratio}")
    return min(math.ceil(total_steps * ratio), total_steps)


def lower_controls(params: Mapping[str, Any]) -> dict[str, Any]:
    """Lower registered snake_case parameters into the job document's `training` fragment.

    The one rule worth stating twice: `early_stopping_patience: 0` lowers to **no**
    `earlyStopping` key. The form cannot express absence and says 0; the job document says
    off by omitting the block, and its schema refuses a patience below 1. Emitting
    `{"patience": 0}` would be a second spelling of off, which is how a run ends up stopping
    for a reason nobody selected — and here it would also fail validation.

    Absent parameters are left absent rather than defaulted: a control the pipeline registry
    does not carry arrives at the container as nothing at all (`COMPLIANCE.md` C-8), and
    filling it in here would present a value the user never had the chance to set as though
    they had chosen it. Ask `CONTROL_DEFAULTS` for a default; do not have one applied to you.
    """
    fragment: dict[str, Any] = {}

    if "weight_decay" in params:
        fragment["weightDecay"] = params["weight_decay"]
    if "lr_scheduler_type" in params:
        fragment["lrSchedulerType"] = params["lr_scheduler_type"]

    # Ratio wins only because it is what the registration carries; both is refused rather
    # than resolved, for the same reason `resolve_warmup_steps` refuses it.
    has_ratio = "warmup_ratio" in params
    has_steps = "warmup_steps" in params
    if has_ratio and has_steps:
        raise ValueError("give warmup_ratio or warmup_steps, never both")
    if (has_ratio or has_steps) and "lr_scheduler_type" not in params:
        # The schema refuses this too: a warmup leaves the shape that follows it unstated.
        # Refusing here as well means a caller that skips validation still cannot build one.
        raise ValueError("a warmup needs an lr_scheduler_type to warm up into")
    if has_ratio:
        fragment["warmupRatio"] = params["warmup_ratio"]
    elif has_steps:
        fragment["warmupSteps"] = params["warmup_steps"]

    patience = params.get("early_stopping_patience", 0) or 0
    if patience < 0:
        raise ValueError(f"early_stopping_patience must not be negative, got {patience}")
    if patience:
        fragment["earlyStopping"] = {
            "patience": patience,
            "minDelta": params.get("early_stopping_min_delta", 0.0),
            "restoreBestAdapter": bool(params.get("restore_best_adapter", False)),
        }

    return fragment
