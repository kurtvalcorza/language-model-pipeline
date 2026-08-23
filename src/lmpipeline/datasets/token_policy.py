"""Tokenizer-aware dataset limits, owned by one module so two repos cannot disagree.

These checks used to live in the validator. They moved here when the source-verified DIMER
contract established that **the validator never learns which model the user selected**: a
validator Job receives four environment variables and `DIMER_PREPROCESSING_ARGS_JSON` is not
among them (`COMPLIANCE.md` C-1). A tokenizer-specific check cannot be performed by a
container that does not know the tokenizer.

The decision (validator #10) was to make the validator model-agnostic and move these into
the finetuner, which does receive the selection. They live in the shared package rather than
in the finetuner because the reason the registry and the resolver live here applies verbatim:
if the two repositories ever compute a token count differently, validation stops meaning
anything. Today only the finetuner calls this. That is not a reason to bury it in the
finetuner — it is the module's contract that matters, not its current caller count.

**Where the finetuner must call it: immediately after resolving the model and tokenizer, and
BEFORE loading weights or starting training.** An incompatible dataset then costs seconds
rather than a multi-gigabyte download and a training run, which is the whole reason these
checks were worth having when the validator ran them.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

from ..errors import Code, DatasetError
from ..registry import ModelEntry

# Total-token budget across the training split. Guards wall time as much as memory: a
# dataset can be within every per-example limit and still describe a training run nobody
# intended to start.
MAX_TOTAL_TRAIN_TOKENS = 50_000_000


@dataclass(frozen=True)
class TokenStats:
    count: int
    minimum: int
    median: int
    p95: int
    p99: int
    maximum: int
    total: int

    def to_dict(self) -> dict[str, int]:
        return {
            "count": self.count,
            "min": self.minimum,
            "median": self.median,
            "p95": self.p95,
            "p99": self.p99,
            "max": self.maximum,
            "total": self.total,
        }


def _percentile(values: list[int], fraction: float) -> int:
    """Nearest-rank percentile.

    Deliberately not interpolated: these numbers are compared against integer token
    ceilings, and an interpolated p99 that lands between two real examples would describe
    a sequence length no example actually has.
    """
    if not values:
        return 0
    index = max(0, min(len(values) - 1, int(round(fraction * len(values) + 0.5)) - 1))
    return values[index]


def summarize(lengths: list[int]) -> TokenStats:
    ordered = sorted(lengths)
    if not ordered:
        return TokenStats(0, 0, 0, 0, 0, 0, 0)
    return TokenStats(
        count=len(ordered),
        minimum=ordered[0],
        median=int(statistics.median(ordered)),
        p95=_percentile(ordered, 0.95),
        p99=_percentile(ordered, 0.99),
        maximum=ordered[-1],
        total=sum(ordered),
    )


def render_and_count(tokenizer, messages: list[dict[str, str]], entry: ModelEntry) -> int:
    """Apply the model's chat template and return the token count for one example.

    Uses the tokenizer's own template so the count reflects what training will actually
    see, including role markers and special tokens — a raw content-only count would
    understate length and let overlength examples through.
    """
    if entry.chat_template == "tokenizer":
        template = getattr(tokenizer, "chat_template", None)
        if not template:
            raise DatasetError(
                f"Model {entry.key!r} declares chat_template=tokenizer but its tokenizer "
                "carries no chat template, so conversational data cannot be rendered.",
                code=Code.DATASET_CHAT_TEMPLATE_MISSING,
                details={"model_key": entry.key},
            )
        rendered = tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=False
        )
    else:  # pragma: no cover - no registry entry uses a custom template yet
        rendered = "\n".join(m["content"] for m in messages)

    return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])


def resolve_ceiling(entry: ModelEntry, requested: Any) -> int | None:
    """The limit the TRAINER will actually enforce, not merely the model's ceiling.

    `max_sequence_length` is a user parameter, so a dataset checked only against the model's
    4096 could still be rejected mid-training at a configured 2048. Resolved through the
    registry entry so every caller computes the same number.

    Note that `clamp_sequence_length` does not clamp despite its name: a request above the
    model's ceiling raises CONFIG_OUT_OF_BOUNDS. That is the right behaviour and worth not
    "fixing" — silently reducing a requested 8192 to 4096 would train something the user did
    not ask for and then report success.
    """
    if requested is None:
        return entry.max_sequence_length
    return entry.clamp_sequence_length(int(requested))


@dataclass
class TokenPolicyReport:
    """What the policy measured, per split, plus the ceiling it measured against."""

    ceiling: int | None
    stats: dict[str, TokenStats]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ceiling": self.ceiling,
            "budget": MAX_TOTAL_TRAIN_TOKENS,
            "splits": {name: s.to_dict() for name, s in sorted(self.stats.items())},
        }


def enforce_token_policy(
    splits: dict[str, list],
    *,
    tokenizer,
    entry: ModelEntry,
    requested_max_sequence_length: Any = None,
    train_split: str = "train",
) -> TokenPolicyReport:
    """Reject a dataset the selected model cannot train on, and return what was measured.

    `splits` maps a split name to a list of normalized examples, each exposing `.messages`
    and `.line_number`. Raises `DatasetError` with a stable code; returns the measurements
    when everything fits, so the caller can report real numbers rather than "ok".

    Line numbers travel in the error details, never content — the same privacy rule the
    validator applied when it owned these checks.
    """
    ceiling = resolve_ceiling(entry, requested_max_sequence_length)

    stats: dict[str, TokenStats] = {}
    overlength: dict[str, list[int]] = {}
    for name, examples in sorted(splits.items()):
        lengths: list[int] = []
        for example in examples:
            count = render_and_count(tokenizer, list(example.messages), entry)
            lengths.append(count)
            if ceiling is not None and count > ceiling:
                overlength.setdefault(name, []).append(example.line_number)
        stats[name] = summarize(lengths)

    overlength_total = sum(len(v) for v in overlength.values())
    if overlength_total:
        worst = max(s.maximum for s in stats.values() if s.count)
        raise DatasetError(
            f"{overlength_total} example(s) exceed the {ceiling}-token limit for "
            f"{entry.key!r} (longest is {worst} tokens). Shorten them and re-upload; "
            "the pipeline does not truncate silently.",
            code=Code.DATASET_SEQUENCE_TOO_LONG,
            details={
                "ceiling": ceiling,
                "overlengthCount": overlength_total,
                "longest": worst,
                "linesBySplit": {n: v[:50] for n, v in sorted(overlength.items())},
            },
        )

    train = stats.get(train_split)
    if train is not None and train.total > MAX_TOTAL_TRAIN_TOKENS:
        raise DatasetError(
            f"The training split totals {train.total} tokens, above the "
            f"{MAX_TOTAL_TRAIN_TOKENS} budget for this pipeline.",
            code=Code.DATASET_TOKEN_BUDGET_EXCEEDED,
            details={"total": train.total, "budget": MAX_TOTAL_TRAIN_TOKENS},
        )

    return TokenPolicyReport(ceiling=ceiling, stats=stats)
