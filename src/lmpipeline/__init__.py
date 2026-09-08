"""Shared contracts for the DIMER language-model fine-tuning capability.

This package is consumed identically by `language-model-dataset-validator` and
`language-model-finetuner`. Everything that both containers must agree on — the DIMER
runtime contract, the model registry, dataset resolution and normalization, the result
shape, and the error-code namespace — lives here and nowhere else.

`CONTRACT_VERSION` is the contract version. Bump the minor for additive changes; bump the
major for anything that changes how an existing field is interpreted, and update the pinned
version in both consumer repos in the same change.
"""

from __future__ import annotations

from .errors import Code, ConfigError, DatasetError, ModelError, PipelineError, Stage
from .registry import ModelEntry, ModelRegistry
from .result import Check, Result, write_result
from .training_controls import (
    CONTROL_DEFAULTS,
    LR_SCHEDULER_TYPES,
    STOPPING_REASONS,
    lower_controls,
    optimizer_steps_per_epoch,
    resolve_warmup_steps,
    total_optimizer_steps,
)

__version__ = "0.1.0"

CONTRACT_VERSION = "1.1"

__all__ = [
    "CONTRACT_VERSION",
    "CONTROL_DEFAULTS",
    "LR_SCHEDULER_TYPES",
    "STOPPING_REASONS",
    "Check",
    "Code",
    "ConfigError",
    "DatasetError",
    "ModelEntry",
    "ModelError",
    "ModelRegistry",
    "PipelineError",
    "Result",
    "Stage",
    "lower_controls",
    "optimizer_steps_per_epoch",
    "resolve_warmup_steps",
    "total_optimizer_steps",
    "write_result",
    "__version__",
]
