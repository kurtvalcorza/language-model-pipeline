#!/usr/bin/env python
"""Generate the DIMER `fineTunableModels` registration from the model registry.

C-2 is the finding that the platform resolves `model_id` against a pipeline's registered
`schema.model.fineTunableModels`, and that nothing registers our keys. This script produces
that block so the values are DERIVED from `data/model-registry.yaml` rather than typed into
the Pipeline Builder from memory. A registration that drifts from the registry fails at
`ModelRegistry.resolve` inside the container, after the Job has already been scheduled.

    python scripts/build_registration.py                 # the block, as JSON
    python scripts/build_registration.py --check         # non-zero if it would not resolve

See MODEL_REGISTRATION.md for what each field means, which ones the backend ignores, and
what it silently substitutes when one is omitted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lmpipeline.registry import ModelRegistry  # noqa: E402
from lmpipeline.training_controls import CONTROL_DEFAULTS  # noqa: E402

# Our task is not one of the three the backend recognizes, so `_normalize_task_type` folds it
# to "object_detection". We send the honest value anyway: the normalization is lossy on the
# platform's side, not ours, and the recorded intent is what a future backend change would be
# read against. Nothing in our containers reads this field -- both bake `language_model_sft`.
TASK_TYPE = "language_model_sft"

# MUST be set explicitly. Omitted, the backend substitutes "ultralytics" for both.
FRAMEWORK = "transformers-peft"

# MUST be set explicitly. Omitted, the backend substitutes "yolo".
SUPPORTED_DATASET_FORMAT = "jsonl_messages"

# Which entry the Builder pre-selects, and -- more importantly -- the one the platform
# falls back to when `model_id` is absent. It also travels to the VALIDATOR as
# `defaultFineTunableModelId` in DIMER_PIPELINE_METADATA_JSON, which is the one model
# channel that does reach it. The validator is model-agnostic by the C-1 decision and
# ignores it; this is the mechanism that made option 1 viable, recorded so the choice is
# re-arguable. Kept equal to dimer-pipeline.json's documented default.
DEFAULT_MODEL_ID = "qwen3-1.7b"

# Registry-independent training defaults, matching dimer-pipeline.json's envelope. MUST be
# set explicitly: omitted, the backend substitutes YOLO's COMMON_TRAINING_DEFAULTS, whose
# `mosaic`, `degrees` and `fliplr` are meaningless here and whose learning rate is 50x ours.
COMMON_DEFAULTS: dict[str, Any] = {
    "epochs": 3,
    "learning_rate": 0.0002,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "per_device_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "seed": 42,
    # The training controls of #54 -- weight decay, LR schedule, early stopping -- are
    # SPREAD IN from the shared package rather than repeated here. They are contract, not
    # registration policy: the finetuner reads the same constant to know what a control
    # means, so a value typed twice is a value that eventually disagrees with itself. Each
    # default reproduces pre-control behaviour, and the reasoning is recorded next to them
    # in `lmpipeline/training_controls.py`.
    **CONTROL_DEFAULTS,
}


def _default_method(entry) -> str:
    """QLoRA where the registry only measured QLoRA, LoRA where it is genuinely available.

    Above roughly 1B parameters QLoRA is not an optimization but the only method that fits
    the measured hardware (COMPATIBILITY.md), so the default has to follow the measurement
    rather than a blanket preference.
    """
    methods = list(entry.training_methods)
    if "lora" in methods and entry.resource_profile.for_method("lora").min_vram_gb:
        return "lora"
    return "qlora" if "qlora" in methods else methods[0]


def _resource_profile(entry, method: str) -> dict[str, Any]:
    """Advisory only -- see MODEL_REGISTRATION.md.

    The backend sizes the pod from cluster-wide WORKBENCH_FINE_TUNING_* environment
    variables and never reads this. It is carried so the value reaches the container in
    DIMER_MODEL_CONFIG_JSON, where it can be compared against the hardware actually
    provisioned.
    """
    measured = entry.resource_profile.for_method(method).min_vram_gb
    return {
        "accelerator": "nvidia",
        "gpuCount": 1,
        "minVramGb": measured,
        "memory": "16Gi",
    }


def build(registry: ModelRegistry) -> list[dict[str, Any]]:
    block: list[dict[str, Any]] = []
    for key in registry.keys(user_facing=True):
        entry = registry.resolve(key)
        method = _default_method(entry)
        block.append(
            {
                # `id` is what DIMER sends back as `model_id`, and it must equal the registry
                # key or ModelRegistry.resolve fails inside the container.
                "id": entry.key,
                # The Hub repository name, which is what a user recognizes. Not the
                # registry key: those are ours, and two of them differ from the repo name
                # only by case.
                "displayName": entry.model_id.split("/")[-1],
                "taskType": TASK_TYPE,
                "framework": FRAMEWORK,
                "provider": entry.provider,
                # Opaque to the backend: required non-empty, used only as a lookup key for
                # legacy `base_model` translation. Never parsed, never fetched. Carrying the
                # pinned revision makes the registration self-describing.
                "baseWeights": f"{entry.model_id}@{entry.revision}",
                "defaultTrainingParams": {
                    **COMMON_DEFAULTS,
                    "method": method,
                    "max_sequence_length": min(2048, entry.max_sequence_length or 2048),
                },
                "resourceProfile": _resource_profile(entry, method),
                "supportedDatasetFormat": SUPPORTED_DATASET_FORMAT,
            }
        )
    return block


def check(block: list[dict[str, Any]], registry: ModelRegistry) -> list[str]:
    """Re-apply the backend's own acceptance rules, so a bad block fails here not in prod.

    Transcribed from `dimer/workbench/fine_tunable_models.py`
    (`get_pipeline_fine_tunable_models`, `resolve_fine_tunable_model_config`) on the
    `on-prem` branch @ `e06d1a9`.
    """
    problems: list[str] = []
    approved = set(registry.keys(user_facing=True))
    seen_ids: set[str] = set()
    seen_weights: set[str] = set()

    for entry in block:
        identifier = str(entry.get("id", "")).strip()
        weights = str(entry.get("baseWeights", "")).strip()

        # The backend SKIPS an entry with an empty id or baseWeights rather than erroring.
        # A block that silently loses an entry leaves `model_id` unresolvable at launch.
        if not identifier:
            problems.append(f"entry with empty id would be dropped: {entry!r}")
            continue
        if not weights:
            problems.append(f"{identifier}: empty baseWeights would drop this entry")

        if identifier not in approved:
            problems.append(
                f"{identifier}: not an approved registry key, so the container would raise "
                f"MODEL_KEY_MISSING after the Job was already scheduled"
            )
        if identifier in seen_ids:
            problems.append(f"{identifier}: duplicate id; models_by_id would keep only one")
        seen_ids.add(identifier)

        # model_ids_by_base_weights is keyed by baseWeights, so a collision silently
        # remaps one model's legacy selector onto another's id.
        if weights in seen_weights:
            problems.append(f"{identifier}: duplicate baseWeights {weights!r} collides in lookup")
        seen_weights.add(weights)

        # Each of these has a YOLO default the backend substitutes when omitted.
        for field, substituted in (
            ("framework", "ultralytics"),
            ("provider", "ultralytics"),
            ("supportedDatasetFormat", "yolo"),
        ):
            if not str(entry.get(field, "")).strip():
                problems.append(
                    f"{identifier}: missing {field}, backend substitutes {substituted!r}"
                )
        if not isinstance(entry.get("defaultTrainingParams"), dict):
            problems.append(
                f"{identifier}: defaultTrainingParams must be an object, or YOLO's are used"
            )

    missing = approved - seen_ids
    if missing:
        problems.append(f"approved keys absent from the registration: {sorted(missing)}")

    # An unresolvable default is worse than a missing one: the backend uses it whenever
    # `model_id` is absent, and an id outside the block raises ValueError before the Job
    # is created -- no result document, no callback, nothing to diagnose from.
    if DEFAULT_MODEL_ID not in seen_ids:
        problems.append(
            f"defaultFineTunableModelId {DEFAULT_MODEL_ID!r} is not one of the registered ids"
        )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate instead of printing")
    args = parser.parse_args()

    registry = ModelRegistry.load()
    block = build(registry)

    if args.check:
        problems = check(block, registry)
        if problems:
            for problem in problems:
                print(f"ERROR: {problem}", file=sys.stderr)
            return 1
        print(f"registration OK: {len(block)} entries, ids {[e['id'] for e in block]}")
        return 0

    print(json.dumps(
        {"fineTunableModels": block, "defaultFineTunableModelId": DEFAULT_MODEL_ID},
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
