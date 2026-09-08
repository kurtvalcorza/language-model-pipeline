"""The #54 training controls, at the layer this repository actually owns.

The controls *behave* in `language-model-finetuner`; what lives here is the contract they
have to satisfy — the job document's shape, the provenance a finished run reports, the
registered defaults, and the arithmetic both consumers must agree on. So these tests are
deliberately about shape and rules, and none of them trains anything.

Two properties are the ones worth having:

  * **The controls are optional in both directions.** A finetuner that has not gained them
    yet must keep emitting valid documents, and one that has must not be able to emit half
    a control. A schema that only ever accepted the new fields would not prove either.
  * **The layers agree.** The registered snake_case defaults are lowered into a real job
    document and validated against the schema, so the form layer and the document layer
    cannot drift into disagreeing about what a control is called or what "off" is.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from lmpipeline.errors import Code, Stage
from lmpipeline.registry import ModelRegistry
from lmpipeline.result import Result
from lmpipeline.schema import JOB, TRAINING_RESULT, SchemaError, load_schema, validate_document
from lmpipeline.training_controls import (
    CONTROL_DEFAULTS,
    LR_SCHEDULER_TYPES,
    STOPPING_REASONS,
    lower_controls,
    optimizer_steps_per_epoch,
    resolve_warmup_steps,
    total_optimizer_steps,
)

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "build_registration", ROOT / "scripts" / "build_registration.py"
)
build_registration = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_registration)


# -- fixtures ------------------------------------------------------------------


@pytest.fixture(scope="module")
def registry() -> ModelRegistry:
    return ModelRegistry.load()


def _job(registry: ModelRegistry, **training: Any) -> dict:
    """A job document as the finetuner emits one, pinned to a real registry entry.

    The model block comes from the committed registry rather than being invented, so the
    fixture cannot pass a revision pattern the real data would fail.
    """
    entry = registry.resolve("qwen3-1.7b")
    document = {
        "schemaVersion": "1.0",
        "model": {
            "key": entry.key,
            "modelId": entry.model_id,
            "revision": entry.revision,
            "backend": "transformers-peft",
        },
        "training": {
            "method": "qlora",
            "epochs": 3,
            "learningRate": 0.0002,
            "loraRank": 16,
            "loraAlpha": 32,
            "loraDropout": 0.05,
            "perDeviceBatchSize": 1,
            "gradientAccumulationSteps": 16,
            "effectiveBatchSize": 16,
            "seed": 42,
            "maxSequenceLength": 2048,
            "validationSplit": 0.1,
        },
        "device": "cuda:0",
        "runId": "run-0001",
        "sessionId": "session-0001",
    }
    document["training"].update(training)
    return document


def _training_success(controls: dict | None = None) -> dict:
    provenance = {
        "baseModel": "Qwen/Qwen3-1.7B",
        "baseModelRevision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "modelKey": "qwen3-1.7b",
        "quantized": True,
        "datasetDigest": "b" * 64,
        "trustRemoteCode": False,
    }
    if controls is not None:
        provenance["trainingControls"] = controls
    return Result.success(
        stage=Stage.TRAINING,
        message="Training complete.",
        code=Code.TRAINING_SUCCEEDED,
        artifact={
            "format": "peft_adapter",
            "path": "/data/output/model",
            "fileCount": 1,
            "totalBytes": 1234,
            "files": [
                {"path": "adapter_model.safetensors", "bytes": 1234, "sha256": "a" * 64}
            ],
        },
        provenance=provenance,
    ).to_dict()


def _controls(**overrides: Any) -> dict:
    complete = {
        "stoppingReason": "early_stopping",
        "epochsCompleted": 2,
        "bestEpoch": 1,
        "bestValidationLoss": 1.5,
        "restoredBestAdapter": True,
        "lrSchedulerType": "cosine",
        "warmupSteps": 3,
        "weightDecay": 0.01,
    }
    complete.update(overrides)
    return complete


# -- the job document ----------------------------------------------------------


def test_a_job_without_any_control_still_validates(registry):
    """Forward compatibility, and the reason the new fields are optional.

    The finetuner emits none of them today. Requiring one would break the consumer's own
    schema gate the moment this repository's contract was vendored in — the contract has to
    be able to land before the runtime does.
    """
    validate_document(JOB, _job(registry))


def test_a_job_using_every_control_validates(registry):
    validate_document(JOB, _job(
        registry,
        weightDecay=0.0,
        lrSchedulerType="cosine",
        warmupRatio=0.03,
        earlyStopping={"patience": 2, "minDelta": 0.001, "restoreBestAdapter": True},
    ))


@pytest.mark.parametrize("field, value", [
    ("weightDecay", -0.01),
    ("weightDecay", 1.5),
    ("weightDecay", "0.01"),
    ("warmupSteps", -1),
    ("warmupSteps", 1.5),
])
def test_an_out_of_range_control_is_refused(registry, field, value):
    with pytest.raises(SchemaError) as exc:
        validate_document(JOB, _job(registry, lrSchedulerType="linear", **{field: value}))
    assert field in str(exc.value)


def test_weight_decay_of_zero_stays_expressible(registry):
    """0.0 is a legitimate choice for LoRA SFT, where dropout already regularizes.

    An exclusive lower bound here would have made the honest value unsayable and pushed
    people to 1e-8 instead.
    """
    validate_document(JOB, _job(registry, weightDecay=0.0))


@pytest.mark.parametrize("scheduler", LR_SCHEDULER_TYPES)
def test_every_declared_schedule_is_accepted(registry, scheduler):
    validate_document(JOB, _job(registry, lrSchedulerType=scheduler))


@pytest.mark.parametrize("scheduler", ["polynomial", "constant_with_warmup", "", "CONSTANT"])
def test_an_undeclared_schedule_is_refused(registry, scheduler):
    with pytest.raises(SchemaError):
        validate_document(JOB, _job(registry, lrSchedulerType=scheduler))


@pytest.mark.parametrize("ratio", [1.0, 1.5, -0.1])
def test_a_warmup_ratio_outside_the_run_is_refused(registry, ratio):
    """1.0 is refused, not merely clamped: a warmup over every step leaves no step for the
    schedule that follows, so the document would declare a schedule that never takes effect.
    """
    with pytest.raises(SchemaError):
        validate_document(JOB, _job(registry, lrSchedulerType="linear", warmupRatio=ratio))


def test_warmup_may_not_be_spelled_both_ways_at_once(registry):
    """The ambiguity is silent — whichever the runtime reads wins and the document still
    looks well-formed — which is exactly why the schema has to refuse it."""
    with pytest.raises(SchemaError):
        validate_document(JOB, _job(
            registry, lrSchedulerType="linear", warmupRatio=0.03, warmupSteps=10,
        ))


@pytest.mark.parametrize("warmup", [{"warmupRatio": 0.03}, {"warmupSteps": 10}])
def test_a_warmup_without_a_declared_schedule_is_refused(registry, warmup):
    with pytest.raises(SchemaError):
        validate_document(JOB, _job(registry, **warmup))


@pytest.mark.parametrize("dropped", ["patience", "minDelta", "restoreBestAdapter"])
def test_early_stopping_must_be_stated_completely(registry, dropped):
    """A half-stated control is worse than none: the missing half gets a default nobody
    recorded, and the document looks like it described the run."""
    block = {"patience": 2, "minDelta": 0.001, "restoreBestAdapter": True}
    del block[dropped]
    with pytest.raises(SchemaError) as exc:
        validate_document(JOB, _job(registry, earlyStopping=block))
    assert dropped in str(exc.value)


def test_a_patience_of_zero_is_refused_because_absence_already_means_off(registry):
    """The single-representation rule, enforced rather than documented.

    Absence is how a job document says early stopping is off. Accepting `patience: 0` as a
    second spelling is how a run ends up stopping for a reason nobody selected.
    """
    with pytest.raises(SchemaError):
        validate_document(JOB, _job(
            registry,
            earlyStopping={"patience": 0, "minDelta": 0.0, "restoreBestAdapter": False},
        ))


def test_an_unknown_field_inside_early_stopping_is_refused(registry):
    with pytest.raises(SchemaError):
        validate_document(JOB, _job(registry, earlyStopping={
            "patience": 2, "minDelta": 0.001, "restoreBestAdapter": True,
            "monitor": "train_loss",
        }))


def test_a_misspelled_control_is_still_rejected(registry):
    """`additionalProperties: false` has to survive the new `not`/`dependentRequired`
    keywords sitting beside it — a typo that is silently ignored is a control that silently
    does not apply."""
    with pytest.raises(SchemaError):
        validate_document(JOB, _job(registry, weightDeacy=0.01))


# -- what the run reports ------------------------------------------------------


def test_a_training_success_without_the_controls_block_still_validates():
    validate_document(TRAINING_RESULT, _training_success())


@pytest.mark.parametrize("restored", [True, False])
def test_a_training_success_reporting_the_controls_validates(restored):
    validate_document(TRAINING_RESULT, _training_success(
        _controls(restoredBestAdapter=restored)
    ))


@pytest.mark.parametrize("reason", STOPPING_REASONS)
def test_every_declared_stopping_reason_is_accepted(reason):
    validate_document(TRAINING_RESULT, _training_success(_controls(stoppingReason=reason)))


@pytest.mark.parametrize("reason", ["cancelled", "oom", "converged", ""])
def test_an_undeclared_stopping_reason_is_refused(reason):
    """A failure never reaches the success branch, so a stopping reason that describes one
    would be a claim the document cannot support."""
    with pytest.raises(SchemaError):
        validate_document(TRAINING_RESULT, _training_success(_controls(stoppingReason=reason)))


@pytest.mark.parametrize("dropped", [
    "stoppingReason", "epochsCompleted", "bestEpoch", "bestValidationLoss",
    "restoredBestAdapter", "lrSchedulerType", "warmupSteps", "weightDecay",
])
def test_the_controls_block_must_be_complete_once_it_is_present(dropped):
    block = _controls()
    del block[dropped]
    with pytest.raises(SchemaError) as exc:
        validate_document(TRAINING_RESULT, _training_success(block))
    assert dropped in str(exc.value)


def test_epochs_completed_is_what_makes_early_stopping_falsifiable():
    """Named explicitly because it is the field a future refactor would judge redundant.

    Without it, `early_stopping` cannot be checked against anything: there is no way to see
    that the run ended short of the epoch budget it was given.
    """
    assert "epochsCompleted" in _controls()
    block = _controls(stoppingReason="early_stopping")
    del block["epochsCompleted"]
    with pytest.raises(SchemaError):
        validate_document(TRAINING_RESULT, _training_success(block))


def test_a_warmup_ratio_is_not_carried_into_provenance():
    """Provenance records the EFFECTIVE warmup in steps. Carrying the ratio instead would
    leave each reader to divide it by a step count they derived themselves, which is how one
    run acquires two warmup lengths."""
    with pytest.raises(SchemaError):
        validate_document(TRAINING_RESULT, _training_success(
            _controls(warmupRatio=0.03)
        ))


# -- the vocabularies, checked against the schemas ------------------------------


def _find_subschema(node: Any, key: str) -> dict:
    """The training-result block is nested inside an `if`/`then`, so find it by name."""
    if isinstance(node, dict):
        if key in node and isinstance(node[key], dict):
            return node[key]
        for value in node.values():
            found = _find_subschema(value, key)
            if found:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_subschema(item, key)
            if found:
                return found
    return {}


def test_the_schedule_vocabulary_is_the_same_in_code_and_in_both_schemas():
    """Three copies of one enum is three chances to add a schedule to two of them."""
    job_enum = load_schema(JOB)["properties"]["training"]["properties"]["lrSchedulerType"]
    controls = _find_subschema(load_schema(TRAINING_RESULT), "trainingControls")
    assert tuple(job_enum["enum"]) == LR_SCHEDULER_TYPES
    assert tuple(controls["properties"]["lrSchedulerType"]["enum"]) == LR_SCHEDULER_TYPES


def test_the_stopping_vocabulary_is_the_same_in_code_and_in_the_schema():
    controls = _find_subschema(load_schema(TRAINING_RESULT), "trainingControls")
    assert tuple(controls["properties"]["stoppingReason"]["enum"]) == STOPPING_REASONS


# -- the arithmetic both consumers have to agree on ----------------------------


def test_a_partial_final_accumulation_window_counts_as_an_optimizer_step():
    """TRAINING_SPEC flushes it, so the scheduler has to count it.

    17 examples at an effective batch of 16 is two updates, not one. Flooring would leave
    the schedule one step short on every epoch that does not divide evenly.
    """
    assert optimizer_steps_per_epoch(
        17, per_device_batch_size=1, gradient_accumulation_steps=16
    ) == 2
    assert optimizer_steps_per_epoch(
        32, per_device_batch_size=2, gradient_accumulation_steps=8
    ) == 2


def test_each_epoch_flushes_its_own_window():
    """Per-epoch then multiplied, not computed over the concatenated run.

    17 examples for 3 epochs is 6 updates. Treating the run as 51 rows would give 4, and the
    difference is where a warmup ends.
    """
    assert total_optimizer_steps(
        17, per_device_batch_size=1, gradient_accumulation_steps=16, epochs=3
    ) == 6


def test_warmup_is_counted_in_optimizer_steps_not_micro_batches():
    """32 examples at 1x16 is 32 micro-batches but only 2 optimizer updates. A half-run
    warmup is therefore 1 step — a scheduler stepping per micro-batch would warm up for 16
    and spend the whole run doing it."""
    steps = total_optimizer_steps(
        32, per_device_batch_size=1, gradient_accumulation_steps=16, epochs=1
    )
    assert steps == 2
    assert resolve_warmup_steps(steps, warmup_ratio=0.5) == 1


def test_a_warmup_ratio_rounds_up_but_never_outlasts_the_run():
    assert resolve_warmup_steps(6, warmup_ratio=0.03) == 1
    assert resolve_warmup_steps(6, warmup_ratio=0.0) == 0
    assert resolve_warmup_steps(6, warmup_steps=99) == 6


def test_resolving_a_warmup_refuses_both_spellings_and_bad_input():
    with pytest.raises(ValueError):
        resolve_warmup_steps(6, warmup_ratio=0.1, warmup_steps=2)
    with pytest.raises(ValueError):
        resolve_warmup_steps(6, warmup_ratio=1.0)
    with pytest.raises(ValueError):
        resolve_warmup_steps(6, warmup_steps=-1)
    with pytest.raises(ValueError):
        resolve_warmup_steps(0, warmup_ratio=0.1)


def test_step_counts_reject_impossible_input():
    with pytest.raises(ValueError):
        optimizer_steps_per_epoch(0, per_device_batch_size=1, gradient_accumulation_steps=1)
    with pytest.raises(ValueError):
        optimizer_steps_per_epoch(8, per_device_batch_size=0, gradient_accumulation_steps=1)
    with pytest.raises(ValueError):
        total_optimizer_steps(
            8, per_device_batch_size=1, gradient_accumulation_steps=1, epochs=0
        )


# -- lowering the form layer into the document layer ---------------------------


def test_the_registered_defaults_lower_to_no_early_stopping_block():
    """`early_stopping_patience: 0` is how the form says off; the document says off by
    omission. This is the whole point of having a lowering function."""
    fragment = lower_controls(CONTROL_DEFAULTS)
    assert "earlyStopping" not in fragment
    assert fragment["weightDecay"] == 0.01
    assert fragment["lrSchedulerType"] == "constant"
    assert fragment["warmupRatio"] == 0.0


def test_enabling_patience_lowers_to_a_complete_block():
    fragment = lower_controls({
        **CONTROL_DEFAULTS,
        "early_stopping_patience": 2,
        "early_stopping_min_delta": 0.001,
        "restore_best_adapter": True,
    })
    assert fragment["earlyStopping"] == {
        "patience": 2, "minDelta": 0.001, "restoreBestAdapter": True,
    }


def test_lowering_leaves_an_absent_parameter_absent():
    """A control the pipeline registry does not carry arrives as nothing at all (C-8).
    Defaulting it here would present a value the user never had the chance to set as though
    they had chosen it."""
    assert lower_controls({}) == {}
    assert lower_controls({"weight_decay": 0.0}) == {"weightDecay": 0.0}


def test_lowering_refuses_what_the_schema_would_refuse():
    with pytest.raises(ValueError):
        lower_controls({"lr_scheduler_type": "linear", "warmup_ratio": 0.1, "warmup_steps": 2})
    with pytest.raises(ValueError):
        lower_controls({"warmup_ratio": 0.1})
    with pytest.raises(ValueError):
        lower_controls({"early_stopping_patience": -1})


def test_every_registered_default_lowers_into_a_valid_job_document(registry):
    """The cross-layer gate, and the one that would catch a real mistake.

    The registration is what a human pastes into the Pipeline Builder; the job document is
    what the container emits from what comes back. If the two layers disagree about a
    control's name, its bounds, or how off is spelled, this fails — and nothing else in
    either repository would notice until a Job had already been scheduled.
    """
    for entry in build_registration.build(registry):
        params = entry["defaultTrainingParams"]
        validate_document(JOB, _job(registry, **lower_controls(params)))


def test_the_registration_carries_the_shared_defaults_rather_than_its_own(registry):
    """Derived, not transcribed — the same rule the model block already follows."""
    for entry in build_registration.build(registry):
        params = entry["defaultTrainingParams"]
        for name, value in CONTROL_DEFAULTS.items():
            assert params[name] == value, f"{entry['id']} disagrees about {name}"


def test_enabling_early_stopping_through_the_form_also_lowers_into_a_valid_document(registry):
    """The other half of the cross-layer check: off is not the only value that has to work."""
    params = {
        **CONTROL_DEFAULTS,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "early_stopping_patience": 2,
        "early_stopping_min_delta": 0.001,
        "restore_best_adapter": True,
    }
    validate_document(JOB, _job(registry, **lower_controls(params)))
