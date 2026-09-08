# Model Registration Contract

How an approved model in `data/model-registry.yaml` becomes selectable in DIMER, and what
the platform does with each field once it is.

**This resolves `COMPLIANCE.md` C-2 as far as the repository can.** Everything below is
transcribed from `dimer-backend` source, not from the portal documentation:
`dimer/workbench/fine_tunable_models.py` and `dimer/workbench/pipeline_registry.py` on
`on-prem` @ `e06d1a9`, cross-read against `main` @ `7ff6124` and `testingjp1` @ `52e1d28`.

## The chain, end to end

```
data/model-registry.yaml              our source of truth, inside the image
  -> scripts/build_registration.py    generates the block below
    -> Pipeline Builder               pasted into schema.model.fineTunableModels
      -> user picks a model
        -> DIMER_HYPERPARAMETERS_JSON  {"model_id": "qwen3-1.7b", ...}
          -> DIMER_MODEL_CONFIG_JSON   a deepcopy of the matching registration entry
            -> ModelRegistry.resolve("qwen3-1.7b")   back to where we started
```

Two consequences follow from the shape of that chain.

**`id` must equal the registry key exactly.** `resolve_fine_tunable_model_config` looks the
selection up by `id` and hands the container that entry; the container then resolves the same
string against its own registry. A mismatch fails inside the container with
`MODEL_KEY_MISSING`, after the Job is already running.

**An unregistered `model_id` never reaches a container at all.** It raises
`ValueError: Unsupported fine-tunable model id` during Job creation — no result document, no
callback, nothing for the user to read. Approving a model in the registry without
re-registering it produces exactly that.

Never hand-write the block:

```bash
python scripts/build_registration.py            # paste this into the Builder
python scripts/build_registration.py --check    # CI gate; re-applies the backend's own rules
```

`tests/test_registration.py` runs `--check`'s logic on every commit, so a registry edit that
would strand a model fails here rather than on the platform.

## The three questions C-2 asked

### Is a non-YOLO family accepted?

**Yes, and nothing has to change on the platform.** `get_pipeline_fine_tunable_models`
iterates whatever the pipeline's `schema.model.fineTunableModels` holds and requires exactly
two things of an entry: a non-empty `id` and a non-empty `baseWeights`. There is no allow-list,
no family check, and no validation against the built-in YOLO catalog. The catalog is a
*fallback* used only when the configured list is missing or empty.

The trap is in the other direction. Every remaining field has a **YOLO default that is
substituted silently**:

| Field | Omitted → backend substitutes | Why that is wrong here |
|---|---|---|
| `framework` | `"ultralytics"` | mislabels the entry in provenance and in the UI |
| `provider` | `"ultralytics"` | same |
| `supportedDatasetFormat` | `"yolo"` (or `"yolo-seg"`) | the format is `jsonl_messages`; a wrong value can gate the upload |
| `defaultTrainingParams` | YOLO's `COMMON_TRAINING_DEFAULTS` | `learning_rate: 0.01` is 50x ours, plus `mosaic`, `degrees`, `fliplr` |

None of these errors. They arrive in `DIMER_MODEL_CONFIG_JSON` as though we had chosen them,
which is the same failure mode as C-8: a silent fallback is indistinguishable from a working
channel. The generator sets all four explicitly and the test suite asserts it.

An entry missing `id` or `baseWeights` is **skipped**, not rejected — so a typo removes one
model from the dropdown and reports nothing.

`defaultTrainingParams` also carries the training controls from `TRAINING_SPEC.md` —
`weight_decay`, `lr_scheduler_type`, `warmup_ratio`, `early_stopping_patience`,
`early_stopping_min_delta`, `restore_best_adapter`. Registering them is what makes the
regularization and schedule policy visible in the Builder rather than implicit in a library
default, which is the point of #54; every value is chosen to reproduce pre-control behaviour,
so registering them changes no run. A parameter the pipeline registry does not carry arrives
at the container as `{}` with no error (`COMPLIANCE.md` C-8), so a control absent from the
Builder form is a control the user cannot set — not one that falls back to the value here.

### What does `baseWeights` mean for a Hugging Face model at a pinned revision?

**It is an opaque identity string.** The backend requires it to be non-empty and otherwise
uses it for exactly one purpose: as the key of `model_ids_by_base_weights`, which translates
a legacy `base_model` hyperparameter back into a `model_id`. It is never parsed, never
resolved to a path, and never fetched — no code reads it as a filename despite every
built-in value being one.

So `Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` is a perfectly valid value, and
it is what the generator emits: carrying the pinned revision makes the registration
self-describing and lets an auditor confirm from the Builder alone that the platform and the
image agree on which commit is being trained.

Two constraints that do follow: values must be **unique across entries**, or the lookup map
silently remaps one model's legacy selector onto another's id; and `LEGACY_BASE_MODEL_TO_ID`
only maps YOLO `.pt` filenames, so a legacy `base_model` value never resolves for us. That is
harmless — the platform sends `model_id` and pops `base_model` — but it means the legacy
channel is not a fallback we have.

### What `taskType` is accepted?

**None of ours, and this cannot be fixed from this repository.** `_normalize_task_type` is a
closed function with three outcomes:

```python
{"segmentation", "instance_segmentation", "semantic_segmentation"} -> "segmentation"
{"classification", "image_classification"}                         -> "image_classification"
everything else                                                    -> "object_detection"
```

`language_model_sft` falls into *everything else*, so a language-model pipeline is reported by
the platform as `object_detection`. This is not a value we can pick our way out of; adding an
LM task type is a backend change.

**It is inert, and that is a verified claim rather than a hope.** The normalized value is used
in three places, and none of them reaches us:

1. `get_default_fine_tunable_models` — the fallback catalog, unused once our list validates.
2. `get_model_finetuning_hyperparameters_schema` — the default parameter schema, unused once
   parameters are registered (C-8).
3. `build_runtime_pipeline_metadata["taskType"]` — reaches the container in
   `DIMER_PIPELINE_METADATA_JSON`. Neither container reads it: both bake
   `task_type = "language_model_sft"`, precisely because `Custom / Other` resolves to generic
   platform metadata that says nothing true about the task.

The generator therefore sends the honest `language_model_sft` and lets the platform normalize
it. Recording the real intent costs nothing and is what a future backend change would be read
against; sending `object_detection` ourselves would bake in a lie that is currently only a
lossy translation.

## What the platform ignores

`resourceProfile` is **advisory**. `_build_fine_tuning_container_resources` builds the pod's
requests and limits from cluster-wide environment variables only —
`WORKBENCH_FINE_TUNING_CPU_REQUEST` / `_LIMIT`, `WORKBENCH_FINE_TUNING_MEMORY_REQUEST` /
`_LIMIT`, `WORKBENCH_FINE_TUNING_GPU_ENABLED`, `WORKBENCH_FINE_TUNING_GPU_COUNT` — and never
reads the registration.

That has a consequence worth stating plainly, because it is easy to assume otherwise:
**per-model sizing is not possible.** Registering `"gpuCount": 1` does not get you a GPU.
Whether the pod gets one is decided by `WORKBENCH_FINE_TUNING_GPU_ENABLED`, a single
cluster-wide flag that **defaults to `False`**, for every model in every pipeline. Selecting
a 4B model on a cluster where it is unset schedules a CPU pod, and LM SFT on CPU is not a
degraded run — it does not finish. See `COMPLIANCE.md` C-4.

The profile is still worth carrying: `resolve_fine_tunable_model_config` deepcopies the whole
entry into `DIMER_MODEL_CONFIG_JSON`, so the container receives the declared requirement and
can compare it against `DIMER_EXPECTED_ACCELERATOR` — which the backend sets to `"nvidia"` or
`"cpu"` from that same flag — and refuse early with a clear reason instead of dying in the
middle of training.

## `defaultFineTunableModelId`

Set it, and keep it inside the registered ids. Two roles:

- the Builder pre-selection, and the fallback the platform uses when `model_id` is absent
  from `DIMER_HYPERPARAMETERS_JSON`. A default outside the block raises the same
  `ValueError` at Job creation;
- it is placed in `DIMER_PIPELINE_METADATA_JSON`, which **does** reach the validator.

That second role is the mechanism that made option 1 in validator issue #10 viable — validate
against the pipeline's default. It was not chosen: the default is the pipeline's, not the
user's, so a user selecting anything else would be validated against the wrong tokenizer. The
validator is model-agnostic by the C-1 decision and ignores this field entirely. Recorded here
so the choice can be re-argued against the mechanism rather than against its absence.

## What is still open

Registering the block needs Pipeline Builder access, which means it is Kurt's to do, and it
cannot be exercised until `#5` clears — creating a `Custom / Other` pipeline currently fails
on a `runtime_dataset_format` not-null constraint.

`DimerEnv` now parses both `DIMER_MODEL_CONFIG_JSON` and `DIMER_EXPECTED_ACCELERATOR`, so the
chain at the top of this document is executable end to end rather than registered on one side
and unread on the other. Selection precedence is `hyperparameters.model_id`, then
`modelConfig.id`, then the legacy `datasetPreprocessing.model_key`; two channels that disagree
fail the Job before any weight is downloaded, and the error names both the channels and their
values, because a registry key is published in this very block and withholding it would only
cost the operator the diagnosis. Diagnostics expose `modelConfigKeys` — key names — and never
the registration's values, since fields added here later may carry deployment metadata.

That change moved `VENDOR_SHA`, and **both** consumers vendor `src/lmpipeline`. The finetuner
re-vendors because it reads the new fields; `language-model-dataset-validator` re-vendors
because its drift gate hashes the same tree, not because it uses any of this.
