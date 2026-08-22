# DIMER Deployment & Runtime Contract

The observed DIMER platform contract, with each claim's evidence. Anything marked
**unverified** has not been confirmed against a running Job and must be re-checked by the
PR 0 probe before it is relied on.

Sources:
- **Portal** — DIMER AI Engineer documentation, `/developers/ai-engineer/docs/`, read
  2026-08-21. Authoritative documentation.
- **Backend** — `NAIRA-SEU/intern-backend` @ `dd66b8e` (2026-04-20). **Four months stale**;
  corroboration only, never sole evidence.
- **Shipped** — the `mitra-classifier-*` trio, which passed DIMER GPU acceptance.

---

## 1. Injected environment

Confirmed from Portal. Every variable the platform injects:

| Variable | Example | Purpose |
|---|---|---|
| `DIMER_DATASET_DIR` | `/data/dataset` | Mounted dataset location |
| `DIMER_RESULT_PATH` | `/data/.../result.json` | Where to write `result.json` |
| `DIMER_DONE_CALLBACK` | backend URL | POST here when finished |
| `DIMER_PIPELINE_METADATA_JSON` | `{"taskType":"..."}` | Pipeline config |
| `DIMER_TRAIN_DEVICE` | `cuda:0` or `cpu` | Training device |
| `DIMER_HYPERPARAMETERS_JSON` | `{"epochs":10}` | User-selected hyperparameters |
| `DIMER_PREPROCESSING_ARGS_JSON` | `{"image_size":224}` | Preprocessing settings |
| `DIMER_SESSION_ID` | UUID | Workbench session |
| `DIMER_RUN_ID` | UUID | This run |
| `DIMER_OUTPUT_DIR` | `/data/output/...` | Where to save model weights |

`DIMER_TASK_TYPE` is **not** injected by the platform. It is baked into our images as
`language_model_sft`, because DIMER's `Custom / Other` task resolves to generic platform
metadata that says nothing true about the LM task.

### Base Model does not reach the container

**This is the fact the whole architecture turns on.** There is no documented mechanism
delivering the Pipeline Builder's *Base Model* field to a Job:

- Portal's environment table has no Base Model entry; `DIMER_PIPELINE_METADATA_JSON` is
  documented only as `{"taskType":"..."}`.
- Backend injects exactly three variables into validator Jobs (`domain.py:464`):
  `DIMER_RESULT_PATH`, `DIMER_DATASET_DIR`, `DIMER_DONE_CALLBACK`.
- Shipped Mitra reads `DIMER_PIPELINE_METADATA_JSON` defensively (`{}` default) and uses it
  only for `taskType`/`supportedDatasetFormat`. It never needed Base Model because it bakes
  one model per container — a route unavailable to a repo shared across models.

**Therefore:** `model_key` is the authoritative runtime selector, declared under
`datasetPreprocessing` in the finetuner's `dimer-pipeline.json` and delivered via
`DIMER_PREPROCESSING_ARGS_JSON`. DIMER's Base Model field is display/provenance metadata.
When the platform does supply it, `ModelRegistry.reconcile_base_model` asserts agreement and
fails on conflict rather than training the wrong model.

### What to enter in Base Model for a GPU-tier registration

The field is mandatory in the Pipeline Builder, so "leave it blank" is not available. The
rule is: **a tier registration's Base Model names the tier, never a model.**

```
Base Model:  lm-sft-12gb-qlora
```

The `lm-sft-` prefix is recognized by `lmpipeline.registry.is_tier_sentinel`, and
`tier_sentinel("12gb-qlora")` builds the value. Construct one per registered GPU tier.

Why not name the real model, which looks more honest: because nothing can keep it honest.
`model_key` chooses the model that actually trains, and Base Model never reaches the Job, so
a registration reading `Qwen/Qwen3-1.7B` can describe a run that trained Granite with no
mechanism able to detect the divergence — not `reconcile_base_model`, which only fires if
the platform delivers the field, and not the operator, who sees a plausible row. Registering
one pipeline per model does not fix this either: `dimer-pipeline.json` is per **image**, not
per registration, so its `model_key` enum still offers every model regardless of which
registration the user started from. Only one image per model would close it, and that gives
up the model-agnostic design this repo exists to provide.

A sentinel cannot be falsified, because it asserts nothing about the model. The model that
actually ran is recorded where it is machine-verifiable and hash-covered:

| Where | Field |
|---|---|
| `result.json` | `metadata.languageModelPipeline.model` (id + pinned revision) |
| `artifact-manifest.json` | every published file, by SHA-256 |
| `MODEL_CARD.md` | model id, revision, method, measured metrics |

`reconcile_base_model` treats a sentinel as "no claim" and returns it unchanged, so runs keep
working if DIMER later starts delivering the field. A **real** model id that disagrees with
the resolved entry still fails with `MODEL_BASE_MODEL_CONFLICT` — the sentinel path is not a
hole that swallows genuine conflicts.

Why `datasetPreprocessing` and not `modelFinetuning`: it is the only user-parameter channel
proven to reach **both** containers. Shipped Mitra's validator consumes
`DIMER_PREPROCESSING_ARGS_JSON` (`validator.py:245`) for `target_column`, which is declared
in the finetuner's manifest — and that pipeline passed acceptance.

---

## 2. The done callback is mandatory

Confirmed from Portal, which names the failure explicitly:

> **Callback timeout / UI stuck at "Validating..."** — Validator crashed without calling
> `DIMER_DONE_CALLBACK`. Fix: wrap entire `run()` in try/except. Always call
> `notify_done_callback()` in `finally` block.

`lmpipeline.dimer.notify_done_callback` never raises and never logs the URL — it is a signed
token. Call it from `finally`, after `write_result`.

---

## 3. Result contract

`result.json` at `DIMER_RESULT_PATH`. The platform's documented shape is the **outer**
object; our richer fields nest under `metadata.languageModelPipeline` so nothing Workbench
renders is displaced. See `lmpipeline.result`.

`metadata.classNames` is mandatory — the Portal names "Validation passed but no classNames"
as its own failure mode. SFT has no classes, so we emit `[]`, which the Portal explicitly
permits.

---

## 4. Build system

Confirmed from Portal, corroborated by Backend (`build_system/domain.py:181`).

```
Click Build -> download repo via GitHub API -> upload zip to S3
            -> CodeBuild: docker build -t repo:latest . -> push to ECR
```

Consequences that constrain this project:

- **No `--build-arg`.** There is no per-registration build-time differentiation. A single
  repo produces identical images for every registration using it — which is precisely why
  model selection must be a runtime parameter.
- **No git credentials inside the build context.** The GitHub App token authenticates the
  *source download*, not the build. While these repos are private, a
  `pip install git+https://...` line against `language-model-pipeline` **will fail**. The
  shared package is therefore vendored — see `scripts/vendor_sync.py`. When
  `language-model-pipeline` goes public, replace the vendored tree with a pip requirement;
  the import path does not change.
- **15-minute build timeout.** Documented cause: large model downloads or heavy
  compilation. Documented mitigation: a pre-built base image. The finetuner uses
  `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` (cu128 ships sm_120 for Blackwell). Do not
  re-pin torch in `requirements.txt` — pip will replace the cu128 build and break sm_120.
- Repo access order: GitHub App token, then platform token, then public.

---

## 5. Known platform blocker

**Creating a `Custom / Other` pipeline fails** before image build:

```
null value in column "runtime_dataset_format" of relation "workbench_pipeline"
violates not-null constraint
```

Expected value is `runtime_dataset_format=custom`. Observed 2026-08-21. This is a DIMER
backend defect, not a repository defect — do not distort this implementation to work around
it. It blocks the PR 0 runtime probe and all DIMER-side acceptance.

**Tracked as #5.** No upstream fix is known; the issue records the blocker and is where any
platform-side movement should land.

---

## 6. Still unverified — PR 0 must confirm

- Whether `DIMER_PIPELINE_METADATA_JSON` carries anything beyond `taskType`. If it does
  carry Base Model, `reconcile_base_model` starts enforcing and the model-selection design
  could be revisited.
- Actual values and mount lifetimes of `DIMER_OUTPUT_DIR`, and whether artifacts persist
  past Job completion.
- Whether preprocessing args reach the validator for a `Custom / Other` pipeline
  specifically (proven for Mitra's tabular pipeline; assumed identical here).
- Outbound Hugging Face network access and authentication behaviour from inside a Job.
- SIGTERM/cancellation semantics for long-running GPU Jobs.

**Probe safety:** the probe must allowlist environment keys, never dump `os.environ`.
`DIMER_DONE_CALLBACK` is a signed URL. `lmpipeline.dimer.DimerEnv.diagnostics()` implements
the allowlist and redacts the callback; use it rather than writing a new dump.

---

## 7. Where this document supersedes the blueprint (#1)

The blueprint issue #1 is the implementation handoff, and four of its architecture rules
were **overturned by runtime evidence** after it was written. They are listed here rather
than left to be discovered, because the failure mode is specific and expensive: a future
session reads #1, finds this repository doing the opposite, and "corrects" the working
architecture back to an assumption the platform has already falsified. That is a real risk
for an agent-assisted repository, where the blueprint reads as an instruction rather than
as a historical position.

This document is normative on every point below. #1 remains the record of what was
originally designed and why, which is worth keeping — the decisions were sound given what
was known then.

| Blueprint rule (#1) | Superseded by | Evidence |
|---|---|---|
| DIMER registrations are model-specific, one per model | Registrations are **resource-tier** registrations using a Base Model sentinel (`lm-sft-12gb-qlora`) | §1, *What to enter in Base Model*. `dimer-pipeline.json` is per **image**, not per registration, so a per-model registration cannot constrain which model runs anyway |
| Base Model is authoritative for model selection | **`model_key` is the authoritative runtime selector**, declared under `datasetPreprocessing` | §1, *Base Model does not reach the container*. Portal's environment table has no Base Model entry; backend injects three variables into validator Jobs |
| Do **not** expose `model_key` in `dimer-pipeline.json` | `model_key` is exposed there, as an enum of registry keys | Same. It is the only user-parameter channel proven to reach both containers, via `DIMER_PREPROCESSING_ARGS_JSON` |
| Base Model must reach the validator before multi-model validation proceeds | Model provenance is **recorded in result/artifact/model-card output** rather than trusted from the registration row | §1, provenance table. A registration row cannot be kept honest; a hash-covered artifact can |

Two things this supersession does **not** do:

- It does not close **#13**, which asks whether the Pipeline Builder accepts a free-text
  Base Model value at all. The sentinel scheme depends on that and it is unconfirmed — the
  sentinel is a working assumption, not a proven mechanism.
- It does not mark §6 verified. Those items remain open, and most stay blocked behind #5.

