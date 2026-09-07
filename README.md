# language-model-pipeline

Canonical contracts for the DIMER language-model fine-tuning capability.

The platform capability is **language-model fine-tuning** — not a Qwen finetuner or a Granite
finetuner. One implementation serves every compatible causal LM; adding a model should mean a
registry entry and, at most, a narrow backend hook.

## The three repositories

| Repo | Role |
|---|---|
| **language-model-pipeline** (this one) | Contracts, schemas, model registry, shared package, deployment/runtime documentation |
| `language-model-dataset-validator` | DIMER-facing `validate.py`; upload validation and dataset normalization |
| `language-model-finetuner` | DIMER-facing `train.py`; LoRA/QLoRA SFT, artifact and provenance generation, and the root `dimer-pipeline.json` |

## What lives here

```
src/lmpipeline/          the shared package both containers import
  errors.py              stable error-code namespace
  registry.py            model registry loading and resolution
  dimer.py               the DIMER container runtime contract
  result.py              the result.json contract
  datasets/resolver.py   safe archive handling and split resolution
  datasets/normalize.py  schema detection and canonical normalization
  data/model-registry.yaml   THE registry — single copy, no root duplicate
scripts/vendor_sync.py   vendor the package into consumers, with a drift gate
scripts/build_registration.py  generate the DIMER fineTunableModels block from the registry
DEPLOYMENT.md            the observed DIMER runtime contract, with evidence
MODEL_REGISTRATION.md    how a registry entry becomes selectable in DIMER (C-2)
DATASET_SPEC.md          the dataset contract
TRAINING_SPEC.md         methods, loss masking, bounds, metrics, lifecycle
ARTIFACT_SPEC.md         what a run publishes and what must hold first
PROVENANCE_SPEC.md       what is recorded, and why it is observed not requested
COMPATIBILITY.md         measured resource matrix
SECURITY.md              supply-chain and privacy rules
```

The registry deliberately has **no copy at the repo root**. One file, no drift.

## Standalone Colab tutorials

The `tutorials/` directory provides an educational, standalone path through the same
language-model SFT capability without requiring DIMER Workbench.

[![Open Fine-Tuning Tutorial In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/language-model-pipeline/blob/main/tutorials/language_model_finetuning_colab.ipynb)

[![Open Artifact Inference In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/language-model-pipeline/blob/main/tutorials/language_model_artifact_inference_colab.ipynb)

The fine-tuning notebook covers approved-model resolution, sample/BYOD validation,
tokenizer-aware checks, baseline generation, LoRA/QLoRA SFT, before/after generation,
new-prompt inference, adapter-first export and fresh reload. The inference notebook verifies
an exported artifact and reconstructs it from the exact pinned base revision without the
training dataset. See [`tutorials/README.md`](tutorials/README.md).

## Model selection

> **Corrected by the 2026-08-23 source-verified audit** (`COMPLIANCE.md`, normative). Two
> claims this section used to make are false: `dimer-pipeline.json` **is not read by
> anything** — parameters live in the pipeline registry and are set through the Builder UI
> (C-8) — and `DIMER_PREPROCESSING_ARGS_JSON` reaches only the **finetuner**; a validator
> Job receives four environment variables and that is not one of them (C-1).

DIMER's Pipeline Builder *Base Model* field is **not** delivered to Jobs — see DEPLOYMENT.md
§1 for the evidence. What the platform actually sends the finetuner is `model_id` in
`DIMER_HYPERPARAMETERS_JSON`, resolved against the pipeline's registered
`fineTunableModels`, plus the selected entry in `DIMER_MODEL_CONFIG_JSON`. Getting our
registry keys registered so that channel carries a real selection is open as #33 (C-2).
`dimer-pipeline.json` remains in the finetuner repo as a **transcription aid for filling in
the Builder UI** — documentation, not configuration.

Consequences that survive the correction:
- Arbitrary Hugging Face IDs are never user-supplied; only approved registry keys resolve.
- Where the platform *does* expose Base Model, the runtime asserts it agrees with the resolved
  entry and fails on conflict.
- The **validator** needs no selection at all: by the C-1 decision (validator #10, option 2)
  it is model-agnostic, and every tokenizer-specific check runs in the finetuner before
  weights load.

## Consuming the package

While these repositories are private, DIMER's build has no git credentials in the build
context (DEPLOYMENT.md §4), so the package is vendored:

```bash
python scripts/vendor_sync.py sync   --into src/_vendor
python scripts/vendor_sync.py verify --into src/_vendor    # CI gate; non-zero on drift
```

When this repo goes public, delete the vendored tree and add a pip requirement instead.
Import paths do not change.

## Base model weights

Base model weights are not committed to git (see `weights/` in `.gitignore`). Use `scripts/fetch_weights.py`
to download approved causal LM weights at pinned immutable revisions, with automatic `dimer-base-manifest.json`
generation and integrity verification:

```bash
# List all registered base models and status
python scripts/fetch_weights.py --list

# Download default base model (qwen3-1.7b) into weights/
python scripts/fetch_weights.py

# Download a specific registered model (e.g., smollm3-3b)
python scripts/fetch_weights.py --model smollm3-3b

# Check download files and sizes without downloading
python scripts/fetch_weights.py --model qwen3-1.7b --dry-run

# Verify snapshot integrity against dimer-base-manifest.json
python scripts/fetch_weights.py --verify-only --dest weights/
```

## Development

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e ".[dev]"
./.venv/Scripts/python.exe -m pytest
./.venv/Scripts/python.exe -m ruff check .
```

## End-to-end acceptance runs

Measured 2026-08-22. Each run is the real DIMER flow — validator container → finetuner
container — against a committed acceptance profile, not a synthetic fixture. Images were
rebuilt from merged `main` (validator `e92ed1e`, finetuner `402b2f4`).

Four models on tier 1, plus `granite-4.1-3b` across all three training tiers. Six runs, six
passes.

| Run | Model | Method | Tier | Examples | Train loss | Val loss | Peak VRAM | Train wall |
|---|---|---|---|---|---:|---:|---:|---:|
| r1 | `qwen3-0.6b` | LoRA | 1 dolly | 79 | 1.9148 | 1.8592 | 5.72 GB | 15.5s |
| r2 | `qwen3-1.7b` | QLoRA | 1 dolly | 79 | 1.7883 | 1.6412 | 4.46 GB | 83.9s |
| r3 | `qwen3-4b` | QLoRA | 1 dolly | 79 | 1.6501 | 1.4911 | 6.19 GB | 158.3s |
| r4 | `granite-4.1-3b` | QLoRA | 1 dolly | 79 | 1.6245 | 1.5199 | 4.69 GB | 146.9s |
| r5 | `granite-4.1-3b` | QLoRA | 2 tagalog | 179 | 0.2224 | 0.1648 | 3.79 GB | 296.3s |
| r6 | `granite-4.1-3b` | QLoRA | 3 multi | 400 | 0.2138 | 0.1966 | 4.37 GB | 717.5s |

`max_sequence_length=1024`, 1 epoch, batch 1, grad-accum 2, on an RTX 5070 Ti Laptop
(11.94 GB, sm_120). Example counts are post-split — the 0.2 validation fraction takes
100/220/500 down to 79/179/400.

What the six runs establish:

- Every model trains through the real dataset path on real data, `granite-4.1-3b` included.
  A non-Qwen vendor running the identical container with no model-specific handling is the
  model-agnostic claim demonstrated rather than asserted.
- Every acceptance tier works end to end, Tagalog and multilingual among them.
- Every run loaded its registry-pinned revision — `baseModelRevision` matched
  `baseModelRevisionExpected` in all six — so nothing silently fell back to `main`.
- Publication is sound: the clean-load smoke test ran, `os.replace` left no `.staging`
  residue, and adapters landed at mode 644 rather than the 0600 safetensors writes by
  default.

Two things these numbers do **not** mean:

- **Tier 2/3 losses are not "better" than tier 1.** They reflect task shape. UNER targets are
  short structured NER labels at roughly 14 supervised tokens per example against dolly's
  ~79 of free-form answer. Comparing loss across tiers measures the datasets, not the models.
- **They do not supersede `COMPATIBILITY.md`.** That matrix is deliberately measured at
  `seq=2048` padded to full length to capture the worst case a real dataset can reach; these
  ran at `seq=1024` on genuinely short examples. The lower peaks here are expected, and the
  registry's `min_vram_gb` ceilings stand unchanged.

Neither does any of this speak to DIMER's own hardware or mount topology — see issue #7, and
`language-model-finetuner#3` for a mount assumption this matrix surfaced.

### What is refused, and where

The fifth registry entry, `phi-4-mini-instruct`, is absent from the matrix by construction:
`microsoft/Phi-4-mini-instruct` carries the Hub's `custom_code` tag and its documented usage
requires `trust_remote_code=True`, so SECURITY.md holds it at `enabled: false`,
`approval_state: blocked`, `revision: null`. Verified 2026-08-22 that this is enforced rather
than merely declared. Three layers were tested then; the audit later established that the
first is decorative — `dimer-pipeline.json` is read by nothing (C-8) — so enforcement rests
on the two container layers, which is where it belonged all along:

| Layer | Behaviour with `model_key=phi-4-mini-instruct` |
|---|---|
| `dimer-pipeline.json` enum | `['qwen3-1.7b', 'qwen3-4b', 'granite-4.1-3b']` — **decorative**: the file is not read by the platform (C-8) |
| Validator container | exit 1, `MODEL_DISABLED`, structured result written |
| Finetuner container | exit 1, `MODEL_DISABLED`, structured result written |

Both containers raise from `ModelRegistry.resolve`, which runs **before** any weight fetch.
Once the C-1 model-agnostic change lands, the validator layer retires too — it will resolve
no model — leaving the finetuner as the sole and sufficient enforcement point.
Nothing is downloaded and no code path that could honour `trust_remote_code` is ever reached
— the gate is not a download-then-check. The model-registry schema encodes the same invariant
(`requires_trust_remote_code: true` implies the entry is disabled), so the registry cannot
drift into offering it without failing CI.

`qwen3-0.6b` is likewise absent from that enum. It is `internal_only` — the CI and
smoke-test model, reachable only when named directly, never from the Workbench form.

## Status

**All PRs merged.** Contracts, registry, shared package, specs, JSON Schemas, a measured
resource matrix, and the full `validation-datasets/` acceptance suite. 206 tests passing,
2 skipped.

The consumer repositories are correspondingly complete: the validator runs end to end in its
image against real tokenizers, and the finetuner trains, packages, reloads and publishes on
a real GPU — both now confirmed by the acceptance matrix above.

All five acceptance profiles carry committed approvals, so a fresh clone can verify each one
against a recorded expectation without rebuilding from network.

**Blocked:** all DIMER-side integration. Creating a `Custom / Other` pipeline currently fails
on a `runtime_dataset_format` not-null constraint — a platform defect with no known fix,
tracked in #5. See DEPLOYMENT.md §5. Nothing merged here depended on it; PR 0's runtime probe
and every on-platform acceptance step do.

The remaining platform items are sequenced **#13 → #5 → #11 → #7**: confirm whether the
Pipeline Builder's Base Model field accepts free text (which decides whether the tier
sentinel is viable at all), then the `runtime_dataset_format` blocker, then the documented
upload quota, then resource profiles re-measured on approved hardware. Each needs portal
access.
