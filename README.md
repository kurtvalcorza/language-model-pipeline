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
DEPLOYMENT.md            the observed DIMER runtime contract, with evidence
DATASET_SPEC.md          the dataset contract
SECURITY.md              supply-chain and privacy rules
```

The registry deliberately has **no copy at the repo root**. One file, no drift.

## Model selection

DIMER's Pipeline Builder *Base Model* field is **not** delivered to Jobs — see DEPLOYMENT.md
§1 for the evidence. The authoritative runtime selector is `model_key`, declared once in the
finetuner's `dimer-pipeline.json` under `datasetPreprocessing` as an `enum` of internal
registry keys, and delivered through `DIMER_PREPROCESSING_ARGS_JSON` — the only user-parameter
channel proven to reach both containers.

Consequences:
- Arbitrary Hugging Face IDs are never user-supplied; the enum holds approved keys only.
- Where the platform *does* expose Base Model, the runtime asserts it agrees with the resolved
  entry and fails on conflict.
- DIMER registrations map to **GPU/resource tiers**, not to individual models.

## Consuming the package

While these repositories are private, DIMER's build has no git credentials in the build
context (DEPLOYMENT.md §4), so the package is vendored:

```bash
python scripts/vendor_sync.py sync   --into src/_vendor
python scripts/vendor_sync.py verify --into src/_vendor    # CI gate; non-zero on drift
```

When this repo goes public, delete the vendored tree and add a pip requirement instead.
Import paths do not change.

## Development

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e ".[dev]"
./.venv/Scripts/python.exe -m pytest
./.venv/Scripts/python.exe -m ruff check .
```

## Status

**PR 1 of 8.** Contracts, registry, shared package, and tests. 55 tests passing.

Not yet written, and deliberately deferred to the PR that gives them meaning:
`TRAINING_SPEC.md`, `ARTIFACT_SPEC.md`, `PROVENANCE_SPEC.md` (PR 3), JSON Schemas for
job/result documents, and the DIMER registration matrix (PR 4).

**Blocked:** all DIMER-side integration. Creating a `Custom / Other` pipeline currently fails
on a `runtime_dataset_format` not-null constraint — a platform defect with no known fix or
tracking issue. See DEPLOYMENT.md §5. Nothing in PRs 1–4 depends on it; PR 0's runtime probe
and every acceptance step do.
