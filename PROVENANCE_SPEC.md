# Provenance Contract

`provenance.json` records what a run **actually did**, not what it was asked to do. Every
field is observed after the fact where that is possible.

## Recorded

| Field | Notes |
|---|---|
| `baseModel`, `baseModelRevision` | The immutable revision actually loaded |
| `baseModelRevisionExpected` | What the registry pinned; a mismatch fails the run |
| `baseModelLicense` | Carried from the registry for downstream attribution |
| `modelKey`, `backend` | Internal registry identity |
| `loadedDtype`, `quantized` | What the model was really loaded as, not what was requested |
| `loraTargetModules` | Resolved against the loaded architecture, not assumed |
| `trustRemoteCode` | Always `false` |
| `datasetDigest` | SHA-256 over the resolved split files, name-ordered |
| `job` | The complete resolved configuration |
| `packageVersions` | torch, transformers, peft, accelerate, tokenizers, bitsandbytes |
| `platform` | System and machine |
| `dimerBaseModel` | DIMER's Base Model field when the platform supplies it |

## Why "actually loaded" matters

A pinned revision that silently resolved to something else would make every downstream
claim false. `backends.load_base_model` reads the commit hash off the loaded config and
fails with `MODEL_REVISION_MISMATCH` if it disagrees with the registry. Provenance therefore
reports a verified fact, not the request that produced it.

The same reasoning applies to `loraTargetModules`: they are discovered by inspecting the
loaded model's modules, so an architecture that does not expose the expected projections
fails loudly rather than training an adapter attached to nothing.

## Base Model reconciliation

DIMER's Pipeline Builder *Base Model* field is display metadata and is not reliably
delivered to Jobs — see `DEPLOYMENT.md`. The authoritative selector is `model_key`.

When the platform does supply Base Model, it is cross-checked against the resolved registry
entry and a disagreement **fails** with `MODEL_BASE_MODEL_CONFLICT`. Both values are
recorded so an auditor can see which was authoritative.

## Reproducibility

Exact package versions plus the pinned revision plus the dataset digest plus the resolved
job configuration are together enough to re-run a job and expect the same result, subject
to GPU non-determinism — **within one image**.

**Across a rebuild, no such claim is made, because none is possible** (`COMPLIANCE.md`
C-12): the platform build pushes `:latest`, the pipeline stores a bare repository URI with
no tag, and Jobs run with `imagePullPolicy: Always`. A rebuild silently changes the image
every future run uses, and nothing in the platform can pin or roll it back. Provenance
therefore makes a completed run **auditable** — every version, revision and digest that
produced it is recorded — but cannot make two runs separated by a rebuild identical. The
pinned base image in the Dockerfile narrows the drift; it does not close it, because the
application layer is rebuilt on top.

**Bitwise GPU determinism is not claimed** and should not be inferred. Seeds cover data
shuffling, the derived validation split, and adapter initialization.

## Privacy

Provenance carries digests and configuration, never dataset content. `datasetDigest`
identifies a dataset without revealing any of it.
