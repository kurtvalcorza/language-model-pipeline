# Language Model Tutorials

These notebooks are the user-facing executable references for the DIMER language-model
fine-tuning capability. They are governed by **DIMER Notebook Specification 1.0** and execute
the production `language-model-finetuner` modules rather than maintaining notebook-specific
training or serving implementations.

## Notebook registry

| Notebook | Profile | Capability | Release status |
|---|---|---|---|
| [`language_model_finetuning_colab.ipynb`](language_model_finetuning_colab.ipynb) | `E2E` | Production-path QLoRA/PEFT SFT, evaluation, new-input inference, artifact staging and fresh reconstruction | **Candidate — clean-runtime verification required** |
| [`language_model_artifact_inference_colab.ipynb`](language_model_artifact_inference_colab.ipynb) | `ARTIFACT-INFERENCE` | Strict external-artifact validation, production reconstruction, new-input generation and machine-readable export | **Candidate — clean-runtime verification required** |

Each profile is also recorded under `metadata.dimer.notebook_profile`, with
`metadata.dimer.notebook_spec_version = "1.0"`. Clean-runtime release evidence is tracked in
[`RELEASE_VERIFICATION.md`](RELEASE_VERIFICATION.md).

## Runtime and private-source prerequisite

The release profile is **Google Colab with a CUDA GPU**. `language-model-finetuner` is a
private production repository, so an authorized user must create a Colab Secret named
`GITHUB_TOKEN`, grant the notebook access to that secret, and use a token with read access to
the repository. Plain Jupyter may provide the same secret as the `GITHUB_TOKEN` environment
variable, but release verification is performed in the supported Colab profile.

The token is read at runtime and supplied to Git through an ephemeral `extraHeader`. It is not
printed, embedded in the clone URL, or persisted in `.git/config`. The notebook deletes its
local token binding immediately after the immutable finetuner checkout is established.

## Open in Colab

[![Open Fine-Tuning Tutorial In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/language-model-pipeline/blob/main/tutorials/language_model_finetuning_colab.ipynb)

[![Open Artifact Inference In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/language-model-pipeline/blob/main/tutorials/language_model_artifact_inference_colab.ipynb)

The badges target `main`. Release verification must execute the exact candidate commit being
reviewed, not the moving `main` branch.

## Architecture: one production implementation

- `language-model-pipeline` owns shared DIMER contracts, the canonical model registry,
  dataset normalization/resolution primitives, runtime identity checks, secure tutorial
  bootstrap, and hostile external-ZIP validation.
- `language-model-finetuner` owns model loading, masking, training, artifact
  staging/publication, reconstruction, and generation.
- Production publication verification calls the same `finetuner.inference` reconstruction and
  generation surfaces used by the notebooks. The compatibility wrapper under
  `finetuner.artifacts` delegates to that shared path.
- `lmpipeline.tutorial_api` and `lmpipeline.tutorial_runtime` remain support/orchestration
  modules; static validation rejects production trainer/model-loader/generation
  implementations from being reintroduced there.

## Immutable runtime-source identities

Notebook source and runtime source are separate identities. This candidate records:

1. the notebook / release-candidate PR head;
2. pipeline support revision `afaf1f032cd7e9751db1ee8eb71b542f9bd0b15f`;
3. finetuner production revision `3772f0ca4e0f7130ffd5f826ea40f43b7212339e`.

Both notebooks use the same runtime-source revisions. The E2E producer writes those source
SHAs into its artifact provenance. The companion enforces them when present. Ordinary
production artifacts may omit the tutorial-only `runtimeRevisions` extension; in that case
the companion relies on the production contract's canonical model/revision plus critical
package compatibility rather than inventing source provenance that was never recorded.

## E2E workflow and metrics

The E2E notebook resolves a user-facing model through the canonical registry, seeds before
model construction, loads either a pinned public sample or BYOD through production data
normalization, applies production assistant-only masking, runs production QLoRA/PEFT training,
performs deterministic new-input inference, writes JSONL/CSV/JSON outputs, stages and hashes
the PEFT artifact, and reconstructs it through production inference.

The notebook explains its principal optimization metrics: train/validation/test causal
cross-entropy per supervised assistant token, derived perplexity, runtime/throughput, peak GPU
memory, and resolved training controls. These are optimization and execution evidence, not
benchmark accuracy or task-quality evidence.

Prediction files can contain user prompt/output text. Treat them as sensitive whenever the
input is sensitive.

## Artifact contract and trust boundary

The deployable output is a **PEFT adapter, not a complete model**. It requires the exact base
model and immutable revision recorded in provenance. The artifact manifest explicitly records
`format = peft_adapter` and `formatVersion = 1`, plus file sizes and SHA-256 digests.

The artifact-inference notebook accepts one adapter ZIP supplied from outside that notebook
execution. `artifact-manifest.json` must be at the ZIP root. Before model loading, the consumer
rejects traversal, backslash-ambiguous paths, symlinks, duplicate members, excessive expansion,
digest/size mismatches, missing required files, unsupported manifest format/version, and files
not covered by the root manifest.

Manifest consistency is not sender authenticity. If an attacker can replace both payload and
manifest, their hashes can agree. A whole-archive SHA-256 is an authenticity aid only when the
expected digest is obtained through an independently trusted channel. The artifact contract
requires safetensors/JSON and `trustRemoteCode=false`.

## Release gate

Static CI is necessary but is not clean-runtime execution evidence. Before either notebook is
labeled release-grade, run the E2E notebook top-to-bottom in a clean supported Colab GPU
runtime at the exact candidate head, preserve/download its adapter ZIP and digest, then run the
artifact-inference notebook in a **separate clean runtime** using that ZIP as external input.
Record the notebook head, both runtime-source SHAs, package/GPU identity, adapter-activity
evidence, artifact format/version and outcome in
[`RELEASE_VERIFICATION.md`](RELEASE_VERIFICATION.md). Never record `GITHUB_TOKEN` or other
secrets in release evidence.
