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

The badges target `main`. A release reviewer must execute the exact notebook/PR head being
reviewed, not merely whatever `main` contains at the time.

## Architecture: one production implementation

The execution boundary is deliberate:

- `language-model-pipeline` owns shared DIMER contracts, the canonical model registry,
  dataset normalization/resolution primitives, runtime identity checks, secure tutorial
  bootstrap, and hostile external-ZIP validation.
- `language-model-finetuner` owns deployable model loading, masking, training, artifact
  staging/publication, reconstruction, and generation.
- Production publication verification now calls `finetuner.inference.clean_load_smoke_test`,
  which uses the same `load_adapter_for_inference` and `generate_reply` surfaces used by the
  notebooks. `finetuner.artifacts.clean_load_smoke_test` is only a compatibility delegate.
- `lmpipeline.tutorial_api` and `lmpipeline.tutorial_runtime` remain support/orchestration
  modules; the static validator rejects production trainer/model-loader/generation
  implementations from being reintroduced there.

## Immutable runtime-source identities

Notebook source and runtime source are separate identities. This candidate records:

1. the notebook / release-candidate PR head;
2. pipeline support revision `afaf1f032cd7e9751db1ee8eb71b542f9bd0b15f`;
3. finetuner production revision `ecbab8cfbb95c061235f68c087141ab7af462b9d`.

Both notebooks use the same runtime-source revisions. The E2E artifact records them in
`provenance.json`; artifact inference rejects a different pair. Producer provenance also
records all critical compatibility packages required by the consumer, including
`safetensors`.

## E2E workflow and outputs

The E2E notebook resolves a user-facing model through the canonical registry, seeds before
model construction, loads either a pinned public sample or BYOD through production data
normalization, applies production assistant-only masking, runs production QLoRA/PEFT training,
records optimization metrics, performs deterministic new-input inference, writes JSONL/JSON
outputs, stages and hashes the PEFT artifact, and reconstructs it through production inference.
Fresh reconstruction requires both non-zero LoRA B weights and a non-zero adapter-on/off logit
delta. The adapter ZIP is then downloaded for the separate artifact-inference verification.

Prediction JSONL can contain user prompt/output text. Treat it as sensitive whenever the
input is sensitive. Loss/perplexity are optimization evidence, not task-quality evidence.

## Artifact inference and trust boundary

The artifact-inference notebook accepts one adapter ZIP supplied **from outside that notebook
execution**. `artifact-manifest.json` must be at the ZIP root. Before model loading, the
consumer rejects traversal, backslash-ambiguous paths, symlinks, duplicate members, excessive
expansion, digest/size mismatches, missing required files, and every file not covered by the
root manifest. Requiring a root manifest prevents a valid manifested subdirectory from hiding
unmanifested sibling content elsewhere in the archive.

The consumer then requires canonical base-model/revision parity, exact critical package
compatibility, pipeline/finetuner runtime-source parity, production reconstruction, adapter
activity evidence, a real editable prompt, and machine-readable outputs.

Manifest consistency is not sender authenticity. If an attacker can replace both payload and
manifest, their hashes can agree. A whole-archive SHA-256 is an authenticity aid only when the
expected digest is obtained through an independently trusted channel. The artifact contract
requires safetensors/JSON and `trustRemoteCode=false`; safe extraction is not permission to
deserialize arbitrary executable formats.

## Release gate

Static CI is necessary but is not clean-runtime execution evidence. Before either notebook is
labeled release-grade, run the E2E notebook top-to-bottom in a clean supported Colab GPU
runtime at the exact candidate head, preserve/download its adapter ZIP and digest, then run the
artifact-inference notebook in a **separate clean runtime** using that ZIP as external input.
Record the notebook head, both runtime-source SHAs, package/GPU identity, adapter-activity
evidence, and outcome in [`RELEASE_VERIFICATION.md`](RELEASE_VERIFICATION.md). Never record the
`GITHUB_TOKEN` or other secrets in release evidence.
