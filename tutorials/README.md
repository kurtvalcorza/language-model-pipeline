# Language Model Tutorials

These notebooks are the user-facing executable reference implementations for the DIMER
language-model fine-tuning capability. They are governed by **DIMER Notebook Specification
1.0** and use the repository's public `lmpipeline.tutorial_api` surface rather than carrying a
parallel model registry, trainer, or artifact loader inside notebook cells.

## Notebook registry

| Notebook | Profile | Capability | Default runtime | Release status |
|---|---|---|---|---|
| [`language_model_finetuning_colab.ipynb`](language_model_finetuning_colab.ipynb) | `E2E` | QLoRA/PEFT supervised fine-tuning, evaluation, new-prompt inference, adapter export and fresh reconstruction | CUDA GPU; exact `requirements-colab.lock` user-space stack | **Candidate — clean-runtime verification required before release** |
| [`language_model_artifact_inference_colab.ipynb`](language_model_artifact_inference_colab.ipynb) | `ARTIFACT-INFERENCE` | External adapter verification, provenance/runtime validation, reconstruction, new-input generation and machine-readable output export | CUDA GPU; exact `requirements-colab.lock` user-space stack | **Candidate — clean-runtime verification required before release** |

The profile is also recorded in each notebook under `metadata.dimer.notebook_profile`, with
`metadata.dimer.notebook_spec_version = "1.0"`. A notebook is not called release-grade here until
a clean supported runtime has executed the candidate revision and the evidence is recorded in
[`RELEASE_VERIFICATION.md`](RELEASE_VERIFICATION.md).

## Open in Colab

[![Open Fine-Tuning Tutorial In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/language-model-pipeline/blob/main/tutorials/language_model_finetuning_colab.ipynb)

[![Open Artifact Inference In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/language-model-pipeline/blob/main/tutorials/language_model_artifact_inference_colab.ipynb)

The badges target `main`; release reviewers should instead execute the exact candidate commit or
PR head being reviewed.

## Architecture: one implementation, two learning workflows

The tutorials intentionally separate production-facing logic from pedagogy:

- `src/lmpipeline/data/model-registry.yaml` is the canonical model registry.
- `lmpipeline.datasets` owns dataset schema normalization, split/archive resolution and stable
  validation behavior shared with the DIMER consumers.
- `lmpipeline.tutorial_api` exposes the repository-supported tutorial execution surface for
  model loading, assistant-only masking, QLoRA attachment/training, generation, artifact
  packaging and artifact consumption.
- notebook cells explain and orchestrate those APIs; they do not reimplement the model or
  training logic.

This matters because a notebook that independently reproduces the implementation can drift while
still appearing to work. The static validator rejects known parallel-implementation markers.

## E2E fine-tuning workflow

`language_model_finetuning_colab.ipynb` demonstrates:

1. exact tutorial runtime verification;
2. deterministic seeding before model/adapter construction;
3. canonical model resolution to an immutable upstream revision;
4. a pinned public tutorial sample or real BYOD JSONL/ZIP input;
5. canonical data normalization, leakage checks and tokenizer-aware sequence validation;
6. assistant-only loss masking using the model's own chat template;
7. deterministic pre-adaptation generation;
8. QLoRA/PEFT training through `lmpipeline.tutorial_api`;
9. held-out optimization metrics with explicit limits on what they establish;
10. new-prompt inference and JSONL/JSON output export;
11. PEFT adapter, tokenizer, manifest and provenance export; and
12. fresh reconstruction from serialized files against the exact base-model revision.

The tutorial distinguishes optimization evidence from task quality. A lower validation loss or
perplexity does not establish factual accuracy, safety, robustness, fairness, calibration or
production fitness. Generic open-ended generation needs representative independent evaluation
and an application-specific rubric, judge or human assessment.

## Data contract and BYOD

The E2E notebook accepts one schema family per JSONL split:

```json
{"messages":[{"role":"user","content":"Question"},{"role":"assistant","content":"Answer"}]}
```

or `prompt`/`completion`, or `instruction`/`input`/`output` records. A BYOD upload requires
`train.jsonl`; `validation.jsonl`/`val.jsonl` and `test.jsonl` are optional. One ZIP containing
those files at its content root is also supported.

Supplied validation/test splits are preserved. When validation is absent, the tutorial derives
an order-independent validation split using canonical record fingerprints and the explicit seed.
Cross-split duplicates fail. Over-length training examples fail rather than being silently
truncated.

Uploaded dataset bytes stay inside the notebook runtime. The notebook does make external network
requests to acquire its pinned repository code, package stack, public sample and model weights.
Do not place confidential, restricted, personal or sensitive data in a hosted notebook runtime
unless that use is authorized.

## Artifact contract

The E2E notebook exports a PEFT adapter rather than duplicating the base model. The bundle records
the exact base model and immutable revision needed for reconstruction and includes:

- `adapter_model.safetensors`;
- `adapter_config.json`;
- tokenizer/chat-template assets;
- `metrics.json`;
- `provenance.json`;
- `MODEL_CARD.md`; and
- `artifact-manifest.json` with file sizes and SHA-256 digests.

The adapter remains dependent on its base model. It may also encode information learned from the
training data and must be handled according to applicable dataset confidentiality, licensing,
retention and disclosure requirements.

## Artifact-inference workflow and trust boundary

`language_model_artifact_inference_colab.ipynb` accepts an adapter ZIP supplied **from outside the
current notebook execution**. Before model loading, it rejects unsafe paths, traversal, symlinks,
duplicate members, oversized expansion, digest/size mismatches and unexpected unlisted files.
It then checks artifact provenance against the canonical model registry and requires critical
producer/consumer package versions to match.

Manifest consistency is not sender authenticity. If an attacker can replace both payload and
manifest, the hashes can still agree. A whole-archive SHA-256 establishes useful identity only
when the expected digest arrives through a trusted independent channel. Path-safe extraction also
does not make arbitrary executable serialization trustworthy; this contract requires safetensors,
JSON metadata and `trust_remote_code=False`.

The inference notebook provides a real editable new-prompt path, validates the prompt against the
artifact's effective context ceiling, generates through the repository API and writes both
`artifact_inference_predictions.jsonl` and `artifact_inference_provenance.json`.

## Reproducible runtime

[`requirements-colab.lock`](requirements-colab.lock) is the tutorial dependency authority. The
user-space stack is exact. PyTorch is accelerator-coupled, so the notebooks verify the required
semantic Torch version and record the full Torch/CUDA build instead of silently replacing a
working Colab CUDA wheel.

The runtime record includes Python, package versions, CUDA availability/build, GPU identity and
BF16 support. Training also records the explicit seed and names residual nondeterminism such as
hardware/kernel differences; this is reproducibility evidence, not a claim of cross-hardware
bitwise determinism.

## Static validation

Run:

```bash
python scripts/validate_colab_tutorials.py
pytest tests/test_colab_tutorials.py tests/test_tutorial_api.py
```

The validator checks notebook JSON/compilation, clean output state, normative profile metadata,
exact dependency lock syntax, immutable repository bootstrap pins, required workflow markers,
absence of known duplicate core implementations, archive path validation and key learning-contract
language.

Static checks do **not** prove that the current Colab runtime, model host, GPU stack and notebook
execute together. That is a separate release gate.

## Release verification

Before either notebook is marked release-grade, follow
[`RELEASE_VERIFICATION.md`](RELEASE_VERIFICATION.md). At minimum the durable record must identify
the notebook commit/PR head, clean environment, runtime/GPU identity and outcome. For the artifact
workflow, the external ZIP must be produced outside the consumer notebook execution; creating and
loading an artifact inside one notebook is only an integration test.

## Related repository contracts

- [`../README.md`](../README.md) — repository architecture and DIMER capability surface
- [`../DATASET_SPEC.md`](../DATASET_SPEC.md) — dataset contract
- [`../TRAINING_SPEC.md`](../TRAINING_SPEC.md) — training semantics
- [`../ARTIFACT_SPEC.md`](../ARTIFACT_SPEC.md) — deployable adapter contract
- [`../PROVENANCE_SPEC.md`](../PROVENANCE_SPEC.md) — provenance contract
- [`../COMPATIBILITY.md`](../COMPATIBILITY.md) — measured resource compatibility
- [`../SECURITY.md`](../SECURITY.md) — supply-chain and privacy controls
