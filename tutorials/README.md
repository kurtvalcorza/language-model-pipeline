# Language Model Tutorials

These notebooks are the user-facing executable references for the DIMER language-model
fine-tuning capability. They are governed by **DIMER Notebook Specification 1.0** and
explicitly execute the production `language-model-finetuner` modules rather than maintaining
a notebook-specific trainer or serving implementation.

## Notebook registry

| Notebook | Profile | Capability | Release status |
|---|---|---|---|
| [`language_model_finetuning_colab.ipynb`](language_model_finetuning_colab.ipynb) | `E2E` | Production-path QLoRA/PEFT SFT, evaluation, new-input inference, artifact staging and fresh reconstruction | **Candidate — clean-runtime verification required** |
| [`language_model_artifact_inference_colab.ipynb`](language_model_artifact_inference_colab.ipynb) | `ARTIFACT-INFERENCE` | External artifact validation, production reconstruction, new-input generation and machine-readable export | **Candidate — clean-runtime verification required** |

Each profile is also recorded under `metadata.dimer.notebook_profile`, with
`metadata.dimer.notebook_spec_version = "1.0"`. Clean-runtime release evidence is tracked in
[`RELEASE_VERIFICATION.md`](RELEASE_VERIFICATION.md).

## Open in Colab

[![Open Fine-Tuning Tutorial In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/language-model-pipeline/blob/main/tutorials/language_model_finetuning_colab.ipynb)

[![Open Artifact Inference In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/language-model-pipeline/blob/main/tutorials/language_model_artifact_inference_colab.ipynb)

The badges target `main`. A release reviewer must execute the exact notebook/PR head being
reviewed, not merely whatever `main` contains at the time.

## Architecture: production implementation, notebook orchestration

The execution boundary is deliberate:

- `language-model-pipeline` owns shared DIMER contracts, the canonical model registry,
  dataset normalization/resolution primitives, and notebook-only support such as exact runtime
  checks and hostile external-ZIP validation.
- `language-model-finetuner` owns the deployable execution implementation. The E2E notebook
  directly imports `finetuner.backends`, `finetuner.data`, `finetuner.masking`,
  `finetuner.training`, and `finetuner.artifacts`.
- `finetuner.inference` is the production-repository inference surface shared by the E2E and
  artifact-inference notebooks for adapter reconstruction, generation and adapter-activity
  verification.
- `lmpipeline.tutorial_api` is intentionally **support-only**. The static validator rejects
  model loading, masking, training, generation or artifact-publication implementations from
  being reintroduced there.

The notebooks therefore teach and orchestrate production code instead of carrying a second
implementation that can drift independently.

## Immutable runtime-source identities

Notebook source and runtime source are not the same thing. This candidate records three
identities:

1. the notebook / release-candidate PR head;
2. the exact `language-model-pipeline` runtime-support commit installed inside the notebook;
3. the exact `language-model-finetuner` commit checked out and imported inside the notebook.

Both notebooks use the same two runtime-source revisions. The E2E artifact writes them into
`provenance.json`; artifact inference rejects an artifact produced with different runtime
revisions. The exact candidate pins and the required clean-run record format are documented in
[`RELEASE_VERIFICATION.md`](RELEASE_VERIFICATION.md).

## E2E workflow

`language_model_finetuning_colab.ipynb` demonstrates:

1. exact package/runtime checks and immutable production-source checkout verification;
2. deterministic seeding before model/adapter construction;
3. canonical model resolution to an immutable upstream revision;
4. pinned public sample or BYOD JSONL/ZIP input;
5. production dataset normalization and deterministic validation splitting;
6. production assistant-only loss masking with tokenizer chat-template prefix checks;
7. production QLoRA base loading and PEFT attachment;
8. production training, including the same scheduler/warmup/early-stopping/restoration
   configuration surface used by the finetuner;
9. deterministic before/after and new-prompt inference;
10. JSONL/JSON output export;
11. production artifact staging, manifest verification and provenance; and
12. fresh reconstruction with explicit evidence that serialized LoRA B weights are non-zero
    and that adapter-on logits differ from adapter-off logits on the same reloaded model.

Loss/perplexity are optimization evidence. They do not establish factual accuracy, safety,
robustness, fairness, calibration or deployment fitness.

## BYOD and privacy

BYOD accepts `train.jsonl`, optional `validation.jsonl`/`val.jsonl`, optional `test.jsonl`, or
one ZIP containing those files. Supported schema families are the repository's canonical chat,
prompt/completion, and instruction/input/output forms.

Uploaded dataset bytes remain in the notebook runtime. The notebook still makes network
requests for source code, packages, public sample data and pinned model weights. Do not place
confidential, restricted, personal or sensitive data in a hosted notebook unless authorized.

## Artifact inference and trust boundary

`language_model_artifact_inference_colab.ipynb` accepts an adapter ZIP supplied **from outside
that notebook execution**. Before model loading it rejects path traversal, backslash-ambiguous
paths, symlinks, duplicate members, excessive expansion, digest/size mismatches, missing
required PEFT files, and unexpected unlisted files.

It then requires:

- canonical base-model and immutable-revision parity;
- critical producer/consumer package compatibility;
- pipeline-runtime and finetuner-runtime revision parity;
- production `finetuner.inference` reconstruction;
- non-zero adapter-activity evidence;
- a real editable new-user prompt; and
- JSONL/JSON output files with model/artifact/runtime provenance.

Manifest consistency is not sender authenticity. If an attacker can replace both payload and
manifest, the hashes can agree. A whole-archive SHA-256 is only an authenticity aid when the
expected digest is obtained through an independently trusted channel. The contract requires
safetensors/JSON and `trustRemoteCode=false`; safe extraction is not permission to deserialize
arbitrary executable formats.

## Release gate

Static validation is necessary but not execution evidence. Before either notebook is labeled
release-grade, run the E2E notebook from a clean supported GPU runtime at the candidate head,
preserve its adapter ZIP and digest, then run the artifact-inference notebook in a separate
clean runtime using that ZIP as external input. Record the notebook head, pipeline runtime SHA,
finetuner runtime SHA, package/GPU identity, adapter-activity evidence and result in
[`RELEASE_VERIFICATION.md`](RELEASE_VERIFICATION.md).
