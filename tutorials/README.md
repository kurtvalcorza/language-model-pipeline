# Standalone Colab Tutorials

These notebooks exercise the DIMER language-model fine-tuning capability without requiring
DIMER Workbench.

| Notebook | Purpose |
|---|---|
| [Language-model fine-tuning](language_model_finetuning_colab.ipynb) | Approved base model → sample/BYOD → validation → baseline generation → LoRA/QLoRA SFT → before/after generation → new-prompt inference → adapter export → fresh reload |
| [Artifact inference](language_model_artifact_inference_colab.ipynb) | Upload exported PEFT adapter bundle → verify manifest/hashes/provenance → load pinned base → attach adapter → generate |

## Open in Colab

[![Open Fine-Tuning Tutorial In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/language-model-pipeline/blob/main/tutorials/language_model_finetuning_colab.ipynb)

[![Open Artifact Inference In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/language-model-pipeline/blob/main/tutorials/language_model_artifact_inference_colab.ipynb)

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/language-model-pipeline)

## Evidence model

The tutorials keep these claims separate:

1. **Pipeline evidence** — validation, training completion, artifact export, and reload.
2. **Training evidence** — loss, perplexity where finite, supervised tokens, runtime, VRAM.
3. **Task-quality evidence** — requires an independent, task-specific evaluation protocol.

Side-by-side prompt generations are educational and illustrative. They are not a formal
quality benchmark.

## Dataset modes

The fine-tuning notebook supports:

- `Sample: Filipino SFT` — `jpaulpoliquit/ph-sft-ai-authored-v1`; AI-authored Filipino/English
  SFT seed, used as tutorial/training material rather than a benchmark.
- `Sample: Dolly` — `databricks/databricks-dolly-15k`; English SFT sample pinned to the same
  immutable revision used by the acceptance infrastructure.
- `Bring Your Own Dataset` — canonical DIMER JSONL schemas, with optional validation/test
  splits and safe single-ZIP transport.

The tutorial does not depend on non-open iTANONG corpora.

## Model policy

The notebook exposes the current user-facing registry keys only:

- `qwen3-1.7b`
- `qwen3-4b`
- `granite-4.1-3b`

Model IDs and immutable Hugging Face revisions are embedded from the canonical registry.
Arbitrary model IDs and `trust_remote_code=True` are not accepted. A training method with no
measured resource profile fails closed.

## Artifact contract

The training notebook exports an adapter-first PEFT bundle containing adapter weights/config,
the tokenizer used for training, metrics, provenance, a model card, and an SHA-256 manifest.
It does not copy base-model weights or training rows. The separate artifact-inference
notebook verifies the bundle before loading the exact base revision and applying the adapter.

## AI provenance

Tutorial implementation: **OpenAI / ChatGPT — GPT-5.6 Sol High**, Builder role. This records
implementation provenance and is not independent reviewer sign-off.
