# Standalone Colab Tutorials

These notebooks provide an educational, standalone path through the DIMER language-model
fine-tuning capability without requiring DIMER Workbench.

| Notebook | Purpose |
|---|---|
| [Language-model fine-tuning](language_model_finetuning_colab.ipynb) | Base-model acquisition → sample/BYOD → structural/tokenizer validation → baseline generation → QLoRA SFT → before/after comparison → new prompts → adapter export → fresh reload |
| [Artifact inference](language_model_artifact_inference_colab.ipynb) | Upload adapter bundle → verify manifest/provenance → reacquire exact base from Hugging Face or DIMER ZIP → attach adapter → generate |

## Open in Colab

[![Open Fine-Tuning Tutorial In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/language-model-pipeline/blob/main/tutorials/language_model_finetuning_colab.ipynb)

[![Open Artifact Inference In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/language-model-pipeline/blob/main/tutorials/language_model_artifact_inference_colab.ipynb)

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/language-model-pipeline)

## Learning goals

The fine-tuning notebook is written as a tutorial rather than a terse execution script. It
explains:

- the difference between a base model, tokenizer, and PEFT adapter;
- why tokenizer-aware validation happens before multi-gigabyte weight loading;
- how assistant-only loss masking works;
- why QLoRA reduces the VRAM required to adapt a causal LM;
- how to interpret loss/perplexity without treating them as task-quality scores;
- why before/after prompt comparisons are illustrative rather than formal evaluation;
- what belongs in an adapter-first artifact; and
- why a fresh base+adapter reload is a stronger portability check than reusing the training
  object still in memory.

## Model choices

The tutorial currently exposes:

- `smollm3-3b` — **default tutorial model**; Apache-2.0; resource profile still being measured.
- `qwen3-1.7b` — Apache-2.0; measured QLoRA profile.
- `qwen3-4b` — Apache-2.0; measured QLoRA profile.
- `granite-4.1-3b` — Apache-2.0; measured QLoRA profile.
- `llama-3.2-3b-instruct` — **credential-test candidate**. The model is gated on Hugging Face
  and uses the Llama 3.2 Community License. It is selectable in the tutorial so Colab's
  `HF_TOKEN` secret path can be tested, but that does not promote it into DIMER production
  registration or prove that DIMER jobs can receive the same credential.

The tutorial keeps `trust_remote_code=False` for every model.

## Hugging Face secret for gated models

For Llama 3.2 3B Instruct:

1. accept the model terms on Hugging Face;
2. add a Colab secret named `HF_TOKEN`;
3. grant the notebook access to that secret; and
4. select `Pinned Hugging Face` as the model source.

The notebook reads the token through `google.colab.userdata`, passes it directly to Hub/model
loading calls, and never prints or writes the token into an artifact.

A successful Colab credential preflight proves only that **Colab** can access the gated
checkpoint. The separate DIMER build/runtime credential-injection blocker remains a platform
question.

## Base-model acquisition

Two source modes are supported:

- `Pinned Hugging Face` — exact immutable revision.
- `DIMER ZIP` — a DIMER-hosted Hugging Face snapshot with a required
  `dimer-base-manifest.json`.

The DIMER manifest format is:

```json
{
  "format": "dimer_hf_snapshot",
  "formatVersion": 1,
  "modelId": "Qwen/Qwen3-1.7B",
  "revision": "40-character-commit-sha",
  "files": [
    {
      "path": "config.json",
      "bytes": 1234,
      "sha256": "64-character-sha256"
    }
  ],
  "totalBytes": 1234
}
```

Every file in the snapshot must be listed. The notebook rejects absolute/traversal paths,
symlinks, unexpected files, missing files, size mismatches, SHA-256 mismatches, and model/revision
mismatches.

The Llama credential-test candidate intentionally does **not** permit DIMER ZIP in the tutorial
until DIMER hosting/redistribution is explicitly cleared.

## Dataset modes

- `Sample: Filipino SFT` — `jpaulpoliquit/ph-sft-ai-authored-v1`; AI-authored Filipino/English
  SFT seed used for tutorial/training, not as a benchmark.
- `Sample: Dolly` — `databricks/databricks-dolly-15k`; English instruction sample.
- `Bring Your Own Dataset` — DIMER JSONL schema families with optional validation/test splits
  and safe ZIP transport.

The tutorial does not depend on non-open iTANONG corpora.

## Evidence model

Keep three claim types separate:

1. **Pipeline/runtime evidence** — validation, training completion, artifact export/reload,
   observed runtime/VRAM.
2. **Optimization evidence** — train/validation/test loss and perplexity where finite.
3. **Task-quality evidence** — requires an independent task-specific evaluation protocol.

A successful short Colab run on an unmeasured model provides observed VRAM for that workload; it
does not by itself establish a general production `min_vram_gb`.

## AI provenance

Tutorial implementation: **OpenAI / ChatGPT — GPT-5.6 Sol High**, Builder role. This records
implementation provenance and is not independent reviewer sign-off.
