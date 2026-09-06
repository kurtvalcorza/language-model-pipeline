# Qwen3-0.6B & Language Model Fine-Tuning Pipeline

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Qwen%2FQwen3--0.6B-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/Qwen/Qwen3-0.6B)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Open Fine-Tuning Tutorial In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/language-model-pipeline/blob/main/tutorials/language_model_finetuning_colab.ipynb)
[![Open Artifact Inference In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/language-model-pipeline/blob/main/tutorials/language_model_artifact_inference_colab.ipynb)
[![GitHub Repository](https://img.shields.io/badge/GitHub-language--model--pipeline-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/language-model-pipeline)

---

## Executive Summary & Description

This repository profile card provides the complete specification for **Qwen3-0.6B** and the accompanying **Language Model Supervised Fine-Tuning (SFT) and Inference Pipeline**.

**Qwen3-0.6B** is an ultra-compact dense causal language model developed by the **Qwen Team (Alibaba Cloud)** featuring approximately 0.6 billion total parameters (590M total, 440M non-embedding). As the lightweight entry point of the Qwen3 family, it offers state-of-the-art instruction following, multi-turn conversational competence, and dual-mode reasoning capabilities:
- **Thinking Mode:** Dynamically emits step-by-step reasoning tokens enclosed in `<think>...</think>` tags for complex problem solving, logic, and mathematics.
- **Non-Thinking Mode:** Generates direct, low-latency conversational responses without reasoning prefixes, ideal for high-throughput interactive dialogue.

Within this pipeline, `Qwen3-0.6B` serves as the primary open-weights baseline and smoke-testing foundation model. With its compact weights payload (**~1.41 GiB safetensors**), it enables end-to-end execution of 4-bit QLoRA fine-tuning, assistant-only loss masking, chat template normalization, before-and-after evaluation, and portable PEFT adapter packaging in **under one minute** on standard single-GPU environments (such as a Google Colab free T4 15 GB instance).

---

## Interactive Google Colab Tutorials

The repository includes two self-contained, interactive Jupyter notebooks for hands-on execution directly in the browser:

| Tutorial Notebook | Badge Link | Focus & Lifecycle Stages |
|---|---|---|
| **Language-Model Fine-Tuning** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/language-model-pipeline/blob/main/tutorials/language_model_finetuning_colab.ipynb) | Pinned base-model acquisition → Dataset normalization → Chat template formatting & assistant masking → Baseline prompt probe → 4-bit QLoRA SFT → Before/after evaluation → Adapter bundle export → In-memory clean reload |
| **Artifact Inference** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/language-model-pipeline/blob/main/tutorials/language_model_artifact_inference_colab.ipynb) | Standalone adapter package upload → Cryptographic manifest & provenance verification → Base-model resolution → Dynamic PEFT attachment → Multi-turn conversational generation |

---

## Base Model Weights Cache & Repository Architecture

All offline model assets are managed under the repository's [`weights/`](https://github.com/kurtvalcorza/language-model-pipeline/tree/main/weights) directory. Each model is isolated in its own dedicated subfolder corresponding to its canonical registry key in [`src/lmpipeline/data/model-registry.yaml`](https://github.com/kurtvalcorza/language-model-pipeline/blob/main/src/lmpipeline/data/model-registry.yaml):

```
weights/
├── README.md                          <-- Root directory guide & index
└── qwen3-0.6b/                        <-- Model subfolder: Qwen3-0.6B snapshot
    ├── config.json                    <-- Model architecture configuration
    ├── generation_config.json         <-- Default sampling & EOS tokens
    ├── tokenizer.json                 <-- Hugging Face fast tokenizer definition
    ├── tokenizer_config.json          <-- ChatML template & special token mappings
    ├── vocab.json                     <-- BPE token vocabulary (151,936 tokens)
    ├── merges.txt                     <-- BPE subword merge rules
    ├── dimer-base-manifest.json       <-- Cryptographic SHA-256 integrity manifest
    ├── README.md                      <-- Qwen3-0.6B dedicated model card
    ├── LICENSE                        <-- Upstream Apache License 2.0
    └── model.safetensors              <-- (1.50 GB: Excluded from Git; uploaded to DIMER)
```

### Links to In-Repository Model Files
- **Base Weights Root Directory:** [`language-model-pipeline/weights/README.md`](https://github.com/kurtvalcorza/language-model-pipeline/blob/main/weights/README.md)
- **Qwen3-0.6B Dedicated Snapshot:** [`language-model-pipeline/weights/qwen3-0.6b/`](https://github.com/kurtvalcorza/language-model-pipeline/tree/main/weights/qwen3-0.6b)
- **Qwen3-0.6B Model Card:** [`language-model-pipeline/weights/qwen3-0.6b/README.md`](https://github.com/kurtvalcorza/language-model-pipeline/blob/main/weights/qwen3-0.6b/README.md)
- **Cryptographic Manifest:** [`language-model-pipeline/weights/qwen3-0.6b/dimer-base-manifest.json`](https://github.com/kurtvalcorza/language-model-pipeline/blob/main/weights/qwen3-0.6b/dimer-base-manifest.json)

### DIMER Architecture & Git Tracking Strategy
In the DIMER workbench ecosystem:
1. **Large Binary Weights (`model.safetensors`):** The ~1.41 GiB weight file is excluded from Git via `.gitignore` (`weights/**/*.safetensors`) and uploaded directly to DIMER as a model asset or downloaded using `scripts/fetch_weights.py`.
2. **Configuration & Tokenizers:** All accompanying configuration files (`config.json`, `generation_config.json`), BPE tokenizer files (`tokenizer.json`, `vocab.json`, `merges.txt`), and the cryptographic manifest are tracked in GitHub under [`weights/qwen3-0.6b/`](https://github.com/kurtvalcorza/language-model-pipeline/tree/main/weights/qwen3-0.6b). This allows build pipelines and offline Docker containers to initialize the model structure and tokenizers without external network access.

### Snapshot Acquisition & Verification CLI
The repository provides [`scripts/fetch_weights.py`](https://github.com/kurtvalcorza/language-model-pipeline/blob/main/scripts/fetch_weights.py) for managing and verifying base weights:

```bash
# List all registered base models
python scripts/fetch_weights.py --list

# Download and verify Qwen3-0.6B into its dedicated subfolder:
python scripts/fetch_weights.py --model qwen3-0.6b --dest weights/qwen3-0.6b

# Cryptographically verify the snapshot on disk against dimer-base-manifest.json:
python scripts/fetch_weights.py --verify-only --dest weights/qwen3-0.6b
```

---

## Technical Specifications: Qwen3-0.6B

| Specification | Value | Notes |
|---|---|---|
| **Model Key** | `qwen3-0.6b` | Registered pipeline selector key |
| **Hugging Face ID** | `Qwen/Qwen3-0.6B` | Canonical repository on Hugging Face Hub |
| **Pinned Revision** | `c1899de289a04d12100db370d81485cdf75e47ca` | Immutable 40-character commit SHA |
| **Developer / Organization** | Qwen Team, Alibaba Cloud | Original model author |
| **Architecture** | `Qwen3ForCausalLM` | Dense causal language model |
| **Total Parameters** | 590M (~0.6B) | 440M non-embedding parameters |
| **Layers (Hidden Depth)** | 28 | Transformer decoder layers |
| **Hidden Size / Intermediate** | 1,024 / 3,072 | Hidden dimensionality and SwiGLU projection |
| **Attention Mechanism** | Grouped Query Attention (GQA) | 16 Query heads, 8 Key/Value heads |
| **Context Window** | 32,768 tokens | Native sliding-window / full context support |
| **Vocabulary Size** | 151,936 | BPE tokenizer with ChatML special tokens |
| **Weight Serialization** | SafeTensors | Single shard: `model.safetensors` (~1.41 GiB) |
| **License** | Apache License 2.0 | Commercial and research use permitted |
| **Security Guardrail** | `trust_remote_code=False` | Safe standard PyTorch module instantiation |

---

## Hardware & Runtime Profiles

`Qwen3-0.6B` is optimized for resource-constrained environments, edge hardware, and accessible cloud environments:

| Workload Mode | Precision | GPU VRAM Allocated | Supported Hardware |
|---|---|---|---|
| **Standard Inference** | BF16 / FP16 | ~1.41 GiB | Any GPU with >=2 GB VRAM; Apple Silicon (M-series); CPU |
| **Quantized Inference** | 4-bit (NF4 / AWQ / GGUF) | ~0.8 – 1.1 GiB | Edge devices, mobile workstations, Raspberry Pi 5 |
| **QLoRA Fine-Tuning** | 4-bit Base + FP16 LoRA (bs=1, seq=1024) | ~4.2 – 5.5 GiB peak | NVIDIA RTX 3050/4060, T4, L4, V100, A100 |
| **Full Fine-Tuning (SFT)** | 16-bit (bs=1, AdamW) | ~14.0 – 16.0 GiB | NVIDIA RTX 3090/4090, A10G, A100 |

---

## Core Fine-Tuning Methodology

The pipeline follows rigorous machine learning engineering standards to guarantee reproducibility, mathematical correctness, and deployment safety:

### 1. The Tripartite Architecture
- **Frozen Base Model:** Preserves general pre-trained world knowledge and language fluency across billions of fixed parameters.
- **Tokenizer & Chat Template:** Translates natural language into integer sequences and applies Jinja2 templates (ChatML format with `<|im_start|>` and `<|im_end|>`) to enforce conversational role structure (`system`, `user`, `assistant`).
- **Low-Rank Adapters (PEFT):** Modulates target weight matrices through low-rank decomposition ($W = W_0 + \frac{\alpha}{r}BA$), updating less than 1% of total parameters.

### 2. Assistant-Only Loss Masking
Computing cross-entropy loss over entire dialogue sequences teaches the model to memorize user prompts. The pipeline identifies exact token boundaries corresponding to assistant turns using prefix-stable tokenization, assigning non-assistant tokens the label `ignore_index = -100`. Backpropagation updates adapter weights **solely based on the assistant's predicted response tokens**.

### 3. 4-bit Quantization (QLoRA)
Loads base weights in 4-bit NormalFloat (`nf4`) with double quantization and BF16 computation (`torch.bfloat16` or `torch.float16`). This compresses base-model memory consumption by >60% with negligible loss in downstream adaptation capability.

### 4. Recommended Fine-Tuning Hyperparameters
- **LoRA Rank ($r$):** 16 (sufficient capacity for style, formatting, and domain adaptation)
- **LoRA Alpha ($\alpha$):** 32 (standard scaling ratio $\frac{\alpha}{r} = 2.0$)
- **Target Modules:** All linear attention and MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- **Learning Rate:** $2 \times 10^{-4}$ with linear warmup ($3\% - 10\%$) and cosine decay
- **Optimizer:** Paged AdamW 8-bit or standard AdamW with gradient accumulation

### 5. Supply-Chain Integrity & Manifest Verification
Models are cryptographically pinned to immutable commit revisions. A `dimer-base-manifest.json` tracks byte counts and SHA-256 digests for all files. Tampered, missing, or altered files immediately halt execution.

### 6. Optimization Metrics vs. Task Quality
The training loop logs:
- **Optimization Convergence:** Training loss, validation loss, validation perplexity ($e^{\\text{loss}}$), gradient norm, wall-clock time, and peak VRAM.
- **Task Quality:** Evaluated independently using prompt probes, held-out task benchmarks, or human evaluation rubrics.

### 7. Standalone Adapter Packaging & Clean Reload
The fine-tuning deliverable is a self-contained adapter package (typically 50–150 MB) containing:
1. `adapter_model.safetensors` (trainable delta weights)
2. `adapter_config.json` (PEFT configuration & base-model metadata)
3. Tokenizer assets (`tokenizer.json`, `vocab.json`, `merges.txt`, `chat_template.jinja`)
4. Training arguments and runtime environment metadata
5. `artifact-manifest.json` (SHA-256 hashes of all exported files)

A **clean reload test** clears the GPU cache, re-instantiates a pristine base model, and attaches the exported adapter from disk to prove end-to-end deployment readiness.

---

## Dataset Normalization & Data Hygiene

The pipeline standardizes instruction datasets into a canonical conversational schema:

```json
{
  "messages": [
    {"role": "system", "content": "You are a concise, domain-expert assistant."},
    {"role": "user", "content": "What is Parameter-Efficient Fine-Tuning?"},
    {"role": "assistant", "content": "PEFT freezes the base model weights and trains a small set of auxiliary adapter parameters to adapt the model to new tasks with minimal compute and memory."}
  ]
}
```

### Supported Dataset Sources
1. **Sample: Filipino SFT:** `jpaulpoliquit/ph-sft-ai-authored-v1` (bilingual Filipino/English conversational dataset).
2. **Sample: Dolly:** `databricks/databricks-dolly-15k` (general-purpose English instruction following).
3. **Bring Your Own Dataset (BYOD):** Custom JSONL files supporting multi-turn `messages`, `prompt`/`completion`, or Alpaca `instruction`/`input`/`output` records.

### Data Hygiene Guardrails
- **Assistant Target Verification:** Guarantees every training example has a non-empty assistant response.
- **Split Leakage Detection:** Computes deterministic SHA-256 hashes of canonicalized records; halts execution if an example appears in both training and validation sets.
- **Duplicate Awareness:** Flags and logs exact duplicate records within splits.
- **Sequence Length Boundaries:** Rejects examples exceeding `MAX_SEQUENCE_LENGTH` rather than silently truncating them, preventing context truncation artifacts.

---

## Serving & Deployment Architectures

Trained adapters can be deployed across three industry-standard patterns:

```
                          ┌─────────────────────────────┐
                          │   Trained Adapter Bundle    │
                          │ (adapter_model.safetensors) │
                          └──────────────┬──────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
┌──────────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
│  Multi-LoRA Serving  │      │     Local Python     │      │ Zero-Overhead Fusion │
│ (vLLM, SGLang, TGI)  │      │ (PEFT + Transformers)│      │(model.merge_and_... )│
├──────────────────────┤      ├──────────────────────┤      ├──────────────────────┤
│ Concurrent routing of│      │ Dynamic in-memory    │      │ Permanently merge    │
│ hundreds of adapters │      │ attachment on base   │      │ delta weights into   │
│ on 1 shared base LM. │      │ model at runtime.    │      │ standalone weights.  │
└──────────────────────┘      └──────────────────────┘      └──────────────────────┘
```

1. **High-Throughput Multi-LoRA Serving (vLLM / SGLang / TGI):**
   Shares a single frozen base model across memory while dynamically applying user-specific adapters per request.
2. **Local Python Integration (PEFT & Transformers):**
   Loads the base model and dynamically attaches the adapter in evaluation mode (`is_trainable=False`) as demonstrated in the [Artifact Inference](https://colab.research.google.com/github/kurtvalcorza/language-model-pipeline/blob/main/tutorials/language_model_artifact_inference_colab.ipynb) notebook.
3. **Zero-Overhead Weight Merging:**
   Permanently combines the adapter delta weights into the base model parameters via `model.merge_and_unload()`, producing a standard, standalone Hugging Face checkpoint loadable by any engine (llama.cpp, Ollama, ONNX).

---

## Decoding Parameters & Generation Guidelines

### Recommended Sampling Configurations

| Parameter | Thinking Mode | Non-Thinking Mode | Deterministic Smoke-Test |
|---|---|---|---|
| **`temperature`** | `0.6` | `0.7` | `0.0` (Greedy) |
| **`top_p`** | `0.95` | `0.80` | `1.0` |
| **`top_k`** | `20` | `20` | `0` |
| **`presence_penalty`** | `0.0 - 1.5` | `0.0` | `0.0` |
| **`do_sample`** | `True` | `True` | `False` |

> [!TIP]
> **Greedy Decoding in Thinking Mode:** Upstream developers strongly advise against using greedy search (`temperature=0.0`) in thinking mode, as it can induce repetitive reasoning loops. For deterministic regression testing, non-thinking mode with greedy search is recommended.

---

## Intended Use and Limitations

### Primary Intended Uses
- Lightweight base model for domain-specific fine-tuning, style transfer, and conversational alignment.
- CI/CD pipeline automation and smoke-testing of data ingestion, tokenization, training, and export loops.
- Embedded, on-device, and low-latency edge assistants (<1.5 GB VRAM footprint).
- Tool-calling and agentic workflows using structured JSON schemas and Model Context Protocol (MCP).

### Out-of-Scope & Prohibited Uses
- **High-Stakes Decision-Making:** Autonomous medical diagnosis, legal judgment, credit underwriting, or critical infrastructure control without human-in-the-loop validation.
- **Ungrounded Factual Retrieval:** Using the model as an authoritative factual repository without Retrieval-Augmented Generation (RAG); 0.6B models have limited parametric capacity and higher hallucination rates than large frontier models.
- **Malicious Generation:** Creating malware, phishing lures, spam, automated disinformation, or harmful content violating safety guidelines.

---

## Ethical Considerations & Risk Mitigations

- **Multilingual Representation:** Pre-trained on 100+ languages, with highest fidelity in English, Chinese, and major world languages. Low-resource languages and regional dialects require dedicated fine-tuning.
- **Sociodemographic Biases:** Web-scraped pre-training data contains inherent societal disparities; downstream domain models should evaluate demographic fairness.
- **Safety Alignment:** Qwen3 incorporates safety refusal vectors during alignment; our training pipeline strictly isolates assistant response loss to prevent learning adversarial user prompt structures.
- **Supply-Chain Integrity:** Immutable revision pinning and `trust_remote_code=False` protect against remote code execution and model tampering.

---

## Attribution & Citation

The base model weights and architecture are developed by the **Qwen Team (Alibaba Cloud)** under the **Apache License 2.0**. This pipeline provides an independent open-weights training, evaluation, and packaging framework.

If you utilize Qwen3, QLoRA, or this pipeline in your research or applications, please cite:

```bibtex
@article{qwen3,
  title   = {Qwen3 Technical Report},
  author  = {Qwen Team, Alibaba Cloud},
  journal = {arXiv preprint},
  year    = {2025}
}

@article{dettmers2024qlora,
  title   = {QLoRA: Efficient Finetuning of Quantized LLMs},
  author  = {Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal = {Advances in Neural Information Processing Systems},
  volume  = {36},
  year    = {2023}
}
```
