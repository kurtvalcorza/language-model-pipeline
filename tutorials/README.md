# Language Model Fine-Tuning Tutorials

This directory contains comprehensive, standalone educational tutorials on **Supervised Fine-Tuning (SFT)** and **Parameter-Efficient Fine-Tuning (PEFT)** for modern causal language models.

The tutorials guide you through the end-to-end lifecycle of adapting an open foundation language model: from acquiring verified base weights and formatting multi-turn conversational datasets to 4-bit QLoRA training, before-and-after behavioral evaluation, standalone adapter packaging, and fresh reload verification.

| Tutorial Notebook | Purpose | Workflow Stages |
|---|---|---|
| [**Language-Model Fine-Tuning**](language_model_finetuning_colab.ipynb) | End-to-end supervised fine-tuning guide | Base-model acquisition → Dataset validation & normalization → Chat template formatting & assistant masking → Baseline generation → 4-bit QLoRA SFT → Before/after evaluation → Adapter bundle export → Clean reload test |
| [**Artifact Inference**](language_model_artifact_inference_colab.ipynb) | Production adapter verification & inference | Upload adapter bundle → Cryptographic manifest & provenance verification → Base-model resolution → Dynamic PEFT attachment → Multi-turn generation |

## Interactive Runtimes (Google Colab)

Run these interactive tutorials directly in your browser on standard single-GPU environments (e.g., Google Colab T4, V100, L4, or A100):

[![Open Fine-Tuning Tutorial In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/language-model-pipeline/blob/main/tutorials/language_model_finetuning_colab.ipynb)
[![Open Artifact Inference In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/language-model-pipeline/blob/main/tutorials/language_model_artifact_inference_colab.ipynb)
[![GitHub Repository](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/language-model-pipeline)

---

## What is Language Model Fine-Tuning?

Modern causal language models undergo distinct training stages before deployment:

1. **Pre-Training (Foundational Knowledge):** Models learn language structure, syntax, and broad factual knowledge by predicting the next token across trillions of tokens of unstructured web text. Pre-trained base models act as raw text completers rather than conversational assistants.
2. **Supervised Fine-Tuning (SFT / Instruction Tuning):** Adapts the pre-trained base model to follow human instructions and conversational conventions using curated prompt-response pairs. SFT aligns the model's output distribution to produce direct, structured, and helpful responses formatted with standardized chat templates.
3. **Parameter-Efficient Fine-Tuning (PEFT / QLoRA):** Instead of updating all billions of parameters during SFT (which requires prohibitive GPU memory and produces massive redundant checkpoints), PEFT freezes the base model and trains low-rank adapter matrices. QLoRA quantizes the frozen base model to 4-bit NormalFloat (`nf4`), allowing models with billions of parameters to be fine-tuned efficiently on accessible consumer GPUs (such as the free Google Colab 15 GB T4).

---

## Core Concepts Taught in These Tutorials

The fine-tuning tutorial is structured as an in-depth pedagogical guide rather than a black-box execution script. You will explore:

### 1. The Tripartite System: Base Model, Tokenizer, and Adapter
- **Base Model:** Encodes broad language understanding, world knowledge, and reasoning capabilities across billions of frozen parameters.
- **Tokenizer & Chat Template:** Translates natural language strings into integer sequences and standardizes conversational roles (`system`, `user`, `assistant`) using Jinja2 templates (e.g., ChatML, Llama-3 format).
- **PEFT Adapter:** Small, trainable low-rank decomposition matrices ($W = W_0 + \frac{\alpha}{r} B A$) injected into attention and MLP projection layers that adjust model tone, formatting, and task adherence without modifying the base weights.

### 2. Supply-Chain Integrity and Cryptographic Pinning
- Why pinning models to immutable 40-character Git commit SHAs prevents silent upstream modifications, ensures reproducibility, and protects against model tampering.
- Why enforcing `trust_remote_code=False` prevents arbitrary execution of untrusted remote Python code during model loading.

### 3. Assistant-Only Loss Masking
- How computing cross-entropy loss over entire sequences causes models to waste capacity memorizing user prompts.
- How prefix-stable tokenization identifies the exact token boundaries corresponding to assistant turns and assigns non-assistant tokens label value `-100` (PyTorch `ignore_index`).
- Why gradients should only update parameters based on the assistant's predicted response tokens.

### 4. 4-bit Quantization (QLoRA)
- How loading base model weights in 4-bit NormalFloat (`nf4`) with double quantization reduces GPU memory consumption by over 60%.
- How standard 16-bit fine-tuning of 3B+ models requires >24 GB VRAM, whereas QLoRA allows models up to 4B parameters to train within 6 GB to 10 GB of VRAM at standard conversational sequence lengths (≤512 tokens) using memory-efficient attention (SDPA). Note that activation memory scales super-linearly with sequence length—longer contexts (1k–2k+ tokens) require larger GPU memory allocations.

### 5. Recommended Hyperparameters & Configuration
- **LoRA Rank ($r$) & Alpha ($\alpha$):** Setting $r = 16$ and $\alpha = 32$ provides a robust balance between representational capacity and parameter efficiency ($\approx 0.5\% - 2\%$ of total parameters).
- **Learning Rate:** Typically $2 \times 10^{-4}$ with a cosine learning rate decay and $3\% - 10\%$ linear warmup.
- **Target Modules:** Injecting adapters into both attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and feed-forward MLP blocks (`gate_proj`, `up_proj`, `down_proj`) ensures superior adaptation compared to attention-only LoRA.

### 6. Optimization Metrics vs. Task Quality
- **Optimization Evidence:** Measures mathematical convergence on the training distribution (training cross-entropy loss, validation loss, validation perplexity $e^{\text{loss}}$, gradient stability).
- **Task Quality:** Evaluates whether the fine-tuned model actually performs well on intended human tasks, which requires held-out benchmark datasets, rubric scoring, or human evaluation.
- Why prompt probes before and after training illustrate behavioral shifts but do not constitute formal evaluation.

### 7. Adapter-First Export and Clean Reload
- Why exporting only the low-rank delta matrices and tokenizer files (typically 50–150 MB) is dramatically more portable than exporting redundant multi-gigabyte base model copies.
- How an `artifact-manifest.json` tracks byte counts and SHA-256 digests for all exported files.
- Why verifying a fresh reload from disk—clearing memory and attaching the saved adapter to a clean base model—is necessary to prove deployment readiness.

---

## Supported Open-Weights Models

The tutorial includes a curated collection of verified, open-weights causal language models pinned to immutable commit revisions:

| Model Key | Parameters | Primary Strengths & Characteristics | License |
|---|---|---|---|
| **`qwen3-0.6b`** | 0.6B | **Default tutorial candidate.** Ultra-fast download (~1.41 GiB) and rapid training loop (~1–2 min on Colab T4, ~50–100s on modern desktop GPUs). Ideal for quick verification. | Apache-2.0 |
| **`qwen3-1.7b`** | 1.7B | Strong conversational reasoning, multi-turn instruction following, and measured QLoRA memory profile (9.3 GiB at seq 2048, ~4.5–5.5 GiB at seq ≤ 512). | Apache-2.0 |
| **`qwen3-4b`** | 4.0B | High-capacity reasoning and knowledge retrieval; suitable for larger GPUs (>=12 GB VRAM). | Apache-2.0 |
| **`smollm3-3b`** | 3.0B | Balanced modern 3B architecture from Hugging Face TB. | Apache-2.0 |
| **`granite-4.1-3b`** | 3.0B | IBM Granite model with audited enterprise data governance and strong tool-calling performance. | Apache-2.0 |
| **`granite-3.1-2b-instruct`** | 2.5B | Compact enterprise-focused model optimized for instruction following and structured outputs. | Apache-2.0 |
| **`deepseek-r1-distill-qwen-1.5b`** | 1.8B | Open reasoning specialist distilled from DeepSeek-R1; outputs step-by-step `<think>...</think>` traces. | MIT |
| **`qwen2.5-coder-1.5b`** | 1.5B | Dedicated code and structured JSON/SQL generation specialist. | Apache-2.0 |
| **`smollm2-1.7b`** | 1.7B | Lightweight model trained on curated educational corpora (Cosmopedia v2, FineWeb-Edu). | Apache-2.0 |
| **`smollm2-360m`** | 360M | Ultra-compact edge architecture (~720 MiB); trains in seconds even on resource-limited hardware. | Apache-2.0 |
| **`h2o-danube3-4b-chat`** | 4.0B | Mobile- and edge-optimized architecture developed by H2O.ai. | Apache-2.0 |
| **`llama-3.2-3b-instruct`** | 3.2B | Gated community model. Demonstrates authenticated Colab Secrets (`HF_TOKEN`) workflow. | Llama 3.2 Community |

Every model in the registry enforces `trust_remote_code=False` to ensure safe weight instantiation.

---

## Base-Model Acquisition

Two acquisition modes are supported:

- **Pinned Hugging Face (Default):** Downloads the exact immutable revision directly from the Hugging Face Hub using `snapshot_download`.
- **Verified Offline Snapshot (DIMER ZIP):** Loads an offline base-model package (such as one created using our CLI fetch tool). To protect against corrupt or tampered files, the package must contain a `dimer-base-manifest.json` file. The extraction process verifies path safety, checks that every listed file matches its recorded byte count and SHA-256 hash, and confirms the presence of standard `.safetensors` weights.

### Authenticated Token for Gated Models
For gated checkpoints (such as Llama 3.2 3B Instruct):
1. Accept the model terms on Hugging Face;
2. Add a secret named `HF_TOKEN` in Colab Secrets;
3. Grant the notebook access to that secret; and
4. Select `Pinned Hugging Face` as the model source.

The notebook reads the token in memory through `google.colab.userdata` for authenticated preflight checks and never prints or persists the credential into output artifacts.

---

## Dataset Formats & Data Hygiene

The tutorial normalizes diverse instruction-tuning schemas into a canonical conversational representation:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful and concise assistant."},
    {"role": "user", "content": "Explain machine learning in one sentence."},
    {"role": "assistant", "content": "Machine learning is the science of training algorithms to learn patterns from data and make predictions."}
  ]
}
```

### Supported Dataset Modes
- **Sample: Filipino SFT:** `jpaulpoliquit/ph-sft-ai-authored-v1`; bilingual Filipino/English conversational dataset.
- **Sample: Dolly:** `databricks/databricks-dolly-15k`; general-purpose English instruction dataset.
- **Bring Your Own Dataset (BYOD):** Upload custom JSONL files supporting multi-turn `messages`, `prompt`/`completion`, or Alpaca `instruction`/`input`/`output` records, with optional validation and test splits.

### Data Hygiene Checks
- **Assistant Target Verification:** Verifies that every training example contains at least one non-empty assistant response turn to supervise.
- **Split Leakage Detection:** Computes deterministic SHA-256 fingerprints of canonical records and immediately halts training if identical examples appear in both training and validation splits.
- **Duplicate Awareness:** Detects and reports exact duplicate examples within splits for transparency.
- **Sequence Length Boundaries:** Flags and rejects examples exceeding `MAX_SEQUENCE_LENGTH` rather than silently truncating them, preventing context corruption.

---

## Common Fine-Tuning Pitfalls to Avoid

| Pitfall | Consequence | Prevention Strategy |
|---|---|---|
| **Training on Prompt Tokens** | Model spends capacity memorizing user inputs; generates repetitive user questions. | Use assistant-only loss masking with label `ignore_index = -100` on prompt tokens. |
| **Chat Template Mismatch** | Model fails to generate turn-separators or hallucinates conversation roles during inference. | Apply tokenizer chat templates (`tokenizer.apply_chat_template`) during training and inference. |
| **Overfitting & Degradation** | Validation loss increases while training loss drops; general knowledge and reasoning degrade. | Monitor validation loss early-stopping, use modest learning rates ($10^{-4}$ to $2 \times 10^{-4}$), and keep LoRA rank moderate ($r = 8 - 16$). |
| **Train/Validation Split Leakage** | Optimistic validation metrics that fail to reflect real-world generalization. | Deduplicate dataset and hash records to ensure strict zero-overlap between splits. |
| **Unquantized Merging on Quantized Weights** | Fusing FP16 LoRA deltas into 4-bit base weights degrades numerical precision. | Always dequantize base weights to FP16/BF16 before merging adapter matrices. |

---

## Deploying Fine-Tuned Adapters

Once training completes and the self-contained adapter package is exported, you can deploy it using several industry-standard serving architectures:

1. **High-Throughput Multi-LoRA Serving (vLLM / SGLang / TGI):**
   Modern serving engines support dynamic multi-LoRA routing, enabling hundreds of specialized fine-tuned adapters to run concurrently on top of a single shared base model instance with minimal GPU memory overhead.
2. **Local Python Integration (PEFT & Transformers):**
   Load the base model and dynamically attach the adapter in evaluation mode (`is_trainable=False`) as demonstrated in our companion [Artifact Inference](language_model_artifact_inference_colab.ipynb) tutorial.
3. **Zero-Overhead Weight Merging:**
   Fuse the trained low-rank matrices directly into the base model weights using `model.merge_and_unload()`. This produces a standalone, unified checkpoint that can be served by any standard inference framework without PEFT dependencies or adapter latency overhead.
