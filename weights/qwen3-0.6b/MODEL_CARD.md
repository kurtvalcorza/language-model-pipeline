---
license: apache-2.0
base_model: Qwen/Qwen3-0.6B
model_card_spec: "1.1"
model_key: qwen3-0.6b
base_model_revision: c1899de289a04d12100db370d81485cdf75e47ca
approval_state: experimental
---

# Qwen3-0.6B (DIMER Base Snapshot c1899de2) — Language Model (QLoRA Fine-Tuning & Adapter Inference)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Qwen%2FQwen3--0.6B-ffcc4d?style=flat)](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-QwenLM%2FQwen3-181717?style=flat&logo=github&logoColor=white)](https://github.com/QwenLM/Qwen3)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2505.09388-b31b1b.svg)](https://arxiv.org/abs/2505.09388)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Pipeline](https://img.shields.io/badge/Pipeline-language--model--pipeline-2ea44f?style=flat&logo=github)](https://github.com/kurtvalcorza/language-model-pipeline)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

The pipeline repository provides two ready-to-run interactive Google Colab notebooks that fine-tune a pinned base snapshot with QLoRA, export the adapter, and reload it in a fresh process:

- **QLoRA Fine-Tuning Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/language-model-pipeline/blob/main/tutorials/language_model_finetuning_colab.ipynb) [`language_model_finetuning_colab.ipynb`](https://github.com/kurtvalcorza/language-model-pipeline/blob/main/tutorials/language_model_finetuning_colab.ipynb)  
  *Supervised QLoRA fine-tuning of the pinned base model on chat-formatted data (shipped Filipino Q&A pairs), with baseline comparison, adapter export, and fresh reload.*

- **Adapter Artifact Inference Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/language-model-pipeline/blob/main/tutorials/language_model_artifact_inference_colab.ipynb) [`language_model_artifact_inference_colab.ipynb`](https://github.com/kurtvalcorza/language-model-pipeline/blob/main/tutorials/language_model_artifact_inference_colab.ipynb)  
  *Consume an externally supplied PEFT adapter ZIP, verify its manifest and base-model provenance, attach it to the pinned base snapshot, and generate text; no training occurs.*

> [!NOTE]
> Both notebooks require a Colab GPU runtime (*Runtime ▸ Change runtime type ▸ T4 GPU*); QLoRA's 4-bit path uses float16 on a T4 and bfloat16 on Ampere or newer.

---

###### Description

This package contains the instruction-tuned `Qwen/Qwen3-0.6B` release at immutable revision `c1899de289a04d12100db370d81485cdf75e47ca`. It is an approximately 590-million-parameter, decoder-only causal language model that produces text by predicting successive tokens. Its tokenizer can render either thinking or non-thinking chat prompts. The upstream weights provide the language behavior; this repository adds an allowlisted DIMER model key, pinned acquisition, a verified offline snapshot, dataset and training contracts, and LoRA/QLoRA adapter packaging. It does not claim authorship of the upstream model.

#### Intended Use and Limitations

The intended scope is deliberately narrower than every task the upstream model may technically attempt.

###### Primary Intended Uses

The model accepts tokenized conversational or prompt-completion text and generates a text continuation. Within DIMER it is intended as an internal experimental base for LoRA or QLoRA supervised fine-tuning, especially fast smoke tests of dataset ingestion, tokenizer rendering, assistant-only loss masking, training, adapter export, and clean reload. Appropriate domains include language-model engineering, educational demonstrations, low-cost prototyping, and early evaluation of narrowly scoped text transformations. It is a base component in a larger reviewed application, not a standalone factual authority or finished decision service.

###### Primary Intended Users

The intended users are machine-learning engineers, researchers, and educators working in internal development or research settings. The registry marks this entry `internal_only` and `experimental`; it is not offered as a user-facing Workbench selection. Operators are expected to understand causal-language-model prompting, supervised fine-tuning, train/validation/test separation, decoding controls, privacy review, and task-specific evaluation. They must also be able to inspect generated text and the recorded provenance rather than treating fluent output or decreasing optimization loss as proof of application quality.

###### Out-of-scope use cases

- **Workbench production selection:** this key is internal-only and absent from the user-facing model choices; promoting it requires the repository's production approval process.
- **Non-text modalities or calibrated classification:** the generic causal-LM backend accepts text and emits tokens. Images, audio, sensor arrays, calibrated probabilities, and fixed-label decisions require a modality- or task-specific pipeline.
- **Training sequences above 2,048 rendered tokens:** the registry ceiling is 2,048, and overlength examples are rejected rather than silently truncated.
- **Tool/function-call training records:** dataset contract v1 accepts only `system`, `user`, and `assistant` roles; tool-call records are deferred.
- **Autonomous high-impact decisions or authoritative retrieval:** medical, legal, safety, employment, credit, housing, and other consequential uses require independent validation and accountable human review. Generated facts must be checked against authoritative sources.

#### Factors

The following factors describe where behavior can vary and where this repository has no evidence.

###### Groups

This is a human-language model, so its outputs can describe, advise, or otherwise affect people even though it does not take demographic fields as a defined input schema. The complete demographic, cultural, linguistic, and socioeconomic composition of the upstream pretraining and post-training corpora is not audited in this repository. No DIMER group-disaggregated quality or safety result is reported for this snapshot. A downstream operator must evaluate their own prompts and outputs across relevant language varieties and protected or vulnerable groups, document disparities, and define escalation criteria before deployment.

###### Instrumentation

The pipeline consumes UTF-8 JSONL produced by external systems rather than raw physical sensors. Those systems may include application logs, content-management exports, databases, annotation tools, surveys, or human-authored files; this repository cannot infer which instrument produced a submitted record. Encoding errors, changed export schemas, annotation inconsistency, collection bias, or incorrect role mapping propagate into training examples as data error. The pipeline detects malformed schemas, invalid roles, empty assistant targets, invalid UTF-8, duplicate records, cross-split overlap, and rendered-token overflows, but it cannot detect whether the source statement or label was collected accurately.

###### Environment

The packaged configuration declares `bfloat16` base weights. In the DIMER causal-LM backend, **LoRA can execute on CPU or CUDA**, but the repository does not treat CPU execution as a qualified training profile: the compliance analysis notes that a general-purpose CPU node is not expected to finish LM SFT within the platform's six-hour budget. **QLoRA requires an NVIDIA CUDA device** in the current implementation; when `method == "qlora"` and CUDA is unavailable, the finetuner fails early with `RESOURCE_GPU_UNAVAILABLE` rather than falling back to CPU. The registry records a measured **13.3 GiB minimum for LoRA** on an RTX 5070 Ti Laptop at sequence length 2,048 and batch size 1. It records **no validated QLoRA VRAM ceiling** for this model, which means the resource profile is unmeasured—not that QLoRA is hardware-agnostic. ROCm, Metal/MPS, and non-CUDA quantization are not qualified by this DIMER backend. The data environment should resemble the language, register, task, and dialogue structure represented in the operator's evaluated data; domain shift, unfamiliar dialects, adversarial prompts, or longer interactions can degrade behavior even when the software executes successfully.

#### Metrics

Metrics describe optimization and operation; they do not establish whether generated answers are correct, safe, or useful.

###### Performance Measures

The base snapshot itself emits generated tokens and reports no DIMER task-quality score. A fine-tuning run reports `trainLoss`, `validationLoss`, optional `testLoss`, corresponding `trainPerplexity`, `validationPerplexity`, and optional `testPerplexity`, plus `epochsCompleted`, `optimizerSteps`, `examplesProcessed`, `supervisedTokens`, `wallSeconds`, `examplesPerSecond`, and `peakGpuMemoryBytes`. Loss and perplexity measure next-token fit on supervised assistant tokens; counts, time, throughput, and memory measure execution. They are appropriate optimization diagnostics, but none substitutes for a task-specific held-out rubric, factuality check, safety evaluation, or human assessment supplied by the operator.

###### Decision thresholds

The base model applies no classification threshold and emits no calibrated confidence. Generation selects tokens according to caller-supplied decoding settings; deterministic smoke tests may use greedy decoding, while sampled decoding introduces variability. Snapshot acceptance is exact: every manifest-listed file must match its byte count and SHA-256 digest, with zero mismatches tolerated. Early stopping is optional and uses caller-configured `patience` and `minDelta`; no universal loss target is shipped. Deployment acceptance thresholds remain with the operator, who must set them from held-out data and weigh false acceptance, harmful generation, and unnecessary rejection according to the application.

###### Approaches to uncertainty and variability

This repository reports no repeated-run confidence interval or variance for this base snapshot. Fine-tuning loss is estimated from one configured run: training loss is a supervised-token-weighted epoch average, validation loss is measured after each epoch, and an optional test split is evaluated once after model selection. Seeds control data shuffling, derived validation splitting, and adapter initialization, but bitwise GPU determinism is not claimed. Sampling settings, prompt wording, kernel nondeterminism, and training data all contribute variability. Token likelihoods are not calibrated correctness probabilities; operators need repeated seeds, prompt perturbations, representative held-out data, and calibration or human review appropriate to their task.

#### Ethical considerations and biases

No external ethics board or demographic clearance is claimed for this DIMER package.

###### Data

The packaged upstream card does not enumerate the complete Qwen3 pretraining and post-training corpora, so the presence of personal, proprietary, restricted, or otherwise sensitive material cannot be ruled out here. This repository tracks configuration, tokenizer assets, license material, the DIMER card, and cryptographic provenance; large `model.safetensors` weights are excluded from git and acquired separately. DIMER adapter artifacts contain learned adapter parameters and provenance, not raw training examples. Operators remain responsible for reviewing submitted datasets and prompts for consent, ownership, personal information, confidential content, and any restrictions the pipeline cannot infer.

###### Human Life

This package is not intended, independently validated, certified, or cleared for healthcare, life safety, criminal justice, employment, education access, insurance, credit, housing, or other decisions central to human rights and flourishing. A small generative model can still be embedded in such a workflow, so foreseeable sensitive use requires accountable human oversight, representative domain evaluation, documented error and escalation procedures, privacy and security review, and any applicable professional or regulatory authorization. None of those conditions is supplied merely by a valid snapshot manifest, a successful fine-tuning run, or fluent generated text.

###### Mitigations

- **Supply chain:** the registry allowlists `qwen3-0.6b`, pins a 40-character revision, requires SafeTensors, enforces `trust_remote_code=False`, and refuses silent fallback to another revision.
- **Snapshot integrity:** `dimer-base-manifest.json` records the size and SHA-256 digest of every other packaged file, and verification fails on missing, altered, extra, unsafe, or symlinked content.
- **Input integrity:** dataset normalization accepts three declared JSONL families, restricts roles to `system`, `user`, and `assistant`, requires a non-empty assistant target, preserves Unicode, and rejects overlength examples rather than truncating them.
- **Leakage and reproducibility:** canonical fingerprints expose duplicates and reject train/validation or train/test overlap; provenance records the model revision, dataset digest, resolved job configuration, package versions, and seed-controlled operations.
- **Refusals:** arbitrary Hugging Face identifiers and remote-code models are rejected, and this model remains internal-only. No statistical class-balancing mitigation is claimed for generative SFT; operators must justify dataset composition for their use case.

###### Risks and harms

- **Confident fabrication:** next-token generation can produce plausible but false or internally inconsistent text, especially outside evaluated domains. Users or third parties may act on it; likelihood is not measured here, and impact ranges from wasted effort to severe harm in consequential settings.
- **Bias amplification:** patterns in upstream or fine-tuning data can produce stereotyping, disparate quality, or abusive language. People represented by those groups bear the harm; DIMER has no group-disaggregated result for this snapshot.
- **Privacy leakage:** memorized upstream content or sensitive fine-tuning examples may be reproduced under prompting. Data subjects and dataset owners bear the risk, which rises when confidential data is supplied or outputs are logged broadly.
- **Automation bias:** fluent output may cause operators to skip verification. The harm is borne by decision subjects and users when the model is treated as an authority rather than an assistive component.
- **Data and prompt shift:** changed language, format, domain, or adversarial inputs can silently reduce quality after deployment. The pipeline validates structure, not truth or semantic representativeness.

###### Use cases

- The model must not be used to create or operate unlawful surveillance, biometric or demographic profiling, social scoring, or population-control systems.
- It must not be used to discriminate unlawfully in employment, housing, credit, insurance, education, healthcare access, public benefits, or comparable services.
- It must not be used for deceptive impersonation, targeted harassment, coercion, fraud, phishing, spam, non-consensual sexual content, child sexual abuse material, or manipulation of vulnerable people.
- It must not be used to provide autonomous instructions or decisions that create a material risk of death, serious injury, self-harm, or illegal weapons activity.
- Uses must comply with Apache-2.0, applicable law, dataset terms, privacy obligations, and the terms of the deployment environment; technical capability is not authorization.

## Packaged Snapshot Details

| Field | Value |
|---|---|
| DIMER model key | `qwen3-0.6b` |
| Upstream model | `Qwen/Qwen3-0.6B` |
| Revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| Architecture | `Qwen3ForCausalLM` |
| Upstream parameter label | 0.6B total; upstream card reports 0.44B non-embedding |
| Snapshot tensor | `model.safetensors`, 1,503,300,328 bytes, SHA-256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` |
| Model-config context | 40,960 positions |
| DIMER training ceiling | 2,048 rendered tokens |
| Training methods | LoRA and QLoRA |
| Serving profile | PEFT adapter |
| Registry state | Enabled, internal-only, experimental |

## Evidence and References

- [Canonical registry entry](../../src/lmpipeline/data/model-registry.yaml)
- [Snapshot manifest](dimer-base-manifest.json)
- [Packaged configuration](config.json)
- [Packaged upstream model card](README.md)
- [Dataset contract](../../DATASET_SPEC.md)
- [Training and metrics contract](../../TRAINING_SPEC.md)
- [Security and supply-chain controls](../../SECURITY.md)
- [Provenance contract](../../PROVENANCE_SPEC.md)
