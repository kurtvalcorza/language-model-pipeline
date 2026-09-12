---
license: apache-2.0
base_model: HuggingFaceTB/SmolLM2-360M-Instruct
model_card_spec: "1.0"
model_key: smollm2-360m
base_model_revision: a10cc1512eabd3dde888204e902eca88bddb4951
approval_state: experimental
---

# SmolLM2-360M-Instruct — DIMER Base Snapshot a10cc151

[![Hugging Face](https://img.shields.io/badge/Hugging_Face-SmolLM2--360M--Instruct-FFD21E)](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/tree/a10cc1512eabd3dde888204e902eca88bddb4951)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

###### Description

This package contains `HuggingFaceTB/SmolLM2-360M-Instruct` at immutable revision `a10cc1512eabd3dde888204e902eca88bddb4951`. It is the instruction-tuned member of the approximately 360-million-parameter SmolLM2 family, implemented by the packaged configuration as a decoder-only `LlamaForCausalLM`. At inference it renders a chat conversation and predicts successive output tokens. The upstream weights provide English-oriented language behavior; this repository adds an allowlisted DIMER key, pinned and verified offline acquisition, dataset and training contracts, and LoRA/QLoRA PEFT-adapter packaging. It does not retrain or claim authorship of the downloaded base weights.

#### Intended Use and Limitations

The model is packaged for bounded experimentation and pipeline validation, not unrestricted production use.

###### Primary Intended Uses

The model accepts tokenized text conversations or normalized prompt-completion records and generates text continuations. Within DIMER it is intended as an internal experimental base for rapid LoRA and QLoRA fine-tuning, pipeline smoke tests, and low-resource demonstrations. Concrete uses include English rewriting, summarization experiments, short-form conversational prototypes, education, and verification of ingestion, masking, optimization, adapter export, and reload workflows. Its role is a compact base model inside an evaluated application or engineering test harness; it is not a complete factual, safety, or decision-making service.

###### Primary Intended Users

The intended users are machine-learning engineers, researchers, educators, and application developers operating in internal development or research environments. The registry marks this model `internal_only` and `experimental`, so ordinary Workbench users are not the target audience. Operators should understand text-generation prompting, SFT and PEFT, train/validation/test separation, sampling behavior, privacy and license review, and the limitations of optimization metrics. They are expected to design task-specific evaluation and to review generated content rather than interpreting compact size, successful execution, or upstream benchmark tables as deployment approval.

###### Out-of-scope use cases

- **Workbench production selection:** this model is internal-only and has no measured resource profile; it must not be presented as a production-sized user option without the required measurement and approval review.
- **Languages requiring demonstrated parity with English:** the upstream card says the model primarily understands and generates English. Other languages require direct evaluation and potentially targeted adaptation.
- **Non-text modalities or calibrated decisions:** images, audio, sensor arrays, fixed-label classification, ranking, and calibrated probabilities require a suitable task-specific pipeline.
- **Training sequences above 2,048 rendered tokens:** DIMER rejects examples over the registry ceiling instead of truncating them. Tool/function-call records are also outside dataset contract v1.
- **Autonomous high-impact decisions or authoritative retrieval:** generated text must not replace verified sources or accountable experts in medical, legal, safety, employment, credit, housing, or similarly consequential domains.

#### Factors

Behavior depends on both the upstream corpus and the operator's downstream data and runtime choices.

###### Groups

This is a human-language model whose generated outputs may refer to or affect people, although the DIMER input schema does not define demographic features. The upstream card reports English as the primary language and does not provide a demographic composition audit or group-disaggregated safety results for the full pretraining and alignment corpora. This repository reports no DIMER group-sliced evaluation for the snapshot. Downstream operators must test relevant language varieties, cultural contexts, protected characteristics, and vulnerable populations on representative prompts, record disparities, and establish human escalation when output quality or harm differs across groups.

###### Instrumentation

The DIMER pipeline consumes UTF-8 JSONL exported from external systems rather than direct sensor readings. Inputs may originate in application logs, databases, annotation platforms, content repositories, surveys, or manually prepared files; the model card cannot infer the original collection instrument. Encoding, annotation policy, changed export schemas, omitted context, and role assignment materially shape the text seen during training. Structural checks detect invalid UTF-8, mixed or unknown schemas, invalid roles, empty assistant targets, duplicates, split overlap, and token-length overflow, but they do not determine whether source text, annotations, consent, or factual content are correct.

###### Environment

The snapshot configuration declares `bfloat16` base weights and an 8,192-position upstream architecture, while the DIMER registry caps training examples at 2,048 rendered tokens. In the current generic causal-LM backend, **LoRA can execute on CPU or CUDA**, but CPU LM fine-tuning is not a qualified six-hour DIMER training profile. **QLoRA requires an NVIDIA CUDA device** and fails early with `RESOURCE_GPU_UNAVAILABLE` when CUDA is unavailable; it does not silently fall back to CPU. Both LoRA and QLoRA resource-profile measurements are absent for this model, so this card publishes no minimum VRAM estimate. That absence is an evidence gap, not a claim that arbitrary hardware is supported. ROCm, Metal/MPS, and non-CUDA quantization are not qualified by this DIMER backend. The expected data environment is short-form, primarily English text resembling the operator's evaluated task data. Domain shift, other languages, long conversations, adversarial prompts, and differing annotation conventions can reduce quality even when loading and training complete successfully.

#### Metrics

The package separates upstream benchmark evidence, DIMER optimization telemetry, and downstream task quality.

###### Performance Measures

The packaged upstream card reports zero-shot or stated few-shot results produced with `lighteval`, including IFEval, MT-Bench, HellaSwag, ARC, PIQA, MMLU, BBH, and GSM8K; this repository has not reproduced those values. A DIMER fine-tuning run instead reports `trainLoss`, `validationLoss`, optional `testLoss`, `trainPerplexity`, `validationPerplexity`, optional `testPerplexity`, `epochsCompleted`, `optimizerSteps`, `examplesProcessed`, `supervisedTokens`, `wallSeconds`, `examplesPerSecond`, and `peakGpuMemoryBytes`. These measure token prediction fit and execution cost. Operators must supply representative held-out tests or human rubrics for correctness, safety, factuality, and application utility because neither upstream benchmarks nor optimization loss proves those outcomes.

###### Decision thresholds

The base model has no class boundary, pass/fail score, or calibrated confidence threshold. Decoding chooses output tokens using operator-supplied greedy or sampling parameters. Snapshot verification has an exact acceptance rule: every packaged file must match the manifest's byte count and SHA-256 digest, and any missing, altered, extra, unsafe, or linked file fails verification. Optional early stopping uses caller-selected `patience` and `minDelta`; DIMER ships no universal target loss or task-quality cutoff. The deployment owner must set acceptance and escalation thresholds from held-out evidence and explicitly weigh harmful false acceptance against unnecessary refusal.

###### Approaches to uncertainty and variability

No repeated-run DIMER performance distribution is available for this snapshot, and the registry contains no measured resource profile. In fine-tuning, training loss is a supervised-token-weighted epoch average, validation loss is measured after each epoch, and an optional untouched test split is evaluated once after training and model selection. Seeds control shuffling, a derived validation split, and adapter initialization, while bitwise GPU determinism is not promised. Sampling, prompt wording, data composition, and non-deterministic kernels can change generated text. Token scores are not calibrated truth probabilities; repeated seeds, prompt perturbations, representative held-out evaluation, and task-appropriate calibration remain operator responsibilities.

#### Ethical considerations and biases

No external ethics-board review or demographic clearance is claimed for this DIMER package.

###### Data

The upstream card reports four trillion pretraining tokens drawn from FineWeb-Edu, DCLM, The Stack, and additional filtered datasets that were not fully disclosed there. It reports SFT using public and curated data and DPO using UltraFeedback. Because the complete records are not enumerated, personal, proprietary, biased, or otherwise sensitive content cannot be ruled out. This repository stores configuration, tokenizer assets, upstream metadata, this DIMER card, and cryptographic provenance; the large SafeTensors file is git-ignored and acquired separately. Operators must review their own datasets and prompts for consent, ownership, confidentiality, personal data, and license restrictions.

###### Human Life

This package is not intended, independently validated, certified, or cleared for healthcare, life-safety control, criminal justice, employment, education access, insurance, credit, housing, or other high-impact decisions. Sensitive deployment remains foreseeable because generated text can be inserted into many workflows. Such use would require accountable human oversight, representative domain and subgroup validation, documented failure and escalation handling, privacy and security review, and any required professional or regulatory authorization. A valid snapshot, an upstream benchmark result, or a successful fine-tuning job supplies none of those safeguards by itself.

###### Mitigations

- **Supply chain:** the registry allowlists `smollm2-360m`, pins a 40-character revision, requires SafeTensors, enforces `trust_remote_code=False`, and refuses fallback to a moving branch or unrelated cache entry.
- **Snapshot integrity:** `dimer-base-manifest.json` records every other packaged file's byte count and SHA-256 digest; verification rejects missing, changed, extra, unsafe, and symlinked content.
- **Input integrity:** normalization accepts declared conversational, prompt/completion, and instruction JSONL families, permits only `system`, `user`, and `assistant` roles, requires a non-empty assistant target, and rejects overlength examples instead of truncating them.
- **Leakage and reproducibility:** canonical fingerprints report duplicates and reject cross-split overlap; provenance records the pinned model, dataset digest, effective job settings, package versions, and seeded operations.
- **Refusals:** arbitrary Hub identifiers and remote-code models are rejected, and this unmeasured model stays internal-only. No class-balancing mechanism is claimed for generative SFT; dataset composition remains an operator decision.

###### Risks and harms

- **Factual or logical error:** the upstream card warns that outputs may be inaccurate or inconsistent. Users and affected third parties bear the harm when fluent English text is mistaken for verified information; likelihood and magnitude are task-dependent and not measured here.
- **Bias amplification:** upstream and downstream text can encode stereotypes or unequal representation. Harm falls on represented groups, especially when deployment data differs from evaluation data; no group-disaggregated DIMER result currently bounds this risk.
- **Language exclusion:** primarily English behavior can create lower quality, misunderstood instructions, or inequitable service for other-language users. The risk becomes material when operators deploy without multilingual evaluation.
- **Privacy leakage:** training data may contain sensitive content that can be reproduced or inferred under prompting. Data subjects and owners bear the risk when confidential datasets are accepted or outputs are retained broadly.
- **Automation bias and misuse:** compact local execution can encourage unreviewed automation or scaled deceptive generation. Structural validation cannot determine whether an output is truthful, fair, lawful, or benign.

###### Use cases

- The model must not be used for unlawful surveillance, biometric or demographic profiling, social scoring, or systems intended to control or suppress a population.
- It must not be used to discriminate unlawfully in employment, housing, credit, insurance, education, healthcare access, public services, or comparable opportunities.
- It must not be used for fraud, phishing, deceptive impersonation, targeted harassment, coercion, spam, non-consensual sexual content, child sexual abuse material, or manipulation of vulnerable people.
- It must not autonomously provide instructions or decisions creating a material risk of death, serious injury, self-harm, illegal weapons activity, or deprivation of rights.
- Every use must comply with Apache-2.0, applicable dataset terms, privacy and intellectual-property obligations, law, and deployment-provider terms; offline capability does not remove those constraints.

## Packaged Snapshot Details

| Field | Value |
|---|---|
| DIMER model key | `smollm2-360m` |
| Upstream model | `HuggingFaceTB/SmolLM2-360M-Instruct` |
| Revision | `a10cc1512eabd3dde888204e902eca88bddb4951` |
| Architecture | `LlamaForCausalLM` decoder-only causal LM |
| Upstream parameter label | 360M |
| Snapshot tensor | `model.safetensors`, 723,674,912 bytes, SHA-256 `e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e` |
| Model-config context | 8,192 positions |
| DIMER training ceiling | 2,048 rendered tokens |
| Training methods | LoRA and QLoRA |
| Serving profile | PEFT adapter |
| Registry state | Enabled, internal-only, experimental; resource profile unmeasured |

## Evidence and References

- [Canonical registry entry](../../src/lmpipeline/data/model-registry.yaml)
- [Snapshot manifest](dimer-base-manifest.json)
- [Packaged configuration](config.json)
- [Packaged upstream model card](README.md)
- [Dataset contract](../../DATASET_SPEC.md)
- [Training and metrics contract](../../TRAINING_SPEC.md)
- [Security and supply-chain controls](../../SECURITY.md)
- [Provenance contract](../../PROVENANCE_SPEC.md)
- [SmolLM2 paper](https://arxiv.org/abs/2502.02737)
