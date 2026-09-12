# Base Model Weights Cache

This directory holds offline base-model weights, configurations, and cryptographic manifests for the language model pipeline.

Each model is organized in its own isolated subfolder corresponding to its canonical registry key from [`src/lmpipeline/data/model-registry.yaml`](../src/lmpipeline/data/model-registry.yaml):

```
weights/
└── qwen3-0.6b/
    ├── config.json
    ├── generation_config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── vocab.json
    ├── merges.txt
    ├── dimer-base-manifest.json
    ├── README.md (Upstream Model Card)
    ├── MODEL_CARD.md (DIMER Model Card Specification v1.0)
    ├── LICENSE
    └── model.safetensors (Excluded from Git; acquired via fetch_weights.py or DIMER upload)
```

## Available Base Model Snapshots

- [**`qwen3-0.6b`**](qwen3-0.6b/): Dedicated snapshot for the default lightweight causal language model (`Qwen/Qwen3-0.6B`, ~590M parameters).
  - [**DIMER Model Card**](qwen3-0.6b/MODEL_CARD.md): DIMER Model Card Specification v1.0 disclosures, limitations, metrics, and risks.
  - [**Upstream Model Card**](qwen3-0.6b/README.md): Upstream architecture, usage, and Apache-2.0 license information.
  - [**Manifest**](qwen3-0.6b/dimer-base-manifest.json): Cryptographic record of byte counts and SHA-256 hashes for all snapshot files.
- [**`smollm2-360m`**](smollm2-360m/): Dedicated snapshot for `HuggingFaceTB/SmolLM2-360M-Instruct` at the registry-pinned revision.
  - [**DIMER Model Card**](smollm2-360m/MODEL_CARD.md): DIMER Model Card Specification v1.0 disclosures, limitations, metrics, and risks.
  - [**Upstream Model Card**](smollm2-360m/README.md): Upstream training, evaluation, limitations, and Apache-2.0 license information.
  - [**Manifest**](smollm2-360m/dimer-base-manifest.json): Cryptographic record of byte counts and SHA-256 hashes for all snapshot files.

## DIMER Architecture & Git Tracking Strategy

In the DIMER workbench ecosystem:
1. **Large Binary Weights (`model.safetensors`):** The ~1.41 GiB weight payload is excluded from Git via `.gitignore` and uploaded directly to DIMER as a model asset or downloaded using `scripts/fetch_weights.py`.
2. **Configuration & Tokenizers:** All accompanying configuration files (`config.json`, `generation_config.json`), BPE tokenizer vocabularies (`tokenizer.json`, `vocab.json`, `merges.txt`), and cryptographic manifests are version-controlled in the repository so the pipeline and offline Docker containers can initialize models without network dependencies.

## Management & Verification Tooling

Manage, download, and cryptographically verify model snapshots using [`scripts/fetch_weights.py`](../scripts/fetch_weights.py):

```bash
# List all registered base models in the pipeline
python scripts/fetch_weights.py --list

# Download and verify default model into its dedicated subfolder:
python scripts/fetch_weights.py --model qwen3-0.6b --dest weights/qwen3-0.6b

# Verify an existing snapshot on disk:
python scripts/fetch_weights.py --verify-only --dest weights/qwen3-0.6b
```
