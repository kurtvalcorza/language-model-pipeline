# Compatibility & Resource Matrix

Measured figures only. Nothing in this file is estimated — a guessed `min_vram_gb` reads as
authoritative and silently mis-sizes every DIMER registration built on it.

Reproduce with:

```bash
python scripts/measure_resource_profile.py --matrix --sequence-length 2048
```

(in `language-model-finetuner`, on the target hardware)

## Method

Each row is one full run through the real training path: load, attach adapter, train one
epoch, package, verify, clean-load, publish. Sequences are padded to the stated
`max_sequence_length` so peak memory reflects the worst case a real dataset reaches, not
whatever a short sample happened to use.

`min_vram_gb` in the registry is the measured peak allocation times a **1.25 safety margin**,
covering fragmentation, longer datasets and allocator variance. It is recorded **per method**,
not per model: QLoRA and LoRA of the same model differ by more than 2x here, so one number
per model would either block QLoRA where it fits or admit a LoRA run that cannot.

An out-of-memory result is a legitimate recorded outcome, not a failure of the measurement.

## Measurement hardware

| Field | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| VRAM | 11.94 GB |
| Compute capability | sm_120 (Blackwell) |
| bf16 | supported |
| Runtime | `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` in the `nvidia-docker` WSL distro |
| torch / transformers / peft / bitsandbytes | 2.8.0+cu128 / 4.57.1 / 0.18.0 / 0.49.0 |

**This is a single-GPU laptop part and is not DIMER's production hardware.** These numbers
establish relative cost and prove the code paths work; a profile may not be promoted past
`approval_state: experimental` until it has been measured on approved DIMER hardware.

## Results

<!-- MEASURED-MATRIX-START -->
Measured 2026-08-21, `max_sequence_length=2048`, `per_device_batch_size=1`,
`gradient_accumulation_steps=2`, 8 examples, 1 epoch.

| Model | Method | Peak alloc | Peak reserved | tok/s | Wall | Verdict |
|---|---|---:|---:|---:|---:|---|
| qwen3-0.6b | lora | 10.66 GB | 11.89 GB | 939.9 | 5.2s | fits, barely (99.6% of VRAM) |
| qwen3-1.7b | lora | 16.29 GB | 17.79 GB | 30.0 | 164.1s | **spilled — invalid** |
| qwen3-1.7b | qlora | 7.41 GB | 8.25 GB | 341.1 | 14.4s | fits |
| qwen3-4b | lora | — | — | — | — | **out of memory** |
| qwen3-4b | qlora | 9.21 GB | 11.43 GB | 160.3 | 30.7s | fits |
| granite-4.1-3b | lora | 24.35 GB | 26.05 GB | 9.1 | 756.9s | **spilled — invalid** |
| granite-4.1-3b | qlora | 6.96 GB | 9.56 GB | 185.0 | 37.2s | fits |

### The two "spilled" rows are not measurements

They reported success while allocating **more memory than the GPU physically has** — 16.29
and 24.35 GB on an 11.94 GB card. On WDDM/WSL the NVIDIA driver silently satisfies such
allocations from host RAM over PCIe rather than failing.

The throughput collapse is the tell, and it is unambiguous: granite QLoRA runs at 185 tok/s
while granite LoRA on identical data runs at **9.1 tok/s — 20x slower**. That figure measures
bus traffic, not the configuration.

**No registry profile is derived from a spilled run.** `measure_resource_profile.py` now
compares peak against total VRAM and marks such runs `status: spilled`, so this cannot be
recorded by accident again.

`qwen3-4b` LoRA did *not* spill — it hit a hard allocation failure and was reported as a
structured `RESOURCE_OOM`, which is the intended behaviour.

### What this hardware supports

Every QLoRA path fits comfortably. **Only the 0.6B model trains with plain LoRA**, and even
that reserves 99.6% of available VRAM. On a 12 GB card, QLoRA is not an optimization — it is
the only viable method above 1B parameters at 2048 tokens.

The dominant memory term for these models is the logits tensor, not the weights: Qwen3's
vocabulary is ~151k, so a single 2048-token sequence produces a logits matrix in the
gigabytes once upcast for the loss. That is why a 0.6B model can reach 10.66 GB.
<!-- MEASURED-MATRIX-END -->

## How the numbers are used

- **`min_vram_gb`** in `model-registry.yaml` gates the run, looked up by `(model, method)`. `finetuner/resources.py`
  compares it against the detected GPU and refuses **before** downloading weights, so a user
  on a small GPU tier who selects a large model gets an immediate, actionable error instead
  of a job that dies mid-download.
- An entry whose profile is still `null` does **not** block a run. An unmeasured profile is
  not a licence to invent a number; the job proceeds and, if it does not fit, fails as a
  structured `RESOURCE_OOM` carrying real measurements.
- **DIMER registrations map to GPU tiers, not to models.** These figures decide which models
  belong in which tier.
