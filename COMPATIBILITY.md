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
covering fragmentation, longer datasets and allocator variance.

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
_Pending: populated by the measurement run._
<!-- MEASURED-MATRIX-END -->

## How the numbers are used

- **`min_vram_gb`** in `model-registry.yaml` gates the run. `finetuner/resources.py`
  compares it against the detected GPU and refuses **before** downloading weights, so a user
  on a small GPU tier who selects a large model gets an immediate, actionable error instead
  of a job that dies mid-download.
- An entry whose profile is still `null` does **not** block a run. An unmeasured profile is
  not a licence to invent a number; the job proceeds and, if it does not fit, fails as a
  structured `RESOURCE_OOM` carrying real measurements.
- **DIMER registrations map to GPU tiers, not to models.** These figures decide which models
  belong in which tier.
