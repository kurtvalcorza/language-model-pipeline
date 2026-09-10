# Tutorial Release Verification

DIMER Notebook Specification 1.0 requires clean-runtime execution evidence in addition to static
source conformance. This file is the durable repository record for that gate.

## Current candidate status

| Notebook | Profile | Source conformance | Clean-runtime execution | Release status |
|---|---|---|---|---|
| `language_model_finetuning_colab.ipynb` | `E2E` | Enforced by `scripts/validate_colab_tutorials.py` and tests | No passing record has been entered for the current candidate revision | **Blocked from release-grade label** |
| `language_model_artifact_inference_colab.ipynb` | `ARTIFACT-INFERENCE` | Enforced by `scripts/validate_colab_tutorials.py` and tests | No passing record has been entered for the current candidate revision | **Blocked from release-grade label** |

Do not change either status to release-grade until a reviewer enters a concrete passing record
below. Static validation, an old notebook run, or a warmed developer cache is not a substitute.

## Required E2E verification procedure

1. Open `language_model_finetuning_colab.ipynb` at the exact PR head/commit being released.
2. Start a clean supported Colab GPU runtime; do not rely on a previously installed package/model
   cache for the release record.
3. Run the default sample path top-to-bottom without manually editing implementation cells.
4. Record the runtime identity printed by the notebook, including Python, critical package
   versions, Torch/CUDA build and GPU.
5. Confirm canonical model/revision resolution, sample selection/validation, assistant-only
   masking, deterministic baseline generation, QLoRA training, held-out optimization metrics,
   new-prompt JSONL output, adapter manifest/provenance export, and fresh serialized-artifact
   reconstruction all complete without error.
6. Preserve the produced `dimer-language-model-adapter.zip` and its printed SHA-256 for the
   separate artifact-inference verification.

## Required artifact-inference verification procedure

1. Start a **new clean runtime** at the same candidate PR head/commit.
2. Supply the adapter ZIP produced by the E2E verification from outside this notebook execution.
3. If available, provide the ZIP SHA-256 through a trusted independent channel and verify it.
4. Run the notebook top-to-bottom with its default editable new prompt.
5. Confirm archive/manifest validation, canonical model/revision parity, producer/consumer runtime
   compatibility, base + tokenizer + adapter reconstruction, context validation, generation,
   `artifact_inference_predictions.jsonl`, and `artifact_inference_provenance.json` all complete.
6. Confirm the notebook does not create the consumed adapter during the same execution.

## Passing record format

Add one record per notebook and revision. Do not use approximate dates, branch-only identifiers,
or a bare statement such as "works in Colab".

```text
Notebook: <filename>
Profile: <E2E or ARTIFACT-INFERENCE>
Commit/PR head: <40-character SHA>
Executed: <YYYY-MM-DD HH:MM timezone>
Environment: <Colab/Jupyter runtime image if exposed>
Python: <version>
Torch/CUDA: <versions>
GPU: <device>
Critical packages: <transformers / tokenizers / peft / bitsandbytes / safetensors>
Input: <default sample OR external artifact SHA-256 + prompt/input identity>
Outcome: PASS
Evidence notes: <key outputs/artifact identifiers; no secrets or private data>
Reviewer/operator: <name or GitHub identity>
```

## Passing records

No passing records are asserted in this change. The builder environment used to author and
statically validate these files does not provide a clean Colab GPU execution substrate, so
claiming REL1/REL5 evidence here would be false. The release PR must remain non-release-grade
until the procedure above is executed and the resulting evidence is committed or otherwise
recorded durably in the PR/repository.
