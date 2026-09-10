# Tutorial Release Verification

DIMER Notebook Specification 1.0 requires clean-runtime execution evidence in addition to
static source conformance. This file is the durable repository record for that gate.

## Revision model

A notebook release has **three independently immutable source identities**. Do not collapse
them into one ambiguous "current revision":

1. **Notebook / release-candidate revision** — the `language-model-pipeline` PR head that
   contains the notebook, registry, static conformance gate, and this release record.
2. **Pipeline runtime-support revision** — the exact `language-model-pipeline` commit
   installed inside the notebook. For this candidate:  
   `8a9935c20f90d90f333ce2a191eedb001a1f0830`.
3. **Production finetuner revision** — the exact `language-model-finetuner` commit checked
   out and imported by both notebooks. For this candidate:  
   `4db4338b28db2260060ca3642fd29e3ca5e19119` (language-model-finetuner PR #30).

The notebooks print and record the two runtime revisions, and the artifact-inference notebook
requires its consumed artifact to have been produced with the same runtime revisions.

## Current candidate status

| Notebook | Profile | Source conformance | Clean-runtime execution | Release status |
|---|---|---|---|---|
| `language_model_finetuning_colab.ipynb` | `E2E` | Enforced by `scripts/validate_colab_tutorials.py` and tests | No passing record for the candidate head + runtime revisions above | **Blocked from release-grade label** |
| `language_model_artifact_inference_colab.ipynb` | `ARTIFACT-INFERENCE` | Enforced by `scripts/validate_colab_tutorials.py` and tests | No passing record for the candidate head + runtime revisions above | **Blocked from release-grade label** |

Static validation, an old notebook run, or a warmed developer cache is not a substitute for
clean-runtime evidence.

## Required E2E verification procedure

1. Open `language_model_finetuning_colab.ipynb` at the exact PR head/commit being released.
2. Start a clean supported Colab GPU runtime; do not rely on a previously installed
   package/model cache for the release record.
3. Confirm the notebook prints pipeline runtime revision
   `8a9935c20f90d90f333ce2a191eedb001a1f0830` and finetuner runtime revision
   `4db4338b28db2260060ca3642fd29e3ca5e19119`.
4. Run the default sample path top-to-bottom without manually editing implementation cells.
5. Record Python, critical package versions, Torch/CUDA build and GPU.
6. Confirm canonical model/revision resolution, sample selection/validation, production
   assistant-only masking, deterministic baseline generation, production QLoRA training,
   held-out optimization metrics, new-prompt JSONL output, production artifact staging and
   manifest verification all complete without error.
7. Confirm the fresh serialized reconstruction reports non-zero LoRA B weights **and** a
   non-zero adapter-on/off logit delta on the same reloaded model.
8. Preserve `dimer-language-model-adapter.zip` and its SHA-256 for the separate
   artifact-inference verification.

## Required artifact-inference verification procedure

1. Start a **new clean runtime** at the same candidate notebook/PR head.
2. Confirm the same pipeline-support and finetuner runtime revisions listed above.
3. Supply the adapter ZIP produced by the E2E verification from outside this notebook
   execution.
4. If available, provide the ZIP SHA-256 through a trusted independent channel and verify it.
5. Run the notebook top-to-bottom with its default editable new prompt.
6. Confirm archive/manifest validation, canonical model/revision parity, package compatibility,
   runtime-source parity, production base + tokenizer + adapter reconstruction, non-zero
   adapter-activity evidence, context validation, generation, `artifact_inference_predictions.jsonl`,
   and `artifact_inference_provenance.json` all complete.
7. Confirm the notebook does not create the consumed adapter during the same execution.

## Passing record format

Add one record per notebook and release candidate. All three source identities are mandatory.

```text
Notebook: <filename>
Profile: <E2E or ARTIFACT-INFERENCE>
Notebook/PR head: <40-character language-model-pipeline SHA containing the notebook>
Pipeline runtime revision: 8a9935c20f90d90f333ce2a191eedb001a1f0830
Finetuner runtime revision: 4db4338b28db2260060ca3642fd29e3ca5e19119
Executed: <YYYY-MM-DD HH:MM timezone>
Environment: <Colab/Jupyter runtime image if exposed>
Python: <version>
Torch/CUDA: <versions>
GPU: <device>
Critical packages: <transformers / tokenizers / peft / bitsandbytes / safetensors>
Input: <default sample OR external artifact SHA-256 + prompt/input identity>
Adapter activity: <LoRA-B max abs + adapter-on/off logit max abs delta>
Outcome: PASS
Evidence notes: <key outputs/artifact identifiers; no secrets or private data>
Reviewer/operator: <name or GitHub identity>
```

## Passing records

No passing records are asserted in this change. The builder environment does not provide a
clean Colab GPU execution substrate, so claiming REL1/REL5 evidence here would be false. The
release PR must remain non-release-grade until both procedures above are executed and the
resulting evidence is recorded durably.
