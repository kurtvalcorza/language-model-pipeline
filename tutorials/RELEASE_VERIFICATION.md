# Tutorial Release Verification

DIMER Notebook Specification 1.0 requires clean-runtime execution evidence in addition to
static source conformance. This file is the durable repository record for that gate.

## Revision model

A notebook release has three independently immutable source identities:

1. **Notebook / release-candidate revision** — the `language-model-pipeline` PR head that
   contains the exact notebook bytes being executed.
2. **Pipeline runtime-support revision** —
   `afaf1f032cd7e9751db1ee8eb71b542f9bd0b15f`.
3. **Production finetuner revision** —
   `3772f0ca4e0f7130ffd5f826ea40f43b7212339e`.

The finetuner revision contains the shared production reconstruction/generation surface,
complete critical package provenance including `safetensors`, explicit BF16/FP16 CUDA
reconstruction policy, and the versioned PEFT artifact manifest.

Both notebooks record the runtime-source revisions they execute. The E2E producer additionally
writes them into tutorial artifact provenance. Artifact inference enforces those source SHAs
when present; ordinary production artifacts that do not contain the tutorial-only
`runtimeRevisions` extension are validated through the production model/revision and critical
package compatibility contract instead.

## Access prerequisite

`language-model-finetuner` is private. For each clean Colab run, configure a Colab Secret named
`GITHUB_TOKEN`, grant the notebook access, and use a token with read access to that repository.
The notebook passes the credential through an ephemeral Git HTTP header, never prints it or
embeds it in a URL, and deletes its local token binding after checkout.

**Never place the token or any other secret in this release record, notebook output, uploaded
artifact, screenshot, or evidence note.**

## Current candidate status

| Notebook | Profile | Source conformance | Clean-runtime execution | Release status |
|---|---|---|---|---|
| `language_model_finetuning_colab.ipynb` | `E2E` | Enforced by static validator and tests | No passing clean-run record | **Blocked from release-grade label** |
| `language_model_artifact_inference_colab.ipynb` | `ARTIFACT-INFERENCE` | Enforced by static validator and tests | No passing clean-run record | **Blocked from release-grade label** |

Static validation, a previous notebook revision, a warm developer cache, or a blocked attempt
is not a substitute for clean-runtime evidence.

## Required E2E verification procedure

1. Resolve the current PR #62 head and confirm exact-head CI is green.
2. Open `language_model_finetuning_colab.ipynb` at that exact 40-character candidate SHA.
3. Start a new supported Google Colab CUDA runtime with the authorized `GITHUB_TOKEN` Secret;
   do not rely on previously installed package/model/source caches.
4. Confirm the notebook reports pipeline revision
   `afaf1f032cd7e9751db1ee8eb71b542f9bd0b15f` and finetuner revision
   `3772f0ca4e0f7130ffd5f826ea40f43b7212339e`.
5. Run the default `qwen3-1.7b` / pinned Filipino SFT sample path top-to-bottom without editing
   implementation cells.
6. Record Python, critical package versions, Torch/CUDA build, GPU identity, model/revision,
   dataset revision/digest, effective splits, principal optimization metrics and controls.
7. Confirm real new-input inference and machine-readable JSONL/CSV/JSON outputs complete.
8. Confirm the artifact manifest reports `format = peft_adapter`, `formatVersion = 1`, every
   load-bearing file verifies, and fresh reconstruction succeeds.
9. Confirm adapter activity reports a non-zero LoRA-B maximum absolute value and non-zero
   adapter-on/off logit maximum absolute delta.
10. Download and preserve `dimer-language-model-adapter.zip` and its whole-ZIP SHA-256 for the
    separate artifact-inference verification.

## Required artifact-inference verification procedure

1. Start a **different new clean Colab CUDA runtime** at the same candidate SHA with the
   authorized `GITHUB_TOKEN` Secret.
2. Supply the preserved E2E adapter ZIP from outside this notebook execution. Do not recreate
   or repack it.
3. If available, enter the expected whole-ZIP SHA-256 obtained through an independently trusted
   channel.
4. Run the companion notebook top-to-bottom with its editable new prompt.
5. Confirm root-level archive validation, manifest format/version, canonical model/revision,
   package compatibility, applicable runtime-source parity, production reconstruction,
   adapter-activity evidence, context validation, generation and all machine-readable outputs.
6. Confirm this second notebook did not train or create the consumed adapter.

## Passing record format

Add one record per notebook and release candidate. All source identities known to the executed
notebook must be recorded.

```text
Notebook: <filename>
Profile: <E2E or ARTIFACT-INFERENCE>
Notebook/PR head: <40-character language-model-pipeline SHA>
Pipeline runtime revision: afaf1f032cd7e9751db1ee8eb71b542f9bd0b15f
Finetuner runtime revision: 3772f0ca4e0f7130ffd5f826ea40f43b7212339e
Executed: <YYYY-MM-DD HH:MM timezone>
Environment: <Google Colab runtime image if exposed>
Python: <version>
Torch/CUDA: <versions>
GPU: <device and VRAM>
Critical packages: <transformers / tokenizers / peft / bitsandbytes / safetensors>
Model: <model key / model ID / immutable revision>
Input: <dataset revision+digest OR external artifact SHA-256 + prompt/input identity>
Principal metrics: <loss/perplexity/runtime/throughput/memory/training controls as applicable>
Artifact: <filename / peft_adapter formatVersion 1 / bytes / SHA-256>
Adapter activity: <LoRA-B max abs + adapter-on/off logit max abs delta>
Machine-readable outputs: <filenames>
Outcome: PASS
Evidence notes: <key facts only; no secrets or private data>
Reviewer/operator: <name or GitHub identity>
```

## Passing records

No passing records are asserted yet. The most recent attempted external execution was correctly
blocked before GPU use because it resolved the stale candidate
`2c4c8309132022e6aec31e048349c64de9646b34`. That blocked attempt is not REL1/REL5 evidence.
A new run must resolve the post-fix PR head and confirm its exact-head CI before execution.
