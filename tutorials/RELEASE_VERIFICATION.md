# Tutorial Release Verification

DIMER Notebook Specification 1.0 requires clean-runtime execution evidence in addition to
static source conformance. This file is the durable repository record for that gate.

## Revision model

A notebook release has **three independently immutable source identities**:

1. **Notebook / release-candidate revision** — the `language-model-pipeline` PR head that
   contains the notebook, static conformance gate, and this release record.
2. **Pipeline runtime-support revision** —
   `afaf1f032cd7e9751db1ee8eb71b542f9bd0b15f`.
3. **Production finetuner revision** —
   `ecbab8cfbb95c061235f68c087141ab7af462b9d` (`language-model-finetuner` PR #30).

Both notebooks record the runtime-source revisions. The E2E producer writes them into
`provenance.json`; artifact inference rejects a different pair.

## Access prerequisite

`language-model-finetuner` is private. For each clean Colab run, configure a Colab Secret
named `GITHUB_TOKEN`, grant the notebook access, and use a token with read access to that
repository. The notebook passes the credential through an ephemeral Git HTTP header, never
prints it or embeds it in a URL, and deletes its local token binding immediately after clone.

**Never place the token or any other secret in this release record, notebook output, uploaded
artifact, screenshot, or evidence note.**

## Current candidate status

| Notebook | Profile | Source conformance | Clean-runtime execution | Release status |
|---|---|---|---|---|
| `language_model_finetuning_colab.ipynb` | `E2E` | Enforced by static validator and tests | No passing clean-run record | **Blocked from release-grade label** |
| `language_model_artifact_inference_colab.ipynb` | `ARTIFACT-INFERENCE` | Enforced by static validator and tests | No passing clean-run record | **Blocked from release-grade label** |

Static validation, an old notebook run, or a warmed developer cache is not a substitute for
clean-runtime evidence.

## Required E2E verification procedure

1. Open `language_model_finetuning_colab.ipynb` at the exact PR head/commit being released.
2. Start a new supported Colab GPU runtime with the authorized `GITHUB_TOKEN` Secret enabled;
   do not rely on previously installed package/model/source caches.
3. Confirm the private finetuner clone succeeds without the token appearing in output and the
   notebook reports pipeline revision `afaf1f032cd7e9751db1ee8eb71b542f9bd0b15f`
   and finetuner revision `ecbab8cfbb95c061235f68c087141ab7af462b9d`.
4. Run the default sample path top-to-bottom without editing implementation cells.
5. Record Python, critical package versions, Torch/CUDA build, and GPU. Confirm producer
   `packageVersions` includes `safetensors` as well as torch/transformers/tokenizers/peft/
   bitsandbytes.
6. Confirm canonical model/revision resolution, data preparation, assistant-only masking,
   baseline generation, production QLoRA training, held-out optimization metrics,
   machine-readable outputs, artifact staging, and manifest verification complete.
7. Confirm fresh reconstruction reports non-zero LoRA B weights and a non-zero adapter-on/off
   logit delta on the same reconstructed model.
8. Download and preserve `dimer-language-model-adapter.zip` and its SHA-256 for the separate
   artifact-inference verification.

## Required artifact-inference verification procedure

1. Start a **different new clean Colab GPU runtime** at the same candidate notebook/PR head,
   with the authorized `GITHUB_TOKEN` Secret enabled.
2. Confirm the same pipeline and finetuner runtime revisions above and no credential disclosure.
3. Supply the E2E adapter ZIP from outside this notebook execution. The archive must have
   `artifact-manifest.json` at its ZIP root; nested manifested subtrees are rejected.
4. If available, provide the expected ZIP SHA-256 through an independently trusted channel.
5. Run top-to-bottom with the default editable new prompt.
6. Confirm root-level archive/manifest validation, canonical model/revision parity, critical
   package compatibility including `safetensors`, runtime-source parity, production
   reconstruction, adapter-activity evidence, context validation, generation,
   `artifact_inference_predictions.jsonl`, and `artifact_inference_provenance.json` all complete.
7. Confirm the notebook does not create the consumed adapter during this execution.

## Passing record format

Add one record per notebook and release candidate. All three source identities are mandatory.

```text
Notebook: <filename>
Profile: <E2E or ARTIFACT-INFERENCE>
Notebook/PR head: <40-character language-model-pipeline SHA containing the notebook>
Pipeline runtime revision: afaf1f032cd7e9751db1ee8eb71b542f9bd0b15f
Finetuner runtime revision: ecbab8cfbb95c061235f68c087141ab7af462b9d
Executed: <YYYY-MM-DD HH:MM timezone>
Environment: <Colab runtime image if exposed>
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
