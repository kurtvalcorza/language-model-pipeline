# DIMER Compliance Audit — 2026-08-23

Audit of `language-model-dataset-validator` and `language-model-finetuner` against the DIMER
AI Engineer contract **as read from backend source** (`NAIRA-SEU/dimer-backend@on-prem`,
the seven pipeline template repos, and `cloudformation-template`) rather than from the
rendered portal page.

**Why this file exists.** `DEPLOYMENT.md` was transcribed from the portal page. Several of
its claims are now known to be wrong, and two of them are load-bearing for the architecture.
This file records what the source says, what our code does, and where the two disagree.
Where it contradicts `DEPLOYMENT.md`, **this file is normative** until that document is
revised (deliberately deferred: two open PRs are editing it).

Every finding below cites either a file:line in our code or the contract clause it violates.
Findings marked **inferred** could not be tested from here and name what would confirm them.

---

## 1. Contract facts that invalidate prior assumptions

| Prior assumption (`DEPLOYMENT.md`) | Source-verified fact |
|---|---|
| `DIMER_PREPROCESSING_ARGS_JSON` reaches **both** containers — "the only user-parameter channel proven to reach both" | **The validator receives four variables only:** `DIMER_DATASET_DIR`, `DIMER_RESULT_PATH`, `DIMER_DONE_CALLBACK`, `DIMER_PIPELINE_METADATA_JSON`. No preprocessing args, no hyperparameters, no output dir |
| `dimer-pipeline.json` declares our parameters and constrains `model_key` to an enum | **`dimer-pipeline.json` is not read by anything.** Parameters live in the registry (`Pipeline.dp_parameters_schema`, `Pipeline.mf_hyperparameters`) and are set through the Pipeline Builder UI. Unregistered parameters arrive as `{}` with no error |
| Base Model is the only model channel, and it never reaches a Job — hence the tier sentinel | Model selection is a registered `model_id` hyperparameter resolved against `schema.model.fineTunableModels`; the selected entry is delivered to the finetuner as **`DIMER_MODEL_CONFIG_JSON`**. `DIMER_HYPERPARAMETERS_JSON` always carries `model_id` |
| GPU is the deployment target | **GPU is opt-in and off by default.** `WORKBENCH_FINE_TUNING_GPU_ENABLED` defaults to `False` and the CloudFormation template provisions **no GPU node pool** |
| `/data` is an ordinary mounted filesystem | `/data` is a **Mountpoint-for-S3 CSI volume**. `shutil.copy2()` fails; renames and in-place edits are unreliable |
| The finetuner's artifact layout is ours to choose | `export-to-repository` requires **`artifacts/best.pt`** at that exact path, plus `evaluation/report.json` and `logs/run-summary.json` |

Two smaller confirmations: `DIMER_TRAIN_DEVICE` really is the bare string `"0"` (we already
handle this), and an LM pipeline receives `"taskType": "object_detection"` because
`infer_pipeline_task_type()` recognizes only three image tasks and defaults to detection.

---

## 2. Findings

| ID | Sev | Area | Finding | Status |
|---|---|---|---|---|
| C-1 | **P0** | validator | `model_key` cannot reach the validator; every run fails `MODEL_KEY_MISSING` | confirmed |
| C-2 | **P0** | registration | No `fineTunableModels` entries; the backend can refuse to launch the Job | confirmed from source doc |
| C-3 | **P0** | finetuner | Artifact layout and result shape cannot be consumed by `export-to-repository` | confirmed |
| C-4 | **P1** | finetuner | QLoRA refuses CPU, and CPU is the default deployment | confirmed |
| C-5 | **P1** | finetuner | Permission normalization fails closed on a filesystem that cannot store modes | inferred |
| C-6 | **P1** | finetuner | Publication is a rename on a filesystem where renames are unreliable | inferred |
| C-7 | **P1** | finetuner | GPU burst mode unhandled: trains, then loses everything | confirmed |
| C-8 | **P2** | both | Parameters are not registered, so every knob silently falls back | confirmed |
| C-9 | **P2** | security | A merged vulnerability exception rests on a constraint that does not exist | confirmed |
| C-10 | ~~P2~~ **P3** | validator | 300 s margin is 1.7-2.5x on a workstation, unmeasured on the target node | measured, **downgraded** |
| C-11 | **P3** | validator | Nested zip is rejected where the platform prefers unwrapping | by choice |
| C-12 | **P3** | both | `:latest` on the built image means no run is reproducible | confirmed |

### C-1 — The validator cannot learn which model to validate against **(P0)**

`src/validator/core.py:98` resolves the model from `env.model_key`, which reads
`DIMER_PREPROCESSING_ARGS_JSON`. That variable is **not injected into validator Jobs**.
`ModelRegistry.resolve(None)` raises `MODEL_KEY_MISSING`, so every validation on the real
platform fails before it reads a single line of the dataset.

`src/validator/core.py:203` reads `max_sequence_length` from the same channel; it falls back
to a default, so it degrades rather than fails.

This is architectural, not a bug: the validator's token-budget and sequence-length checks are
tokenizer-specific, and no user selection reaches it. **`DIMER_PIPELINE_METADATA_JSON` does
carry `defaultFineTunableModelId`**, which is a channel that reaches the validator — but it
is the pipeline's default, not the user's choice, so a user who picks a different model would
be validated against the wrong tokenizer. Note also that on the `main` backend branch the
validator does not receive pipeline metadata at all, so any use of it needs a baked default.

**Why the wrong inference looked well-evidenced.** `DEPLOYMENT.md` grounded the claim on
shipped Mitra: its validator consumes `DIMER_PREPROCESSING_ARGS_JSON` for `target_column`,
and that pipeline passed DIMER GPU acceptance. Both halves are true and the conclusion still
does not follow, because `mitra-classifier-dataset-validator/validator.py:253` reads

```python
target_column=str(preprocessing.get("target_column") or "target").strip(),
```

**It has a default.** With the variable absent, `preprocessing` is `{}` and the column falls
back to `"target"` — which is what Mitra's acceptance datasets use. It passed acceptance
because the default was correct, not because the channel reached the validator. Ours has no
default and hard-fails, so identical platform behaviour is invisible there and fatal here.

Two consequences outside this repo, both worth checking rather than assuming: Mitra's
`target_column` parameter is probably **inert at validation time**, so a user who sets it to
anything but `target` would have the finetuner honour it while the validator checks a column
named `target` — validation could reject a dataset training would accept (same shape in the
regressor). And the general lesson: a silent fallback makes a missing channel
indistinguishable from a working one, so acceptance passing is not evidence a variable
arrived.

**Needs a decision.** The options are not equivalent and the choice is yours:

1. Validate against `defaultFineTunableModelId`, falling back to a baked default, and state
   in the result that token counts are approximate for any other selection.
2. Make the validator model-agnostic — structural, schema and leakage checks only — and move
   every tokenizer-specific check into the finetuner, which does receive the selection.
3. One validator image per model. Compliant and precise, and it abandons the model-agnostic
   design the whole project exists to provide.

### C-2 — The finetuner Job may never launch **(P0)**

Model selection resolves `model_id` against the pipeline's `schema.model.fineTunableModels`.
The backend ships a catalog covering YOLO variants only. If `model_id` resolves to nothing
and no legacy `base_model` was supplied, it raises `ValueError: Unsupported fine-tunable model
id` **before the container is launched** — a failure with no result document and no callback.

Our registry keys (`qwen3-1.7b`, `granite-4.1-3b`, …) exist only inside our image. Nothing
registers them with DIMER. Each approved model needs a `fineTunableModels` entry carrying at
minimum `id`, `displayName`, `taskType`, `framework`, `provider`, `baseWeights`,
`resourceProfile`, `supportedDatasetFormat`.

This also supersedes the premise of #13. The question is no longer whether the Pipeline
Builder accepts a free-text Base Model value; it is whether `fineTunableModels` entries for a
non-image family are accepted, and what `baseWeights` means for a Hugging Face repo id rather
than a `.pt` file.

### C-3 — The artifact cannot be exported **(P0)**

`export-to-repository` and `.../artifact-info` look for `artifacts/best.pt` — a literal
filename at a literal path. We publish a PEFT adapter directory (`adapter_model.safetensors`,
`adapter_config.json`, `tokenizer/`, `MODEL_CARD.md`, `artifact-manifest.json`,
`provenance.json`, `metrics.json`). Nothing matches, and no `evaluation/report.json`,
`logs/run-summary.json`, or `progress/epoch_*.json` is written at all.

The result document diverges too. The contract's finetuner envelope carries top-level
`metrics` and an `artifacts` object of `modelArtifact` / `evaluationReport` / `logArtifact`,
each with `path` **relative to `/data`**, plus `metadata.baseModel`, `metadata.selectedModelId`
and a `metadata.device` block. `src/lmpipeline/result.py:64` nests all of this under
`metadata.languageModelPipeline` instead. The backend reads only `successful` and `message`
for the *validator*, but the finetuner's shape is what export consumes.

An adapter is not a single checkpoint file, so this needs either a packaged `best.pt`
(a torch-serialized bundle of the adapter, which reintroduces pickle — see `SECURITY.md`) or a
backend-side change. Worth deciding before building either.

### C-4 — CPU is the default deployment, and QLoRA refuses it **(P1)**

`src/finetuner/resources.py` raises `RESOURCE_GPU_UNAVAILABLE` when `method == "qlora"` and no
CUDA device is visible. That is correct behaviour on a GPU-less cluster and it means the
pipeline cannot run there at all. LoRA would proceed and cannot plausibly finish an LM SFT
within the 6 h budget on a general-purpose node.

The contract's instruction — "your image must run correctly on CPU" — is satisfiable for an
image-classification finetuner and is not satisfiable for LM fine-tuning in any useful sense.
The honest resolution is a preflight that refuses early with a specific message naming the
operator action (enable `WORKBENCH_FINE_TUNING_GPU_ENABLED`, provision a GPU node pool),
rather than either pretending CPU works or failing obscurely.

### C-5 — Permission normalization fails closed on a filesystem with no modes **(inferred, P1)**

`src/finetuner/artifacts.py:71` chmods every staged file and then **verifies by `stat()`**,
raising `ARTIFACT_PACKAGING_FAILED` if the mode did not stick. That fail-closed behaviour was
deliberate and correct against a POSIX mount. Mountpoint-for-S3 does not support the metadata
updates that make `shutil.copy2()` fail, so the mode almost certainly will not stick, and the
verification would then refuse to publish **every** artifact.

Confirming this needs one run against a real session PV, or a local mountpoint-s3 mount. If
confirmed, the rule needs to become filesystem-aware rather than being dropped: the serving
uid still has to be able to read the artifact.

### C-6 — Publication renames on a filesystem where renames are unreliable **(inferred, P1)**

`src/finetuner/artifacts.py:290` publishes with `os.replace(staging, output_dir)`, which is
the atomicity guarantee the whole staging design rests on. Guidance for `/data` is explicit:
random writes, renames and in-place edits are unreliable; write whole files once.

If renames do not work there, the fix is not to weaken the guarantee but to move it: stage on
local scratch (`/tmp`), then write final files once into `/data`. That also matches the
guidance to copy the dataset to local scratch rather than reading it repeatedly over S3.

### C-7 — GPU burst mode is unhandled **(P1)**

No source file in any of the three repos references `GPU_BURST_MODE` or any burst variable
(verified by grep across `src/` and root scripts). In burst mode `/data` is not mounted at
all; the container must fetch the dataset from S3 and upload the result and weights back.
A burst-mode run of our finetuner would train for hours and then write everything into a
container filesystem that disappears.

### C-8 — Unregistered parameters fall back silently **(P2)**

`dp_parameters_schema` and `mf_hyperparameters` must be populated in the Pipeline Builder for
`DIMER_PREPROCESSING_ARGS_JSON` and `DIMER_HYPERPARAMETERS_JSON` to carry anything. Until
then both arrive as `{}` and every knob falls to our defaults — with no error anywhere.

Our `dimer-pipeline.json` files are therefore documentation, not configuration. They should
say so, since their present framing implies they configure the platform.

### C-9 — A merged security exception rests on a constraint that does not exist **(P2)**

The committed exception for **PYSEC-2026-2289** (`vulnerability-policy.yaml`) argues that a
malicious `config.json` is unreachable because *"users cannot supply a model id —
dimer-pipeline.json constrains model_key to an enum of registry keys."*

That file is not read by anything. The enum constraint exists only if the equivalent
parameter is registered in the Pipeline Builder, which has not been done. The mitigation is
not absent — `ModelRegistry.resolve()` still rejects any key outside the registry, which is
the control that actually holds — but the *stated* rationale names the wrong mechanism, and a
reachability argument that cites a control that does not exist is exactly what the exception
process is supposed to prevent.

The rationale needs correcting to cite `ModelRegistry.resolve()`. No change in the decision;
a change in what the decision rests on.

### C-10 — The 300 s validator margin is thin, not breached **(measured, P3 — downgraded)**

**Corrected 2026-08-23 by measurement that contradicted the original finding.** This entry
first claimed our cap admits datasets that *cannot* validate inside the budget. Running the
whole validator rather than extrapolating from one stage shows that is wrong on the hardware
available here, so the severity drops from P2 to P3 and the recommended action changes.

The validator's hard timeout is **300 s** and our `MAX_TRAIN_EXAMPLES` is 500 000. Measured
end to end through `validate()` with the real `Qwen/Qwen3-0.6B` tokenizer at its pinned
revision, on this workstation:

| dataset | outcome | wall time | margin vs 300 s |
|---|---|---|---|
| 500 000 short examples, 64 MB | **accepted** — the largest dataset that can legitimately pass | **118.7 s** | 2.5x |
| 500 000 long examples, 278 MB | rejected, `DATASET_TOKEN_BUDGET_EXCEEDED` | **178.7 s** | 1.7x |
| 500 000 examples, tokenizer absent | parse + normalize + fingerprint + leakage only | **8.0 s** | — |

That third row is the correction. The original estimate assumed archive handling, duplicate
fingerprinting and split-leakage comparison would add materially to the ~154 s of
tokenization. They add **about 8 s at 500 000 examples** — tokenization is roughly 93% of the
cost, and the rest is noise.

**What remains true, and is the reason this is not closed.** The margin is 1.7–2.5x on a
workstation, and neither figure was measured on the target: a CodeBuild-class 2 vCPU node,
reading an archive from an S3-backed mount rather than local disk. A node ~2.5x slower than
this one breaches the budget on the accepted path, and a timeout produces no result document
at all — the one outcome the contract cannot report on.

**So the action changes from "lower the cap" to "measure before changing anything."** Lowering
`MAX_TRAIN_EXAMPLES` on the strength of a desktop number would reject datasets the platform
accepts, which is the failure `resolver.py` already warns about for the byte bounds (#11). A
wall-clock guard that fails with a stable code before the platform kills the Job is cheap and
useful regardless of where the true ceiling sits, and does not require knowing it.

### C-11 — Nested zip: we reject, the platform prefers unwrapping **(P3, by choice)**

`resolve_dataset` raises `DATASET_ARCHIVE_NESTED`. Guidance is to unwrap the inner archive
rather than erroring. Our choice was deliberate — unwrapping arbitrary nested archives widens
the zip-bomb surface — and the divergence is worth re-deciding rather than leaving implicit,
because the platform documents unwrapping as expected behaviour.

### C-12 — Nothing is reproducible across a rebuild **(P3)**

The build pushes `:latest` and the pipeline stores a bare repository URI; Jobs run with
`imagePullPolicy: Always`. A rebuild silently changes every future run with no way to pin or
roll back. Our `provenance.json` records exact package versions, which makes a completed run
*auditable* after the fact, but it cannot make two runs identical. Worth stating plainly in
`PROVENANCE_SPEC.md` rather than letting the pinned base image imply more than it delivers.

---

## 3. Where we already comply

Recorded because an audit that lists only failures misrepresents the state.

- **`DIMER_TRAIN_DEVICE` normalization.** `dimer.py::_normalize_device` converts a bare `"0"`
  to `cuda:0`. We handled this before it was documented as a correction.
- **Callback from `finally`, even when env parsing failed.** Both entrypoints write a result
  and POST the callback in `finally`, and `notify_done_callback_url` exists precisely so the
  callback fires when `DimerEnv.from_environ()` is what raised. The contract notes all three
  reference validators get this wrong.
- **`classNames` on every write path.** Emitted as `[]` on success and on the crash path.
- **Exit code with a well-formed envelope.** We exit non-zero on failure with a valid result,
  which the contract names as the correct failure mode.
- **Wrapper-folder unwrapping**, and a `.zip` or a directory accepted interchangeably.
- **Wrong-pipeline detection** (PR #28, pending) — the contract asks for exactly this, and
  names the segmentation-into-detection case we abstain on.
- **Pinned amd64 base, no `--build-arg` reliance, heavy dependencies pre-built.**
- **No `os.environ` dump; the signed callback URL is never logged.**

---

## 4. Suggested order

C-1 and C-2 gate everything: without them no run starts or completes. C-3 gates whether a
successful run is worth anything. C-5 and C-6 should be tested against a real session PV
before either is redesigned — both are inferred, and redesigning on an inference is how a
working mechanism gets replaced by a worse one.
