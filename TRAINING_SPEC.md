# Training Contract

Implemented by `language-model-finetuner`. Read alongside `DATASET_SPEC.md` (what goes in)
and `ARTIFACT_SPEC.md` (what comes out).

## Methods

**LoRA** is the default: the base model is frozen in bf16 where supported, and only adapter
parameters train. **QLoRA** additionally loads the frozen base in 4-bit NF4 with double
quantization, trading throughput for memory.

Full-weight fine-tuning is **not** available and is not merely unimplemented — it would
change the artifact contract from an adapter to a full model copy, which serving,
provenance and storage policy all assume does not happen.

On measured hardware QLoRA is not an optimization but the only viable method above ~1B
parameters at 2048 tokens. See `COMPATIBILITY.md`.

## Loss masking

**Only assistant tokens are supervised.** Prompts, system messages and padding are excluded.

Spans are computed by rendering the conversation incrementally with the model's own chat
template:

```
prefix_i = template(messages[:i], add_generation_prompt=True)
upto_i   = template(messages[:i+1])
supervised span = [len(tokens(prefix_i)), len(tokens(upto_i)))
```

This requires the template to be **prefix-stable** — each incremental render must be a
token-level prefix of the next. That holds for every registry model but is not guaranteed in
general, so it is asserted per example. A template that violates it fails the run rather
than silently producing misaligned labels.

Prompt/completion and instruction records normalize to `[user, assistant]`, so completion-only
loss falls out of the same code path rather than needing its own.

An example that ends up with no supervised tokens fails; it would contribute nothing.

## Batching

Right-padded to the longest sequence in the batch. Padding is excluded from **both** the
attention mask and the loss — a model that learns to predict pad tokens is learning an
artifact of batching.

Effective batch size is `per_device_batch_size x gradient_accumulation_steps`. A partial
accumulation window at the end of an epoch is flushed, so the last examples are not silently
discarded.

## Training controls

> **Status.** The contract below is defined here — job schema, result provenance and
> registration defaults — and is **not yet implemented in `language-model-finetuner`**
> (#54). The schemas accept a document that omits every field in this section for exactly
> that reason: the contract lands first and the runtime follows, without either step
> breaking the other. Until the runtime lands, a job document carries no scheduler or
> early-stopping block and weight decay stays at torch's implicit value.

Three controls beyond learning rate and epoch count. Each one **defaults to the behaviour
of runs taken before it existed**, so re-running an existing registration today produces
the run it produced then. A default that quietly changed what a re-run does would
invalidate the measured matrix in `COMPATIBILITY.md` without anyone editing it.

### Weight decay

Stated, not inherited. It was previously whatever `torch.optim.AdamW` defaults to — the
finetuner constructs the optimizer with a learning rate and nothing else — which is
regularization policy nobody chose and provenance cannot show.

The documented default is that same value, `0.01`, and it is unchanged for that reason
rather than as a recommendation. `0.0` is legitimate for LoRA SFT, where dropout already
regularizes, and stays expressible.

### Learning-rate schedule

`constant`, `linear` or `cosine`. Warmup is given as **either** `warmup_ratio` or
`warmup_steps`, never both: with both present, whichever the runtime happened to read
would win and the document would still look well-formed.

Warmup is counted in **optimizer updates, not micro-batches**, so raising
`gradient_accumulation_steps` does not silently shorten the warmup. The partial
accumulation window flushed at the end of an epoch is one such update and counts as one.

A warmup with no declared schedule is refused — it leaves the shape after warmup unstated.

`constant` stays supported because every figure in `COMPATIBILITY.md` was measured at a
constant rate, and it remains the default for the same reason: a warmup-capable schedule
becomes the default when a measurement says it should be, not before.

### Early stopping

Off unless asked for. When on, validation loss is measured after each epoch exactly as it
already is; the best epoch is tracked; and the run stops once `patience` consecutive epochs
fail to beat the best by at least `min_delta`. Patience is spent on epochs, not evaluations,
because that is the granularity validation is measured at.

`restore_best_adapter` decides **which adapter is published** — the best epoch's or the
last completed one's. It defaults to false. The two are different artifacts and the file
manifest cannot tell them apart, so publishing the other one has to be something a user
selected rather than something a default did to them.

### One spelling of off

The registered parameter form has no way to express an absent value, so it says off with
`early_stopping_patience: 0`. The job document says off by **omitting**
`training.earlyStopping`, and its schema refuses a patience below 1. So the lowering rule
is:

| `early_stopping_patience` | `training.earlyStopping` in the job document |
|---|---|
| `0` | absent |
| `n >= 1` | `{"patience": n, "minDelta": ..., "restoreBestAdapter": ...}` |

One state, one representation. Two spellings of off in the same document is how a run ends
up stopping for a reason nobody selected.

The lowering is a function, not a paragraph: `lmpipeline.training_controls.lower_controls`,
in the package the finetuner vendors. So is the step arithmetic a schedule is stretched over
(`total_optimizer_steps`) and the ratio-to-steps conversion recorded in provenance
(`resolve_warmup_steps`). A rule each repository implements from a spec is a rule that
eventually differs between them.

## Bounds

Two layers, and the registry always wins:

- `dimer-pipeline.json` documents one envelope shared by every registration using the
  repository, so it must be safe for the **tightest** enabled model. It documents rather
  than enforces: the platform does not read the file (`COMPLIANCE.md` C-8) — the envelope
  it describes must be entered in the Builder UI, and the registry below is what actually
  refuses an out-of-bounds value at runtime.
- The model registry holds the authoritative per-model ceiling.

A request above the ceiling **fails**. Nothing is silently clamped: a user who asked for
4096 tokens and got 2048 would be told their run succeeded while their long examples were
quietly mangled.

## Splits

A supplied validation split is used exactly as given. When absent, one is derived by stable
content hashing — order-independent, so re-uploading the same rows shuffled produces the
same split. A supplied test split **is actually evaluated and reported**.

Datasets of two or more examples always yield a non-empty validation split, so a small
upload cannot silently produce a meaningless loss curve.

A test split is **optional, and stays optional.** Train and validation are sufficient for an
operational fine-tuning run: validation is what the loss curve, early stopping and
best-epoch selection all read. A test split is for formal evaluation or benchmarking, where
the number has to come from data no part of the run could have selected on.

When one is supplied it is excluded from optimization **and from model selection**, and is
evaluated once, after training has finished. Nothing in early stopping or best-adapter
restoration reads it: a held-out split that model selection consults is not held out.

## Metrics

Reported: train / validation / test loss, perplexity where finite, supervised token counts,
examples processed, throughput, wall time, peak GPU memory.

Losses are weighted by supervised tokens, so a batch of short examples does not count the
same as a batch of long ones.

**Train loss is an epoch average; validation loss is measured after the epoch.** A
fast-converging run legitimately shows validation well below train.

A run using the training controls additionally records, in the training result's
provenance: the stopping reason, the epochs actually completed, the best epoch and its
validation loss, whether the best adapter was restored, and the resolved schedule —
scheduler type plus the **effective** warmup in optimizer steps. What was *requested* is
already legible in the embedded job document; this is what the controls did with it.

`epochsCompleted` is there so that `early_stopping` is falsifiable — without it there is no
way to see that the run ended short of its budget. The effective warmup is recorded in steps
rather than as a ratio because two readers dividing a ratio by a step count they each
derived is how one run acquires two warmup lengths.

Not reported, deliberately: any generic "accuracy", and any claim about task quality.
Pipeline correctness and model quality are separate concerns, and a falling loss is evidence
of the former only.

## Resources

Preflight compares the registry's measured per-`(model, method)` requirement against the
detected GPU and refuses **before** weights are downloaded. A measurement bounds only jobs
at least as demanding as the configuration it was taken at.

A CUDA OOM becomes a structured `RESOURCE_OOM` carrying peak allocation, total VRAM,
sequence length and batch size, and names the lever that helps: QLoRA when on LoRA, a
smaller model when already on QLoRA.

## Lifecycle

SIGTERM and SIGINT raise, converting cancellation into the normal failure path so the run
still writes a result and calls the DIMER callback. A job that vanishes silently is
indistinguishable from one still working.

Resume is **not** supported in v1. A cancelled run fails and must be restarted.
