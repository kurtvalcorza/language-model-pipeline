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

## Bounds

Two layers, and the registry always wins:

- `dimer-pipeline.json` advertises one envelope shared by every registration using the
  repository, so it must be safe for the **tightest** enabled model.
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

## Metrics

Reported: train / validation / test loss, perplexity where finite, supervised token counts,
examples processed, throughput, wall time, peak GPU memory.

Losses are weighted by supervised tokens, so a batch of short examples does not count the
same as a batch of long ones.

**Train loss is an epoch average; validation loss is measured after the epoch.** A
fast-converging run legitimately shows validation well below train.

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
