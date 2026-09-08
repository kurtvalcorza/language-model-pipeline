# Dataset Contract

Implemented by `lmpipeline.datasets`, shared verbatim by the validator and the finetuner.
Both resolve and normalize through this one module, so they cannot disagree about what a
dataset contains.

## Transport

A mounted directory or a single `.zip`. No other archive formats.

One level of directory wrapping is unwrapped automatically — users routinely zip the folder
rather than its contents. A zip containing another zip is rejected with a dedicated,
actionable message (`DATASET_ARCHIVE_NESTED`); the Portal documents this as a common user
error. Two or more archives is ambiguous and fails.

## Resource bounds

Ingestion is bounded **before** any row is parsed, because a dataset that is merely
transport-valid can still be large enough to exhaust RAM. An OOM kill is the one failure the
contract cannot report on: the Job dies without writing a result document, so the user sees
an infrastructure error rather than a reason.

| Bound | Value | Applies to | Override |
|---|---|---|---|
| `MAX_SPLIT_BYTES` | 512 MiB | each resolved split file | `LM_MAX_SPLIT_BYTES` |
| `MAX_DATASET_BYTES` | 1 GiB | all splits together | `LM_MAX_DATASET_BYTES` |
| `MAX_MEMBER_BYTES` | = `MAX_SPLIT_BYTES` | each archive member | follows |
| `MAX_UNCOMPRESSED_BYTES` | = `MAX_DATASET_BYTES` | total archive expansion | follows |
| `MAX_LINE_BYTES` | 4 MiB | one record | — |

> **The two byte values are provisional (issue #11).** The agreed policy is to set them to
> DIMER's documented maximum upload size, so this pipeline never rejects a dataset the
> platform itself accepted. That quota has not been read out of the portal yet, and putting
> an invented number in a shared contract is the failure mode COMPATIBILITY.md exists to
> prevent. They are conservative in the meantime and overridable by environment variable, so
> setting the real quota needs no code change. What ingest actually costs in RAM is now
> measured rather than assumed — see below, and note that it is the example cap, not these
> byte values, that governs peak memory.

Two rules follow, and both are load-bearing:

- **The byte bounds apply to mounted directories, not only archives.** Archive members carry
  metadata that can be checked cheaply; a mounted directory carries none, and DIMER mounts
  whatever the user uploaded. Bounding only the archive path leaves the directory path
  completely open. Violations raise `DATASET_SPLIT_TOO_LARGE`.
- **Consumers must not materialize a split unbounded.** `iter_examples` streams, but every
  consumer needs the examples more than once, so each one reached for `list(...)` — which
  restores the unbounded allocation the streaming reader exists to prevent, and defers the
  example-count policy until after the allocation it was supposed to guard. Use
  `load_examples(path, max_examples=...)`, which raises `DATASET_TOO_MANY_EXAMPLES` on the
  example that would exceed the cap. Peak memory is then bounded by the cap, not by the file.

Archive bounds are aligned to the split bounds deliberately: if a member limit were looser
than the split limit, the pipeline would pay to extract bytes that split resolution then
rejects.

### What ingest actually costs, measured

Issue #11 records that a large quota should not simply be adopted, because peak memory is a
multiple of file size and a bound set above the survivable point converts a structured
`DATASET_SPLIT_TOO_LARGE` into an OOM kill. That needed a number rather than an estimate.
`scripts/measure_ingest_memory.py` produces one: it runs the real `load_examples` and
`iter_examples` paths over synthesized splits, each point in a fresh process because
`ru_maxrss` is a high-water mark that never falls.

Measured on `Linux-6.18.44-x86_64` / CPython 3.11.15, retaining via `load_examples`:

| file bytes/example | examples | split file | retained RSS | streamed RSS |
|---|---|---|---|---|
| 220 | 100,000 | 20.9 MiB | 95.0 MiB | 3.1 MiB |
| 573 | 100,000 | 54.6 MiB | 127.9 MiB | 3.1 MiB |
| 1,868 | 100,000 | 178.2 MiB | 254.3 MiB | 4.1 MiB |
| 616 | 500,000 | 307.9 MiB | 673.4 MiB | 3.1 MiB |

Which fits, across shapes:

```
retained RSS  ~=  the split's file bytes  +  ~770 bytes per example held
```

The fit predicts 685 MiB for the 500k row above against 673 MiB measured, so it is accurate
to a few percent and errs high. Streaming is flat at a few MiB at every size — it is the
retained list that grows, which is exactly what an example cap can bound and a byte bound
cannot.

**The consequence inverts the intuition behind a byte bound.** Because ~770 bytes of that cost
is fixed CPython object overhead per example — an `Example`, its message tuple, a dict per
message, a list slot — the RSS-to-file ratio is **worst for the shortest rows**: 4.5x file size
at 220 bytes per example, but only 1.4x at 1,868. A byte limit is therefore loosest precisely
where the memory risk is highest, and short structured rows are not hypothetical — that is
UNER's shape.

So the two bounds do different jobs, and only one of them is a memory instrument:

- the **byte bounds** protect transport, extraction and disk;
- the **example cap** (`max_examples`, passed by each consumer) is what governs peak RAM.

Taken together, the shipped values admit a worst case of roughly **886 MiB retained** — a
512 MiB split of 500,000 rows, which works out to ~1,074-byte rows, an entirely ordinary size
rather than a pathological one. A validator container therefore needs on the order of a
gigabyte for ingest alone, before its own working set.

**What this settles for #11, and what it does not.** It does not supply the quota; only the
portal can. What it does is make the judgement call arithmetic rather than a guess: if the
documented quota turns out to be far above these values, the correct response is not to
refuse the quota but to set the **example cap** from the container's memory budget using the
model above, and let the byte bound follow the platform. The two questions are separable, and
the cap is the one that decides whether the container survives.

Two limits on these figures. They were taken on a general-purpose cloud container, **not on
approved DIMER hardware** (#7) and not in the validator's real 2-vCPU container, whose own
measurement at the 500k cap is `language-model-dataset-validator#14`. The per-example fixed
cost is a property of CPython's object layout and transfers across machines running the same
interpreter and architecture; the absolute memory budget does not.

## Split resolution

| Split | Filename | Required |
|---|---|---|
| train | `train.jsonl` | yes |
| validation | `validation.jsonl` (alias `val.jsonl`) | no |
| test | `test.jsonl` | no |

If both `validation.jsonl` and `val.jsonl` are present the input is **ambiguous and fails**.
Ambiguity is never resolved by preference order — a silent guess here would train on a
different dataset than the user validated.

A supplied validation split is used exactly as given. A supplied test split MUST actually be
evaluated and reported. When validation is absent the trainer may derive one deterministically
by stable hashing — reproducible, and not sensitive to input order.

## Record schemas

One family per file. A file that mixes families fails (`DATASET_SCHEMA_MIXED`) with the line
number where the family changed.

**Conversational** (preferred, canonical form)
```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

**Prompt / completion**
```json
{"prompt":"...","completion":"..."}
```

**Instruction**
```json
{"instruction":"...","input":"...","output":"..."}
```

All three normalize to canonical `messages`. Normalization is content-preserving, so the same
content expressed in two families produces the same fingerprint — which is what makes
cross-split leakage detection work when the user changes format between splits.

## Rules

- v1 roles are `system`, `user`, `assistant`. Tool/function-call training is deferred.
- Every example needs at least one non-empty assistant target.
- Valid UTF-8 required; Unicode is preserved exactly.
- Every error carries a 1-based line number.
- **No record is ever silently dropped, truncated, or mutated.** Blank lines are skipped
  without shifting the line numbers of real records.
- Per-line byte cap, and min/max example-count policy.

## Tokenizer-aware checks

Applied by the validator using the chat template of the resolved registry model at its pinned
revision — which is why `model_key` must reach the validator (DEPLOYMENT.md §1).

Token-length statistics (min/median/p95/p99/max) are computed and checked against the model's
`max_sequence_length` ceiling. Overlength samples are **rejected, never silently truncated**.

## Duplicates and leakage

Fingerprints are computed over the canonical normalized form. Exact duplicates within a split
are reported. **Exact train/validation or train/test overlap fails the run.** Training data is
never silently deduplicated — that would change the dataset the user believes they trained on.

## Privacy

Result payloads carry aggregate statistics and line/index references only. Raw examples never
appear in results or ordinary logs. See SECURITY.md.
