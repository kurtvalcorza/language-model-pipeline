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
