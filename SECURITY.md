# Security & Supply Chain

## Model acquisition

- **Allowlist only.** Only `model_key`s present and `enabled` in the registry may be loaded.
  Arbitrary Hugging Face IDs are never accepted from end users: the manifest exposes an
  `enum` of internal registry keys, and the runtime resolves through
  `ModelRegistry.resolve`, which fails closed.
- **Immutable revisions only.** Every enabled entry pins a 40-character commit SHA.
  `resolve()` rejects an entry whose revision is absent or not a SHA, so a model cannot
  silently float on `main`. Enforced by `tests/test_registry.py`.
- **No fallback.** A pinned revision that cannot be fetched is a failure, never a silent
  fall back to `main` or to whatever the cache happens to hold.
- **`trust_remote_code=False`.** Any entry with `requires_trust_remote_code: true` is
  rejected by `resolve()` even if someone flips `enabled`. `microsoft/Phi-4-mini-instruct`
  is kept in the registry, disabled, so its status is auditable rather than forgotten.
- **Safetensors only** for model and adapter weights. No pickle-based artifacts.

## Secrets and private data

- **Never dump `os.environ`.** `DIMER_DONE_CALLBACK` is a signed URL; the container may also
  hold registry or Hub credentials. `DimerEnv.diagnostics()` allowlists keys and redacts the
  callback. This applies with full force to the PR 0 probe, whose entire purpose is to
  enumerate the runtime — see DEPLOYMENT.md §6.
- **Never log raw dataset content.** Datasets are user-private. Errors reference line
  numbers and field names, never values. Enforced by
  `test_error_messages_never_embed_record_content`.
- **Unexpected exceptions do not surface their message.** Arbitrary exception text can embed
  signed URLs, absolute paths, or dataset rows, so `Result.from_exception` reports the
  exception *type* and directs the reader to Job logs. Enforced by
  `test_unexpected_exception_does_not_leak_its_message`.
- **Artifacts never contain training data.**

## Untrusted input handling

User-supplied archives are hostile until proven otherwise. `datasets/resolver.py` enforces,
before writing any bytes: path traversal and absolute/drive-prefixed member rejection,
symlink member rejection, per-member and total uncompressed size caps, member count caps, and
a compression-ratio ceiling. Bounds are checked against the central directory first and
re-checked against bytes actually written, so a lying header cannot slip past.

## Review gates

A model may not move to `approval_state: production` without an explicit license, security,
and serving review. Anything requiring remote code stays disabled until separately reviewed
and approved — adding one model is never a reason to weaken the loading policy.
