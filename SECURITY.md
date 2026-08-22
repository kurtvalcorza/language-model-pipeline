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

## Vulnerability policy

`src/lmpipeline/data/vulnerability-policy.yaml` is the single authority for what counts as
a release-blocking finding, and `lmpipeline.vulnpolicy` applies it. Both live in the shared
package so the two deployable images cannot drift on that question — the same reason the
model registry lives there.

**Strictness tracks controllability.**

| Scanner | Covers | Blocks on |
|---|---|---|
| `pip-audit` | the Python dependencies each image installs | any finding **with a fix available** |
| Trivy | the built validator and finetuner images | **CRITICAL** with a fix available |

We pin our own dependencies and can bump them, so a fixable finding there is actionable.
Base-image OS packages come from an upstream image pinned for correctness — the CUDA/torch
base is why sm_120 works at all — and cannot be patched from this repository.

That split is measured, not assumed. On 2026-08-22 `python:3.12-slim` carried 194 findings:
3 CRITICAL (none fixable), 50 HIGH (36 fixable), 141 lower. A "fail on any HIGH with a fix"
rule would have blocked 36 Debian findings on day one, none actionable from here — which is
how a scanner becomes noise everyone learns to ignore. "Fail on CRITICAL with a fix" blocks
zero today and would still catch a real one.

### Exceptions

An exception is a decision with an owner and an end date, not a suppression. Every entry
records **id, package, scope, owner, expires, and a rationale**, and the loader refuses a
policy where any of those is missing or where `expires` is unparseable.

The rationale must give the **mechanism** by which the code path is unreachable or
mitigated here. "Not exploitable in practice" is not a rationale.

**An expired exception blocks.** Renewal is a deliberate act carrying a fresh reachability
argument; nothing lapses into permanence by being forgotten.

Exceptions are scoped per image, because an argument about the finetuner says nothing about
the validator, and match on aliases as well as ids — scanners disagree about whether a
finding is `PYSEC-…`, `CVE-…` or `GHSA-…`.

### Reading a finding

Before excepting anything, establish reachability against this codebase rather than the
advisory's worst case. The two live exceptions are worked examples: `CVE-2026-1839` targets
`transformers.Trainer._load_rng_state()`, and this finetuner implements its own training
loop and never instantiates `Trainer` — there is no `torch.load` call anywhere in
`src/finetuner`. `CVE-2026-4372` is a genuine config-loading path, mitigated rather than
unreachable, because configs are fetched only for registry-approved models at immutable
40-hex revisions that users cannot override.

State what a mitigation buys, and what it does not. Pinning means a user cannot redirect the
Job to another revision and upstream cannot silently change the bytes we fetch — content
addressing gives **immutability, not trustworthiness**. It does not establish that the
pinned commit was benign when it was pinned. That residual supply-chain assumption about the
publishers belongs *in* the exception, because it is the part that could stop being true:
a publisher compromise, or a new entry from a less established source, is grounds to
reassess the exception rather than renew it.
