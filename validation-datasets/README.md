# Validation Datasets

Reproducible data for validating the DIMER language-model fine-tuning capability, per
issue #2.

**This is validation infrastructure, not a benchmark.** Nothing here supports a claim about
model quality. It exists so that "which data validated which pipeline?" has an exact,
reproducible answer.

## Layout

```
registry.yaml          sources, licences, pinned revisions, profiles
build_fixtures.py      regenerates the fixture corpus below
synthetic/             committed valid fixtures — CI never needs network
  conversational/      train + validation + test, incl. multi-turn, no-system, Tagalog
  prompt-completion/
  instruction/
adversarial/           one directory per failure case
  expected-codes.json  the contract: fixture -> stable error code
approved/              committed expectations: digests + fingerprints, never rows
build/                 generated profiles (git-ignored; recipes are committed, not rows)
```

## Fixtures are generated, not hand-written

A hand-maintained corpus of adversarial archives drifts from the expected-code map beside
it. `build_fixtures.py` regenerates everything, and a test asserts regeneration is
**byte-identical** — which caught two real determinism bugs, both zip timestamps leaking
the current time.

```bash
python validation-datasets/build_fixtures.py
```

Content is synthetic and non-sensitive. The Tagalog strings exercise Unicode and tokenizer
handling; they are ordinary civics phrasing, not copied from any corpus.

## The adversarial contract

`adversarial/expected-codes.json` maps each fixture to the stable code it must produce.
Tests assert on the code, never on message text. Three assertions guard the map itself: every
declared code must be real, every fixture directory must be declared, and every declared
fixture must actually raise its code.

Cases requiring a tokenizer, a model entry, or cross-split state are asserted in
`language-model-dataset-validator`, which owns those checks.

## Determinism

Every rule here exists because an acceptance dataset that changes silently is worse than no
acceptance dataset.

- **Pinned revisions only.** Every source is a 40-character commit SHA. `verify` fails if
  the registry's pin no longer matches what a build used, rather than silently accepting
  drifted data.
- **Selection is by stable content hash, never "first N".** Upstream ordering is an accident
  and a subset that depends on it changes when upstream reorders. Row identity is content
  plus an *occurrence counter*, not an absolute index — using the index made selection
  order-dependent, which a test caught.
- **`fingerprintCount` below `exampleCount` is expected, not a defect.** Because identity
  carries an occurrence counter, both copies of a row that upstream duplicates are eligible,
  and a large enough draw will sometimes take both. `uner-multilingual@dc9e8b0f` holds 54,957
  rows but 53,825 distinct contents — 1,132 redundant rows in 697 groups, or 3,758 duplicate
  pairs. Each pair survives a draw of `n` with probability `(n/54957)²`, so `multi_2000`
  expects ~5 collapsed pairs and records 1997/2000, while `multi_500` expects 0.31 and
  records 500/500. The profiles are not behaving differently; the smaller one is too small
  to hit it. Deduplicating would mean giving up occurrence-based identity, which is the
  thing keeping selection independent of upstream ordering.
- **Profiles are salted by name**, so `smoke_500` is not a superset of `smoke_100` — a nested
  prefix would make the larger profile's extra rows systematically unlike its first hundred.
- **Canonical serialization is pinned**: sorted keys, no ASCII escaping, LF endings.
- **Packages are byte-identical**: fixed member order, fixed timestamps, stored rather than
  deflated so the digest does not depend on the zlib version.

## Licensing

`redistribution` in the registry decides whether converted rows may be committed here.
`unclear` and gated sources mean: commit the recipe and the digest, never the data. The
`build/` directory is git-ignored for that reason.

| Source | Licence | Gated | Status |
|---|---|---|---|
| `dolly-15k` | CC-BY-SA-3.0 | no | enabled |
| `uner-tagalog` | CC-BY-SA-4.0 | no | enabled — **upstream split is `test`**, used as training |
| `uner-multilingual` | CC-BY-SA-4.0 | no | enabled; tier-3 builds blocked on `language_field` |
| `sea-instruct-2602` | ODC-BY | yes | terms accepted; **still disabled** — no subsetting policy, no inspected schema |
| `kalahi` | CC-BY-4.0 | yes | terms accepted; fetchable — **evaluation only, never training** |

Upstream terms for both gated sources were accepted on 2026-08-22 (issue #10). That changed
exactly one thing: they are now *fetchable*. It did not make them *usable*, and it did not
make their rows *redistributable* — three separate questions, and only the first was answered.

Three policies are enforced in code, not merely documented: an evaluation-only source cannot
be requested for training, a disabled source fails before anything is fetched, and a gated
source may never have its converted rows committed here.

## The acceptance ladder

Four tiers, each answering a different question. Decided 2026-08-22, closing issue #6.

| Tier | Source | Question it answers |
|---|---|---|
| 1 | `dolly-15k` | Does ordinary English SFT work? |
| 2 | `uner-tagalog` | Does the Philippine-language path work? |
| 3 | `uner-multilingual`, **excluding `tl`** | Does general multilingual handling work, independently of tier 2? |
| 4 | `kalahi` | Held-out Filipino **evaluation**. Never training. |

Tiers 1–3 are pipeline **acceptance** runs — they establish that training mechanics work on
that kind of text. Tier 4 is the only **evaluation** tier. Keeping those two ideas apart is
the point of the ladder: UNER Tagalog validates training mechanics using Filipino text,
while Kalahi independently evaluates the resulting model.

### Independence is structural, then proved

`uner-tagalog` converts the Tagalog portion of Universal NER v1, and `uner-multilingual`
converts the same v1 corpus including `tl`. Rather than build both and then detect the
overlap, **tier 3 excludes `tl`** — so the tiers mean different things by construction.

The exclusion then gets a continuous guard, because a future source bump or converter change
could quietly reintroduce the leakage:

```bash
python -m validation_datasets disjoint uner-tagalog:full uner-multilingual:multi_500
```

It compares committed fingerprint files, so it needs no network and no build, and CI can
enforce it on every push. **Live and passing:** tier 2 (220 examples) and tier 3 (500) are
built, approved and committed, and share no canonical example.

The filter itself fails closed: no usable exclusion mechanism, a field missing from the
fetched rows, or a removal count that does not match the recorded expectation all raise
rather than building a "tier 3" profile that still contains Tagalog.

### How `tl` is excluded, given there is no language column

Resolved 2026-08-22 by inspecting the pinned revision (issue #12). **Universal NER in the
Aya format has no language column** — every row is just `inputs` and `targets`. A
`language_field` filter was therefore impossible, and guessing a column name would have
produced a filter that silently matched nothing.

What the corpus does carry is a **per-language instruction preamble**, byte-identical across
every row in that language. Danish rows all open `Angiv venligst alle navngivne enheder…`;
Tagalog rows all open `Sa aktibidad na ito, kailangan mong hanapin…`. So the shared template
of the single-language `uner-tagalog` corpus identifies Tagalog rows inside the multilingual
one — including any that a fingerprint subtraction against its 220 rows would have missed.

The template is derived, never hardcoded: 545 characters, computed as the longest common
prefix across all 220 tier-2 rows, and rejected if it comes out shorter than 100.

### Tier 3 is already Tagalog-free, and the filter still runs

Measured across every split at the pinned revision:

| split | rows | matching the Tagalog template |
|---|---:|---:|
| train | 54,957 | **0** |
| validation | 7,788 | **0** |
| test | 14,950 | **220** — exactly the tier-2 corpus |

Tagalog lives only in `test`, and tier-3 profiles draw from `train`, so the tiers are
disjoint upstream before any filtering. The exclusion still runs on **every** build, against
an exact recorded expectation of `expected_removals: 0`. Exact in both directions: removing
*more* than expected stops the build too, because it means upstream moved a language between
splits and that should be a human decision rather than quietly becoming the new acceptance
data.

### Tier 2 uses an upstream `test` split, deliberately

`uner_llm_inst_tagalog` publishes only a `test` split. This suite imports it as **source
material for an SFT acceptance run**, not as an evaluation set, and the manifest records both
facts side by side:

```
source_split:    test        # what upstream calls it
pipeline_usage:  training    # what we do with it
```

The deterministic tooling derives its own train/validation partition from it as needed.
**Results on this source must never later be reported as an independent Tagalog quality
benchmark** — that role belongs to tier 4.

## Approved profiles

`build/` is git-ignored, so a manifest written there only ever proved that a run agreed with
itself. A fresh clone needs a committed expectation to check a rebuild against, or
"the same pinned source still produces the dataset we approved" is not a checkable claim.

Each approved profile commits two small files, neither containing dataset rows:

```
approved/<dataset>/<profile>.json           digests, counts, pinned revision, usage
approved/<dataset>/<profile>.fingerprints   sorted canonical fingerprints, truncated
```

`build` and `verify` both compare against the committed `.json` and fail on any difference —
a moved revision, a changed converter version, a different selected-row set, a changed split
digest or example count. Recording a new expectation is a separate, deliberate command:

```bash
python -m validation_datasets approve dolly-15k --profile smoke_100 --date 2026-08-22
```

Separate on purpose. If `build` wrote the approval itself, the expectation would follow the
code around instead of pinning it, and a converter change would re-approve its own output.

Fingerprints are truncated to 64 bits. A collision at this suite's scale is ~1e-13, and would
only ever report an overlap that is not there — the gate fails safe.

## Usage

```bash
python -m validation_datasets list
python -m validation_datasets build   dolly-15k --profile smoke_100
python -m validation_datasets verify  dolly-15k --profile smoke_100
python -m validation_datasets package dolly-15k --profile smoke_100
python -m validation_datasets approve dolly-15k --profile smoke_100 --date 2026-08-22
python -m validation_datasets disjoint uner-tagalog:full uner-multilingual:multi_500
```

`build` and `approve` require `pip install datasets` and network access. `verify` and
`disjoint` run offline against committed files, which is what lets CI enforce them.

## Verified end to end

The Dolly `smoke_100` profile has been built from the pinned revision, rebuilt to an
identical digest, packaged, and **accepted by the validator container**:

```
fetched 15011 rows from databricks/databricks-dolly-15k@bdd27f4d94b9
built 100 examples, sha256 0dacfbf2af8b3c54...
rebuild digest identical
package sha256 2fc68577a344b79e...
validator: 8/8 checks pass, median 112 tokens, longest 943 of 4096
approved 2026-08-22 -> approved/dolly-15k/smoke_100.json
```

The drift gate was also exercised against its failure case, not only its success case: a
build tampered so that its own manifest still agreed with it — the exact scenario the
git-ignored manifest could not catch — was rejected by `verify` with exit 1 and a named
digest mismatch.

### The package digest changed once, for a real reason

It was `62913fa5...` until 2026-08-22. A zip member records the OS that created it, and
Python fills that byte in from the host — 0 on Windows, 3 on Linux — so the same inputs
produced different archive bytes depending on who built them. Every digest claim here was
therefore only true per-platform. CI caught it on the first Linux run; no amount of local
rebuilding could have.

`create_system` is now pinned in both the packager and the fixture generator, verified by
regenerating the whole corpus on Linux and on Windows and comparing: **9 of 9 archives
byte-identical**. The `.jsonl` digests were never affected — only archives.
