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
enforce it on every push. The filter itself fails closed: an unknown language field, a field
missing from the fetched rows, or an exclusion that removes nothing all raise rather than
building a "tier 3" profile that still contains Tagalog.

> **Tier 3 profiles cannot be built yet.** `language_field` for
> `universalner/uner_llm_instructions` has not been confirmed against the pinned revision,
> and `convert.py` does not guess field names. Until it is recorded, any tier-3 build fails
> with an actionable error instead of producing a profile that silently skipped the filter.

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
package sha256 62913fa53fb407da...
validator: 8/8 checks pass, median 112 tokens, longest 943 of 4096
approved 2026-08-22 -> approved/dolly-15k/smoke_100.json
```

The drift gate was also exercised against its failure case, not only its success case: a
build tampered so that its own manifest still agreed with it — the exact scenario the
git-ignored manifest could not catch — was rejected by `verify` with exit 1 and a named
digest mismatch.
