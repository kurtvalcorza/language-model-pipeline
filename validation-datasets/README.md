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
| `uner-tagalog` | CC-BY-SA-4.0 | no | enabled — **upstream split is `test`, not train** |
| `uner-multilingual` | CC-BY-SA-4.0 | no | enabled |
| `sea-instruct-2602` | ODC-BY | yes | disabled pending terms acceptance |
| `kalahi` | CC-BY-4.0 | yes | disabled; **evaluation only, never training** |

Two policies are enforced in code, not just documented: an evaluation-only source cannot be
requested for training, and a disabled or gated source fails before anything is fetched.

## Known overlap

`uner-tagalog` converts the Tagalog portion of Universal NER v1; `uner-multilingual`
converts the same v1 corpus and includes `tl`. They are therefore **not independent
acceptance tiers**, and the registry records the overlap in both directions. `build` warns
when a source declares one. A model trained on the multilingual profile and then "accepted"
on the Tagalog profile is being evaluated on its own training data.

Resolving this needs a decision: a disjointness check, excluding `tl` from the multilingual
profile, or documented acceptance that the tiers are related.

## Usage

```bash
python -m validation_datasets list
python -m validation_datasets build   dolly-15k --profile smoke_100
python -m validation_datasets verify  dolly-15k --profile smoke_100
python -m validation_datasets package dolly-15k --profile smoke_100
```

Fetching requires `pip install datasets` and network access.

## Verified end to end

The Dolly `smoke_100` profile has been built from the pinned revision, rebuilt to an
identical digest, packaged, and **accepted by the validator container**:

```
fetched 15011 rows from databricks/databricks-dolly-15k@bdd27f4d94b9
built 100 examples, sha256 0dacfbf2af8b3c54...
rebuild digest identical
package sha256 62913fa53fb407da...
validator: 8/8 checks pass, median 112 tokens, longest 943 of 4096
```
