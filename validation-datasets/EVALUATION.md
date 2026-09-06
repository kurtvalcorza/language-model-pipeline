# Held-out evaluation datasets

Evaluation is a separate evidence class from SFT acceptance and training. The first Tier 4
asset is **Kalahi**, the pinned MCQ-compatible SEA-HELM representation from
`aisingapore/Cultural-Evaluation-Kalahi`.

## Kalahi contract

The registry pins:

- revision `236decdf946498dd5f1c4efa270ba169fe725ad3`;
- config `tl`;
- split `eval`;
- exact full-profile size: 150 rows;
- upstream fields `id`, opaque `label`, `prompts[{question, mcq_options, mcq}]`, and
  `metadata{language, category, topic}`.

The source is gated. Upstream terms were accepted on the account recorded in issue #10, but
that does **not** grant this repository permission to redistribute the rows. Generated
`validation-datasets/build/` content stays git-ignored. Commit recipes, manifests, digests
and truncated fingerprints only.

## Build

Install the optional fetch dependency and authenticate the Hugging Face account that accepted
the source terms, then run:

```bash
pip install datasets
python scripts/build_kalahi_evaluation.py kalahi --profile full
```

The builder:

1. requires an enabled, explicitly evaluation-only registry source;
2. fetches only the exact pinned revision/config/split;
3. requires the `full` profile to still contain exactly 150 rows;
4. validates every source row against the pinned MCQ shape;
5. preserves `label` and every prompt/metadata field without assigning new semantics;
6. writes deterministic `evaluation.jsonl`, `manifest.json`, and `fingerprints` under
   `validation-datasets/build/kalahi/full/`;
7. **does not create `train.jsonl` or a DIMER training ZIP**.

If the source shape or row count differs, the build fails and requires a deliberate source
review rather than silently creating a new evaluation set.

## Approve and verify

After reviewing a successful gated-source build:

```bash
python -m validation_datasets approve kalahi --profile full --date YYYY-MM-DD
python -m validation_datasets verify  kalahi --profile full
```

`approve` commits only expectation metadata and fingerprints, never the gated rows. `verify`
compares a rebuild against that committed expectation.

`python -m validation_datasets package kalahi --profile full` intentionally produces no
package: the DIMER package command recognizes only SFT `train`/`validation`/`test` splits,
and Kalahi must never enter that path.

## What this does not establish

This loader prepares reproducible held-out evaluation records. It does **not** define what the
upstream `label` means, reproduce SEA-HELM scoring, choose a model-quality threshold, or turn
Kalahi into training data. Those belong to a separately versioned evaluation harness whose
metric semantics can be reviewed independently.
