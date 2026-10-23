# Release verification

`tutorials/language_model_finetuning_colab.ipynb` (`E2E`, kind `TUTORIAL`) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, and `tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence
under DIMER Notebook Specification 3.0. This file is the durable release-gate record for the notebook.

**Notebook Specification 3.0 migration (2026-10-22).** The artifact-inference companion
`tutorials/language_model_artifact_inference_colab.ipynb` (`ARTIFACT-INFERENCE`) is retired (KND11, §32.2); its
consumer path is now Section 11 of the E2E notebook, the external-artifact section (§19.1). Section 11 reads the
release asset `sample-bundle-v1` (`language_model_finetuning_adapter_bundle.zip`, 22,231,859 bytes, SHA-256
`f99348aab5537bbccc26a0283f0fd6e714c0e3b58d27beb8c3d2586951ffaa35`), which is the export of this notebook's
default path from the recorded 2026-10-07 Colab T4 run at `d185817` / blob `8a2c6ad4886d` (table below), so the
existing pin meets SART8–SART10 and is kept (§32.2 item 3); no new release was made. The companion's executions
below stay as history (REL14) and are not evidence for the migrated notebook. From the migration on, Section 10
writes the bundle ZIP deterministically (sorted members, fixed timestamp and mode), so identical bundle files give
identical ZIP bytes. LoRA training on a T4 is not bit-reproducible, so a new run's bundle files, and therefore its
ZIP digest, will differ from the pinned asset; the pin is not expected to be reproduced by a re-export.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- the notebook parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs
  or execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory
  markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its profile, the notebook-spec
  version and the standalone carrier; `metadata.dimer` declares the profile, spec `3.0`, kind `TUTORIAL`, `standalone: true`
  and `generated_from` (repository, generating commit, module SHA-256, generator);
- the §7.1 section order (Export and fresh reload → Interpretation → Conclusion → Section 11 external artifact → Section 12 optional activity → Troubleshooting → References) and the Section 11 markers (pinned digest before extraction, `verify_artifact_bundle`, the data-contract check, `outputs/external_artifact/`, the trust-boundary statements);
- ST1: the only `github.com/kurtvalcorza` reference allowed in code is the exact pinned sample-artifact URL, compared as a whole token, in the carried module cell; a reference holding `..`, `%2e`, `?` or `#` is refused;
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository
  import on the primary path; exactly one cell tagged `embedded_module` equal to `src/lmpipeline/pipeline.py`
  after the generator's documented rewrite; the inline `MANIFEST` equal to
  `weights/smollm2-360m/dimer-base-manifest.json` and the inline `PINS` equal to the `pyproject.toml` runtime
  pins; the notebook byte-identical to `tools/build_notebook.py` output for its template; exactly one
  kernel cell, which builds (or reuses) the hash-locked isolated environment from
  `tutorials/requirements-isolated.lock.txt` and routes every later cell to it (no in-kernel install, no restart); `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cell (and repeated in the inline manifest,
  which the notebook asserts against the module before fetching), the revision is a 40-hex immutable commit,
  and the same identity string appears in `README.md`, `weights/smollm2-360m/MODEL_CARD.md` and
  `weights/README.md` with no stray revisions beyond the other registry entries and the two pinned sample
  datasets;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `LanguageModelPipeline.from_pretrained(weights_dir=...)`, `validate_inputs`, `prepare_splits`, `generate`,
  `evaluate_loss`, `evaluation_report`, `export_adapter_bundle`, `reload_base`, `load_adapter`,
  `validate_prompts`, `verify_artifact_bundle`, `extract_zip_safely`, the base-model validation loss before
  adaptation, the reload reproduction check, the adapter-activity readout, the seeded sampling), the ceiling prints, the exports, the
  learner-facing statements (what is trained and what is not, assistant-only masking, manufactured validation
  split, optimisation metrics versus task quality, adapters incomplete without their base, trust boundary,
  no network fallback) and the gated-off defaults (`USE_BYOD`, `USE_OWN_ARTIFACT` and the optional exercises are
  `False` on a form line); forbidden patterns (credential-in-URL, any
  `git clone` / `github.com/kurtvalcorza` / repository import on the primary path, a mutable `revision='main'`
  or unpinned `git+https` dependency, `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`,
  and — outside the carried module cell — `from transformers import`, `AutoModelForCausalLM`,
  `PeftModel.from_pretrained(`, `huggingface_hub`, `HF_TOKEN`, worker/subprocess calls);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes
  an unsupported release-grade, production-readiness or benchmark claim;
- `weights/smollm2-360m/MODEL_CARD.md` front matter (spec 1.1, `base_model`), single H1, required heading
  order, and the `## Packaged Snapshot Details` provenance section.

CI also installs the light development pins (no torch), runs `ruff`, `tools/build_notebook.py --check`, the offline unit suite (`tests/test_snapshot_helpers.py`, `tests/test_role_helpers.py`,
`tests/test_notebook_parity.py`, `tests/test_external_artifact_section.py`, `tests/test_colab_tutorials.py`; fake
tokenizer, injected downloader, no weights) and the repository's contract suite. These are source/provenance
and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab T4 GPU runtime | The runtime the tutorials are written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel | Kaggle GPU kernel (T4 / P100), Python 3.12 image | Reproducible clean-room executor of the same class; the notebook is pushed verbatim plus one leading shim cell that provides `google.colab` and chdirs to a scratch directory (no repository checkout is needed — the notebook is standalone). The 2026-09-07 Kaggle T4 run of the previous, repository-installing revision (52.7 s training loop, 1.52 GiB peak, Qwen3-0.6B) does **not** cover the standalone carrier or the pinned SmolLM2-360M snapshot |
| Local workstation (pre-flight only) | RTX 5070 Ti, sequential cell executor with a `google.colab` shim | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open the exact E2E notebook revision in a new GPU runtime (Colab, or the Kaggle executor above) with
   **no repository checkout** and a clean model cache;
3. run it top-to-bottom without editing implementation cells (form parameters at their defaults for the
   sample path: `USE_BYOD = False`, `DATA_SOURCE = "Sample: Filipino SFT"`, `SAMPLE_LIMIT = 120`,
   `MAX_SEQUENCE_LENGTH = 512`, `EPOCHS = 1`, `RUN_NEW_PROMPT_INFERENCE = True`);
4. verify that Section 1 builds (or reuses) the isolated environment from `tutorials/requirements-isolated.lock.txt`
   with **no restart** and no install into the kernel, that the runtime cell reports
   `NOTEBOOK_SOURCE.repository_revision` equal to the commit recorded in `metadata.dimer.generated_from`, and that
   the versions it prints equal the inline `PINS` (= `pyproject.toml`) — `bitsandbytes==0.49.0`, `peft==0.18.0`,
   `transformers==4.57.1` and `torch==2.14.0` on the CUDA runtime are the pins most likely to need a
   wheel-availability check;
5. verify every default-path stage completes:
   - isolated environment built from the hash-locked pins with no GitHub access and no token;
   - the carried module cell executes (defines `LanguageModelPipeline` and the helpers) with no import of
     `lmpipeline`;
   - the inline `MANIFEST` is asserted against the module identity and written to `weights/smollm2-360m/`;
     `stage_missing_files(WEIGHTS_DIR, allow_download=True)` reports all 13 manifest entries on a clean
     runtime, `verify_snapshot` returns the manifest dict, and `from_pretrained(weights_dir=WEIGHTS_DIR)`
     reports `source == 'local-snapshot'`, `quantized == True` on CUDA;
   - the pinned sample loads at its dataset revision; `{'train': 96, 'validation': 24}` with the defaults
     and a printed dataset digest; the ceilings surfaced;
   - `validate_inputs` writes `outputs/language_model_finetuning_input_manifest.json` (verdict `accepted`,
     one recorded rejection finding from the leaked-split probe); `prepare_splits` prints supervised-token
     counts and the bracketed first example;
   - two greedy baseline answers and the base model's validation loss (`base_validation_loss`); LoRA attached
     with about 1.2 % trainable parameters; one epoch with train and validation loss printed and the
     `validation loss base → adapted` line; adapted answers and `METRICS` (`baseValidationLoss`, `trainLoss`,
     `validationLoss`, `validationPerplexity`, `epochs`, `wallSeconds`, `peakGpuMemoryBytes`);
   - `evaluation_report` writes `outputs/language_model_finetuning_evaluation_report.json` with verdict
     `sample-sanity` and a `baselines` entry for `baseValidationLoss`, stated as optimisation evidence only;
   - new-prompt answers with the adapter off and on; `export_adapter_bundle` writes the bundle and its
     `artifact-manifest.json`; the fresh reload (into a separate object; the trained model stays in memory) passes
     the three hard checks and prints the reproduction line — reloaded validation loss, its absolute difference
     from the in-memory loss against the 0.05-nat tolerance, and whether the 16-token greedy openings are identical
     (record both numbers and the verdict; `NOT reproduced` is a finding, not a pass); `outputs/language_model_finetuning_result.json`,
     `outputs/language_model_finetuning_probes.csv` and `outputs/language_model_finetuning_adapter_bundle.zip`
     written with `NOTEBOOK_SOURCE`, model revision, model licence, runtime versions and device;
6. verify Section 11, the external-artifact section, on the default path (same run): it prints the pinned URL,
   SHA-256 and producer of `sample-bundle-v1` before downloading; the whole-ZIP digest is checked before
   extraction; `verify_artifact_bundle` passes and the bundle's base model and revision equal the pinned ones; the
   data-contract check passes (the bundle names the default `Sample: Filipino SFT` dataset and revision); the
   adapter is attached to a fresh base with the printed `lora_b_max_abs`; `validate_prompts` writes the input
   manifest with the blank-prompt finding; greedy answers with the adapter off and on and `answers_differ`;
   `evaluation_report` verdict `not-measurable`; `input_manifest.json`, `evaluation_report.json`, `result.json` and
   `generations.csv` under `outputs/external_artifact/`; this run's own bundle ZIP is never read. Optional
   own-bundle path, in a separate session: `ARTIFACT_SOURCE = 'Your own bundle ZIP'`, `ARTIFACT_ZIP_PATH` set to the
   step-5 ZIP and `EXPECTED_ARTIFACT_ZIP_SHA256` to its printed digest;
7. verify the exports exist and the interpretation sections match the observed path;
8. record the notebook Git blob ids, commit, runtime (platform, Python, PyTorch, transformers, peft, GPU),
   model identifier and immutable revision, whether the model cache was clean, **whether a kernel restart or a
   `google.colab` shim was needed** (either one disqualifies the run as Run-all evidence), the sample-artifact
   SHA-256 Section 11 checked, outcome, produced outputs, and any warning or applicable `SHOULD` deviation in
   the table below;
9. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of the notebook (verify with
`git rev-parse <commit>:tutorials/<notebook>`). Wall times, when recorded, are the sum of per-cell times
reported by the executor and include installs and the model download; they are measurements for the stated
runtime, not general estimates.

### Manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | `e9d7e6b` / `469911578d05` | Kaggle T4 (`kurtvalcorza/dimer-nb2-language-model-finetuning` v2) | E2E default sample path (`language_model_finetuning_colab.ipynb`), an earlier revision that still installed into the kernel | 360.9 s | **PASSED only after one kernel restart** — pass 1 stopped at the install cell (`Core dependencies changed while older modules were loaded: cuda-bindings` → restart instruction); pass 2, after the executor restarted the kernel, ran 10/10 code cells with a leading `google.colab` shim cell. 28 files, 727 MB staged; train loss 3.266, validation loss 3.261, perplexity 26.1, 1.186 % trainable, `bfloat16` compute on the T4, 55.5 s loop, 0.66 GiB peak. The in-memory and reloaded 16-token openings differed and the cell still printed PASS (now reported explicitly by the reload check). **Not Run-all evidence** (RUN10/ENV6), and not evidence for the current isolated-runtime blob |
| 2026-10-07 | `d185817` / `8a2c6ad4886d` | Colab CLI 0.7.4 sequential execution (`colab exec -f`), fresh Colab Tesla T4 — not a browser Run all; no execution counts, order from `exec.log` | E2E default sample path (`language_model_finetuning_colab.ipynb`), default settings only | 241.5 s (session wall) | **one pass, no restart, 0 errors** — 12/12 code cells in order (cell `language_model_finetuning-08`, the carried module definitions, prints nothing). Python 3.12.12, torch 2.14.0+cu130, transformers 4.57.1, peft 0.18.0, 4-bit, `bfloat16` compute, 1.186 % trainable. Validation loss base → adapted 3.5010 → 3.2640 (−0.2370 nats; perplexity 33.1 → 26.2); train loss 3.2654; 55.8 s loop; 0.66 GiB peak. Reload: 3.2659 vs 3.2640, \|Δ\| 0.0019 ≤ 0.05 → reproduced; greedy 16-token openings differ by one token (`ng`/`ang`), reported. Bundle ZIP 22,231,859 bytes, SHA-256 `f99348aab5537bbccc26a0283f0fd6e714c0e3b58d27beb8c3d2586951ffaa35`. Evidence: `docs/verification/2026-10-07-colab-t4/language_model_finetuning_colab/` — executed notebook SHA-256 `f2535a6be4b74673ba080fc56d185265b560cf61a83f8ea11e6eee7d2ea92ac9`, `run_summary.json` `d1a45c35621a115d768fdb2bf7de50a1c317b186e33c4c8edba5efe5e236807a`, `exec.log` `55f372582297a1d0011c0e8bcfc10343e3ab262d8b3181af77acae9fe496f4b2` (as first committed; redacted 2026-10-22, now `28f46472c6a560c080be0ed545e061f89f078b2d6aac3990fc0eb25c1bb693f7`). Not exercised: Section 10 re-run, BYOD (REL12), the optional activity, forms and download dialogs. The previous attempt at `f2053aa` failed in Section 3 (`google.colab.__spec__ is None` in the isolated worker), fixed in `d185817` |
| 2026-10-07 | `70fd000` / `f3be29a5d95b` | Colab CLI 0.7.4 sequential execution (`colab exec -f`), fresh Colab Tesla T4 — not a browser Run all; no execution counts, order from `exec.log` | Companion default path (`language_model_artifact_inference_colab.ipynb`): pinned sample bundle `sample-bundle-v1`, default settings only | 148.3 s (session wall) | **one pass, no restart, 0 errors** — 9/9 code cells in order (cell `language_model_artifact_inference-08`, the carried module definitions, prints nothing). Section 4 downloaded the release asset, matched the pinned whole-ZIP SHA-256 `f99348aab5537bbccc26a0283f0fd6e714c0e3b58d27beb8c3d2586951ffaa35` before extraction, `artifact_source` = `sample artifact: …/releases/download/sample-bundle-v1/…` (no upload dialog); manifest `peft_adapter` v1, 12 files, 22,228,333 bytes; producer provenance printed. `lora_b_max_abs` 0.00816; `next_token_logit_max_abs_delta` 2.871; greedy answers differ on 2 of 2 prompts → `ADAPTER_ACTIVITY` `active`; evaluation verdict `not-measurable`; the blank-prompt probe rejected. Evidence: `docs/verification/2026-10-07-colab-t4/language_model_artifact_inference_colab/` — executed notebook SHA-256 `50dc3019a597c77b914877df6f5a3507fd6b486ae752acc2413e4b514c516a0c`, `run_summary.json` `1253047314ebf9fd33934ac4c4246f6bf0a29a38284f24cf519b3d1c263e828f`, `exec.log` `00072424801f8f9c3aae4247420a0831d5b5ff0d3f19bd0bb79d76af715927f8` (as first committed; redacted 2026-10-22, now `4561f9c9572297b4355bd0d0572329721a8bf87c06ff4cd694f2d923cd79db96`). Not exercised: own-bundle path (`USE_OWN_ARTIFACT`, upload, `ARTIFACT_ZIP_PATH`/`ARTIFACT_DIR`), `CUSTOM_PROMPT`, the Section 8 exercise, Section 7 re-run, forms. A first attempt at the same head stopped before cell 1 (`colab exec`: `Connection was lost`; 0 cells run) and was repeated on a fresh VM |
| 2026-10-22 | `50f07ea` / `b8df6bc79be5` | Colab CLI 0.7.4 sequential execution (`colab exec -f`), fresh Colab Tesla T4 — not a browser Run all | E2E default sample path of the **Notebook Spec 3.0 revision** (PR #79; the companion folded into Section 11; the second pass of §32.2), default settings only | 252.1 s (session wall) | **PASS, one pass, exit 0, input unchanged** — executed cell sources equal the blob (36 cells). The fresh reload reproduced (validation loss 3.2646 vs in-memory 3.2627, |Δ| 0.0018 ≤ 0.05); this run's own export printed SHA-256 `00df5a9a1c0941e12d1cb2cf8de514170d3e057ac23cdc9a572c2296dcb8db74` (LoRA training is not bit-reproducible, so it differs from the pin). **Section 11 ran end to end on the default path:** it downloaded `sample-bundle-v1`, checked the whole-ZIP SHA-256 `f99348aab5537bbccc26a0283f0fd6e714c0e3b58d27beb8c3d2586951ffaa35` before extraction and every file against the bundle manifest, matched the bundle base `HuggingFaceTB/SmolLM2-360M-Instruct` @ `a10cc1512eabd3dde888204e902eca88bddb4951` (`trustRemoteCode: False`), attached the adapter to the 4-bit base on `cuda:0` (r 8, alpha 16), generated for the two sample prompts with the adapter off and on (`answers_differ` 2 of 2; verdict `not-measurable`, one finding) and wrote four files under `outputs/external_artifact/` (printed; the files stayed on the released VM). Evidence: `docs/verification/2026-10-22-colab-t4/language_model_finetuning_colab/` — executed notebook SHA-256 `1133910e98efd2a67ccd903f2956d3ee22446e763c23b6fed2129d2ff6c99a80`, `run_summary.json` `f2c4a7262edb19e08bb66250440ee48d81e07abe4ae62de5c7c0449d1e1b4a44`, `exec.log` captured `e87b33606da02821b6c7071a2ffef53c1b0f5b6a6127cd4e885af478a7c822da`, committed `154e90744be68e3328e0fd68fb83907a474b21fcd6eb5306c12f3f637b90383a` after the local run directory was replaced by `<local-run-dir>` (2 occurrences) |
| 2026-10-22 | `edcb9e3` / committed blob `b8df6bc79be5`, executed as a run copy with one override, `RUN_DECODING_EXPERIMENT = False` → `True` (run copy blob `253eb0989746`, SHA-256 `db15d1a3ee28372406ea50939cfd7d0a0f647453f21149bc2e2bb00429220488`; it differs from the committed blob only in that assignment literal) | Colab CLI 0.7.5 `colab exec -f` (a scripted run, not a browser Run all), fresh Colab Tesla T4 | E2E default sample path, then the optional Section 12 decoding activity on the adapted model; every other setting at its default | 385.4 s (session wall) | **PASS, exit 0, sequential, input (the run copy) unchanged, 15/15 code cells in the executor log, 0 errors**, started 2026-10-22T13:42:49Z. Default path: validation loss base → adapted 3.5010 → 3.2636; the fresh reload reproduced (3.2680 vs 3.2636, \|Δ\| 0.0044 ≤ 0.05). Section 12 printed the greedy answer's longest repeated 4-word phrase "occurs 3x", the two seeded sampled answers' (seeds 42 and 43) "occurs 2x" each, and `{'sampled_answers_identical': False}`. Evidence: `docs/verification/2026-10-22-colab-t4/language_model_finetuning_colab-edcb9e3-activity/` (SHA-256 table below) |
| 2026-10-23 | `6c22e80` / `4651e699c994` | Colab CLI `colab exec -f` (a scripted run, not a browser Run all), fresh Colab Tesla T4 | E2E default sample path of the **refusal-display revision** (PR #83), default settings only | 266.3 s (session wall) | **PASS, exit 0, sequential, input unchanged, 15/15 code cells in the executor log, 0 errors**, started 2026-10-23T09:06:59Z. Validation loss base → adapted 3.5010 → 3.2631 (−0.2378 nats per supervised token; perplexity 33.1 → 26.1); train loss 3.2653; 1.186 % trainable; 51.9 s loop; 0.66 GiB peak; evaluation verdict `sample-sanity`. The fresh reload reproduced (3.2648 vs in-memory 3.2631, \|Δ\| 0.0016 ≤ 0.05; greedy 16-token openings differ, reported). This run's own bundle printed `Artifact: outputs/language_model_finetuning_adapter_bundle.zip (21.2 MB) SHA-256 025d7a4f931e7a6bded10f08b3b79de4f564113ad349dfc6d476437b9195869b` (LoRA training is not bit-reproducible, so it differs from the pin). Section 11 checked the whole-ZIP SHA-256 of `sample-bundle-v1` (`f99348aab553…`) before extraction, matched the bundle base identity, attached the adapter on `cuda:0` and printed `{'verdict': 'not-measurable', 'answers_differ': '2 of 2', 'findings': 1}`. Section 12 printed its off message (`RUN_DECODING_EXPERIMENT is off`). Evidence: `docs/verification/2026-10-23-colab-t4/language_model_finetuning_colab-6c22e80-default/` (SHA-256 table below) |
| 2026-10-23 | `6c22e80` / committed blob `4651e699c994`, executed as a run copy with one override, `RUN_DECODING_EXPERIMENT = False` → `True` (run copy blob `c6b43c5a6b55`, SHA-256 `40519a855e48c0a27b45e2328f4249963e8630f9a96eada4148a0a7b7b37449d`; it differs from the committed blob only in that assignment literal) | Colab CLI `colab exec -f` (a scripted run, not a browser Run all), fresh Colab Tesla T4 | E2E default sample path, then the optional Section 12 decoding activity on the adapted model; every other setting at its default | 303.8 s (session wall) | **PASS, exit 0, sequential, input (the run copy) unchanged, 15/15 code cells in the executor log, 0 errors**, started 2026-10-23T09:11:39Z. Default path: validation loss base → adapted 3.5010 → 3.2631 (−0.2379 nats; train loss 3.2651); the fresh reload reproduced (3.2656 vs 3.2631, \|Δ\| 0.0026 ≤ 0.05); Section 11 verdict `not-measurable`, `answers_differ` 2 of 2. Section 12 printed the greedy answer's longest repeated 4-word phrase "occurs 3x", the seeded sampled answers' "occurs 1x" (seed 42) and "occurs 3x" (seed 43), and `{'sampled_answers_identical': False}`. Evidence: `docs/verification/2026-10-23-colab-t4/language_model_finetuning_colab-6c22e80-activity/` (SHA-256 table below) |

**Optional-activity run evidence (`edcb9e3`, 2026-10-22).** The run copy is committed beside the evidence as `run_copy.ipynb`. Local paths were replaced by `<local-run-dir>` before the first commit:

| File (under `docs/verification/2026-10-22-colab-t4/language_model_finetuning_colab-edcb9e3-activity/`) | Captured | Committed |
|---|---|---|
| `language_model_finetuning_colab_edcb9e3_colab-cli-t4_activity.ipynb` | `5732cdc2067cc80dadfbd396cc4b75c5dd16b45cb0616ca1ee7c3e48a60ff083` | unchanged (no local path) |
| `exec.log` | `abf2113badeba77762d0729501bea93cfeb22c96bdf05bbdf2c1203897d0256c` | `5183e603db92d2c4525aa4121a71a1da5fcdb3634310776395838afe4d340e00` (2 replacements) |
| `run_summary.json` | `8af92ee771ba5f40e211dcc5b5eda4dd81f797feba650f9aafcdaa86f92a19f3` | unchanged (no local path) |
| `run_copy.ipynb` | `db15d1a3ee28372406ea50939cfd7d0a0f647453f21149bc2e2bb00429220488` | unchanged (no local path) |

**Refusal-display revision run evidence (`6c22e80`, blob `4651e699`, 2026-10-23).** The activity run copy is committed beside its evidence as `run_copy.ipynb`. The executed notebooks and the run copy were checked against `run_summary.json` (`executed_sha256`, `run_copy_sha256`) before any edit. Local paths were replaced by `<local-run-dir>` before the first commit (byte-wise; no other byte changed):

| File (under `docs/verification/2026-10-23-colab-t4/`) | Captured | Committed |
|---|---|---|
| `language_model_finetuning_colab-6c22e80-default/language_model_finetuning_colab_6c22e80_colab-cli-t4_default.ipynb` | `38941bf4dc7e516bd1871311a76492f706aecab3ecfd3b1ec55d738e48a41de9` | unchanged (no local path) |
| `language_model_finetuning_colab-6c22e80-default/exec.log` | `1083d74e32be9de085904f9e8b8ab159e28c48f0f5a49922b304277c01a443b9` | `6c0d74a8231c881d2441618cd5f9fc1e3b21d2d43987eb38d0621679919b1250` (2 replacements) |
| `language_model_finetuning_colab-6c22e80-default/run_summary.json` | `beb4beee25f3993a7ca216c6042a429a8ce5701c519c838bcabd442ec1c327e1` | unchanged (no local path) |
| `language_model_finetuning_colab-6c22e80-activity/language_model_finetuning_colab_6c22e80_colab-cli-t4_activity.ipynb` | `49e60f88bc37b90080d8f03617081477209f4c80f30f85314d7fa98bed94ce32` | unchanged (no local path) |
| `language_model_finetuning_colab-6c22e80-activity/exec.log` | `a95ff1ec23917ec8c7a62bdc4e202696c4dcadb55597ed928b9441d1c5482e5d` | `b9b8f1859ff5752bbf71e5c79e786c347b065ec73e15854734bc1b7fb277da0f` (2 replacements) |
| `language_model_finetuning_colab-6c22e80-activity/run_summary.json` | `868ef0d628e2afbef2348acb426cd8472889e8c69943e7d6862e0d89cca0d1b8` | unchanged (no local path) |
| `language_model_finetuning_colab-6c22e80-activity/run_copy.ipynb` | `40519a855e48c0a27b45e2328f4249963e8630f9a96eada4148a0a7b7b37449d` | unchanged (no local path) |

### Local-path redaction of earlier evidence (2026-10-22)

The local run directory in these earlier committed evidence files was replaced by `<local-run-dir>/`, the
convention the later evidence in this repository already uses. Only those bytes changed: line endings and every
other byte are as committed. Git history is not rewritten, so earlier commits still hold the original files.
SHA-256 before (committed on `main`) → after:

| File | Replacements | SHA-256 before | SHA-256 after |
| :--- | ---: | :--- | :--- |
| `docs/verification/2026-10-07-colab-t4/language_model_artifact_inference_colab/exec.log` | 2 | `00072424801f8f9c3aae4247420a0831d5b5ff0d3f19bd0bb79d76af715927f8` | `4561f9c9572297b4355bd0d0572329721a8bf87c06ff4cd694f2d923cd79db96` |
| `docs/verification/2026-10-07-colab-t4/language_model_finetuning_colab/exec.log` | 2 | `55f372582297a1d0011c0e8bcfc10343e3ab262d8b3181af77acae9fe496f4b2` | `28f46472c6a560c080be0ed545e061f89f078b2d6aac3990fc0eb25c1bb693f7` |

Where this document quotes one of these files' earlier hash as its committed hash, the new hash is now
given beside it.


## SART7 deviation (2026-10-22)

- **Tag:** `sample-bundle-v1` (asset `language_model_finetuning_adapter_bundle.zip`).
- **Mutable release:** the GitHub release is not immutable (`gh api repos/kurtvalcorza/language-model-pipeline/releases/tags/sample-bundle-v1 --jq .immutable` returns `false`), so SART7 (SHOULD) is not met.
- **Pinned digest:** the notebook pins the asset by `SAMPLE_ARTIFACT_SHA256 = "f99348aab5537bbccc26a0283f0fd6e714c0e3b58d27beb8c3d2586951ffaa35"` (`src/lmpipeline/pipeline.py`, carried into the notebook); the GitHub asset digest is the same value.
- **Integrity:** integrity rests on that pinned SHA-256 check. `fetch_pinned_artifact` and `open_artifact_zip` hash the whole ZIP and refuse it before extraction if the digest differs, so a replaced asset is refused, not used.
- **Owner acceptance:** the owner accepted this deviation on 2026-10-22. The release is not re-tagged and the pin is not changed.

## Current status

**Current status (2026-10-23): Candidate.** The current E2E notebook blob is `4651e699`
(`tutorials/language_model_finetuning_colab.ipynb` at `6c22e80` on branch `fix/refusal-display`, PR #83; Notebook Specification 3.0,
with Section 11), produced by the 2026-10-23 refusal-display fix (branch `fix/refusal-display`, PR #83; NOTEBOOK_SPEC 3.0 DAT19/DAT20: a refused user input is shown as one `Refused: <file name>: <rule>` line with no traceback, and the run stops at that cell). It has two recorded hosted runs, the two `6c22e80`
rows of the clean-runtime table, both Colab CLI `colab exec -f` (scripted, not a browser Run all) on a fresh Colab Tesla T4,
exit 0, 15/15 code cells in the executor log, 0 error outputs, verdict PASS:

- **Default path** (266.3 s, input unchanged; evidence `docs/verification/2026-10-23-colab-t4/language_model_finetuning_colab-6c22e80-default/`): the QLoRA training ran,
  the fresh reload reproduced the validation loss, and Section 11 checked the whole-ZIP SHA-256 of `sample-bundle-v1` before extraction.
- **Optional Section 12 decoding activity** (a `RUN_DECODING_EXPERIMENT = True` run copy, 303.8 s, run copy unchanged; evidence
  `docs/verification/2026-10-23-colab-t4/language_model_finetuning_colab-6c22e80-activity/`), which satisfies the owner's 2026-10-22 ruling that it runs before promotion.

The notebook stays **Candidate** pending only these two for blob `4651e699`:

- **REL12:** a BYOD run in which one representative input is accepted through the downstream stages and one incompatible input is refused;
- **REL2:** a browser run of the interactive journeys (BYOD upload dialog, optional-activity toggle, own-artifact ZIP); the scripted CLI runs count for the default path and the activity only.

Promotion is an integrator decision; the builder does not change the status.

**History (REL14; not evidence for `4651e699`).** The previous blob `b8df6bc7` (commit `50f07ea`) has one hosted default-path run
(the `50f07ea` row, 2026-10-22, PASS, evidence `docs/verification/2026-10-22-colab-t4/language_model_finetuning_colab/`) and one hosted
run of the optional activity (the `edcb9e3` row, 2026-10-22, PASS). They are history for `b8df6bc7`.

**Earlier history (before the Notebook Specification 3.0 migration; not evidence for `b8df6bc7` or `4651e699`, REL14).** The previous E2E
blob has one recorded one-pass execution at blob `8a2c6ad4886d` (2026-10-07, Colab CLI sequential
execution on a fresh Colab Tesla T4, default path only; table above). It is **not** a browser Run all, and the
Section 10 re-run, BYOD (REL12) and the optional activity were not exercised. The companion has one recorded
one-pass default-path execution at blob `f3be29a5d95b` (2026-10-07, same executor, on the pinned sample bundle;
own-bundle path not exercised). The earlier 2026-09-14 Kaggle T4 run of an older E2E blob needed a kernel
restart and a `google.colab` shim and does not satisfy RUN10/ENV6. The
current revisions replace the in-kernel install with the isolated environment, add the base validation loss, the
reload reproduction check and the companion's sample-artifact slot (`SAMPLE_ARTIFACT`, now
pinned to release asset `sample-bundle-v1` (whole-archive SHA-256 `f99348aab553…`), the bundle written by the 2026-10-07 Colab T4 E2E run at `d185817`). Static
validation (`tools/validate_release_assets.py`), nbformat validation, a
`compile()` sweep over every code cell, and the offline unit suite passed on the tutorial source at the
candidate revision, which is necessary but not sufficient. The registry status remains **Candidate** until a
reviewer confirms recorded runs against the notebook blobs under review and an integrator promotes them;
promotion is not performed by the builder. Facts a reviewer should weigh: `stage_missing_files` was
exercised only with an injected downloader in the unit suite (the real `hf_hub_download` fetch of all 13
manifest entries into a fresh `weights/smollm2-360m/` has not been executed); `LanguageModelPipeline`'s
model-loading, generation, loss, export and reload methods were not executed here (no model was loaded in
this pass — the helpers are tested with a fake tokenizer and the notebook was never run); the QLoRA loop was
carried over from the previous revision's executed notebook but now consumes the module's API; the standalone
carrier itself — executing the carried module cell in a runtime that has no repository checkout — has been
validated statically only (parity PASS) and through a carrier probe that executes the install, module and
manifest cells with the repository package blocked, never end-to-end. The clean run will be the first
execution of the standalone path, of the staging path, of the SmolLM2-360M pin under this module, and of the
`torch==2.14.0` / `bitsandbytes==0.49.0` pin set on a T4 inside the isolated environment. `tutorials/RELEASE_VERIFICATION.md`
mirrors this status (one Colab CLI default-path run per current blob; no browser Run all, BYOD or own-bundle run)
and holds the per-run record format.