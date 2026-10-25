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
| 2026-10-25 | `4f3a907` / `d88be9f90a1c` | Colab CLI `colab exec -f` (a scripted run, not a browser Run all), fresh Colab Tesla T4 | E2E default sample path of the **tier C wording revision** (PR #84), default settings only | 259.8 s (session wall) | **PASS, exit 0, sequential, input unchanged, 15/15 code cells in the executor log, 0 errors**, started 2026-10-25T09:06:08Z. Validation loss base → adapted 3.5010 → 3.2604 (−0.2406 nats per supervised token; perplexity 33.1 → 26.1); train loss 3.2646; 1.186 % trainable; 0.66 GiB peak; evaluation verdict `sample-sanity`. The fresh reload reproduced (3.2627 vs in-memory 3.2604, \|Δ\| 0.0023 ≤ 0.05; greedy 16-token openings differ, reported). This run's own bundle printed `Artifact: outputs/language_model_finetuning_adapter_bundle.zip (21.2 MB) SHA-256 50f205aaacc6025d8ac238ca04e8327d9139a16c10862c4adbb7f07a760eb7d8`. Section 11 checked the whole-ZIP SHA-256 of `sample-bundle-v1` (`f99348aab553…`) before extraction, attached the adapter on `cuda:0` and printed `{'verdict': 'not-measurable', 'answers_differ': '2 of 2', 'findings': 1}`. Section 12 printed its off message. The prose's only quoted training numbers are labelled "for comparison only" from the 2026-09-14 Kaggle run of `46991157` (train 3.266 / validation 3.261); this run printed 3.2646 / 3.2604. Evidence: `docs/verification/2026-10-25-colab-t4/language_model_finetuning_colab-4f3a907-default/` (SHA-256 table below) |
| 2026-10-25 | `4f3a907` / committed blob `d88be9f90a1c`, executed as a run copy with one override, `RUN_DECODING_EXPERIMENT = False` → `True` (run copy blob `8b0a80d35913`, SHA-256 `3ecea916c0bb53db29a7b6fa3021027528cb58a50567b262c13f74eb6060e7f5`; it differs from the committed blob only in that assignment literal) | Colab CLI `colab exec -f` (a scripted run, not a browser Run all), fresh Colab Tesla T4 | E2E default sample path, then the optional Section 12 decoding activity on the adapted model; every other setting at its default | 297.1 s (session wall) | **PASS, exit 0, sequential, input (the run copy) unchanged, 15/15 code cells in the executor log, 0 errors**, started 2026-10-25T09:10:47Z. Default path: validation loss base → adapted 3.5010 → 3.2637 (−0.2372 nats; train loss 3.2650); the fresh reload reproduced (3.2678 vs 3.2637, \|Δ\| 0.0041 ≤ 0.05); Section 11 verdict `not-measurable`, `answers_differ` 2 of 2. Section 12 printed the greedy answer's longest repeated 4-word phrase "occurs 3x", the seeded sampled answers' "occurs 3x" (seed 42) and "occurs 3x" (seed 43), and `{'sampled_answers_identical': False}` (the 2026-10-23 run printed 1x for seed 42). Evidence: `docs/verification/2026-10-25-colab-t4/language_model_finetuning_colab-4f3a907-activity/` (SHA-256 table below) |
| 2026-10-25 | `c745a70` / `90a46c1dd0c9` | Colab CLI `colab exec -f` (a scripted run, not a browser Run all), fresh Colab Tesla T4 | E2E default sample path of the **prose-mismatch revision** (PR #84), default settings only | 249.6 s (session wall) | **PASS, exit 0, sequential, input unchanged, 15/15 code cells in the executor log, 0 errors**, started 2026-10-25T13:13:41Z; captured executed-notebook SHA-256 equals `run_summary.json` `executed_sha256`. Validation loss base → adapted 3.5010 → 3.2620 (−0.2390 nats per supervised token; perplexity 33.1 → 26.1); train loss 3.2665; `wallSeconds` 53.1; 1.186 % trainable; 0.66 GiB peak; evaluation verdict `sample-sanity`. The fresh reload reproduced (3.2655 vs in-memory 3.2620, \|Δ\| 0.0035 ≤ 0.05; greedy 16-token openings differ, reported). This run's own bundle printed `Artifact: outputs/language_model_finetuning_adapter_bundle.zip (21.2 MB) SHA-256 2e8dcce8728b020ba714495be43e8b966be63d83a1e221c4bfe5151a2e8b4e52`. Section 11 checked the whole-ZIP SHA-256 of `sample-bundle-v1` (`f99348aab553…`) before extraction, attached the adapter and printed `{'verdict': 'not-measurable', 'answers_differ': '2 of 2', 'findings': 1}`. Section 12 printed its off message. Evidence: `docs/verification/2026-10-25-colab-t4/language_model_finetuning_colab-c745a70-default/` |
| 2026-10-25 | `c745a70` / committed blob `90a46c1dd0c9`, executed as a run copy with one override, `RUN_DECODING_EXPERIMENT = False` → `True` (run copy blob `ad832897204e`, SHA-256 `3553c055da7e8df371c1b26c3d2d1edff226e887c61d8db7582fccbd728f4c51`; it differs from the committed blob only in that assignment literal) | Colab CLI `colab exec -f` (a scripted run, not a browser Run all), fresh Colab Tesla T4 | E2E default sample path, then the optional Section 12 decoding activity on the adapted model; every other setting at its default | 300.1 s (session wall) | **PASS, exit 0, sequential, input (the run copy) unchanged, 15/15 code cells in the executor log, 0 errors**, started 2026-10-25T13:18:14Z. Default path: validation loss base → adapted 3.5010 → 3.2618 (−0.2391 nats; train loss 3.2653; `wallSeconds` 53.3); the fresh reload reproduced (3.2651 vs 3.2618, \|Δ\| 0.0033 ≤ 0.05); Section 11 verdict `not-measurable`, `answers_differ` 2 of 2. Section 12 printed the greedy answer's longest repeated 4-word phrase "occurs 3x", the seeded sampled answers' "occurs 1x" (seed 42) and "occurs 3x" (seed 43), and `{'sampled_answers_identical': False}`. Evidence: `docs/verification/2026-10-25-colab-t4/language_model_finetuning_colab-c745a70-activity/` |
| 2026-10-23 | `6c22e80` / `4651e699c994` | Colab CLI `colab exec -f` (a scripted run, not a browser Run all), fresh Colab Tesla T4 | E2E default sample path of the **refusal-display revision** (PR #83), default settings only | 266.3 s (session wall) | **PASS, exit 0, sequential, input unchanged, 15/15 code cells in the executor log, 0 errors**, started 2026-10-23T09:06:59Z. Validation loss base → adapted 3.5010 → 3.2631 (−0.2378 nats per supervised token; perplexity 33.1 → 26.1); train loss 3.2653; 1.186 % trainable; 51.9 s loop; 0.66 GiB peak; evaluation verdict `sample-sanity`. The fresh reload reproduced (3.2648 vs in-memory 3.2631, \|Δ\| 0.0016 ≤ 0.05; greedy 16-token openings differ, reported). This run's own bundle printed `Artifact: outputs/language_model_finetuning_adapter_bundle.zip (21.2 MB) SHA-256 025d7a4f931e7a6bded10f08b3b79de4f564113ad349dfc6d476437b9195869b` (LoRA training is not bit-reproducible, so it differs from the pin). Section 11 checked the whole-ZIP SHA-256 of `sample-bundle-v1` (`f99348aab553…`) before extraction, matched the bundle base identity, attached the adapter on `cuda:0` and printed `{'verdict': 'not-measurable', 'answers_differ': '2 of 2', 'findings': 1}`. Section 12 printed its off message (`RUN_DECODING_EXPERIMENT is off`). Evidence: `docs/verification/2026-10-23-colab-t4/language_model_finetuning_colab-6c22e80-default/` (SHA-256 table below) |
| 2026-10-23 | `6c22e80` / committed blob `4651e699c994`, executed as a run copy with one override, `RUN_DECODING_EXPERIMENT = False` → `True` (run copy blob `c6b43c5a6b55`, SHA-256 `40519a855e48c0a27b45e2328f4249963e8630f9a96eada4148a0a7b7b37449d`; it differs from the committed blob only in that assignment literal) | Colab CLI `colab exec -f` (a scripted run, not a browser Run all), fresh Colab Tesla T4 | E2E default sample path, then the optional Section 12 decoding activity on the adapted model; every other setting at its default | 303.8 s (session wall) | **PASS, exit 0, sequential, input (the run copy) unchanged, 15/15 code cells in the executor log, 0 errors**, started 2026-10-23T09:11:39Z. Default path: validation loss base → adapted 3.5010 → 3.2631 (−0.2379 nats; train loss 3.2651); the fresh reload reproduced (3.2656 vs 3.2631, \|Δ\| 0.0026 ≤ 0.05); Section 11 verdict `not-measurable`, `answers_differ` 2 of 2. Section 12 printed the greedy answer's longest repeated 4-word phrase "occurs 3x", the seeded sampled answers' "occurs 1x" (seed 42) and "occurs 3x" (seed 43), and `{'sampled_answers_identical': False}`. Evidence: `docs/verification/2026-10-23-colab-t4/language_model_finetuning_colab-6c22e80-activity/` (SHA-256 table below) |

**Optional-activity run evidence (`edcb9e3`, 2026-10-22).** The run copy is committed beside the evidence as `run_copy.ipynb`. Local paths were replaced by `<local-run-dir>` before the first commit:

| File (under `docs/verification/2026-10-22-colab-t4/language_model_finetuning_colab-edcb9e3-activity/`) | Captured | Committed |
|---|---|---|
| `language_model_finetuning_colab_edcb9e3_colab-cli-t4_activity.ipynb` | `5732cdc2067cc80dadfbd396cc4b75c5dd16b45cb0616ca1ee7c3e48a60ff083` | unchanged (no local path) |
| `exec.log` | `abf2113badeba77762d0729501bea93cfeb22c96bdf05bbdf2c1203897d0256c` | `5183e603db92d2c4525aa4121a71a1da5fcdb3634310776395838afe4d340e00` (2 replacements) |
| `run_summary.json` | `8af92ee771ba5f40e211dcc5b5eda4dd81f797feba650f9aafcdaa86f92a19f3` | unchanged (no local path) |
| `run_copy.ipynb` | `db15d1a3ee28372406ea50939cfd7d0a0f647453f21149bc2e2bb00429220488` | unchanged (no local path) |

**Tier C wording revision run evidence (`4f3a907`, blob `d88be9f9`, 2026-10-25).** The activity run copy is committed beside its evidence as `run_copy.ipynb`. The executed notebooks and the run copy were checked against `run_summary.json` (`executed_sha256`, `run_copy_sha256`) before any edit. Local paths were replaced by `<local-run-dir>` before the first commit (byte-wise; no other byte changed):

| File (under `docs/verification/2026-10-25-colab-t4/`) | Captured | Committed |
|---|---|---|
| `language_model_finetuning_colab-4f3a907-default/language_model_finetuning_colab_4f3a907_colab-cli-t4_default.ipynb` | `60acc5c23b2c6c25e9b8fe93e1eefeec4bb178419a72c96724b33e011b6d1a7e` | unchanged (no local path) |
| `language_model_finetuning_colab-4f3a907-default/exec.log` | `31c3dff542c183437c81c9e69f2a2d7d14d48b49e5915a20ad6c8ec51f91e5b4` | `d76ec982b80ad46e2b9c18d577f3a63c425a487737d40bfc91ab07f14783ed98` (2 replacements) |
| `language_model_finetuning_colab-4f3a907-default/run_summary.json` | `9c78e22d13093247dc71a4f90292db61945d233665e9498339a2dceb63b8a7f7` | unchanged (no local path) |
| `language_model_finetuning_colab-4f3a907-activity/language_model_finetuning_colab_4f3a907_colab-cli-t4_activity.ipynb` | `5f80d289f33ebf22ebca39fe4caea9d6d2534824489c649090b0c1e814039f94` | unchanged (no local path) |
| `language_model_finetuning_colab-4f3a907-activity/exec.log` | `71c1eb4757e96bf6a005efb62bbc3bef514e2512e0900681105c22c35c79f447` | `8cc65565b2a3e0b95c5c989307ef2ca8849481947020f7eaa93bdaa87b54df46` (2 replacements) |
| `language_model_finetuning_colab-4f3a907-activity/run_summary.json` | `37eb0a975d732509ba89f939a5a882d68431f10cda3c3cebff84cb0847e95fce` | unchanged (no local path) |
| `language_model_finetuning_colab-4f3a907-activity/run_copy.ipynb` | `3ecea916c0bb53db29a7b6fa3021027528cb58a50567b262c13f74eb6060e7f5` | unchanged (no local path) |

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
| 2026-10-25 (18:12:52 UTC start) | `b64f689` / `0272ec08` (PR #84 head; historical wording revision; final confirming run; fetched at the full SHA and blob-verified before the session) | Colab CLI sequential execution, fresh Colab Tesla T4, via the workspace `colab-cli-serial-test-suite`. Not a browser Run all; the CLI records no execution counts, so order is evidenced by its `Executing cell k/N` log | Default sample path, no overrides; input unchanged; captured executed-notebook SHA-256 `3063076cf6f17241550d9633c4fcb7aa6ba231be15394533def592e2ec9ddc19` equals `run_summary.json` `executed_sha256` | 295.2 s session wall | **PASSED — one pass, no restart, 0 errors, 15/15 code cells.** Isolated environment, 71 locked packages, setup 59 s; Tesla T4, 4-bit, compute `bfloat16`. 4,341,760 of 366,162,880 parameters trainable (1.186 %). Validation loss base → adapted 3.5010 → 3.2633 (−0.2376 nats per supervised token; perplexity 33.1 → 26.1); train loss 3.2664; `wallSeconds` 54.4; peak 0.66 GiB. Fresh reload: 3.2679 vs 3.2633 (|Δ| 0.0046 ≤ 0.05), reproduced; greedy 16-token openings differ (near-tie under the dtype difference). Section 12 decoding activity off (default). Not exercised: BYOD (REL12), the custom-prompt and external-bundle journeys, a browser Run all (REL2). Final confirming run under the 2026-10-26 historical-wording ruling (drift recorded under that revision). Evidence `docs/verification/2026-10-25-colab-t4/language_model_finetuning_colab-b64f689-default/` — SHA-256 captured → committed: `language_model_finetuning_colab_b64f689_colab-cli-t4_default.ipynb` `3063076cf6f17241550d9633c4fcb7aa6ba231be15394533def592e2ec9ddc19` → unchanged (no local path); `exec.log` `050b9c971c2d0ec1c92ad68a933e011c46872de8074d7ac2965d5445aae47946` → `2eaf0edc88f22682be10f1da68d46bd9a619af8bbbfebf9d4ed7fdafc36c1f99` (2 local-path replacements); `run_summary.json` `802c63829f50b095b013b2b296b0b3e615e68036cab1c354a10612c97aef4ec8` → unchanged (no local path) |
| 2026-10-25 (18:12:52 UTC start) | `b64f689` / `0272ec08` (PR #84 head; historical wording revision; final confirming run; fetched at the full SHA and blob-verified before the session) | Colab CLI sequential execution, fresh Colab Tesla T4, via the workspace `colab-cli-serial-test-suite`. Not a browser Run all; the CLI records no execution counts, so order is evidenced by its `Executing cell k/N` log | Default path with one override (`RUN_DECODING_EXPERIMENT` False → True) applied to a run copy; the run-copy diff against the blob is that one line only; input unchanged; captured executed-notebook SHA-256 `46b0352cc9efd4c6c9ca8c7c0adbf1150e121133297bffe46da152c8c9f321c0` equals `run_summary.json` `executed_sha256` | 333.9 s session wall | **PASSED — one pass, no restart, 0 errors, 15/15 code cells.** Isolated environment, 71 locked packages, setup 57 s. Default path: validation loss 3.5010 → 3.2632 (−0.2377 nats; perplexity 33.1 → 26.1); train loss 3.2656; `wallSeconds` 55.8; peak 0.66 GiB; fresh reload 3.2657 vs 3.2632 (|Δ| 0.0025), reproduced. Section 12 decoding activity on the adapted model: longest repeated 4-word phrase greedy 2x, sampled (seed 42) 1x, sampled (seed 43) 3x. Not exercised: BYOD (REL12), the custom-prompt and external-bundle journeys, a browser Run all (REL2). Final confirming run under the 2026-10-26 historical-wording ruling (drift recorded under that revision). Evidence `docs/verification/2026-10-25-colab-t4/language_model_finetuning_colab-b64f689-activity/` — SHA-256 captured → committed: `language_model_finetuning_colab_b64f689_colab-cli-t4_activity.ipynb` `46b0352cc9efd4c6c9ca8c7c0adbf1150e121133297bffe46da152c8c9f321c0` → unchanged (no local path); `exec.log` `6c2369db6ece307412520aa36211bf8d6278bb56b8545340c27e4641903938a3` → `fa21358c16dbb81d636a89932633580f992c41d184052fbdf2120513e59e37b8` (2 local-path replacements); `run_summary.json` `47eb16cd54693698a1d448ad5f77749a1a22e201ce34a525ce7f998c367d4e01` → unchanged (no local path); `run_copy.ipynb` `d5c767ad2d82b6dd4b7a876a57f86baea81c45db9645bc8e092ba635a1217e88` → unchanged (no local path) |

Where this document quotes one of these files' earlier hash as its committed hash, the new hash is now
given beside it.


## SART7 deviation (2026-10-22)

- **Tag:** `sample-bundle-v1` (asset `language_model_finetuning_adapter_bundle.zip`).
- **Mutable release:** the GitHub release is not immutable (`gh api repos/kurtvalcorza/language-model-pipeline/releases/tags/sample-bundle-v1 --jq .immutable` returns `false`), so SART7 (SHOULD) is not met.
- **Pinned digest:** the notebook pins the asset by `SAMPLE_ARTIFACT_SHA256 = "f99348aab5537bbccc26a0283f0fd6e714c0e3b58d27beb8c3d2586951ffaa35"` (`src/lmpipeline/pipeline.py`, carried into the notebook); the GitHub asset digest is the same value.
- **Integrity:** integrity rests on that pinned SHA-256 check. `fetch_pinned_artifact` and `open_artifact_zip` hash the whole ZIP and refuse it before extraction if the digest differs, so a replaced asset is refused, not used.
- **Owner acceptance:** the owner accepted this deviation on 2026-10-22. The release is not re-tagged and the pin is not changed.

## Current status

**Current status (2026-10-26, final prose pass): Candidate, hosted run pending.** **Code commit `23ae24a`, notebook commit `fc30deb`.** The current blob of `tutorials/language_model_finetuning_colab.ipynb` is `eba2f1c07f0da28fe9485208436f0bb14b064220`, **Candidate**, with its hosted Colab T4 run pending. Kurt's 2026-10-26 ruling: fix only the prose rule breaks the final evidence flagged; recorded drift stays recorded, not fixed. Markdown only: the Section 12 answer no longer says sampled answers loop less and the Section 9 answer no longer predicts which prompts change. The runs of blob `0272ec08` and earlier blobs are history (REL14); the records below are kept.

**Historical wording revision (2026-10-26, code commit `3e42337`, regenerated in `65d71cb`).** Markdown only, code cells unchanged. Following the 2026-10-26 ruling that run-derived numbers in the notebook are records, never expectations: the how-to-use paragraph no longer carries a run-status sentence ("hosted runs of this revision are pending"; the run status lives here and in `STATUS.md`); the Section 8 comparison note now records all nine Colab T4 runs (adding the two `c745a70` runs of blob `90a46c1d`): train loss 3.2646–3.2665, validation loss 3.2604–3.2640, perplexity 26.1–26.2, 51.9–55.8 s of training (`wallSeconds`), 0.66 GiB peak, labelled as records of those runs, not targets; the Section 7 and Section 11.4 answers state the val-below-train pattern and the adapted register as what the named runs showed, which can vary between runs. The regenerated `E2E` notebook blob `0272ec08385f543e05bc8b07e652ff6a68cfed37` was **Candidate**; its default path and, on a run copy with `RUN_DECODING_EXPERIMENT = True`, its optional Section 12 decoding activity each ran in one pass with no restart and 0 errors on fresh Colab Tesla T4s on 2026-10-25 (commit `b64f689`; Colab CLI sequential execution, 15/15 code cells each, default 295.2 s and activity 333.9 s; validation loss 3.5010 → 3.2633 / 3.2632, reload reproduced; see their rows under "Recorded executions"). The paths this blob's runs cover: the default path and the Section 12 decoding activity. Remaining gates: REL12 (a hosted BYOD run, one input accepted and one refused) and REL2 (a browser run of the interactive journeys). The 2026-10-25 runs of blob `90a46c1d` (`c745a70`) and every earlier record below are kept as history (REL14). Remaining gates unchanged: REL12 and REL2.

Evidence files for the 2026-10-25 default run of blob `0272ec08` (final confirming run), copied from the run directory and redacted byte-wise (local run paths replaced by `<local-run-dir>`, the Colab CLI install path by `<colab-cli-tool-dir>`; no JSON re-serialised; the executed notebook's captured hash equals `run_summary.json` `executed_sha256`):

| File (under `docs/verification/2026-10-25-colab-t4/language_model_finetuning_colab-b64f689-default/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `language_model_finetuning_colab_b64f689_colab-cli-t4_default.ipynb` | `3063076cf6f17241550d9633c4fcb7aa6ba231be15394533def592e2ec9ddc19` | `3063076cf6f17241550d9633c4fcb7aa6ba231be15394533def592e2ec9ddc19` (unchanged, no local path) |
| `exec.log` | `050b9c971c2d0ec1c92ad68a933e011c46872de8074d7ac2965d5445aae47946` | `2eaf0edc88f22682be10f1da68d46bd9a619af8bbbfebf9d4ed7fdafc36c1f99` (2 local-path replacements) |
| `run_summary.json` | `802c63829f50b095b013b2b296b0b3e615e68036cab1c354a10612c97aef4ec8` | `802c63829f50b095b013b2b296b0b3e615e68036cab1c354a10612c97aef4ec8` (unchanged, no local path) |

Evidence files for the 2026-10-25 activity run of blob `0272ec08` (final confirming run), copied from the run directory and redacted byte-wise (local run paths replaced by `<local-run-dir>`, the Colab CLI install path by `<colab-cli-tool-dir>`; no JSON re-serialised; the executed notebook's captured hash equals `run_summary.json` `executed_sha256`):

| File (under `docs/verification/2026-10-25-colab-t4/language_model_finetuning_colab-b64f689-activity/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `language_model_finetuning_colab_b64f689_colab-cli-t4_activity.ipynb` | `46b0352cc9efd4c6c9ca8c7c0adbf1150e121133297bffe46da152c8c9f321c0` | `46b0352cc9efd4c6c9ca8c7c0adbf1150e121133297bffe46da152c8c9f321c0` (unchanged, no local path) |
| `exec.log` | `6c2369db6ece307412520aa36211bf8d6278bb56b8545340c27e4641903938a3` | `fa21358c16dbb81d636a89932633580f992c41d184052fbdf2120513e59e37b8` (2 local-path replacements) |
| `run_summary.json` | `47eb16cd54693698a1d448ad5f77749a1a22e201ce34a525ce7f998c367d4e01` | `47eb16cd54693698a1d448ad5f77749a1a22e201ce34a525ce7f998c367d4e01` (unchanged, no local path) |
| `run_copy.ipynb` | `d5c767ad2d82b6dd4b7a876a57f86baea81c45db9645bc8e092ba635a1217e88` | `d5c767ad2d82b6dd4b7a876a57f86baea81c45db9645bc8e092ba635a1217e88` (unchanged, no local path) |

**Final confirming run: drift record (2026-10-26).** Under the 2026-10-26 ruling ("historical wording, then stop") the notebook prose is not changed again; a value this run printed that differs from the values the prose attributes to the named earlier runs is recorded here as drift, not fixed. Checked against both runs' output: train loss 3.2664 / 3.2656, validation loss 3.2633 / 3.2632, perplexity 26.1, `wallSeconds` 54.4 / 55.8 and peak 0.66 GiB all lie inside the spans the prose records for the nine named Colab T4 runs (train 3.2646–3.2665, validation 3.2604–3.2640, perplexity 26.1–26.2, 51.9–55.8 s, 0.66 GiB); the trainable count 4,341,760 of 366,162,880 (1.186 %) and the `bfloat16` compute dtype equal the values quoted from the named 2026-10-23 run (the trainable share is a deterministic fact of the configuration).

Drift (this run versus the prose's named-run values):

- Section 12 answer (md33): the prose says the sampled answers "loop less" than the greedy one. In this activity run the longest repeated 4-word phrase occurred 2x in the greedy answer, 1x in the seed-42 sample and 3x in the seed-43 sample, so one sampled answer looped more than the greedy one. (Recorded here as drift per the ruling; the sentence is also listed below as a rule break.)

Prose statements that break the historical rules themselves (reported, not fixed here):

- md33 (Section 12 worked answer): "The greedy answer, if any: it always takes the most likely token, so once a phrase becomes the most likely continuation of itself it repeats. The sampled answers usually differ from each other … and loop less." A claim about which decoding loops more, stated as a general expectation with no named run (rule 2); this run's seed-43 sample looped more (3x) than greedy (2x).
- md21 (Section 9 worked answer), borderline: "Often, but not always: about a hundred training rows shift the *style* of answers, so expect some prompts to change register or language and others to look much like the base." A hedged expectation about the adapted answers rather than a record of the named runs (rule 2).

**Prose-mismatch revision (2026-10-25, code commit `fd68f6d`).** Markdown only, code cells unchanged: the Section 8 comparison note quoted only the 2026-09-14 Kaggle T4 run of `e9d7e6b` (train loss 3.266, validation loss 3.261, 55.5 s); it now gives the span of the seven recorded Colab T4 runs of this notebook (`d185817` 2026-10-07; `50f07ea` and `edcb9e3` 2026-10-22; `6c22e80` default and activity 2026-10-23; `4f3a907` default and activity 2026-10-25): train loss 3.2646–3.2657, validation loss 3.2604–3.2640, perplexity 26.1–26.2, about 52 to 56 s of training (`wallSeconds` 51.9–55.8), 0.66 GiB peak in every run, and keeps the Kaggle figures labelled as that earlier run. The regenerated `E2E` notebook blob `90a46c1dd0c97a90128233f1c16cc67dcea18c97` (PR #84 head `c745a70`) is **Candidate**. It has two recorded hosted runs, the two `c745a70` rows of the clean-runtime table, both Colab CLI `colab exec -f` (scripted, not a browser Run all) on a fresh Colab Tesla T4 on 2026-10-25, exit 0, 15/15 code cells, 0 errors, verdict PASS: the default path (249.6 s; validation loss 3.5010 → 3.2620, reload reproduced, Section 11 whole-ZIP SHA-256 checked) and the optional Section 12 decoding activity (300.1 s, `RUN_DECODING_EXPERIMENT = True` run copy). The paths this blob's runs cover: default and activity. Remaining gates: REL12 (a hosted BYOD run with one accepted and one refused input) and REL2 (a browser run of the interactive journeys). Under REL14 the 2026-10-25 runs of blob `d88be9f9` are history for that blob.

Evidence files for the 2026-10-25 default run of blob `90a46c1d`, copied from the run directory and redacted byte-wise (local run paths replaced by `<local-run-dir>`, the Colab CLI install path by `<colab-cli-tool-dir>`; no JSON re-serialised; the executed notebook's captured hash equals `run_summary.json` `executed_sha256`):

| File (under `docs/verification/2026-10-25-colab-t4/language_model_finetuning_colab-c745a70-default/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `language_model_finetuning_colab_c745a70_colab-cli-t4_default.ipynb` | `85c31ec06cdb39d2f28074d1570c417b01534c4e9af3a4bc21f7cc49ee8ff0b8` | `85c31ec06cdb39d2f28074d1570c417b01534c4e9af3a4bc21f7cc49ee8ff0b8` (unchanged, no local path) |
| `exec.log` | `73b0b33f75ce518316efdcc421f7bb82ad5651993073856ab5949a5c84cb76d3` | `3abaf9e3494df896d9d8be7a7b37de166da4a0cd2f2b3746757436edace2f97e` (2 local-path replacements) |
| `run_summary.json` | `ebe40a133cd652a8d90cad3a2f4da8e5916204dae8e89bd95ed1a73730828f44` | `ebe40a133cd652a8d90cad3a2f4da8e5916204dae8e89bd95ed1a73730828f44` (unchanged, no local path) |

Evidence files for the 2026-10-25 activity run of blob `90a46c1d`, copied from the run directory and redacted byte-wise (local run paths replaced by `<local-run-dir>`, the Colab CLI install path by `<colab-cli-tool-dir>`; no JSON re-serialised; the executed notebook's captured hash equals `run_summary.json` `executed_sha256`):

| File (under `docs/verification/2026-10-25-colab-t4/language_model_finetuning_colab-c745a70-activity/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `language_model_finetuning_colab_c745a70_colab-cli-t4_activity.ipynb` | `276e377f7bb1dba346e7d9ac362375aef8c377eafe53a886525f7f362a930b52` | `276e377f7bb1dba346e7d9ac362375aef8c377eafe53a886525f7f362a930b52` (unchanged, no local path) |
| `exec.log` | `67748fe3779e4d7f7078276e320b9cff544e705724c2ac27710fd6d1a4a6ee78` | `d837ed628329efb79a6a93e9cd3bd0185f572da359fe3e8d6c30e7617ebc11d9` (2 local-path replacements) |
| `run_summary.json` | `2e398578e014b321a8b0c2ec0d9a7c75489fb88ccd215889491b9bcbb2a4f477` | `2e398578e014b321a8b0c2ec0d9a7c75489fb88ccd215889491b9bcbb2a4f477` (unchanged, no local path) |
| `run_copy.ipynb` | `3553c055da7e8df371c1b26c3d2d1edff226e887c61d8db7582fccbd728f4c51` | `3553c055da7e8df371c1b26c3d2d1edff226e887c61d8db7582fccbd728f4c51` (unchanged, no local path) |

**Prose confirmation (2026-10-26).** This run is the confirming run for the "prose kept true across runs" revision. Every run-derived number in the notebook prose was checked against what this run printed: each number given as agreed across runs equals this run's printed value, and each stated range contains it. These still fail (the notebook is unchanged in this evidence commit):

- Section 8 comparison note (md17): the seven recorded Colab T4 runs "printed train loss 3.2646–3.2657". This default run printed train loss 3.2665, outside the range (the activity run's 3.2653 is inside; validation loss 3.2620 / 3.2618 is inside 3.2604–3.2640, perplexity 26.1, `wallSeconds` 53.1 / 53.3 inside "about 52 to 56 s", and 0.66 GiB peak hold).
- How-to-use paragraph (md1): "hosted runs of this revision are pending." These two runs are hosted runs of the current blob `90a46c1d`, so the sentence is now false.

- Holding in both runs: 120 rows split 96 / 24; the leaked-split probe rejected; compute dtype `bfloat16` on the T4; trainable 4,341,760 of 366,162,880 (1.186 %); validation loss below training loss after one epoch (3.2620 < 3.2665; 3.2618 < 3.2653); reload reproduced within 0.05 (|Δ| 0.0035 and 0.0033); Section 11 `r` 8, `lora_alpha` 16, `answers_differ` 2 of 2. Activity: greedy longest repeated 4-word phrase 3x, sampled 1x (seed 42) and 3x (seed 43), `sampled_answers_identical: False`, consistent with the hedged "usually differ ... and loop less".


**Tier C wording revision (2026-10-25, branch `fix/language-model-tier-c-2026-10-25`, code commit `bc6c04e`).** Markdown, header and metadata only for `tutorials/language_model_finetuning_colab.ipynb`: pretraining-overlap statement (DAT9), run-labelled numbers (REL13), use notice, IEEE references and the environment-variable notes. The regenerated notebook blob `d88be9f90a1c2c3805bac88ddc502b23e2297e85` is **Candidate**. It has two recorded hosted runs, the two `4f3a907` rows of the clean-runtime table, both Colab CLI `colab exec -f` (scripted, not a browser Run all) on a fresh Colab Tesla T4, exit 0, 15/15 code cells, 0 errors, verdict PASS: the default path (259.8 s; validation loss 3.5010 → 3.2604, reload reproduced, Section 11 whole-ZIP SHA-256 checked; evidence `docs/verification/2026-10-25-colab-t4/language_model_finetuning_colab-4f3a907-default/`) and the optional Section 12 decoding activity (297.1 s, `RUN_DECODING_EXPERIMENT = True` run copy; evidence `docs/verification/2026-10-25-colab-t4/language_model_finetuning_colab-4f3a907-activity/`). Remaining gates: REL12 (a hosted BYOD run with one accepted and one refused input) and REL2 (a browser run of the interactive journeys). The 2026-10-23 runs of `6c22e80` are evidence for blob `4651e699` only and are history (REL14).

**Status before the tier C revision (2026-10-23, history).** The E2E notebook blob before the tier C revision is `4651e699`
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