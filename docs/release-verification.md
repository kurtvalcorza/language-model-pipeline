# Release verification

`tutorials/language_model_finetuning_colab.ipynb` (`E2E`) and `tutorials/language_model_artifact_inference_colab.ipynb`
(`ARTIFACT-INFERENCE`) are **release candidates** until the exact notebook revisions have executed
top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell compilation, and
`tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence under DIMER
Notebook Specification 2.2. This file is the durable release-gate record for both notebooks.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- both notebooks parse; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs
  or execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory
  markdown cell;
- exactly the two tutorial notebooks, named in `tutorials/README.md` with their profiles, the notebook-spec
  version and the standalone carrier; `metadata.dimer` declares the profile, spec `2.2`, `standalone: true`
  and `generated_from` (repository, generating commit, module SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3) for each notebook: no clone, repository install or repository
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

CI also installs the light development pins (no torch), runs `ruff`, `tools/build_notebook.py --check` for
both templates, the offline unit suite (`tests/test_snapshot_helpers.py`, `tests/test_role_helpers.py`,
`tests/test_notebook_parity.py`, `tests/test_companion_parity.py`, `tests/test_colab_tutorials.py`; fake
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
6. open the exact companion notebook revision in a **separate** clean runtime. Default path: leave every field
   alone; Section 4 reads the trusted sample bundle pinned in `SAMPLE_ARTIFACT` (until that slot is filled,
   Run all stops there with the message naming it — record that as the outcome). Own-bundle path: set
   `USE_OWN_ARTIFACT = True` and `ARTIFACT_ZIP_PATH` to the ZIP from step 5 (Kaggle executor), or use the
   upload dialog (Colab), with `EXPECTED_ARTIFACT_ZIP_SHA256` set to the step-5 digest. Verify: whole-ZIP digest
   checked before extraction, bundle verified before load; adapter attached with the printed `lora_b_max_abs`;
   `validate_prompts` writes the input manifest with the blank-prompt finding; the adapter-activity readout
   (`next_token_logit_max_abs_delta`, `answers_differ`, verdict `active`); two seeded sampled answers;
   `evaluation_report` verdict `not-measurable` with `sample_kind: sample`; result JSON and generations CSV written;
7. verify the exports exist and the interpretation sections match the observed path;
8. record the notebook Git blob ids, commit, runtime (platform, Python, PyTorch, transformers, peft, GPU),
   model identifier and immutable revision, whether the model cache was clean, **whether a kernel restart or a
   `google.colab` shim was needed** (either one disqualifies the run as Run-all evidence), the sample-artifact
   SHA-256 for the companion, outcome, produced outputs, and any warning or applicable `SHOULD` deviation in
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
| 2026-10-07 | `d185817` / `8a2c6ad4886d` | Colab CLI 0.7.4 sequential execution (`colab exec -f`), fresh Colab Tesla T4 — not a browser Run all; no execution counts, order from `exec.log` | E2E default sample path (`language_model_finetuning_colab.ipynb`), default settings only | 241.5 s (session wall) | **one pass, no restart, 0 errors** — 12/12 code cells in order (cell `language_model_finetuning-08`, the carried module definitions, prints nothing). Python 3.12.12, torch 2.14.0+cu130, transformers 4.57.1, peft 0.18.0, 4-bit, `bfloat16` compute, 1.186 % trainable. Validation loss base → adapted 3.5010 → 3.2640 (−0.2370 nats; perplexity 33.1 → 26.2); train loss 3.2654; 55.8 s loop; 0.66 GiB peak. Reload: 3.2659 vs 3.2640, \|Δ\| 0.0019 ≤ 0.05 → reproduced; greedy 16-token openings differ by one token (`ng`/`ang`), reported. Bundle ZIP 22,231,859 bytes, SHA-256 `f99348aab5537bbccc26a0283f0fd6e714c0e3b58d27beb8c3d2586951ffaa35`. Evidence: `docs/verification/2026-10-07-colab-t4/language_model_finetuning_colab/` — executed notebook SHA-256 `f2535a6be4b74673ba080fc56d185265b560cf61a83f8ea11e6eee7d2ea92ac9`, `run_summary.json` `d1a45c35621a115d768fdb2bf7de50a1c317b186e33c4c8edba5efe5e236807a`, `exec.log` `55f372582297a1d0011c0e8bcfc10343e3ab262d8b3181af77acae9fe496f4b2`. Not exercised: Section 10 re-run, BYOD (REL12), the optional activity, forms and download dialogs. The previous attempt at `f2053aa` failed in Section 3 (`google.colab.__spec__ is None` in the isolated worker), fixed in `d185817` |
| | | | Companion path (`language_model_artifact_inference_colab.ipynb`) | | no run recorded at any revision |

## Current status

The E2E notebook has one recorded one-pass execution at blob `8a2c6ad4886d` (2026-10-07, Colab CLI sequential
execution on a fresh Colab Tesla T4, default path only; table above). It is **not** a browser Run all, and the
Section 10 re-run, BYOD (REL12) and the optional activity were not exercised. The earlier 2026-09-14 Kaggle T4 run
of an older E2E blob needed a kernel restart and a `google.colab` shim and does not satisfy RUN10/ENV6; the
companion has never been run. The
current revisions replace the in-kernel install with the isolated environment, add the base validation loss, the
reload reproduction check and the companion's sample-artifact slot (`SAMPLE_ARTIFACT`, still empty: it must be
filled from a recorded hosted run of the E2E notebook before the companion's default path can complete). Static
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
mirrors this status (no passing clean-run record for either current blob) and holds the per-run record format.