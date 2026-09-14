# Release verification

`tutorials/language_model_finetuning_colab.ipynb` (`E2E`) and `tutorials/language_model_artifact_inference_colab.ipynb`
(`ARTIFACT-INFERENCE`) are **release candidates** until the exact notebook revisions have executed
top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell compilation, and
`tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence under DIMER
Notebook Specification 1.1. This file is the durable release-gate record for both notebooks.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- both notebooks parse; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs
  or execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory
  markdown cell;
- exactly the two tutorial notebooks, named in `tutorials/README.md` with their profiles, the notebook-spec
  version and the standalone carrier; `metadata.dimer` declares the profile, spec `1.1`, `standalone: true`
  and `generated_from` (repository, generating commit, module SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3) for each notebook: no clone, repository install or repository
  import on the primary path; exactly one cell tagged `embedded_module` equal to `src/lmpipeline/pipeline.py`
  after the generator's documented rewrite; the inline `MANIFEST` equal to
  `weights/smollm2-360m/dimer-base-manifest.json` and the inline `PINS` equal to the `pyproject.toml` runtime
  pins; the notebook byte-identical to `tools/build_notebook.py` output for its template; the pinned-install
  cell with its restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cell (and repeated in the inline manifest,
  which the notebook asserts against the module before fetching), the revision is a 40-hex immutable commit,
  and the same identity string appears in `README.md`, `weights/smollm2-360m/MODEL_CARD.md` and
  `weights/README.md` with no stray revisions beyond the other registry entries and the two pinned sample
  datasets;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `LanguageModelPipeline.from_pretrained(weights_dir=...)`, `validate_inputs`, `prepare_splits`, `generate`,
  `evaluate_loss`, `evaluation_report`, `export_adapter_bundle`, `reload_base`, `load_adapter`,
  `validate_prompts`, `verify_artifact_bundle`, `extract_zip_safely`), the ceiling prints, the exports, the
  learner-facing statements (what is trained and what is not, assistant-only masking, manufactured validation
  split, optimisation metrics versus task quality, adapters incomplete without their base, trust boundary,
  no network fallback) and the gated-off BYOD default; forbidden patterns (credential-in-URL, any
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
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the commit recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`) — `bitsandbytes==0.49.0`, `peft==0.18.0`, `transformers==4.57.1` and `torch==2.14.0`
   on the CUDA runtime are the pins most likely to need a wheel-availability check;
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access and no token;
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
   - two greedy baseline answers; LoRA attached with under 1 % trainable parameters; one epoch with train
     and validation loss printed; adapted answers and `METRICS` (`trainLoss`, `validationLoss`,
     `validationPerplexity`, `wallSeconds`, `peakGpuMemoryBytes`);
   - `evaluation_report` writes `outputs/language_model_finetuning_evaluation_report.json` with verdict
     `sample-sanity`, stated as optimisation evidence only;
   - new-prompt answers with the adapter off and on; `export_adapter_bundle` writes the bundle and its
     `artifact-manifest.json`; the fresh reload passes all three checks; `outputs/language_model_finetuning_result.json`,
     `outputs/language_model_finetuning_probes.csv` and `outputs/language_model_finetuning_adapter_bundle.zip`
     written with `NOTEBOOK_SOURCE`, model revision, model licence, runtime versions and device;
6. open the exact companion notebook revision in a **separate** clean runtime, supply the bundle produced
   in step 5 (upload, or `ARTIFACT_DIR` for the Kaggle executor), leave the prompts at their defaults, and
   verify: bundle verified before load; adapter attached with non-zero `B` matrices; `validate_prompts`
   writes the input manifest with the blank-prompt finding; greedy adapter-off/on answers differ; two
   sampled answers; `evaluation_report` verdict `not-measurable`; result JSON and generations CSV written;
7. verify the exports exist and the interpretation sections match the observed path;
8. record the notebook Git blob ids, commit, runtime (platform, Python, PyTorch, transformers, peft, GPU),
   model identifier and immutable revision, whether the model cache was clean, outcome, produced outputs,
   and any warning or applicable `SHOULD` deviation in the table below;
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
| 2026-09-14 | `e9d7e6b` / `469911578d05` | Kaggle T4 (`kurtvalcorza/dimer-nb2-language-model-finetuning` v2) | E2E default sample path (`language_model_finetuning_colab.ipynb`) | 360.9 s | **PASSED** — 10/10 ok code cells executed cleanly, 28 files, 727 MB staged |
| | | | Companion path fed by the E2E bundle (`language_model_artifact_inference_colab.ipynb`) | | pending — queued to the GPU lane |

## Current status

No clean-runtime execution of either standalone notebook has been recorded yet; clean GPU execution evidence for the E2E path is now recorded below. Static validation (`tools/validate_release_assets.py`), nbformat validation, a
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
`torch==2.14.0` / `bitsandbytes==0.49.0` pin set on a T4.