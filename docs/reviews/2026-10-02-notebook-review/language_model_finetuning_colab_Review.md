# Language-Model Fine-Tuning (QLoRA, E2E) Notebook — Review

**Verdict: Needs revision**
**Review date:** 3 October 2026 (relay batch 2026-10-02, row 38)
**Repository:** `kurtvalcorza/language-model-pipeline`
**Notebook:** `tutorials/language_model_finetuning_colab.ipynb`
**Reviewed commit:** `311b3c0cd4153c7300040d5880c12616f15dc9c5` (origin/main, confirmed via GitHub API)
**Notebook Git blob:** `2dbb503910594840ab6f7c5313f68647dfc68e8e`
**Finding prefix:** `LMF` (the companion `language_model_artifact_inference_colab` review in this folder uses `LMA`)

## Executive assessment

The notebook is a real, standalone QLoRA workflow. It carries the pipeline module byte-for-byte (the generator `--check` passes), pins the base snapshot by 13 per-file SHA-256 digests, rejects split leakage and over-length rows before training, masks the loss to assistant tokens using the model's own chat template, trains one bounded epoch in a plain PyTorch loop, exports a manifested PEFT adapter bundle and reloads it from disk. Its prose about what the loss and perplexity can and cannot show is unusually honest.

Four Major problems remain. (1) On the one hosted record, **Run all stopped at the install cell with a restart instruction**, and the run only passed after the executor restarted the kernel. The release record does not mention that restart. (2) The opening promises loss and perplexity "base vs adapted", but **no pre-adaptation loss is ever computed**, so a learner cannot tell whether the printed validation loss of 3.26 is any change at all. (3) **The reloaded adapter does not reproduce the in-memory adapted output.** In the recorded run the 16-token openings differ, yet the cell prints `PASS` and nothing compares them. (4) **The documented "change a knob and re-run from that cell down" instruction breaks after a full run.** The custom-prompt cell raises `NameError`, and a rerun records its "baseline" on a model that already carries the trained adapter.

Two things this review does **not** establish: that a Colab T4 runtime needs the same restart (the only hosted run was on Kaggle T4), and that the in-memory/reload divergence comes from the dtype mechanism described under LMF-M3 (that cause is inferred).

## 1. Review contract and evidence

| Item | Scope |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED`; declares NOTEBOOK_SPEC **2.0** (`metadata.dimer.notebook_spec`). Reviewed against the fleet spec **2.2 (2026-09-26)**, ml-worker origin/main `b1cfe13` |
| Intended learner | Knows Python functions/dicts/comprehensions and that an LM predicts the next token; transformer internals not required (Prerequisites cell) |
| Supported runtime | Colab T4 GPU "or another CUDA machine"; CUDA is mandatory (4-bit `bitsandbytes`) |
| Default task | QLoRA SFT of `HuggingFaceTB/SmolLM2-360M-Instruct@a10cc1512eab…` on 120 rows of `jpaulpoliquit/ph-sft-ai-authored-v1@8333699…` (96 train / 24 manufactured validation), 1 epoch, r=8, α=16, lr 2e-4, seed 42 |
| Promised outcomes | Pinned install; digest-verified snapshot; normalised and validated data with an input manifest; assistant-only masking; deterministic baseline; readable QLoRA loop; loss/perplexity "base vs adapted"; evaluation report; base/adapted answers on new prompts; manifested adapter bundle; fresh reload against the verified base |
| Optional paths | `USE_BYOD` upload (JSONL or ZIP); `Sample: Dolly`; `CUSTOM_PROMPT`; Next experiments (EPOCHS=3, LORA_RANK=16) |
| Generator | `tools/build_notebook.py` (build_notebook.py/2) + `tools/notebook_template.py`; carried module `src/lmpipeline/pipeline.py` sha256 `95f4b2b0…` (matches origin/main) |
| Relationship | Produces `outputs/language_model_finetuning_adapter_bundle.zip`, the input the companion notebook needs. LMA-B1 (companion has no automatic sample bundle) is fed by this notebook, but only through a browser download (LMF-m2) |

### Evidence actually obtained

**Source inspection:** all 23 cells, the carried module, the generator template, `tutorials/README.md`, `docs/release-verification.md`, `tutorials/RELEASE_VERIFICATION.md`.

**Documented execution evidence:** Kaggle T4, commit `e9d7e6b`, notebook blob `469911578d05`, kernel `dimer-nb2-language-model-finetuning` v2, 14 Sep 2026 (`.agent/backups/kaggle-pass-2026-09-14/out/.../v2/evidence`, sha256s in `source_manifest.json`). Pass 1 stopped at cell 3 with `RuntimeError: Core dependencies changed while older modules were loaded: cuda-bindings: loaded=12.9.4, installed=13.4.1. Restart the runtime, then rerun from the top.` The executor then restarted the kernel, and pass 2 ran 10/10 code cells, using a `google.colab` shim. The current blob `2dbb503` differs from `469911578d05` only in embedded revision strings and one wrapped line (`git diff e9d7e6b HEAD`: 14 lines changed in the notebook, 3 in the module). The code is semantically the same, but this is **not** the reviewed blob.

**Direct execution (CPU, this review):** Python 3.12 conda env `eo-notebook-test` (torch 2.13.0+cpu, transformers 4.57.6). `peft 0.18.0`, `accelerate 1.11.0`, `pyarrow` and `nbformat` were installed `--no-deps` into a scratch target directory, not into a shared environment. These are not the notebook's pins. Probes P01–P09 (`run_probes.py`) cover:
- nbformat validation and compilation;
- generator parity;
- AST checks of rerun state;
- the **real pinned tokenizer** (5 files, sha256 verified against the manifest) with the **real pinned sample parquet**: the cell 9/11 data path reproduced `{'train': 96, 'validation': 24}`, digest `b744d708c47877e0`, 21 714/5 634 tokens and 14 826/3 840 supervised tokens, all identical to the Kaggle record;
- the BYOD reader on synthetic files;
- a **stand-in** PEFT state probe (tiny random bf16 Llama, not SmolLM2, not 4-bit).

**Not verified:** any Colab run; any GPU training, 4-bit load, generation, export or reload by this review; the BYOD branch end to end; `Sample: Dolly`; learner understanding.

## 2. Separate judgments

| Dimension | Assessment |
|---|---|
| Technical correctness | The data, validation and masking path is correct and reproducible (P07 matches the hosted record exactly). The install path needed a restart on its only hosted run (M1). Rerun state is unsafe (M4). The reload check verifies that the adapter is present and active, but not that it reproduces the in-memory model (M3). |
| Promise fulfillment | Most stages are delivered. The "loss/perplexity base vs adapted" promise is not (M2). The "prove the bundle is usable" claim is narrower than it reads (M3). "Another CUDA machine" ends in `ModuleNotFoundError` (m2). |
| Scientific/experimental validity | Leakage rejection, assistant-only masking and honest optimisation-vs-quality framing are strong. Missing base-loss baseline (M2). Reported metrics and answers come from a different numeric configuration than the exported artifact (M3). Rerun baselines are contaminated (M4). "Keep the lowest epoch" is not implemented (m3). |
| Learner orientation | Clear prerequisites and an excellent masking/LoRA explanation. Several "what to expect" notes contradict the pinned model's recorded output (m1). |
| Explanations / interpretation | The interpretation section is accurate about limits. The recorded adapted output is a repetition loop ("kung ano ang kultura" ×10), and the stated expectations do not prepare the learner for that. |
| Meaningful activity | Next experiments exist only as prose. The one interactive form (`CUSTOM_PROMPT`) fails after Run all (M4). No predict/checkpoint activities (m6). |
| Interaction / recovery | Failure messages for leakage and over-length rows are actionable. BYOD JSON and schema errors do not name the file or row (m4). No troubleshooting section (m6). |
| Completion / transfer | Machine-readable result JSON, probes CSV, input manifest, evaluation report and manifested bundle are written. The bundle reaches the companion only through a Colab browser download (m2). |
| Spec conformance (2.2) | Unmet MUSTs: **RUN10/ENV6/RUN1** (restart, documented on Kaggle), **REL10** (no record for the current blob), **EVAL8 + VER5** (evaluation config ≠ reload config; loading not distinguished from reproducing), **DAT19** (BYOD errors partly non-actionable), **SRC3** (knowingly stale instructions: m1, m5). SHOULD gaps: EVAL10, VER4, EXE2, SPL2/SPL3, GDL4/7/9/11/13/14, UX5. |

## 3. Prioritized findings

### LMF-M1 — Major: Run all stops at the install cell and needs a manual restart on the recorded hosted run

**Cell/section:** Section 1 install cell (cell 3); generator `tools/notebook_template.py` / `tools/build_notebook.py` install block.

**Observed issue:** The cell pip-installs 13 pins (including `torch==2.14.0` cu130, `pandas==3.0.5`) into the live kernel. On Kaggle T4 the install replaced a distribution that was already loaded (`cuda-bindings 12.9.4 → 13.4.1`). The stale-import guard then raised its restart instruction, as designed. The run passed only after a kernel restart (`run_summary.json`: `restarted_after_install_cell: true`, pass 1 `ok: false` 188.8 s, pass 2 `ok: true`). `docs/release-verification.md` records this run as "**PASSED** — 10/10 ok code cells executed cleanly" without the restart or the `google.colab` shim.

**Consequence:** A first-time learner's Run all ends at cell 3 with an error. The spec says a notebook that needs a second manual execution after installation is not Run-all conformant (§5, RUN10, ENV6). The release record overstates the evidence. Pip also reported conflicts with the preinstalled image (`google-colab requires pandas==2.2.2`, `cudf-cu12 requires pandas<2.4`), which is the same class of conflict a Colab image is likely to hit.

**Evidence:** Documented execution evidence (Kaggle T4, `e9d7e6b`/`469911578d05`; P06). Colab behaviour: **not verified**.

**Recommended correction:** Use the fleet's uv isolated-environment pattern instead of installing into the kernel. A carrier cell bootstraps uv, creates `uv venv --managed-python --python 3.12.12 <ROOT>/env`, installs a hash-locked `requirements.txt` with `uv pip install --require-hashes --only-binary :all:`, and runs the workload in that environment. The kernel's preloaded NumPy/torch/CUDA bindings are then never replaced. Reference: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` (origin/main). Apply it in the generator, not in the cell. Also correct the release-verification row to state the restart and the shim.

**Acceptance check:** On a fresh Colab T4 runtime and a fresh Kaggle T4 kernel, Run all with default fields completes every code cell in one pass, with no restart and no error output. The release record names the exact blob, states "no restart", and states whether a shim was used.

### LMF-M2 — Major: The promised base-vs-adapted loss/perplexity comparison is never computed

**Cell/section:** Opening Run-all paragraph (template line 44: "evaluates loss/perplexity base vs adapted"); Section 6 baseline (cell 13); Section 7 loop (cell 15); Section 8 report (cell 17).

**Observed issue:** `pipe.evaluate_loss` is called only after training, on the adapted model (validation and test; P05). Nothing scores the base model on the same validation split. `evaluation_report` always emits `"baselines": []`, and its own `needs` field asks for "a pre-adaptation baseline scored the same way". The only pre-adaptation evidence is two greedy answers.

**Consequence:** The central quantitative result, validation loss 3.261 / perplexity 26.1 (Kaggle record), cannot be interpreted. A learner cannot tell whether one epoch lowered the loss, raised it, or did nothing, so the loss half of §21.7 "compare adapted and base behavior" is missing. The header misdescribes what the notebook measures.

**Evidence:** Source inspection (P05); documented run shows `baselines: []`.

**Recommended correction:** In Section 6, compute `BASE_VALIDATION_LOSS = pipe.evaluate_loss(MASKED['validation'])` (with `MASKED` from cell 11) before LoRA is attached. Record it in `METRICS` under a distinct id (e.g. `baseValidationLoss` / `baseValidationPerplexity`) and pass it to `evaluation_report` as a baseline. Print a base → adapted line, and add a "what to notice" note that does not hard-code the direction. Make the change in `tools/notebook_template.py` (cells 13/15) and `evaluation_report` in `src/lmpipeline/pipeline.py`.

**Acceptance check:** The executed notebook prints base and adapted validation loss on the same 24 rows. The evaluation report JSON lists a base-model baseline entry with the same units and estimation text. The opening sentence matches what is computed.

### LMF-M3 — Major: The reload check prints PASS although the reloaded adapter does not reproduce the in-memory adapted output

**Cell/section:** Section 10 (cell 21), template lines 436–449; Section 7 (cell 15) `prepare_model_for_kbit_training`.

**Observed issue:** Cell 21 computes `expected_opening` from the in-memory model, then reloads base + adapter and checks only three things: LoRA-B is non-zero, adapter-on ≠ adapter-off, and output is non-empty. It never compares `reloaded_opening` with `expected_opening`. In the recorded run they differ:

- in-memory: `Kung ano ang machine learning, ipakita ang kultura,`
- reloaded: `Ipaliwanag ang machine learning na hindi nang maging k`

The cell still printed `PASS: fresh base + adapter reload ... adapter weights present and active` and wrote both strings to `result.json` without comment. A likely cause (**inferred**): `prepare_model_for_kbit_training` casts every non-4-bit bf16/fp16 parameter to fp32 (peft 0.18.0 `utils/other.py` 170–175). That includes the embeddings, norms and the tied LM head. The training-time model, its validation loss and its adapted answers therefore ran with fp32 non-quantized layers. The reloaded artifact runs with the bf16 base that `reload_base` loads. The stand-in probe confirms the dtype change (bf16 → fp32) and a non-zero logit difference between the in-memory and reloaded models (P09: max |Δlogit| 0.0089 on a tiny model). Greedy decoding turns small differences into different text.

**Consequence:** The notebook teaches that the export is "proven usable from disk". It does not show that the artifact a downstream consumer loads reproduces the reported metrics and answers. Evaluation and inference use different effective model configurations, and loading is not distinguished from reproducing (EVAL8, VER5 MUST; VER4 SHOULD). A learner who notices the different openings has no explanation.

**Evidence:** Documented execution evidence (Kaggle `result.json` `reload` block; P06). The mechanism comes from source inspection plus a CPU stand-in (P09). It was not verified on SmolLM2/nf4.

**Recommended correction:** Make evaluation and reload use the same numeric configuration. Either re-evaluate the reloaded artifact (validation loss plus the probe answers) and report those as the artifact's numbers, or restore the base dtype before evaluation and export. Then compare in-memory and reloaded outputs explicitly: exact match for greedy tokens, or a stated logit/loss tolerance. Say which quantity the reload check reproduces. Change it in `tools/notebook_template.py` around lines 436–449, and in `export_adapter_bundle`/`load_adapter` if the bundle should carry dtype provenance.

**Acceptance check:** A fresh GPU run shows a reload check that either (a) reports reloaded validation loss within a stated tolerance of the reported `validationLoss` and identical greedy openings, or (b) fails loudly when they differ. The `PASS` text names exactly what was reproduced.

### LMF-M4 — Major: "Change the knobs and re-run from that cell down" leaves the notebook in an invalid state after a full run

**Cell/section:** Prerequisites "How to use this notebook" (cell 1, template line 97); Section 6 baseline (cell 13); Section 7 (cell 15); Section 9 `CUSTOM_PROMPT` (cell 19); Section 10 (cell 21); Next experiments (cell 22).

**Observed issue:**
1. Cell 21 runs `del model` (P03). Editing the `CUSTOM_PROMPT` form in cell 19 after Run all and re-running that cell, as the instruction says, raises `NameError: name 'model' is not defined` at `model.disable_adapter()`.
2. Cell 21 also sets `pipe.model = pipe.reload_base()`, and then `pipe.load_adapter` calls `PeftModel.from_pretrained(self.model, …)`, which injects the trained LoRA layers **into `pipe.model` in place**. When a learner follows Next experiments (`EPOCHS = 3`, `LORA_RANK = 16`) and re-runs from cell 9 down, cell 13's "deterministic baseline" calls `pipe.generate(prompt)` on that adapted model. Cell 15 then wraps an already-wrapped model; PEFT warns "You are trying to modify a model with PEFT for a second time…". The stand-in reproduces this: the re-run baseline equals the adapter-on output exactly, and the true baseline differs by 1.36 in max |logit| (P09).

**Consequence:** The single typed-input activity fails at the moment a learner would use it. Every suggested experiment produces a before/after table whose "before" is the previously trained adapter, which invalidates the comparison the experiment exists to show. The notebook gives no warning (framework dimension 7; GDL10, UX7, SRC2).

**Evidence:** Source inspection (P03) and direct CPU stand-in execution (P09). Not executed on GPU with SmolLM2.

**Recommended correction:** Keep a separate `reloaded` handle and do not `del model` until the end, or make cell 19 use whichever adapted model is current. Load the reload base into a fresh object (`load_adapter(..., base_model=pipe.reload_base())`) so `pipe.model` stays the untouched base. At the top of cell 13, re-create or verify a clean base (assert no `BaseTunerLayer` in `pipe.model`), or tell the learner to restart from Section 3 for a new experiment. Fix it in `tools/notebook_template.py` and `load_adapter`.

**Acceptance check:**
- After Run all, setting `CUSTOM_PROMPT` and re-running cell 19 prints base/adapted answers.
- After Run all, setting `LORA_RANK = 16` and re-running from cell 9 records baseline answers identical to the first run's baseline, with no PEFT "second time" warning, and the trainable-parameter count reflects r=16.

### LMF-m1 — Minor: "What to expect" notes contradict the pinned model's recorded run

**Cell/section:** Opening (template line 66), Sections 4, 6 and 7 markdown (lines 105, 244, 247, 271, 278).

**Observed issue:**
- "a fraction of one percent" / "Expect **under 1 %** … trainable": the recorded figure is **1.186 %** (4 341 760 / 366 162 880).
- "float16 on a T4": the recorded run reports `compute_dtype: 'bfloat16'` on Tesla T4.
- "expect English or mixed-language replies": the recorded base model **echoed the Filipino prompt**.
- "non-thinking mode" is a Qwen3 concept; SmolLM2 has no thinking mode.
- Section 7 gives Qwen3-0.6B reference numbers (loss ≈ 3.22, 1.52 GiB). A SmolLM2 measurement exists: train 3.266 / val 3.261 / ppl 26.1, 55.5 s, 0.66 GiB peak.

**Consequence:** The learner's interpretation checkpoints contradict what they see. This is knowingly stale text left after the base-model switch (SRC3, GDL8, UX4).

**Evidence:** Documented execution evidence (P06) and source inspection.

**Recommended correction:** Regenerate the template prose from the SmolLM2 record: ~1.2 % trainable; bf16 is selected whenever `torch.cuda.is_bf16_supported()` is true, including emulated on T4; base answers may echo the prompt; drop "non-thinking"; replace the Qwen3 figures with the measured SmolLM2 run and its environment.

**Acceptance check:** Each expected-result note in Sections 0/4/6/7 is consistent with a fresh run's printed values.

### LMF-m2 — Minor: An unconditional `google.colab` import ends Run all with an error outside Colab

**Cell/section:** Cell 21, last two lines (template 489–490).

**Observed issue:** `from google.colab import files; files.download(str(ARTIFACT_ZIP))` runs on every path (P04). The Prerequisites name "another CUDA machine" as a supported runtime, but on any non-Colab Jupyter it raises `ModuleNotFoundError` as the final statement. The outputs are already written by then. The recorded Kaggle run only passed because of an injected shim.

**Consequence:** Run all on a supported non-Colab runtime ends in an error. The bundle that the companion needs (see LMA-B1) is delivered only by a browser download.

**Evidence:** Source inspection (P04) and documented evidence (shim noted in `docs/release-verification.md`).

**Recommended correction:** Guard the import (`try: from google.colab import files` … `except ImportError: print(path)`). Print the bundle path and SHA-256 as the hand-off to the companion.

**Acceptance check:** On a non-Colab CUDA Jupyter kernel, Run all ends without error and prints the ZIP path and digest. On Colab the download is still offered.

### LMF-m3 — Minor: "Keep the lowest-validation-loss epoch" is advice the notebook cannot follow

**Cell/section:** Section 8 markdown and Next experiments (template line 511); cell 15 loop.

**Observed issue:** Validation loss is printed per epoch, but `METRICS` keeps only the last epoch's values and the exported adapter is the last epoch's. Nothing checkpoints or selects. If a learner did select by validation loss, the reported validation loss would become a selection metric on the same manufactured split, with no independent test (SPL6, EVAL14, ART8). The claim that validation loss "usually … turns up around epoch 3–4" is unverified for SmolLM2.

**Evidence:** Source inspection.

**Recommended correction:** Either record per-epoch losses plus a best-epoch adapter (and the selection basis in provenance), noting that the selected-epoch validation loss is then no longer an unbiased estimate, or reword the advice as something to do outside this notebook.

**Acceptance check:** With `EPOCHS = 3`, `result.json` lists the loss for every epoch. The exported bundle's provenance names the selected epoch and the selection split, or the prose no longer promises selection.

### LMF-m4 — Minor: BYOD path gaps (no location field, non-actionable parse errors, unshuffled head split)

**Cell/section:** Cell 9 BYOD branch (template lines 142–166).

**Observed issue (P08):**
- There is no BYOD location field, so BYOD works only through the Colab upload widget (EXE2).
- A malformed line raises `JSONDecodeError: … line 1 column 17` with no file name or line number. Unknown schemas raise `Unsupported SFT schema …` with no file or row (DAT19).
- When `validation.jsonl` is absent, validation is the **first fifth in file order**, not shuffled, and the markdown does not say so. A topic-sorted file yields a single-topic validation set (SPL3).
- An over-length BYOD row rejects the whole dataset, while the sample path quietly sets such rows aside. The difference is not explained (VAL7).
- The BYOD branch has never been executed on a hosted runtime (REL12).

**Evidence:** Direct execution of the cell 9 reader logic on synthetic files (CPU); source inspection.

**Recommended correction:**
- Add `BYOD_PATH = ''  # @param`.
- Wrap per-line parsing so errors name `file:line` and the failing rule.
- Shuffle with `SEED` (or hash-order as in the sample path) before manufacturing validation, and say so.
- State the over-length policy for BYOD.
- Record one hosted BYOD run, including one rejected input.

**Acceptance check:**
- A ZIP with a malformed line 5 is rejected with a message naming `train.jsonl:5`.
- A topic-sorted file yields a mixed validation set.
- A hosted BYOD run reaches train → evaluate → export.

### LMF-m5 — Minor: Execution record and release docs do not describe the current blob consistently

**Cell/section:** `docs/release-verification.md` (Recorded executions; Current status); `tutorials/RELEASE_VERIFICATION.md`; `tutorials/README.md`; notebook metadata.

**Observed issue:**
- The only record covers blob `469911578d05` at `e9d7e6b`, not the current blob `2dbb503` (REL10).
- "Current status" begins "No clean-runtime execution of either standalone notebook has been recorded yet; clean GPU execution evidence for the E2E path is now recorded below" in one sentence.
- `tutorials/RELEASE_VERIFICATION.md` says "No passing clean-run record" for this notebook and still describes passing a credential through a Git HTTP header (pre-standalone text).
- `tutorials/README.md` says "verified".
- The notebook declares spec 2.0, and the release doc's static-check section cites spec 1.1.

This overlaps LMA-m7 from the companion review.

**Evidence:** Source inspection.

**Recommended correction:** Re-run the current blob on a fresh hosted runtime and record it with blob, restart status and shim status. Reconcile the three documents to one status. Regenerate with the current spec version once LMF-M1–M4 are fixed.

**Acceptance check:** All three documents state the same status for the same blob, and no sentence contradicts another.

### LMF-m6 — Minor: The GUIDED layer is thin for spec 2.2

**Cell/section:** Whole notebook.

**Observed issue:**
- No Input → Model → Output contract (GDL4).
- No prediction prompts before the baseline/after-training comparison (GDL7).
- No interpretation checkpoints with sample answers (GDL9).
- The 953-line carried module cell is not labelled Infrastructure or "safe to run without studying" (GDL11).
- No troubleshooting section for the failures this notebook actually hits: install restart, CUDA missing, bitsandbytes CUDA libs, HF download, OOM, BYOD errors (GDL13).
- No evidence-based conclusion template (GDL14).
- The optional experiments exist only as prose, with no runnable switch (UX5).

**Evidence:** Source inspection.

**Recommended correction:** Add these in the template, with priority on a troubleshooting table and one runnable Predict → Change → Run → Observe → Explain activity, which depends on LMF-M4 being fixed.

**Acceptance check:** Each GDL item above can be pointed to a cell, and the activity runs to completion after Run all.

### Suggestions

- **LMF-S1** — Check for CUDA before Section 3 stages and loads the 727 MB snapshot. Today the "clear message" (RUN11) arrives at cell 9, after the download and a CPU float32 model load.
- **LMF-S2** — The pinned sample publishes its own `validation`/`test` parquet splits plus `contamination_verdict`/`judge_score` columns. Only `train` is used and validation is manufactured (SPL2). Consider using the published splits, or say why not. P07 found 2 identical first-user-line templates shared across train and validation (e.g. "Ipaliwanag ang sagot nang hakbang-hakbang."). Exact-duplicate leakage is checked; near-duplicate templating is not mentioned.
- **LMF-S3** — The SmolLM2 chat template injects its default system prompt ("You are a helpful AI assistant named SmolLM…") into every training render, visible in the `⟦ ⟧` print. Say so, and say what happens with BYOD rows that carry their own system turn.
- **LMF-S4** — Relationship to LMA-B1: this notebook is the natural producer of a pinned, published sample bundle for the companion's default path. Consider publishing one recorded bundle (with its ZIP SHA-256) from a verified run.

## 4. Promise and objective trace

| Claim / objective | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| Pinned, digest-verified base snapshot | cell 7 `stage_missing_files` + `verify_snapshot` | 13 files fetched/verified (Kaggle) | identity printed | Delivered |
| Normalise 3 schemas → chat; validate → input manifest | cells 9, 11 | 96/24, digest `b744d708…`, leak probe rejected (Kaggle and P07) | clear | Delivered |
| Assistant-only loss masking | cell 11 `prepare_splits`, `show_supervision` | `⟦answer<|im_end|>⟧` only (P07) | excellent prose | Delivered |
| Deterministic pre-adaptation baseline | cell 13 | 2 greedy answers | expected-result note wrong (m1); contaminated on rerun (M4) | Partly delivered |
| Readable QLoRA loop | cell 15 | 1 epoch, losses printed | trainable % claim wrong (m1) | Delivered |
| Loss/perplexity base vs adapted | none | only adapted | cannot interpret 3.26 | **Not delivered (M2)** |
| Evaluation report | cell 17 | `sample-sanity` | honest | Delivered |
| Base vs adapted on new prompts | cell 19 | side-by-side answers | `CUSTOM_PROMPT` fails after Run all (M4) | Delivered on first pass only |
| Manifested adapter bundle | cell 21 export | 12 files, 22.2 MB, zip SHA printed | clear | Delivered |
| Fresh reload proves usability | cell 21 reload | PASS, but openings differ | unexplained | **Overstated (M3)** |
| Run all in a fresh runtime | whole notebook | needed 1 restart (Kaggle) | stops at cell 3 | **Not met (M1)** |
| BYOD reaches train→evaluate→export | cell 9 branch | not executed hosted | parse errors weak | Not verified (m4) |

## 5. Journeys

| Journey | Evidence basis | Outcome |
|---|---|---|
| First-time learner | Source inspection + documented run | Strong concept prose. Run all stops at cell 3 (M1, Kaggle). Expectations contradict output (m1). Thin guided layer (m6). |
| Clean default | Documented execution evidence (Kaggle T4, older blob) + direct CPU execution of the data path | PASS after one restart and with a shim. CPU data/masking path reproduces the hosted record exactly. GPU stages not run by this review; Colab not verified. |
| Active learning | Source inspection + CPU stand-in | `CUSTOM_PROMPT` rerun → `NameError`. Knob rerun → contaminated baseline and double PEFT wrap (M4). Lowest-epoch advice unimplementable (m3). |
| Reuse and recovery | Direct CPU execution of the BYOD reader + documented reload | Leakage and over-length rejections are actionable. JSON and schema errors lack file/row (m4). Reload diverges from in-memory output (M3). Hosted BYOD not verified. |

## 6. Readiness

**Needs revision.** Remaining gates:
- fix LMF-M1–M4;
- record one fresh-runtime Run all of the fixed blob with no restart, on Colab or Kaggle with the record stating which;
- record one hosted BYOD run (REL12);
- reconcile the release documents (m5).

## 7. Verified vs inferred

- **Verified by direct execution:** nbformat validity, all 10 code cells compile, no outputs persisted; generator `--check` OK and module sha256 matches; sample data path (splits, digest, token and supervised counts) identical to the hosted record; leak-probe message; BYOD reader messages; PEFT in-place injection, double-wrap warning and the fp32 upcast on a stand-in model.
- **Verified from documented evidence:** restart on pass 1; trainable 1.186 %; bf16 on T4; prompt-echo baseline; in-memory vs reloaded opening mismatch; metrics.
- **Inferred:** that Colab needs the same restart; that the fp32 upcast causes the reload divergence; that a SmolLM2 rerun reproduces the stand-in contamination exactly.
- **Only Kurt or a hosted run can confirm:** Colab behaviour of the install cell, and whether fixed reload numbers match.
- **Most likely to be wrong:** the M3 mechanism. The divergence is documented; its cause is not. bitsandbytes kernel nondeterminism or a tokenizer reload difference could also explain it, but the missing comparison stands either way.

## Probes

`language_model_finetuning_colab_Review_Probes.zip` contains `run_probes.py` (P01–P09), `results.json` and `source_manifest.json` (sha256 of every inspected source and of the Kaggle evidence files).
