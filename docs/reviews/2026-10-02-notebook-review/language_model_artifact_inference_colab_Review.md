# Language-model adapter artifact-inference tutorial — Review

**Verdict: Needs revision**  
**Review date:** 3 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/language-model-pipeline`  
**Notebook:** `tutorials/language_model_artifact_inference_colab.ipynb`  
**Reviewed commit:** `311b3c0cd4153c7300040d5880c12616f15dc9c5` (`main`, merge of PR #75)  
**Notebook Git blob:** `41b88e8015411ba4129c9d5524b7d0813b565509`  
**Finding prefix:** `LMA`  
**Framework:** Notebook Review Framework v1; requirements baseline DIMER Notebook Specification **2.2** (2026-09-26, `ml-worker` `origin/main` `b1cfe13`)

## Executive assessment

The verification core of this notebook is solid. The carried module matches the repository
(`tools/build_notebook.py --check` OK; offline suite 49 passed; `validate_release_assets.py` PASS). On a CPU run with
a stand-in bundle, every check the notebook describes fired with the message it documents. That covers whole-ZIP digest
mismatch, zip-slip, byte flip, extra file, other base revision, branch-name revision, `trustRemoteCode`, all-zero
adapter, blank prompt, over-long prompt and `max_new_tokens=0` (probes `P6`–`P13`). Re-running Sections 4–5 with a
different bundle in the same kernel attaches the new adapter correctly: logits are identical to a fresh attach (`P14`).
The notebook is also candid about what it does not prove. The trust boundary, the `not-measurable` evaluation and the
"does not authenticate the sender" caveat are all stated.

**The default `Run all` path cannot complete by itself.** With `ARTIFACT_DIR` empty, which is the default, Section 4
opens a Colab upload dialog for a bundle that the learner must first produce by running the E2E notebook in another
session. On a non-Colab Jupyter kernel, which the Prerequisites list as supported, the same default raises
`ModuleNotFoundError: No module named 'google.colab'` (`P5`). The notebook discloses this itself as a "Known NOTEBOOK_SPEC
2.0 gap (§19, SART1/RUN2)" and labels itself `Candidate`. The disclosure is honest, but it does not satisfy the
`ARTIFACT-INFERENCE` contract: a trusted sample artifact obtained automatically, with no upload on the default path.
This is **LMA-B1**.

There is no execution record of this notebook at any revision. Seven Minor findings cover that gap and six others: an
unchecked "the adapter changes the answers" claim, a non-interactive path that bypasses the archive checks, mislabelled
evaluation-report fields, unseeded sampling with an over-strong "none should loop", a thin guided layer, and stale spec
and status declarations. There are also three Suggestions.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Revision | `311b3c0` (merge of PR #75). Notebook last changed in `534fc40` (2026-09-14). `metadata.dimer.generated_from.revision` = `8062ec17` (module SHA-256 `95f4b2b0…`, equal to `src/lmpipeline/pipeline.py` on `main`) |
| Profile / mode | `ARTIFACT-INFERENCE` / `GUIDED` (opening and `metadata.dimer`) |
| Spec declared / applied | 2.0 / 2.2 |
| Intended audience | Self-paced technical learner who receives an adapter ZIP and must verify and use it (opening narrative); familiar with Colab and basic Python |
| Prerequisites (stated) | Colab or Jupyter, Python 3.12; CUDA recommended (4-bit `nf4`), CPU float32 "slower but works"; an adapter bundle produced by the E2E tutorial; Hugging Face Hub access only, no credentials |
| Supported runtime | Google Colab (badge); Jupyter named as supported |
| Promised outcomes | Verify an external PEFT bundle before deserialisation; inspect provenance; attach it to the digest-verified pinned base with no network fallback; validate new prompts into an input manifest; generate greedy adapter-off/on and sampled answers; produce a `not-measurable` evaluation report; export result JSON, generations CSV, input manifest and evaluation report |
| Generator | `tools/build_notebook.py --template tools/notebook_template_artifact_inference.py` (carried module `src/lmpipeline/pipeline.py`) |

**Existing execution evidence.** `docs/release-verification.md` → "Recorded executions" lists the companion path as
"pending — queued to the GPU lane". The only recorded run in the repository is the **E2E** notebook on Kaggle T4
(2026-09-14, `e9d7e6b`, blob `469911578d05`, 10/10 cells, PASSED). That blob is not the current E2E blob
(`2dbb5039`), and that run does not cover this notebook. `tutorials/RELEASE_VERIFICATION.md` says neither notebook has
a passing record. There is no `docs/execution-evidence/` directory.

### Evidence actually obtained

- **Source inspection:** all 17 cells; the generator template; `pipeline.py`; the E2E template's bundle export and ZIP
  layout (`tools/notebook_template.py` ~L420–460); the three release/status documents.
- **Static checks (not execution evidence):** `build_notebook.py --check` OK; `validate_release_assets.py` PASS;
  `pytest` offline suite 49 passed (`test_companion_parity`, `test_colab_tutorials`, `test_role_helpers`,
  `test_snapshot_helpers`, `test_notebook_parity`), exit 0.
- **Direct execution (CPU, labelled):** `run_probes.py` executes the notebook's own code cells 3, 5, 7, 9, 11, 13 and 15
  in order, in one namespace, and changes form fields by rewriting the `# @param` line. Environment: Windows, Python
  3.12.14, torch 2.13.0+cpu, transformers 4.57.6, peft 0.18.0. These are **not** the notebook's pins: the install
  cell ran its `DIMER_NOTEBOOK_CI_PREINSTALLED=1` skip path. The pinned SmolLM2-360M snapshot was hard-linked from a
  local copy, and `verify_snapshot` re-hashed all 13 files against the notebook's inline manifest (`P3`, no download).
  The adapter bundles are **stand-ins** built with PEFT and the carried `export_adapter_bundle` (random LoRA-B on
  `q_proj`/`v_proj`, ZIP layout copied from the E2E template). They are not E2E-trained bundles. The upload branch
  ran through a `google.colab.files` shim. Total probe wall time was 52 s.
- **Not verified:** any Colab or Jupyter `Run all`; the pinned install (`torch==2.14.0` etc.) and its restart guard on
  the Colab kernel; the CUDA 4-bit `nf4` load path (`_preload_nvidia_libs`, bitsandbytes); a real Hub fetch of the
  snapshot; an E2E-produced bundle; answer quality or language shift from a trained adapter; learner understanding.

### Journeys

| Journey | Evidence basis | Result |
|---|---|---|
| First-time learner | Source inspection | Well oriented, with a clear trust-boundary story. The default path needs an E2E bundle the learner does not have, and the notebook says so (LMA-B1). The guided layer is thin (LMA-m6) |
| Clean default | Not verified as `Run all` (the default opens an upload dialog by design; no hosted record). Direct CPU execution with `ARTIFACT_DIR` set to a stand-in bundle | All cells completed. Generation took 8 s; 4 exports written; blank-prompt finding recorded; verdict `not-measurable` (`P9`). On non-Colab Jupyter the true default stops at Section 4 (`P5`) |
| Active learning | Direct CPU execution | `CUSTOM_PROMPT` reaches validation, generation and all exports (3 generations, custom row present, `P10`). Invalid custom prompt and `MAX_NEW_TOKENS=0` are rejected, naming the ceiling (`P10b`/`P10c`). Next experiments are prose only (LMA-m6) |
| Reuse and recovery | Direct CPU execution (upload via shim) | Upload good/cancel/wrong-SHA/zip-slip, 5 directory tamper controls and the zero adapter all rejected with the documented messages (`P6`–`P8`, `P13`). Re-attaching a second bundle in the same kernel is correct (`P14`). Directory mode silently ignores a wrong `EXPECTED_ARTIFACT_ZIP_SHA256` and never runs `extract_zip_safely` (LMA-m3) |

## 2. Promise and objective tracing

| Claim | Implementation (cell) | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| "Once the bundle is present, **Run all** … all inside this kernel" | 9 (upload unless `ARTIFACT_DIR`) | Upload dialog on the default; `ModuleNotFoundError` outside Colab (`P5`) | The opening explains the gap | **Not delivered on the default path** (LMA-B1) |
| Bundle verified before any state is deserialised | 9 `verify_artifact_bundle`, before 11 `load_adapter` | 5/5 tamper controls rejected; messages match the "When a check fails" table (`P8`) | Table in Interpretation | Delivered |
| Archive path safety, 512 MiB cap, whole-ZIP digest | 9 upload branch, `extract_zip_safely` | zip-slip rejected, nothing escaped; wrong SHA rejected (`P6`) | Section 4 prose | Delivered on the upload branch only; not on the `ARTIFACT_DIR` branch (LMA-m3) |
| Attach to the verified base, no network fallback | 7 `from_pretrained(local_files_only)`, 11 `PeftModel.from_pretrained` | Attached; non-zero B check passes; zero adapter rejected (`P9b`, `P13`) | Section 5 prose | Delivered (CPU fp32; nf4 not verified) |
| Prompts validated into an input manifest with a rejection finding | 13 `validate_prompts` | Manifest written; blank-prompt finding recorded (`P9c`) | Section 6 prose | Delivered |
| Greedy adapter off vs on, "the only difference … is the adapter" | 15 `disable_adapter()` | Both columns produced. With a stand-in adapter whose on/off logits differ by up to 1.02, the greedy answers were identical (`P9d`, `P14`) | "Expect the answers … to differ in language, tone or format" | Mechanism delivered; that the answers differ is not checked (LMA-m2) |
| "Each sampled answer differs and none should loop" | 15 `SAMPLING`, no seed | 2 different answers (`P9d`) | Section 7 prose | Not reproducible, and the claim is over-strong (LMA-m5) |
| Evaluation report is `not-measurable` and says what would make it measurable | 15 `evaluation_report(None, sample_kind='BYOD', …)` | Verdict `not-measurable`, `needs` present; `task` = "supervised fine-tuning (QLoRA adapter)", cross-entropy `score_semantics`, `sample_kind` `BYOD` (`P9d`) | Section 7 prose | Delivered, with mislabelled fields (LMA-m4) |
| Exports carry artifact identity, provenance, source and runtime, with no credentials | 15 payload | Result JSON contains `artifact.manifest`, `provenance`, `NOTEBOOK_SOURCE`, model id/revision/licence, runtime (`P9d`) | Section 7 prose | Delivered |
| "A successful run proves … [the adapter] changes the model's answers when switched on" | none | A run with a ~1e-9 LoRA-B adapter succeeds with identical answers (`P11`) | Interpretation | **Unsupported** (LMA-m2) |

| Learning objective | Learner activity | Evidence the objective was exercised |
|---|---|---|
| Install the pinned runtime; read what the module guarantees | Run cells 3, 5 | Version dict printed. Reading a 953-line cell is not scaffolded (LMA-m6) |
| Resolve and digest-verify the immutable base snapshot | Run cell 7 | `verified_files: 13` printed |
| Supply and verify an external bundle (path safety, digests, format, provenance, base identity) | Upload or set `ARTIFACT_DIR` | Verified on the upload branch. The learner exercises no failure; the "When a check fails" table is the only route to seeing one |
| Inspect provenance | Read cell 9 prints | Provenance printed; no question asks the learner what to check in it |
| Attach with no network fallback | Run cell 11 | Adapter config printed |
| Validate new prompts | Type `CUSTOM_PROMPT` | Custom prompt reaches all exports (`P10`) |
| Generate off/on, greedy then sampled | Run cell 15 and compare | Side-by-side print. No predict-first prompt and no "what to notice" check on the actual difference |
| Produce a `not-measurable` report; export results and provenance | Run cell 15 | Files listed |

## 3. Separate judgments

- **Technical correctness:** good on the paths exercised. Verification, safe extraction, attach, re-attach, prompt
  validation and exports behaved as documented on CPU. Two issues: the directory path bypasses the archive checks
  without telling the user (LMA-m3), and the evaluation-report fields are mislabelled (LMA-m4). The CUDA `nf4` path
  and the pinned install are not verified.
- **Promise fulfilment:** the core verify-and-attach promise is met. The `Run all` promise is not met on the default
  path (LMA-B1), and the claim that a run "proves" the adapter changes answers is unsupported (LMA-m2).
- **Learner experience:** clear narrative, an honest trust boundary, and a good failure-message table. But a learner
  without an E2E bundle cannot start. Learner activity is limited to typing one prompt, and the guided-layer
  expectations of 2.2 are mostly absent (LMA-m6).
- **Spec conformance (2.2):** **fails** RUN1, RUN2, RUN5, SART1–SART4 and §19 (automatic trusted sample artifact; no
  upload on the default path) (LMA-B1), and REL1/REL2/REL10 (no clean-runtime evidence, LMA-m1). The ENV7 seed
  requirement for sampled generation is arguable (LMA-m5). SHOULD deviations: EXE2/EXE3 and the §20 whole-archive
  digest (LMA-m3); UX5 and GDL7–GDL11 (LMA-m6). The declared spec version is 2.0 against the 2.2 baseline (LMA-m7).

## 4. Findings

### Blocker

#### LMA-B1 — The default `Run all` path has no sample artifact: it opens an upload dialog (and fails outside Colab)

**Cell/section:** opening "Run all" paragraph; Section 4, cell 9 (`ARTIFACT_DIR = ''` → `from google.colab import files; files.upload()`).

**Observed issue:** Nothing on the default path obtains an adapter bundle. With the documented defaults, cell 9 blocks on
an upload dialog for `outputs/language_model_finetuning_adapter_bundle.zip`, which only the E2E notebook produces, in a
separate session. On a Jupyter kernel, which the Prerequisites list as supported, the default raises
`ModuleNotFoundError: No module named 'google.colab'`. The notebook discloses the gap ("Known NOTEBOOK_SPEC 2.0 gap
(§19, SART1/RUN2)") and calls itself `Candidate`.

**Consequence:** A learner who opens the badge and chooses `Run all` cannot complete the core workflow. They must first
run a different, GPU-recommended fine-tuning notebook to completion, download its ZIP, and come back. For an
`ARTIFACT-INFERENCE` notebook this defeats the profile. The spec states plainly that "external" does not mean the
learner uploads the artifact (§7.4). Every downstream stage (attach, prompts, generation, exports) is unreachable on the
default path.

**Evidence:** Source inspection of cell 9 and the opening (generator `tools/notebook_template_artifact_inference.py`
L28, L96–L118). Direct execution: `P5` (default cell 9 on a non-Colab kernel → `ModuleNotFoundError`). `P6a` (cancelled
upload → `ValueError: Upload exactly one adapter bundle ZIP`).

**Recommended correction:** Publish one E2E-produced bundle as a trusted sample artifact at an immutable location. Two
options: a Hugging Face Hub repo at a pinned commit, or a release asset whose bytes are pinned. Record its whole-archive
SHA-256 and producer provenance (E2E notebook blob, commit, runtime) in the template. When no user artifact is
requested, cell 9 downloads that ZIP, checks the pinned SHA-256 before extraction, and runs it through
`extract_zip_safely` and `verify_artifact_bundle` exactly as an uploaded bundle would be. Make the user artifact an
opt-in branch (for example `USE_OWN_ARTIFACT = False  # @param`, plus a ZIP location field; see LMA-m3) so that
`google.colab` is imported only when the user asks for the upload dialog. Then remove the "Known gap" paragraph and
restate the opening per spec §29.

**Acceptance check:** In a fresh Colab runtime and in a fresh Jupyter kernel, with no field edited, `Run all` completes
every cell with no upload dialog and no `google.colab` import. Cell 9 prints the sample artifact's source URL, its
pinned SHA-256 and `artifact_source` ≠ `upload dialog`. Changing one byte of the pinned SHA constant makes cell 9 fail
before extraction. Setting the opt-in flag opens the upload dialog on Colab.

**Spec:** RUN1, RUN2, RUN5, SART1–SART4, §7.4, §19, DAT1/DAT4 (artifact side).

### Major

None.

### Minor

#### LMA-m1 — No execution evidence for this notebook at any revision

**Cell/section:** repository release records (`docs/release-verification.md` → Recorded executions;
`tutorials/RELEASE_VERIFICATION.md` → Passing records).

**Observed issue:** The companion row is "pending — queued to the GPU lane". No run of blob `41b88e80` (or any earlier
blob) is recorded. The PEFT attach path in the standalone carrier, the pinned install on a hosted kernel, the CUDA
`nf4` load and the Hub fetch have never been recorded as executed for this notebook.

**Consequence:** Nothing shows that the supported runtime runs this notebook. The CPU probes in this review use other
library versions and stand-in bundles and cannot substitute. The 2026-09-14 E2E Kaggle PASS (same `PINS`, so the
install guard did not fire on that image) is the closest evidence, but it covers a different notebook.

**Evidence:** Source inspection of both records; `git log` shows the notebook unchanged since `534fc40` (2026-09-14).

**Recommended correction:** After LMA-B1, run the notebook at the fixed head in a fresh Colab GPU runtime: `Run all`,
no fields edited, followed by one optional user-artifact run. Record the commit, blob, runtime (Python, torch,
transformers, peft, GPU), sample-artifact SHA-256, outcome and outputs in `docs/release-verification.md`.

**Acceptance check:** `docs/release-verification.md` has a PASS row whose blob equals
`git rev-parse HEAD:tutorials/language_model_artifact_inference_colab.ipynb` at the release commit, with the runtime
fields filled in. The run shows no restart and no upload.

**Spec:** REL1, REL2, REL4, REL5, REL10.

#### LMA-m2 — "A successful run proves … [the adapter] changes the model's answers" is never checked

**Cell/section:** Section 7 cell 15; Interpretation, first paragraph; Section 7 prose "Expect the answers … to differ".

**Observed issue:** Cell 11 rejects only an all-zero LoRA-B. Cell 15 prints base and adapted answers but never compares
them. A bundle whose adapter has no visible effect still runs to completion, and the Interpretation then tells the
learner the run "proves" the adapter changes the answers. The repository's own release procedure (step 6) expects
"greedy adapter-off/on answers differ", and the E2E notebook raises when its reloaded on/off answers are identical. The
companion does neither.

**Consequence:** If the learner's bundle is weak or mis-trained, the learner sees identical columns next to text that
says the run proved a difference. There is no prompt telling them how to read that.

**Evidence:** Direct execution with stand-in bundles. `P11`: LoRA-B ≈ 1e-9 passes the non-zero check; cells 9–15
succeed; both greedy pairs are identical. `P9d`/`P14`: a stand-in with on/off next-token logits differing by up to
1.02 still gave identical greedy answers on both default prompts, because the base model echoes these Filipino prompts.
That is stand-in behaviour, not an E2E bundle.

**Recommended correction:** In the template (L188–L193, L216, L252), add an explicit adapter-activity readout to cell
15: the max-abs difference of adapter-on vs adapter-off next-token logits on the first prompt, plus a per-prompt
`answers_differ` flag recorded in the result JSON. Rewrite the Interpretation so that it claims only what was checked,
for example "the adapter changed next-token scores by X; the greedy answers differed on k of n prompts". Either fail
when the logit delta is ~0, or tell the learner what an identical pair means.

**Acceptance check:** Running with the `P11` tiny-B bundle either stops with a message naming the adapter as inactive,
or prints `answers_differ: [False, False]` and a near-zero logit delta that the Interpretation text explains. The
Interpretation contains no "proves … changes the model's answers" sentence unless a check backs it.

**Spec:** §21.7 ("compare adapted and base behavior"), RUN8, UX1.

#### LMA-m3 — The non-interactive `ARTIFACT_DIR` path skips the archive checks and silently ignores the trusted digest

**Cell/section:** Section 4, cell 9 (`if ARTIFACT_DIR:` branch).

**Observed issue:** The only non-interactive input is a **directory**. In that branch `extract_zip_safely` never runs,
and `EXPECTED_ARTIFACT_ZIP_SHA256` is ignored with no warning. A wrong 64-hex value is accepted (`P7`). There is no ZIP
location field, so an executor holding the E2E ZIP has to unzip it outside the notebook, which bypasses the path-safety
and whole-archive-digest checks that Section 4 teaches. The upload branch, the only one that exercises them, is
Colab-only. The release procedure tells the Kaggle executor to use `ARTIFACT_DIR`, so a recorded run would never
exercise extraction.

**Consequence:** A user who pastes the producer's whole-ZIP SHA-256 and points `ARTIFACT_DIR` at an unpacked folder
believes the digest was checked when it was not. Release runs cannot cover the archive-safety stage.

**Evidence:** Direct execution `P7` (wrong digest + `ARTIFACT_DIR` → accepted). Source inspection of cell 9 (template
L96–L118) and `docs/release-verification.md` step 6.

**Recommended correction:** Add `ARTIFACT_ZIP_PATH = ''  # @param` (EXE2/EXE3). When it is set, read the ZIP from that
path, check `EXPECTED_ARTIFACT_ZIP_SHA256`, and run `extract_zip_safely` without importing `google.colab`. Keep the
directory option only if it is still needed, and raise (or print a visible notice) when `EXPECTED_ARTIFACT_ZIP_SHA256`
is set but no archive is being read. Point the release procedure at the ZIP field.

**Acceptance check:** `ARTIFACT_ZIP_PATH=<good zip>` with the correct digest passes. With a wrong digest it fails with
the whole-ZIP mismatch message. With a zip-slip archive it fails with `Unsafe archive path`. None of these imports
`google.colab`. Setting the digest together with a directory-only input never passes silently.

**Spec:** EXE2, EXE3 (SHOULD); §20 (trusted whole-archive digest, SHOULD); DAT19.

#### LMA-m4 — The exported evaluation report is labelled as a fine-tuning report and `BYOD`

**Cell/section:** Section 7 cell 15, `evaluation_report(None, sample_kind='BYOD', probes=rows)`; module `evaluation_report`.

**Observed issue:** The inference notebook's report says `task: "language-model supervised fine-tuning (QLoRA
adapter)"` and `score_semantics: "mean cross-entropy per supervised assistant token; perplexity = exp(loss)"`, and
records `sample_kind: "BYOD"`. The default run uses the notebook's two prefilled sample prompts, not user data, and
computes no loss.

**Consequence:** The machine-readable report, which the notebook presents as the evaluation-stage output, describes a
different task and a metric nobody computed. It also labels sample input as user data. Downstream readers of
`language_model_artifact_inference_evaluation_report.json` are misled. The verdict itself (`not-measurable`) is correct.

**Evidence:** Direct execution `P9d` (report fields as quoted). Source: `pipeline.py` `evaluation_report` (`task` and
`score_semantics` are hard-coded); template L216.

**Recommended correction:** Give `evaluation_report` a `task`/`score_semantics` parameter, or add an inference variant
in `pipeline.py`. Use "adapter-attached generation on new prompts; no score computed" for the companion. Pass
`sample_kind='sample'` when only the prefilled prompts ran, and `'sample+BYOD'` when `CUSTOM_PROMPT` was used.

**Acceptance check:** On a default run the exported report has `sample_kind: "sample"`, no cross-entropy wording, and a
`task` naming inference with an attached adapter. With `CUSTOM_PROMPT` set it says so. The parity tests still pass.

**Spec:** INF3, OUT3, EVAL6.

#### LMA-m5 — Sampled generation is unseeded, and "none should loop" is stated as an expectation

**Cell/section:** Section 7 prose and cell 15 (`SAMPLING`, two `pipe.generate(..., do_sample=True)` calls).

**Observed issue:** No seed is set before sampling, so the sampled answers, and the CSV/JSON that record them, change on
every run. The prose says "each sampled answer differs and none should loop". Neither half is guaranteed: two draws can
coincide, and a 360M model can still loop at temperature 0.7. The "Next experiments" line then asks the learner to
"note which loops".

**Consequence:** Re-running the notebook gives different exports with no record of why, and a learner who sees a
sampled loop is told it should not happen.

**Evidence:** Source inspection; direct execution `P9d` (two distinct sampled answers on CPU; reproducibility across runs
not measured).

**Recommended correction:** Seed sampling explicitly (`torch.manual_seed(SEED)` before each draw, with `SEED` recorded in
`payload['sampled']`). State that sampled text varies with the seed, device and versions. Change the claim to "sampling
usually avoids greedy repetition loops; note any that remain".

**Acceptance check:** Two CPU runs with the same seed give identical `sampled.answers`; the seed appears in the result
JSON; the prose contains no unconditional "none should loop".

**Spec:** ENV7, ENV8, INF6.

#### LMA-m6 — The `GUIDED` layer is thin: no prediction, checkpoint or exercise cell; infrastructure is unlabelled

**Cell/section:** whole notebook. Section 2 (953-line carried module), Sections 6–7, "Next experiments".

**Observed issue:** The learner's only activity is optionally typing `CUSTOM_PROMPT`. Nothing asks for a prediction
before the off/on comparison, gives an interpretation checkpoint with a sample answer, or offers a runnable
change-one-thing exercise. The three "Next experiments" are prose only. The carried module cell is the
inference learner's biggest obstacle: it includes dataset normalisation, loss masking, the training registry and
`manufacture_validation`, and its docstring cites "NOTEBOOK_SPEC 1.1 §3.6". It is not labelled as infrastructure the
learner may run without studying (Section 2 says only "Nothing in these cells runs a model yet"). The objectives are
mostly code actions ("install", "resolve", "attach") rather than observable learner outcomes.

**Consequence:** The notebook works as a reference, but a self-paced learner gets little practice in the stated
skills, such as reading provenance, recognising a refused bundle, or judging whether the adapter did anything.

**Evidence:** Source inspection of all markdown cells.

**Recommended correction:** In the template, label Section 2 as **Infrastructure — run without studying**, and state
which five functions matter for this notebook. Add a "Predict" line before cell 15 (for example "will the adapted
answer be in Filipino? longer?") and a collapsible sample answer after it. Turn one Next experiment into a gated-off
runnable cell: a deliberately tampered copy of the bundle (one byte flipped) that the learner verifies and sees refused,
or the English-prompt comparison. Phrase objectives as observable actions (GDL5).

**Acceptance check:** The notebook contains a cell labelled Infrastructure for the carried module; at least one
predict → run → explain activity with worked guidance; and one optional runnable exercise that is off by default, so
`Run all` is unchanged.

**Spec:** UX5 (SHOULD); GDL5, GDL7–GDL11 (SHOULD, 2.2).

#### LMA-m7 — Spec version and release-status records disagree

**Cell/section:** `metadata.dimer.notebook_spec` and the opening ("Specification 2.0"); `docs/release-verification.md`;
`tutorials/RELEASE_VERIFICATION.md`; carried module docstring.

**Observed issue:** The notebook declares spec 2.0; the baseline is 2.2. `docs/release-verification.md` says the
validator checks "spec `1.1`" and that static checks are "not runtime evidence under DIMER Notebook Specification 1.1",
although `validate_release_assets.py` enforces `NOTEBOOK_SPEC = "2.0"`. Its "Current status" paragraph opens "No
clean-runtime execution of either standalone notebook has been recorded yet; clean GPU execution evidence for the E2E
path is now recorded below", which contradicts itself. `tutorials/RELEASE_VERIFICATION.md` lists the E2E as "No passing
clean-run record" while `docs/release-verification.md` records an E2E PASS.

**Consequence:** A reviewer cannot tell from the records which spec applies or what has been run. That is the
release-record drift §32 item 6 was written to stop.

**Evidence:** Source inspection; `grep` of `tools/validate_release_assets.py` L202 (`NOTEBOOK_SPEC = "2.0"`).

**Recommended correction:** When LMA-B1 is fixed, migrate the declaration to 2.2 (template, `NOTEBOOK_SOURCE`,
validator constant and tests in the same change set). Correct the spec references in `docs/release-verification.md`.
Make the two release documents agree on one status per notebook.

**Acceptance check:** `metadata.dimer.notebook_spec`, the opening, `NOTEBOOK_SOURCE`, the validator constant and both
release documents name the same spec version. `grep -n "1\.1" docs/release-verification.md` finds no notebook-spec
reference. Both release documents give the same status for each notebook.

**Spec:** §32 items 1 and 6; REL10; SRC3 (knowingly stale instructions).

### Suggestions

- **LMA-S1** — Record and print the "adapter activity" figure that the release template already asks for
  (`tutorials/RELEASE_VERIFICATION.md`: "LoRA-B max abs + adapter-on/off logit max abs delta"). Cell 11 already
  computes the LoRA-B max; print it rather than only testing it against zero.
- **LMA-S2** — Add a short troubleshooting note on two things. First, CPU runtime: Section 3 loads float32 and generation
  is slower. Second, re-running Section 5 with a new bundle prints PEFT's "multiple adapters" warning. In this review
  that warning was harmless, because the re-attach replaced the `default` adapter and gave logits identical to a fresh
  attach (`P14`). Telling the learner so prevents a needless restart.
- **LMA-S3** — Consider carrying only the inference-relevant subset of `pipeline.py` in the companion (snapshot
  verification, archive safety, bundle verification, prompt validation, generation, evaluation report). That would
  shrink the 953-line cell, and parity could still be enforced per function.

## 5. Readiness

**Needs revision.** Open gates:

1. **LMA-B1:** the default path must obtain a trusted sample artifact automatically, with no upload and no
   `google.colab` import. Until then the notebook is not `Run all` conformant (RUN1/RUN2/RUN5/SART1/§19).
2. **LMA-m1:** a recorded fresh-runtime `Run all` PASS for the release blob (REL1/REL2/REL10).
3. The Minor findings (LMA-m2 to LMA-m7) should be fixed when practical. LMA-m7 touches the release records and is
   cheapest to fix alongside the other two gates.

There are no Major findings. Suggestions are optional.

## 6. Verified vs inferred

- **Verified by direct CPU execution (stand-in bundles, non-pinned library versions):** every rejection path listed in
  the "When a check fails" table, plus zip-slip, `trustRemoteCode`, the zero adapter, a blank or over-long prompt and
  `max_new_tokens=0`. Also: all four exports are written; `CUSTOM_PROMPT` reaches the exports; re-attaching in the same
  kernel is correct; directory mode ignores the digest; the default fails outside Colab; the report fields are as quoted.
- **Verified statically:** generator parity, validator PASS, offline suite 49 passed, pins unchanged since the
  2026-09-14 E2E Kaggle run.
- **Inferred, not verified:** that the pinned install completes on the current Colab kernel without a restart
  (inferred from the E2E Kaggle PASS with the same `PINS`; the Colab kernel's Python version was not checked for this
  notebook); that the CUDA `nf4` path loads; that an E2E-trained adapter visibly changes the greedy answers.
- **Only Kurt or a hosted run can confirm:** a Colab `Run all` once LMA-B1 is fixed; learner comprehension.
- **Most likely to be wrong:** the LMA-m2 mechanism. Identical greedy answers came from stand-in random adapters on a
  CPU float32 base that echoes the Filipino prompts. An E2E-trained adapter on the `nf4` base may differ visibly on
  both prompts, which would turn the "unchecked claim" into a rarely triggered edge case. The missing check and the
  overstated sentence remain either way.
