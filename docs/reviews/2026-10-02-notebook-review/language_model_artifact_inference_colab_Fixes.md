# Language-model artifact-inference notebook — review fixes

**Review:** `language_model_artifact_inference_colab_Review.md` (Notebook Review Framework v1, prefix `LMA`, reviewed
commit `311b3c0`, verdict *Needs revision*; PR #76).
**Sweep record:** `../2026-10-05-fleet-sweep/language_model_artifact_inference_colab_Fixes.md` (commit `22338c5`: isolated
runtime, guided layer, guarded `ARTIFACT_DIR`/upload).
**Notebook:** `tutorials/language_model_artifact_inference_colab.ipynb`, regenerated with
`tools/build_notebook.py --template tools/notebook_template_artifact_inference.py`; `--check` clean.
**Readiness:** **Verification pending.** **Status labels:** unchanged (`Candidate`).

All changes are in the generator, the templates, the carried module (`evaluation_report` gains `task`, `score_semantics`
and `baselines`; `INFERENCE_TASK` / `INFERENCE_SCORE_SEMANTICS` constants), the validator, the release documents and the
tests. No model stage ran here (no torch, no GPU, the Hugging Face Hub is unreachable).

## Findings

| ID | Status | Change | Cells / files touched | Evidence |
|---|---|---|---|---|
| LMA-B1 | **Partly fixed — the hosted run must produce the sample bundle** | The default path now reads a **trusted sample artifact slot**: `SAMPLE_ARTIFACT = {'url', 'sha256', 'producer'}` in Section 4. The ZIP is downloaded (reused if present and matching), its whole-archive SHA-256 is compared with the pinned digest *before extraction*, then `extract_zip_safely` and `verify_artifact_bundle` run exactly as for an uploaded bundle; `artifact_source` prints `sample artifact: <url>` with the pinned digest and producer provenance. The user bundle is the opt-in `USE_OWN_ARTIFACT = False  # @param` branch (`ARTIFACT_ZIP_PATH`, `ARTIFACT_DIR`, or the Colab upload); `google.colab` is imported only there. **The slot is empty**: the bundle can only come from a recorded hosted run of the fine-tuning notebook and cannot be produced offline. While empty, the cell uses `outputs/language_model_finetuning_adapter_bundle.zip` if the E2E notebook wrote it in the same runtime, else stops with a message naming `SAMPLE_ARTIFACT` and `USE_OWN_ARTIFACT`. The "Known gap" paragraph is replaced by a §29-style `run_all` that states the remaining gap. | Section 4 (md + code); opening; prerequisites; `tutorials/README.md`; both release docs | `test_lma_b1_default_path_downloads_and_digest_checks_the_pinned_sample_artifact`, `test_lma_b1_a_wrong_pinned_digest_fails_before_extraction`, `test_lma_b1_empty_slot_falls_back_to_a_same_runtime_e2e_bundle_or_stops_with_a_named_message`, `test_lma_b1_the_default_path_never_imports_google_colab_and_the_gate_is_off`; probe re-run `new_sample_slot_pinned_default_path` (stand-in ZIP served to the cell: extracted, `artifact_source` = sample artifact), `new_sample_slot_wrong_digest` (mismatch, nothing extracted), `P5_default_no_colab` before: `ModuleNotFoundError: google.colab` → after: the named `RuntimeError` |
| LMA-m1 | Not fixed — needs hosted run | No run can be recorded here. The release documents now say what must be recorded (restart/shim status, sample-artifact SHA-256, adapter activity, `sample_kind`). | `docs/release-verification.md` steps 6 and 8; `tutorials/RELEASE_VERIFICATION.md` | — |
| LMA-m2 | Fixed | Section 7 measures adapter activity instead of assuming it: `next_token_logits` (first prompt, adapter off vs on) → `next_token_logit_max_abs_delta`; per prompt `answers_differ`; `ADAPTER_ACTIVITY` with `lora_b_max_abs` and a verdict (`inactive:` when the delta is < 1e-6, else `active: … greedy answers differ on k of n prompts`), printed and written to the result JSON. The Interpretation no longer says the run "proves" the adapter changes the answers; it explains identical columns next to a clear delta and a delta near zero. | Section 5 (prints `lora_b_max_abs`), Section 7 (md + code), closing | `test_lma_m2_readout_records_the_logit_delta_and_which_answers_differ`, `test_lma_m2_a_tiny_b_adapter_that_moves_nothing_is_reported_inactive` (the review's P11 case: delta 0 → `inactive`, `answers_differ: [False, False]`), `test_lma_m2_interpretation_claims_only_what_is_checked` (stand-in model; the logit values on the real model are hosted evidence) |
| LMA-m3 | Fixed | `ARTIFACT_ZIP_PATH` form field: a ZIP in the runtime is digest-checked (`EXPECTED_ARTIFACT_ZIP_SHA256`) before `extract_zip_safely`, without `google.colab`. `ARTIFACT_DIR` is kept for an unpacked folder, prints that the archive checks do not apply, and a digest paired with it is **refused**. The release procedure points the Kaggle executor at `ARTIFACT_ZIP_PATH`. | Section 4; `docs/release-verification.md` step 6 | `test_lma_m3_artifact_zip_path_runs_the_digest_and_archive_checks_without_google_colab` (good digest passes, wrong digest `Whole-ZIP SHA-256 mismatch`, zip-slip `Unsafe archive path`, nothing escapes), `test_lma_m3_a_digest_paired_with_a_folder_is_refused_and_the_folder_route_says_what_it_skips`; probe re-run `P7_dir_mode_wrong_digest` before: accepted silently → after: `ValueError: EXPECTED_ARTIFACT_ZIP_SHA256 is set but ARTIFACT_DIR names an unpacked folder` |
| LMA-m4 | Fixed | `evaluation_report(..., task=, score_semantics=)` in the module; the companion passes `INFERENCE_TASK` ("adapter-attached generation on new prompts …; no score computed") and `INFERENCE_SCORE_SEMANTICS` (no cross-entropy wording), and `sample_kind='sample'` or `'sample+BYOD'` when `CUSTOM_PROMPT` is set. Fine-tuning defaults unchanged. | `src/lmpipeline/pipeline.py`; Section 7 | `test_lma_m4_evaluation_report_takes_task_and_score_semantics_and_the_companion_passes_them`; parity tests pass (module re-carried) |
| LMA-m5 | Fixed | `SEED = 42  # @param`; `torch.manual_seed(SEED + attempt)` before each draw; `payload['sampled']['seeds']`; prose: sampled text is reproducible on the same device and versions, "sampling usually avoids greedy repetition loops; note any that remain", two draws can coincide. | Section 7 (md + code) | `test_lma_m5_sampled_generation_is_seeded_and_the_seeds_are_recorded` (stand-in `torch.manual_seed`; identical answers across two real runs is hosted evidence) |
| LMA-m6 | Fixed | Section 2 carries a **What matters for this notebook** note naming the five functions this notebook exercises (generator key `carrier_note`); the Section 7 Predict now asks about language and length and the answer is worked; objectives rephrased as observable actions ("by the end you can …"); new **Section 8 exercise**, off by default (`RUN_TAMPER_EXERCISE = False  # @param`): copies the verified bundle, flips one bit of `adapter_model.safetensors`, shows `SHA-256 mismatch: adapter_model.safetensors`, and re-verifies the bundle in use. | Section 2 note, opening objectives, Sections 7–8, Change one thing | `test_lma_m6_carrier_cell_says_which_functions_matter`, `test_lma_m6_tamper_exercise_refuses_a_one_bit_change_and_is_off_by_default` (real `verify_artifact_bundle` on a stand-in bundle), `test_lma_m6_objectives_are_observable_and_the_default_path_has_no_extra_cells`; validator enforces the gate is `False` on a form line |
| LMA-m7 | Fixed | Spec declaration migrated to **2.2** in the generator (`NOTEBOOK_SPEC`), the validator, `metadata.dimer.notebook_spec`, `NOTEBOOK_SOURCE`, the header and `tutorials/README.md` / `README.md`. `docs/release-verification.md`: spec references corrected (no notebook-spec "1.1"), the contradictory "Current status" sentence rewritten, the Kaggle row states the restart and the shim. `tutorials/RELEASE_VERIFICATION.md` rewritten for the standalone notebooks (no `GITHUB_TOKEN`, no finetuner revision) with the same status per notebook as the other documents. `STATUS.md` untouched (status rule); its "Specification 1.1 §3.6" wording is left for the maintainer. | `tools/build_notebook.py`, `tools/validate_release_assets.py`, both release docs, both READMEs | `test_lma_m7_spec_version_is_2_2_in_generator_validator_metadata_and_records`, `test_lmf_m5_release_records_tell_one_story_about_the_kaggle_run_and_the_current_blobs`; `validate_release_assets.py` PASS |
| LMA-S1 | Fixed (in a touched cell) | Section 5 prints `lora_b_max_abs`; it is recorded under `adapter_activity`. | Section 5 | covered by the LMA-m2 tests |
| LMA-S2 | Fixed (in a touched cell) | Troubleshooting: CPU runtime timing; the PEFT "multiple adapters" warning on re-attach is harmless, no restart. | Section 5 md, Troubleshooting | static |
| LMA-S3 | Not done | Carrying a subset of `pipeline.py` would change the parity contract; left for the maintainer. | — | — |
| Sweep SWP-R/G/B | Unchanged | Isolated runtime, guided layer and guarded upload stay as in `22338c5`; the Section 4 guard test was rewritten for the opt-in branch. | — | `test_swp_*` |

## User-visible changes

- Section 4: new fields `USE_OWN_ARTIFACT` (default `False`), `ARTIFACT_ZIP_PATH`; `ARTIFACT_DIR` and
  `EXPECTED_ARTIFACT_ZIP_SHA256` kept. The default path no longer opens the upload dialog: it reads the pinned sample
  artifact (slot empty in this revision → same-runtime E2E bundle, or a stop with a named message). A digest paired with
  `ARTIFACT_DIR` is refused.
- Section 7: `SEED` field; seeded sampling; `adapter_activity` and `sampled.seeds` in the result JSON; `answers_differ` on
  each generation row; the evaluation report's `task`, `score_semantics` and `sample_kind` describe inference.
- New optional Section 8 exercise (`RUN_TAMPER_EXERCISE`, off).
- Spec declaration 2.2; release documents reconciled.

## Verification (offline; not clean-runtime evidence)

- Real input: none (no snapshot, no bundle from a recorded run). Synthetic: stand-in bundles built with the module's own
  `write_artifact_manifest` (random bytes as adapter weights), stand-in ZIPs, a zip-slip archive. Stand-in model objects
  for the Section 7 readout and sampling (labelled in the tests; not pretrained-inference evidence).
- `build_notebook.py --check` (both templates) OK; `validate_release_assets.py` PASS; `scripts/validate_colab_tutorials.py`
  OK; `scripts/build_registration.py --check` OK; `ruff check .` clean; `pytest -p no:cacheprovider` with CI's pins:
  560 passed, 1 skipped → 586 passed, 1 skipped.
- Review probes re-run where they apply offline (a marker-based adapter around the review's probe logic, since the cell indices changed): P5, P6a,
  P7 before/after as in the table; the new ZIP-path and sample-slot cases pass. P8–P14 need torch and the snapshot: not
  re-run.

## Remaining gates

- **Hosted E2E run → publish the bundle → pin `SAMPLE_ARTIFACT`** (url, sha256, producer provenance) in
  `tools/notebook_template_artifact_inference.py`, regenerate, and record the run. Until then the default path stops at
  Section 4 (LMA-B1 remains open on the default path).
- A fresh-runtime Run all of this notebook in one pass (Colab and Jupyter) with the pinned sample; one own-bundle run
  (`USE_OWN_ARTIFACT`, `ARTIFACT_ZIP_PATH` or upload) with the step-5 digest; record the adapter-activity figures (LMA-m1).
- The real-model values of `next_token_logit_max_abs_delta` and `answers_differ` for a trained bundle.
