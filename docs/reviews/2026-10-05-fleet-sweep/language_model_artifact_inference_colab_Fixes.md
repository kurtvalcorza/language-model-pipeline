# Language-model artifact-inference notebook — fleet-sweep fixes

**Source:** the 2026-10-05 static fleet sweep of `main` flagged this notebook for the old restart guard and a missing
guided layer (0 of 9 guided markers; declares `GUIDED`). Targeted fix; **no full Notebook Review Framework v1 review
has been done**.
**Base:** `main` at `311b3c0`.
**Notebook:** `tutorials/language_model_artifact_inference_colab.ipynb`, regenerated with
`tools/build_notebook.py --template tools/notebook_template_artifact_inference.py` (`build_notebook.py/2.2`).
**Readiness:** **Verification pending.** **Status labels:** unchanged (the `run_all` text keeps its known SART1/RUN2 gap).

## Findings

| ID | Status | Change | Cells / files touched | Evidence |
|---|---|---|---|---|
| SWP-R | Fixed — hosted confirmation pending | Same generator `/2.2` isolated runtime and lock as the fine-tuning notebook (the companion template now inherits `isolated_runtime`, `infrastructure_labels`, `managed_python`, `uv` and `lock` from it); no in-kernel install, no restart; Section 1 reuses a lock-keyed environment and keeps a live worker on re-run; `MPLBACKEND=Agg`, `PYTHONPATH`/`PYTHONHOME`/`PYTHONSTARTUP` dropped. | `tools/notebook_template_artifact_inference.py` (+ shared generator, validators, lock) | `test_swp_r_*` (parametrised over both notebooks) |
| SWP-G | Fixed | Guided layer: audience, Input → Model → Output, How to use, roadmap, four **Predict** / **Check your reasoning** pairs (bundle refusal point, adapter config, blank-prompt probe, sampled versus greedy and the `not-measurable` verdict), What to notice, Troubleshooting (isolated runtime + the existing bundle-check messages), Change one thing, Glossary, Conclusion. No recorded run exists, so answers are qualitative. Infrastructure cells collapsed. | template | `test_swp_g_*` (parametrised); guided markers 0/9 → 9/9 |
| SWP-A | Not applicable | No quality `assert`. | — | — |
| SWP-F | Not applicable | No training. | — | — |
| SWP-B | Fixed | `ARTIFACT_DIR` already existed; it is now checked to be a folder (named refusal), the upload fallback is guarded outside Colab, and a cancelled or multi-file upload reports what it got. | Section 4 | `test_swp_b_artifact_dir_and_upload_are_guarded` |

## User-visible changes

- Isolated Section 1 (as in the fine-tuning notebook); guided-layer markdown; clearer Section 4 refusals.

## Verification (offline; not clean-runtime evidence)

- No model stage ran. Stand-ins: Section 1 bootstrap; the Section 4 guard block with temporary folders and a fake
  upload. `--check` OK; validators PASS; `pytest` 541 passed, 1 skipped → 560 passed, 1 skipped (repository total).

## Remaining gates

- A hosted Run all in one pass with a bundle from the fine-tuning notebook (upload, or `ARTIFACT_DIR`), then a re-run
  of the export cell.
- A full Notebook Review Framework v1 review has not been done.
