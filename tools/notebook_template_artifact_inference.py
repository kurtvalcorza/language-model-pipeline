"""Companion template for tools/build_notebook.py — ARTIFACT-INFERENCE (NOTEBOOK_SPEC 2.2 §4, §19).

Generate with ``python tools/build_notebook.py --template tools/notebook_template_artifact_inference.py``
(the generator resolves ``--out`` to ``tutorials/<notebook_name>``). The notebook carries the same pipeline
module and the same pinned base snapshot as the E2E notebook; it consumes an adapter bundle produced by a
*separate* execution and never creates one. The default path reads the trusted sample bundle pinned in
``SAMPLE_ARTIFACT`` (an E2E-produced ZIP at an immutable location, whole-archive SHA-256 checked before
extraction); ``USE_OWN_ARTIFACT`` switches to a user bundle (ZIP path, unpacked folder, or the Colab upload).

``SAMPLE_ARTIFACT`` is pinned to release asset ``sample-bundle-v1`` of this repository: the bundle written by the
fine-tuning notebook (blob ``8a2c6ad4886d``, commit ``d185817``) in the 2026-10-07 Colab CLI T4 run recorded in
``docs/verification/2026-10-07-colab-t4/``. If the slot is emptied, the default path falls back to the E2E
notebook's bundle when it exists in the same runtime, and otherwise stops with a message saying what to set.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

import importlib.util
from pathlib import Path

# The E2E template next to this file is the source of the shared keys (loaded by path so that the
# generator, the validator and the tests resolve it from any working directory).
_spec = importlib.util.spec_from_file_location("_e2e_notebook_template", Path(__file__).with_name("notebook_template.py"))
assert _spec and _spec.loader
_e2e_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e2e_module)
BADGES, REPO, _E2E = _e2e_module.BADGES, _e2e_module.REPO, _e2e_module.TEMPLATE

TEMPLATE = {
    **{k: _E2E[k] for k in ("package", "repo_name", "pipeline_class", "weights_key", "modules", "entry_module", "runtime_imports", "isolated_runtime", "infrastructure_labels", "managed_python", "uv", "lock")},
    "stem": "language_model_artifact_inference",
    "notebook_name": "language_model_artifact_inference_colab.ipynb",
    "profile": "ARTIFACT-INFERENCE",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated environment from the hash-locked pins (nothing is installed into the notebook's own Python, so no restart is needed), stages and digest-verifies the pinned base snapshot, obtains the trusted sample adapter bundle from the location pinned in Section 4 (`SAMPLE_ARTIFACT`: one bundle produced by the fine-tuning notebook, whose whole-archive SHA-256 is checked before extraction and whose files then pass the same archive, manifest, provenance and base-identity checks as any other bundle), attaches the adapter to the verified base, validates new prompts into an input manifest, generates with the adapter off and on and with seeded sampling, records whether the adapter changed anything, writes the evaluation report, and exports outputs and provenance — all inside this runtime, with no DIMER worker or service, no credential, no upload dialog and no `google.colab` import on the default path. A user-supplied bundle is the opt-in `USE_OWN_ARTIFACT` branch. The pinned sample bundle is release asset `sample-bundle-v1` of this repository, written by the fine-tuning notebook in a recorded Colab T4 run (blob `8a2c6ad4886d`, commit `d185817`; see `docs/release-verification.md`). This notebook remains a `Candidate` until a clean run of it is recorded and reviewed."
    ),
    "byod": (
        "New-input BYOD is the `CUSTOM_PROMPT` form field in Section 6 (empty by default): your own prompt passes through the same validation, generation and export cells as the sample prompts. A user-supplied adapter bundle is the separate opt-in `USE_OWN_ARTIFACT` branch in Section 4 (`ARTIFACT_ZIP_PATH` or `ARTIFACT_DIR` on Kaggle and Jupyter, the upload dialog on Colab), digest-checked and verified by `verify_artifact_bundle` before deserialisation. Uploads stay inside this runtime; do not upload confidential or restricted data unless you are authorised to process it here."
    ),
    "guided": {
        "opening": [
            (
                "**Who this notebook is for.** A learner (or a reviewer receiving someone else's fine-tune) who knows basic Python, has run a Colab notebook, and wants to use an adapter bundle safely: check what it is, check that it belongs to this base model, attach it, and see what it changes — without training anything. It is the companion of the fine-tuning notebook, which produces the bundle in a separate session. *Adapter*, *manifest*, *provenance*, *greedy* and *sampled decoding* are explained where they first matter and again in the **Glossary**. A T4 GPU is recommended; CPU works for a few prompts. The intended audience is learners and reviewers; a passing run does not certify a bundle for production.\n\n**Input → Model → Output.**\n\n| | What it is in this notebook |\n|---|---|\n| Input | an adapter bundle produced outside this run — by default the pinned sample bundle (`SAMPLE_ARTIFACT`, made by the fine-tuning notebook), or your own ZIP or folder with `USE_OWN_ARTIFACT` — and new prompts typed into the form |\n| Model | the pinned SmolLM2-360M-Instruct base from the digest-verified snapshot, with the bundle's LoRA deltas attached in eval mode (no training) |\n| Output | the verified bundle provenance, base versus adapted answers (greedy) with an adapter-activity readout (next-token score delta, which answers differ), two seeded sampled answers, an input manifest, an evaluation report whose verdict is `not-measurable`, a result JSON and a generations CSV |\n\n**How to use this notebook.** Choose **Runtime → Change runtime type → T4 GPU** (CPU also works), then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed. Sections 1–3 are **infrastructure** — the isolated environment, the carried pipeline module and the pinned base snapshot — and their cells are collapsed. The learning path starts in Section 4, where the bundle arrives: by default the pinned sample bundle (nothing to upload); set `USE_OWN_ARTIFACT = True` and `ARTIFACT_ZIP_PATH` (or `ARTIFACT_DIR`, or leave both empty on Colab for the upload dialog) to verify your own. Form fields (`# @param`) are the knobs; after a full run you can change `CUSTOM_PROMPT` or switch on the Section 8 exercise and re-run that cell alone. Before each principal result the notebook asks you to **Predict**; after it come **What to notice** and a collapsible **Check your reasoning**. No passing clean-run record exists yet for this notebook (`tutorials/RELEASE_VERIFICATION.md`), so the worked answers state what the code guarantees and what to expect qualitatively, not recorded numbers. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end.\n\n**Roadmap:** 1–3 infrastructure → 4 obtain the bundle and verify it before anything is loaded *(core concept: the trust boundary)* → 5 attach the adapter to the verified base → 6 validate new prompts into an input manifest → 7 adapter off versus on with an activity readout, greedy versus seeded sampling, and why nothing is measured *(evaluation practice)*; export *(engineering)* → 8 optional exercise: see a tampered copy refused → interpretation, troubleshooting, glossary, conclusion."
            )
        ]
    },
    "carrier_note": (
        "**What matters for this notebook.** Of everything the module carries, this notebook exercises five things: `verify_snapshot` / "
        "`stage_missing_files` (Section 3), `extract_zip_safely` and `verify_artifact_bundle` (Section 4), `LanguageModelPipeline.load_adapter` "
        "(Section 5), `validate_prompts` (Section 6) and `generate` with `evaluation_report` (Section 7). The dataset normalisation, loss masking, "
        "training registry and `manufacture_validation` code belongs to the fine-tuning notebook and never runs here; you may skip it."
    ),
    "title": "Language-Model Adapter — DIMER artifact inference tutorial (standalone)",
    "badges": [
        badge
        if badge[0] != "Open In Colab"
        else (
            badge[0],
            badge[1],
            f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/language_model_artifact_inference_colab.ipynb",
        )
        for badge in BADGES
    ],
    "capability": "consume an externally supplied PEFT adapter bundle, verify its manifest and provenance against the pinned `HuggingFaceTB/SmolLM2-360M-Instruct` base snapshot, attach it, and generate text for new prompts",
    "intro": (
        "Someone hands you an adapter ZIP and says \"this is the fine-tuned model\". It is not a model; it is a **PEFT "
        "adapter**: a few million low-rank delta weights that only mean something on top of one specific base model at "
        "one specific revision. This notebook consumes a bundle produced **outside this execution** (for example by the "
        "E2E tutorial in a separate session), treats it as untrusted input, verifies every file against the bundle's "
        "manifest and its provenance against the carried module's pinned identity, attaches it to the digest-verified base "
        "snapshot, and generates with the adapter switched off and on. No training happens and **no artifact is created "
        "here**. The carried module supplies `extract_zip_safely`, `verify_artifact_bundle`, `LanguageModelPipeline.load_adapter`, "
        "`validate_prompts` and `evaluation_report`.\n\n"
        "**Trust boundary.** Manifest and digest checks establish that the bundle is internally consistent and names this "
        "pipeline's base model; they do not authenticate the sender. The bundle format carries no pickle and no Python: "
        "adapter weights are `safetensors`, the tokenizer and configs are JSON/text, `trust_remote_code` is never enabled, "
        "and the base weights are acquired separately in Section 3 and digest-verified before use. Use only bundles from "
        "a producer you trust, and paste the whole-archive SHA-256 they gave you into `EXPECTED_ARTIFACT_ZIP_SHA256`."
    ),
    "learning_objectives": (
        "by the end you can: name the checks a bundle passes before any model state is deserialised (archive path safety, "
        "whole-archive and per-file digests, format, provenance, pinned base identity) and say which of them a tampered or "
        "mismatched bundle fails; read a bundle's provenance and state which base model and revision it was trained on, from "
        "what data and with which hyperparameters; attach the adapter to the digest-verified base and show that it has "
        "non-zero weights and no network fallback; validate new prompts into an input manifest and explain a recorded "
        "rejection; compare adapter-off and adapter-on answers under the greedy decoding rule and read the adapter-activity "
        "readout (next-token score delta, which answers differ) to judge whether the adapter did anything; explain why the "
        "evaluation report is `not-measurable` and what would make it measurable; and export machine-readable results with "
        "provenance."
    ),
    "exclusions": (
        "artifact creation, fine-tuning of any kind, merging the adapter into the base weights, serving, quality claims "
        "of any kind (without labelled prompts nothing is measured), and any base model other than the pinned "
        "`smollm2-360m` snapshot — a bundle trained on another base or revision is refused, not adapted."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). A CUDA device is recommended (the base loads in 4-bit `nf4`, as it was trained); on CPU the module loads the base in float32, which is slower but works for a few prompts. Section 1 builds a separate environment from the hash-locked pins (nothing is installed into the notebook's own Python, so no restart is needed); its PyTorch CUDA wheels are the largest download of the run.",
        "- **Artifact:** an externally produced adapter bundle. The default is the trusted sample bundle pinned in Section 4 (`SAMPLE_ARTIFACT`: a bundle the fine-tuning notebook wrote as `outputs/language_model_finetuning_adapter_bundle.zip` in a recorded run, published at an immutable location with its whole-archive SHA-256); with `USE_OWN_ARTIFACT = True` it is your own bundle, as a ZIP path or unpacked folder already in the runtime (Kaggle, Jupyter) or through the Colab upload dialog. Nothing in this notebook manufactures it.",
        "- **Data:** new prompts typed into the form (two Filipino prompts are prefilled). No dataset is bundled, because scoring self-generated rows would not be external-artifact evidence. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Obtain the external adapter bundle and verify it before any model state is loaded\n\n"
                "By default the cell reads the **trusted sample bundle** pinned in `SAMPLE_ARTIFACT`: one bundle produced by the "
                "fine-tuning notebook in a recorded run, published at an immutable location together with its whole-archive SHA-256 "
                "and producer provenance. The ZIP is downloaded (or reused if already present), its SHA-256 is compared with the pinned "
                "digest **before extraction**, and it is then extracted with `extract_zip_safely` and verified exactly as an uploaded "
                "bundle would be. Nothing is uploaded and `google.colab` is not imported on this path. The pinned bundle is release asset "
                "`sample-bundle-v1` of this repository, written by the fine-tuning notebook in a recorded Colab T4 run; the printed "
                "`producer` names the notebook blob, commit and run. If you empty the slot, the cell uses "
                "`outputs/language_model_finetuning_adapter_bundle.zip` when the fine-tuning notebook wrote it in this runtime, and "
                "otherwise stops with a message saying what to set.\n\n"
                "To verify **your own** bundle, set `USE_OWN_ARTIFACT = True` and give `ARTIFACT_ZIP_PATH` (a bundle ZIP already in "
                "the runtime: Kaggle, Jupyter), or `ARTIFACT_DIR` (an unpacked bundle folder), or leave both empty on Colab to get the "
                "upload dialog. If the sender gave you the whole-archive SHA-256, paste it into `EXPECTED_ARTIFACT_ZIP_SHA256`: a "
                "substituted archive then fails before extraction. The folder route skips the archive checks (there is no archive), "
                "so the cell says so, and it refuses a digest paired with a folder rather than silently ignoring it. Every archive is "
                "extracted with `extract_zip_safely`: every member must be a relative path without `..`, a leading `/` or backslashes "
                "that resolves inside the extraction folder; symlinks are refused; the archive may not expand past 512 MiB; "
                "`extractall` is never used.\n\n"
                "`verify_artifact_bundle` then runs **before** any state is deserialised: exactly the manifest format the "
                "module writes, every listed file present with its recorded size and SHA-256, no file on disk the manifest "
                "does not list, the required members (`adapter_config.json`, `adapter_model.safetensors`, `provenance.json`, "
                "the tokenizer), `trustRemoteCode` false, a 40-character `baseModelRevision` (a branch name like `main` is "
                "rejected — the Hub can move it), and the base model and revision equal to the ones carried by the module. "
                "A mismatch stops the notebook: an adapter attached to weights it was not trained on produces confident "
                "nonsense rather than an error. The cell prints where the bundle came from (`artifact_source`), the archive "
                "digest, and the provenance a consumer needs: format, base model and revision, dataset digest and licence, "
                "training hyperparameters and the producer's runtime.\n\n"
                "**Predict:** a bundle trained on a different base model or revision arrives. At which step does the notebook stop it? "
                "And if one byte of the pinned sample digest were wrong, would anything be extracted?\n\n"
                "<details><summary>Check your reasoning</summary>\n\n"
                "At `verify_artifact_bundle`, in this cell, before any weight is deserialised: the provenance's `baseModel` and "
                "`baseModelRevision` must equal the identity carried by the module, otherwise it refuses to attach. Digests "
                "prove the bundle is internally consistent; they do not prove who made it. A wrong pinned digest stops the cell at "
                "`Whole-ZIP SHA-256 mismatch`, before `extract_zip_safely` runs, so nothing is written to `work/external-artifact`.\n\n"
                "</details>"
            ),
            "code": (
                "import shutil\n"
                "import urllib.request\n\n"
                "USE_OWN_ARTIFACT = False  # @param {{type:\"boolean\"}}\n"
                "# Your own bundle (USE_OWN_ARTIFACT = True): a bundle ZIP already in the runtime, an unpacked bundle folder, or (both empty, on Colab) the upload dialog.\n"
                "ARTIFACT_ZIP_PATH = ''  # @param {{type:\"string\"}}\n"
                "ARTIFACT_DIR = ''  # @param {{type:\"string\"}}\n"
                "EXPECTED_ARTIFACT_ZIP_SHA256 = ''  # @param {{type:\"string\"}}\n"
                "# The trusted sample artifact: one bundle produced by the fine-tuning notebook in a recorded hosted run, published at an\n"
                "# immutable location (a pinned release asset), with its whole-archive SHA-256 and producer provenance (notebook blob,\n"
                "# repository commit, runtime) recorded here (release sample-bundle-v1); the digest is checked before extraction.\n"
                "SAMPLE_ARTIFACT = {{'url': 'https://github.com/kurtvalcorza/language-model-pipeline/releases/download/sample-bundle-v1/language_model_finetuning_adapter_bundle.zip', 'sha256': 'f99348aab5537bbccc26a0283f0fd6e714c0e3b58d27beb8c3d2586951ffaa35', 'producer': {{'notebook': 'tutorials/language_model_finetuning_colab.ipynb', 'notebook_blob': '8a2c6ad4886d1947f3b2fb8e8e33b3eb7e1317a6', 'commit': 'd185817b4a45049d51cbfa752737e31bf82b0762', 'run': '2026-10-07 Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4, default path', 'evidence': 'docs/verification/2026-10-07-colab-t4/language_model_finetuning_colab/', 'release': 'sample-bundle-v1', 'asset_id': 619976907}}}}\n"
                "E2E_BUNDLE = Path('outputs') / 'language_model_finetuning_adapter_bundle.zip'  # same-runtime fallback if the slot is emptied\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "Path('work').mkdir(exist_ok=True)\n\n\n"
                "def bundle_from_zip(archive_path, expected_sha, source):\n"
                "    \"\"\"Whole-archive digest check (when a digest is known) BEFORE extraction, then safe extraction; the manifest check follows.\"\"\"\n"
                "    archive_sha = sha256_of_file(archive_path)\n"
                "    if expected_sha and archive_sha.lower() != expected_sha.strip().lower():\n"
                "        raise ValueError('Whole-ZIP SHA-256 mismatch: this is not the archive you were told to expect')\n"
                "    print(f'archive {{archive_path.name}}: {{archive_path.stat().st_size / 1024**2:.1f}} MB, sha256 {{archive_sha}}')\n"
                "    extraction_root = extract_zip_safely(archive_path, Path('work') / 'external-artifact', size_limit_bytes=512 * 1024**2)\n"
                "    manifests = list(extraction_root.rglob(ARTIFACT_MANIFEST_NAME))\n"
                "    if len(manifests) != 1:\n"
                "        raise ValueError(f'Expected exactly one {{ARTIFACT_MANIFEST_NAME}} in the archive')\n"
                "    return manifests[0].parent, archive_sha, source\n\n\n"
                "if USE_OWN_ARTIFACT:\n"
                "    if ARTIFACT_ZIP_PATH.strip():\n"
                "        zip_path = Path(ARTIFACT_ZIP_PATH.strip()).expanduser()\n"
                "        if not zip_path.is_file():\n"
                "            raise FileNotFoundError(f'ARTIFACT_ZIP_PATH {{str(zip_path)!r}} is not a file: give the adapter bundle ZIP, or set ARTIFACT_DIR to an unpacked bundle folder')\n"
                "        bundle_dir, archive_sha, artifact_source = bundle_from_zip(zip_path, EXPECTED_ARTIFACT_ZIP_SHA256, f'zip: {{zip_path}}')\n"
                "    elif ARTIFACT_DIR.strip():\n"
                "        if EXPECTED_ARTIFACT_ZIP_SHA256.strip():\n"
                "            raise ValueError('EXPECTED_ARTIFACT_ZIP_SHA256 is set but ARTIFACT_DIR names an unpacked folder, so no archive would be digest-checked: set ARTIFACT_ZIP_PATH to the ZIP instead, or clear the digest')\n"
                "        bundle_dir = Path(ARTIFACT_DIR.strip()).expanduser()\n"
                "        if not bundle_dir.is_dir():\n"
                "            raise FileNotFoundError(f'ARTIFACT_DIR {{str(bundle_dir)!r}} is not a folder: give the folder that holds the bundle (with {{ARTIFACT_MANIFEST_NAME}}), or set ARTIFACT_ZIP_PATH')\n"
                "        print('Notice: ARTIFACT_DIR is an unpacked folder, so the archive checks (safe extraction, whole-ZIP digest) do not apply; the manifest, provenance and base-identity checks still run.')\n"
                "        artifact_source, archive_sha = f'directory: {{bundle_dir}}', None\n"
                "    else:\n"
                "        try:\n"
                "            from google.colab import files\n"
                "        except ImportError:\n"
                "            raise RuntimeError('USE_OWN_ARTIFACT = True but ARTIFACT_ZIP_PATH and ARTIFACT_DIR are empty and this runtime has no Colab upload dialog: set ARTIFACT_ZIP_PATH to the adapter bundle ZIP') from None\n"
                "        uploaded = files.upload() or {{}}\n"
                "        if len(uploaded) != 1:\n"
                "            raise ValueError(f'Upload exactly one adapter bundle ZIP; got {{len(uploaded)}} ({{sorted(uploaded) or \"upload cancelled or empty\"}})')\n"
                "        archive_path = Path('work') / Path(next(iter(uploaded))).name\n"
                "        archive_path.write_bytes(next(iter(uploaded.values())))\n"
                "        bundle_dir, archive_sha, artifact_source = bundle_from_zip(archive_path, EXPECTED_ARTIFACT_ZIP_SHA256, 'upload dialog')\n"
                "elif SAMPLE_ARTIFACT['url']:\n"
                "    archive_path = Path('work') / 'sample-adapter-bundle.zip'\n"
                "    if not (archive_path.is_file() and sha256_of_file(archive_path) == SAMPLE_ARTIFACT['sha256']):\n"
                "        with urllib.request.urlopen(SAMPLE_ARTIFACT['url'], timeout=120) as response, open(archive_path, 'wb') as handle:\n"
                "            shutil.copyfileobj(response, handle)\n"
                "    print({{'sample_artifact': SAMPLE_ARTIFACT['url'], 'pinned_sha256': SAMPLE_ARTIFACT['sha256'], 'producer': SAMPLE_ARTIFACT['producer']}})\n"
                "    bundle_dir, archive_sha, artifact_source = bundle_from_zip(archive_path, SAMPLE_ARTIFACT['sha256'], f\"sample artifact: {{SAMPLE_ARTIFACT['url']}}\")\n"
                "elif E2E_BUNDLE.is_file():\n"
                "    print(f'SAMPLE_ARTIFACT is empty; using the fine-tuning notebook bundle found in this runtime at {{E2E_BUNDLE}}')\n"
                "    bundle_dir, archive_sha, artifact_source = bundle_from_zip(E2E_BUNDLE, EXPECTED_ARTIFACT_ZIP_SHA256, f'same-runtime E2E output: {{E2E_BUNDLE}}')\n"
                "else:\n"
                "    raise RuntimeError('No trusted sample artifact is pinned (SAMPLE_ARTIFACT is empty) and no fine-tuning bundle exists at outputs/language_model_finetuning_adapter_bundle.zip in this runtime. To verify your own bundle, set USE_OWN_ARTIFACT = True and ARTIFACT_ZIP_PATH (Kaggle, Jupyter) or use the upload dialog (Colab).')\n"
                "artifact_manifest, provenance = verify_artifact_bundle(bundle_dir)\n"
                "print({{'artifact_source': artifact_source, 'zip_sha256': archive_sha, 'format': artifact_manifest['format'], 'formatVersion': artifact_manifest['formatVersion'], 'files': len(artifact_manifest['files']), 'total_bytes': artifact_manifest['totalBytes']}})\n"
                "print({{'baseModel': provenance['baseModel'], 'baseModelRevision': provenance['baseModelRevision'], 'baseModelLicense': provenance.get('baseModelLicense'), 'quantized': provenance.get('quantized'), 'datasetDigest': provenance.get('datasetDigest'), 'dataset': provenance.get('dataset')}})\n"
                "print({{'training': provenance.get('training'), 'producer_runtime': provenance.get('runtime')}})"
            ),
        },
        {
            "md": (
                "## 5. Attach the adapter to the verified base model\n\n"
                "`pipe.load_adapter` re-verifies the bundle, loads the tokenizer **from the bundle** (so prompts render with "
                "exactly the chat template the adapter saw in training), and attaches the deltas to the base model that "
                "Section 3 loaded from the digest-verified snapshot with `PeftModel.from_pretrained(..., is_trainable=False)` "
                "in eval mode. There is **no network fallback**: the only acceptable base weights are the files named by the "
                "inline manifest. The cell prints the adapter configuration and the largest absolute value in the adapter's `B` "
                "matrices (`lora_b_max_abs`), and refuses an all-zero adapter — an untrained or mis-saved adapter would be "
                "indistinguishable from the base model. A tiny but non-zero `B` passes this check; Section 7 then measures whether "
                "it actually changes anything. Re-running this section with a new bundle prints PEFT's \"multiple adapters\" "
                "warning; it is harmless — the re-attach replaces the `default` adapter — so no restart is needed.\n\n"
                "**Predict:** will the printed `r` and `lora_alpha` match the fine-tuning notebook's defaults?\n\n"
                "<details><summary>Check your reasoning</summary>\n\n"
                "For a bundle made by the fine-tuning notebook with its defaults, yes: `r` 8 and `lora_alpha` 16, on the "
                "attention and MLP projections. Whatever the values, they come from the bundle's own `adapter_config.json`, "
                "which the manifest has already digest-checked.\n\n"
                "</details>"
            ),
            "code": (
                "model = pipe.load_adapter(bundle_dir)\n"
                "lora_b_matrices = [param for name, param in model.named_parameters() if 'lora_b' in name.lower()]\n"
                "LORA_B_MAX_ABS = max(p.abs().max().item() for p in lora_b_matrices) if lora_b_matrices else 0.0\n"
                "if not lora_b_matrices or LORA_B_MAX_ABS == 0:\n"
                "    raise RuntimeError('Adapter weights are zero or missing')\n"
                "adapter_config = json.loads((Path(bundle_dir) / 'adapter_config.json').read_text(encoding='utf-8'))\n"
                "print({{'adapter_attached': True, 'r': adapter_config.get('r'), 'lora_alpha': adapter_config.get('lora_alpha'), 'target_modules': adapter_config.get('target_modules'), 'lora_b_max_abs': LORA_B_MAX_ABS, 'device': pipe.device, 'quantized_4bit': pipe.quantized, 'source': pipe.source}})"
            ),
        },
        {
            "md": (
                "## 6. Validate new prompts → input manifest\n\n"
                "`validate_prompts` is the pipeline's public validation stage for inference requests: it applies exactly the "
                "checks `generate` applies — each prompt a non-empty user turn (or a list of turns ending with one), turn "
                "content within `MAX_PROMPT_CHARS`, `max_new_tokens` within `MAX_NEW_TOKENS_CEILING`, and the rendered prompt "
                "within `MAX_SEQUENCE_LENGTH_CEILING` tokens measured with the bundle's tokenizer — and returns an **input "
                "manifest** naming the schema, ceilings, per-prompt turn/character/token counts and the verdict. It is "
                "written to `outputs/{stem}_input_manifest.json`. To show what rejection looks like, the cell also validates "
                "a blank prompt and records the pipeline's own error message as a finding. Type your own prompt into "
                "`CUSTOM_PROMPT`; the two prefilled prompts are new to the adapter (they were not in the tutorial's training "
                "sample).\n\n"
                "**Predict:** will the blank-prompt probe be accepted?\n\n"
                "<details><summary>Check your reasoning</summary>\n\n"
                "No: a prompt must be a non-empty user turn, so the probe is rejected and the pipeline's own message is "
                "recorded under `findings`, while the real prompts are accepted with their token counts.\n\n"
                "</details>"
            ),
            "code": (
                "CUSTOM_PROMPT = ''  # @param {{type:\"string\"}}\n"
                "MAX_NEW_TOKENS = 96  # @param {{type:\"integer\"}}\n"
                "PROMPTS = [\n"
                "    'Ipaliwanag sa simpleng Filipino kung ano ang machine learning.',\n"
                "    'Sumulat ng maikling payo para sa isang estudyanteng nagsisimula sa AI.',\n"
                "]\n"
                "if CUSTOM_PROMPT.strip():\n"
                "    PROMPTS.append(CUSTOM_PROMPT.strip())\n"
                "print({{'ceilings': {{'MAX_NEW_TOKENS_CEILING': MAX_NEW_TOKENS_CEILING, 'MAX_PROMPT_CHARS': MAX_PROMPT_CHARS, 'MAX_SEQUENCE_LENGTH_CEILING': MAX_SEQUENCE_LENGTH_CEILING}}, 'decoding_rule': DECODING_RULE}})\n"
                "input_manifest = validate_prompts(PROMPTS, MAX_NEW_TOKENS, token_length=pipe.prompt_token_length, names=[f'prompt-{{i}}' for i in range(len(PROMPTS))])\n"
                "# Demonstrate rejection on a blank prompt; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_prompts(['   '], MAX_NEW_TOKENS)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'blank-prompt-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "## 7. Generate with the adapter off and on, then with sampling; report what cannot be measured; export\n\n"
                "The first comparison uses **greedy decoding** (`do_sample=False`, the module's `DECODING_RULE`) so that the "
                "result is deterministic and the only difference between the two columns is the adapter itself: "
                "`model.disable_adapter()` switches it off for the base answer. A small adapter trained on ~100 rows usually "
                "changes language, tone or format rather than substance — but it may also change nothing visible on a given "
                "prompt, so the cell **measures adapter activity** instead of assuming it: the largest absolute change the "
                "adapter makes to the next-token scores of the first prompt (`next_token_logit_max_abs_delta`), and per prompt "
                "whether the greedy answers differ (`answers_differ`). Both go into the result JSON with a verdict. A delta near "
                "zero means the adapter is inactive on this base, whatever its `B` matrices contain; identical answers with a "
                "clear delta mean the adapter shifted the scores without changing the most likely token for these prompts.\n\n"
                "Greedy decoding is right for a reproducible check and wrong for a product: small models fall into repetition "
                "loops under it, so the cell re-asks the first prompt twice with the direct-answer sampling settings "
                "(`do_sample=True, temperature=0.7, top_p=0.8, top_k=20`), each draw seeded (`SEED`, `SEED + 1`) so the "
                "exported answers are reproducible on the same device and library versions; a different device or version can "
                "give different text. Sampling usually avoids greedy repetition loops; note any that remain, and note that two "
                "draws can coincide.\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and is produced even here: with no labelled "
                "prompts its verdict is `not-measurable` and it states what would make the task measurable; its `task` names "
                "adapter-attached generation (no score is computed), and `sample_kind` is `sample` for the prefilled prompts or "
                "`sample+BYOD` when you typed a `CUSTOM_PROMPT`. It is written to `outputs/{stem}_evaluation_report.json`. The "
                "result JSON records the artifact identity and digests, the provenance the bundle carried, every prompt with its "
                "base, adapted and sampled answers, the adapter-activity readout, the sampling seeds, the input manifest, the "
                "notebook's source, the model identity, licence and runtime; the CSV keeps one row per prompt and answer kind. "
                "No credentials are recorded.\n\n"
                "**Predict:** will the adapted answer be in Filipino? Longer or shorter than the base answer? Will the greedy "
                "answers differ on both prompts? Which verdict will the evaluation report give?\n\n"
                "**What to notice:** language and register in each BASE/ADAPTED pair, `answers_differ` against the printed logit "
                "delta, and any repeated phrase in the greedy answers.\n\n"
                "<details><summary>Check your reasoning</summary>\n\n"
                "For a bundle from the fine-tuning notebook, expect the adapted answer to lean towards Filipino and the register "
                "of the training rows; the base model often echoes a Filipino prompt back. The greedy pair is reproducible on the "
                "same weights and runtime, and the two seeded sampled answers are reproducible too, but usually differ from each "
                "other and from the greedy answer. `answers_differ` can be `False` on a prompt even when the logit delta is "
                "clearly non-zero (the scores moved, the top token did not); a delta near zero means the adapter did nothing. The "
                "verdict is `not-measurable`: no labelled prompts exist here, so the answers are qualitative evidence of what the "
                "adapter changes, not a score.\n\n"
                "</details>"
            ),
            "code": (
                "import csv\n\n"
                "SEED = 42  # @param {{type:\"integer\"}}\n\n\n"
                "def base_and_adapted(prompt):\n"
                "    with model.disable_adapter():\n"
                "        base_answer = pipe.generate(prompt, model=model, max_new_tokens=MAX_NEW_TOKENS)\n"
                "    return base_answer, pipe.generate(prompt, model=model, max_new_tokens=MAX_NEW_TOKENS)\n\n\n"
                "def next_token_logits(prompt):\n"
                "    \"\"\"Scores for the first generated token of the rendered prompt: the first decision greedy decoding makes.\"\"\"\n"
                "    text = pipe.render_chat([{{'role': 'user', 'content': prompt}}], add_generation_prompt=True)\n"
                "    inputs = pipe.tokenizer(text, return_tensors='pt', add_special_tokens=False).to(next(model.parameters()).device)\n"
                "    with torch.inference_mode():\n"
                "        return model(**inputs).logits[0, -1].float()\n\n\n"
                "with model.disable_adapter():\n"
                "    logits_off = next_token_logits(PROMPTS[0])\n"
                "LOGIT_MAX_ABS_DELTA = (next_token_logits(PROMPTS[0]) - logits_off).abs().max().item()\n"
                "rows = []\n"
                "for prompt in PROMPTS:\n"
                "    base_answer, adapted_answer = base_and_adapted(prompt)\n"
                "    rows.append({{'prompt': prompt, 'base': base_answer, 'adapted': adapted_answer, 'answers_differ': base_answer != adapted_answer}})\n"
                "    print(f'PROMPT:  {{prompt}}\\nBASE:    {{base_answer}}\\nADAPTED: {{adapted_answer}}\\n')\n"
                "n_differ = sum(row['answers_differ'] for row in rows)\n"
                "ADAPTER_ACTIVITY = {{\n"
                "    'lora_b_max_abs': LORA_B_MAX_ABS,\n"
                "    'next_token_logit_max_abs_delta': LOGIT_MAX_ABS_DELTA,\n"
                "    'answers_differ': [row['answers_differ'] for row in rows],\n"
                "    'verdict': ('inactive: the adapter leaves the next-token scores unchanged on the first prompt, whatever its B matrices hold' if LOGIT_MAX_ABS_DELTA < 1e-6 else f'active: the adapter moved the next-token scores of the first prompt by up to {{LOGIT_MAX_ABS_DELTA:.4g}}; the greedy answers differ on {{n_differ}} of {{len(rows)}} prompts'),\n"
                "}}\n"
                "print(ADAPTER_ACTIVITY)\n"
                "SAMPLING = {{'do_sample': True, 'temperature': 0.7, 'top_p': 0.8, 'top_k': 20}}\n"
                "sampled, sampling_seeds = [], []\n"
                "for attempt in range(2):\n"
                "    torch.manual_seed(SEED + attempt)\n"
                "    sampling_seeds.append(SEED + attempt)\n"
                "    sampled.append(pipe.generate(PROMPTS[0], model=model, max_new_tokens=MAX_NEW_TOKENS, **SAMPLING))\n"
                "for attempt, answer in enumerate(sampled, start=1):\n"
                "    print(f'sampled answer {{attempt}} (seed {{sampling_seeds[attempt - 1]}}): {{answer}}\\n')\n"
                "report = evaluation_report(None, sample_kind='sample+BYOD' if CUSTOM_PROMPT.strip() else 'sample', probes=rows, task=INFERENCE_TASK, score_semantics=INFERENCE_SCORE_SEMANTICS)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "payload = {{\n"
                "    'artifact': {{'source': artifact_source, 'bundle_dir': str(bundle_dir), 'zip_sha256': archive_sha, 'manifest': artifact_manifest, 'provenance': provenance, 'adapter_config': adapter_config}},\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'generations': rows,\n"
                "    'adapter_activity': ADAPTER_ACTIVITY,\n"
                "    'sampled': {{'prompt': PROMPTS[0], 'settings': SAMPLING, 'seeds': sampling_seeds, 'answers': sampled}},\n"
                "    'inference': {{'decoding_rule': DECODING_RULE, 'max_new_tokens': MAX_NEW_TOKENS}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': importlib.metadata.version('peft'), 'device': pipe.device, 'quantized_4bit': pipe.quantized, 'compute_dtype': pipe.compute_dtype}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "with open('outputs/{stem}_generations.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['prompt', 'kind', 'answer'])\n"
                "    for row in rows:\n"
                "        writer.writerow([row['prompt'], 'base-greedy', row['base']])\n"
                "        writer.writerow([row['prompt'], 'adapted-greedy', row['adapted']])\n"
                "    for answer in sampled:\n"
                "        writer.writerow([PROMPTS[0], 'adapted-sampled', answer])\n"
                "print(json.dumps({{k: v for k, v in report.items() if k != 'probes'}}, indent=2))\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
        {
            "md": (
                "## 8. Exercise (optional): see a tampered copy of the bundle refused\n\n"
                "Section 4 described the refusals; this cell lets you watch one. It is **off by default** so that Run all is "
                "unchanged. Set `RUN_TAMPER_EXERCISE = True` and re-run the cell: it copies the verified bundle, flips one bit in "
                "the last byte of `adapter_model.safetensors` (same size, different content), runs `verify_artifact_bundle` on the "
                "copy and prints the refusal. The bundle in use is not touched, and the cell re-verifies it afterwards.\n\n"
                "**Predict:** which message will the tampered copy produce — `Manifest/file-set mismatch`, `SHA-256 mismatch: "
                "<file>`, or `refusing to attach`? Would changing the file's *size* produce a different one?\n\n"
                "<details><summary>Check your reasoning</summary>\n\n"
                "`SHA-256 mismatch: adapter_model.safetensors`: the file is still listed and still the recorded size, but its digest "
                "no longer matches the manifest entry. A size change fails the same per-file check (size and digest are compared "
                "together); adding or removing a file fails `Manifest/file-set mismatch`; a different base revision in "
                "`provenance.json` fails `refusing to attach` — unless the digest check catches the edited provenance file first, "
                "which it does, because the manifest covers `provenance.json` too.\n\n"
                "</details>"
            ),
            "code": (
                "RUN_TAMPER_EXERCISE = False  # @param {{type:\"boolean\"}}\n\n"
                "if RUN_TAMPER_EXERCISE:\n"
                "    tampered = Path('work') / 'tampered-bundle'\n"
                "    shutil.rmtree(tampered, ignore_errors=True)\n"
                "    shutil.copytree(bundle_dir, tampered)\n"
                "    target = tampered / 'adapter_model.safetensors'\n"
                "    data = bytearray(target.read_bytes())\n"
                "    data[-1] ^= 0x01  # one bit of the last byte: same size, different content\n"
                "    target.write_bytes(bytes(data))\n"
                "    try:\n"
                "        verify_artifact_bundle(tampered)\n"
                "    except ValueError as exc:\n"
                "        print(f'Refused, as it should be: {{exc}}')\n"
                "    else:\n"
                "        raise RuntimeError('The tampered copy was accepted: the per-file digest check did not fire')\n"
                "    print({{'bundle_in_use_still_verifies': verify_artifact_bundle(bundle_dir)[0]['totalBytes'] == artifact_manifest['totalBytes']}})\n"
                "else:\n"
                "    print('RUN_TAMPER_EXERCISE is off: set it to True and re-run this cell to watch a one-bit change in the bundle be refused.')"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "A successful run proves that an independently supplied adapter bundle is internally consistent, names the "
        "carried module's pinned base model at its immutable revision, and attaches to the digest-verified base without any "
        "network fallback — without the repository being reachable. Whether the adapter *changed* anything is not assumed: "
        "the Section 7 readout records how far it moved the next-token scores on the first prompt and on how many prompts the "
        "greedy answers differed, and its verdict says `active` or `inactive`. Read identical BASE/ADAPTED columns next to a "
        "clear logit delta as \"the scores moved, the most likely token did not\" for those prompts; read a delta near zero as an "
        "adapter that does nothing on this base, which a non-zero `B` check alone cannot detect. The run does **not** "
        "authenticate the producer, and it establishes no task quality: the evaluation report says `not-measurable` because "
        "no labelled prompts exist here, the greedy answers are a reproducibility check rather than a product setting, and "
        "the seeded sampled answers are illustrations. Never bypass a failed archive, digest, "
        "provenance or base-identity check; obtain a correct bundle from a trusted producer. If base-model acquisition "
        "fails, the only acceptable weights are the files named by the inline manifest, never a substitute.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, can "
        "acquire and digest-verify the pinned base snapshot, validate and attach an external adapter bundle, validate the "
        "supplied prompts, execute the public generation path with the adapter off and on, and emit the shown "
        "machine-readable outputs in the tested runtime. It does **not** establish benchmark superiority, deployment "
        "calibration, safety for high-consequence decisions, or production fitness on an unseen domain.\n\n"
        "## Troubleshooting\n\n"
        "Section 1 stops with `This notebook needs a Linux x86_64 runtime`: use Google Colab, Kaggle or a Linux Jupyter host. "
        "`The pinned uv wheel failed its size/SHA-256 check`: run Section 1 again; if it repeats, the download is being "
        "altered. `The isolated environment's Python process exited`: the worker crashed, usually out of memory — restart "
        "the session and choose **Run all**. `No trusted sample artifact is pinned`: `SAMPLE_ARTIFACT` was emptied, so the default path has no "
        "bundle to read — restore it, or set `USE_OWN_ARTIFACT = True` with `ARTIFACT_ZIP_PATH`, or run the fine-tuning notebook in this "
        "runtime first. `… this runtime has no Colab upload dialog`, `ARTIFACT_ZIP_PATH … is not a file` or `ARTIFACT_DIR … is "
        "not a folder`: point `ARTIFACT_ZIP_PATH` at the bundle ZIP. `EXPECTED_ARTIFACT_ZIP_SHA256 is set but ARTIFACT_DIR names "
        "an unpacked folder`: a folder has no archive to digest-check; give the ZIP or clear the digest. `Upload exactly one "
        "adapter bundle ZIP`: the upload was cancelled, empty or held several files. On a CPU runtime Section 3 loads the base in "
        "float32 and each answer takes seconds rather than a fraction of a second; that is expected. Re-running Section 5 with a "
        "new bundle prints PEFT's \"multiple adapters\" warning; the re-attach replaces the previous adapter and needs no restart.\n\n"
        "When a bundle check fails: `Whole-ZIP SHA-256 mismatch` — the archive is not the one whose digest you were given; get "
        "it again, do not \"fix\" the expected hash. `SHA-256 mismatch: <file>` — a file inside the bundle differs from its "
        "manifest entry; the bundle was altered or corrupted in transit. `Manifest/file-set mismatch` — files were added or "
        "removed. `refusing to attach` — the adapter was trained on a different base model or revision than this pipeline "
        "pins; use the matching pipeline. `40-character commit SHA` — the producer recorded a branch name; the bundle is "
        "not reproducibly attributable.\n\n"
        "## Change one thing (next experiments)\n\n"
        "Switch on the Section 8 exercise and watch a one-bit change refused; change `SEED` and see which sampled answers "
        "change (and whether a greedy answer does — it should not); compare the greedy adapted answer with the sampled ones and "
        "note which loops; type a prompt in English into `CUSTOM_PROMPT` and see whether the adapter still answers in Filipino "
        "and whether `answers_differ` flips; hand the same prompts and a labelled answer key to your own evaluation to obtain a "
        "measurable verdict.\n\n"
        "## Glossary\n\n"
        "- **Adapter (PEFT / LoRA):** small low-rank delta weights that only mean something on top of one base model at one revision.\n"
        "- **Adapter bundle:** the adapter weights, config, tokenizer, metrics and provenance plus `artifact-manifest.json`.\n"
        "- **Manifest:** the list of every bundle file with its byte size and SHA-256; any extra, missing or altered file is refused.\n"
        "- **Provenance:** who-made-it-from-what facts carried by the bundle: base model and revision, dataset digest and licence, hyperparameters, producer runtime.\n"
        "- **Trust boundary:** the line between checks that prove consistency (digests, identity) and the trust you place in the producer, which no check provides.\n"
        "- **Greedy decoding:** always the most likely next token; reproducible, used for the off/on comparison.\n"
        "- **Sampled decoding:** drawing the next token from the distribution (`temperature`, `top_p`, `top_k`); what a product would use; seeded here so the export is reproducible.\n"
        "- **Adapter activity:** the largest change the adapter makes to the next-token scores, plus whether each greedy answer changed; the evidence that the adapter did something.\n"
        "- **Trusted sample artifact:** the bundle pinned in `SAMPLE_ARTIFACT` by location and whole-archive SHA-256, so the default path needs no upload.\n"
        "- **Input manifest:** the JSON record of which prompts were validated, against which ceilings, with what verdict.\n"
        "- **Isolated environment:** the separate hash-locked Python environment built in Section 1; every later cell runs there.\n\n"
        "## Conclusion (your notes)\n\n"
        "1. Which checks ran before any adapter weight was loaded, and what does each one rule out?\n"
        "2. What did the adapter-activity readout say, and what did the adapter change in the answers? How sure can you be without labelled prompts?\n"
        "3. Which of your predictions were wrong?\n"
        "4. What would you ask the bundle's producer for before using it in production?\n\n"
        "**Your notes:**\n\n"
        "## References\n\n"
        f"- Repository README: https://github.com/kurtvalcorza/{REPO}/blob/main/README.md\n"
        f"- Repository model card: https://github.com/kurtvalcorza/{REPO}/blob/main/weights/smollm2-360m/MODEL_CARD.md\n"
        f"- Weight provenance: https://github.com/kurtvalcorza/{REPO}/blob/main/weights/README.md\n"
        f"- E2E companion (produces the bundle): https://github.com/kurtvalcorza/{REPO}/blob/main/tutorials/language_model_finetuning_colab.ipynb\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/huggingface/smollm\n"
        "- PEFT: https://github.com/huggingface/peft"
    ),
}
