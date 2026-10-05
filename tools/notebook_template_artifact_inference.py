"""Companion template for tools/build_notebook.py — ARTIFACT-INFERENCE (NOTEBOOK_SPEC 1.1 §3.6, §18).

Generate with ``python tools/build_notebook.py --template tools/notebook_template_artifact_inference.py``
(the generator resolves ``--out`` to ``tutorials/<notebook_name>``). The notebook carries the same pipeline
module and the same pinned base snapshot as the E2E notebook; it consumes an adapter bundle produced by a
*separate* execution (upload, or an explicit directory for non-interactive executors) and never creates one.
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
        "**Known NOTEBOOK_SPEC 2.0 gap (§19, SART1/RUN2):** the default path does not yet obtain a trusted sample adapter bundle automatically — with `ARTIFACT_DIR` empty, Section 4 opens an upload dialog for a bundle produced by the E2E tutorial; an executor sets `ARTIFACT_DIR` to a directory already in the runtime to skip the dialog. Until a published sample bundle is wired in, this notebook is a `Candidate`, not release-grade. Once the bundle is present, **Run all** builds an isolated environment from the hash-locked pins (nothing is installed into the notebook's own Python, so no restart is needed), stages and digest-verifies the pinned base snapshot, verifies the bundle manifest before any state is deserialised, attaches the adapter to the verified base, validates new prompts into an input manifest, generates with the adapter off and on and with sampling, writes the evaluation report, and exports outputs and provenance — all inside this runtime, with no DIMER worker or service and no credential."
    ),
    "byod": (
        "New-input BYOD is the `CUSTOM_PROMPT` form field in Section 6 (empty by default): your own prompt passes through the same validation, generation and export cells as the sample prompts. A user-supplied adapter bundle is the separate optional `ARTIFACT_DIR`/upload branch in Section 4, verified by `verify_artifact_bundle` before deserialisation. Uploads stay inside this runtime; do not upload confidential or restricted data unless you are authorised to process it here."
    ),
    "guided": {
        "opening": [
            (
                "**Who this notebook is for.** A learner (or a reviewer receiving someone else's fine-tune) who knows basic Python, has run a Colab notebook, and wants to use an adapter bundle safely: check what it is, check that it belongs to this base model, attach it, and see what it changes — without training anything. It is the companion of the fine-tuning notebook, which produces the bundle in a separate session. *Adapter*, *manifest*, *provenance*, *greedy* and *sampled decoding* are explained where they first matter and again in the **Glossary**. A T4 GPU is recommended; CPU works for a few prompts. The intended audience is learners and reviewers; a passing run does not certify a bundle for production.\n\n**Input → Model → Output.**\n\n| | What it is in this notebook |\n|---|---|\n| Input | an adapter bundle ZIP or folder produced outside this run (for example `language_model_finetuning_adapter_bundle.zip`), and new prompts typed into the form |\n| Model | the pinned SmolLM2-360M-Instruct base from the digest-verified snapshot, with the bundle's LoRA deltas attached in eval mode (no training) |\n| Output | the verified bundle provenance, base versus adapted answers (greedy) and two sampled answers, an input manifest, an evaluation report whose verdict is `not-measurable`, a result JSON and a generations CSV |\n\n**How to use this notebook.** Choose **Runtime → Change runtime type → T4 GPU** (CPU also works), then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed. Sections 1–3 are **infrastructure** — the isolated environment, the carried pipeline module and the pinned base snapshot — and their cells are collapsed. The learning path starts in Section 4, where you supply the bundle: leave `ARTIFACT_DIR` empty for the Colab upload dialog, or set it to a folder already in the runtime (Kaggle, Jupyter). Form fields (`# @param`) are the knobs. Before each principal result the notebook asks you to **Predict**; after it come **What to notice** and a collapsible **Check your reasoning**. No passing clean-run record exists yet for this notebook (`tutorials/RELEASE_VERIFICATION.md`), so the worked answers state what the code guarantees and what to expect qualitatively, not recorded numbers. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end.\n\n**Roadmap:** 1–3 infrastructure → 4 supply the bundle and verify it before anything is loaded *(core concept: the trust boundary)* → 5 attach the adapter to the verified base → 6 validate new prompts into an input manifest → 7 adapter off versus on, greedy versus sampled, and why nothing is measured *(evaluation practice)*; export *(engineering)* → interpretation, troubleshooting, glossary, conclusion."
            )
        ]
    },
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
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify the "
        "immutable base snapshot, supply an externally produced adapter bundle and verify it — archive path safety, "
        "per-file digests, format, provenance, pinned base identity — before any model state is deserialised, inspect its "
        "provenance, attach the adapter to the verified base with no network fallback, validate new prompts into an input "
        "manifest, generate with the adapter off and on under an explicit greedy decoding rule and then with sampling, "
        "produce an evaluation report that is `not-measurable` because no labelled data exists, and export machine-readable "
        "results plus provenance."
    ),
    "exclusions": (
        "artifact creation, fine-tuning of any kind, merging the adapter into the base weights, serving, quality claims "
        "of any kind (without labelled prompts nothing is measured), and any base model other than the pinned "
        "`smollm2-360m` snapshot — a bundle trained on another base or revision is refused, not adapted."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). A CUDA device is recommended (the base loads in 4-bit `nf4`, as it was trained); on CPU the module loads the base in float32, which is slower but works for a few prompts. Section 1 builds a separate environment from the hash-locked pins (nothing is installed into the notebook's own Python, so no restart is needed); its PyTorch CUDA wheels are the largest download of the run.",
        "- **Artifact:** an externally produced adapter bundle — the E2E tutorial writes `outputs/language_model_finetuning_adapter_bundle.zip` — supplied through the upload dialog, or as a directory already present in the runtime via `ARTIFACT_DIR` for non-interactive execution. Nothing in this notebook manufactures it.",
        "- **Data:** new prompts typed into the form (two Filipino prompts are prefilled). No dataset is bundled, because scoring self-generated rows would not be external-artifact evidence. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Supply the external adapter bundle and verify it before any model state is loaded\n\n"
                "Leave `ARTIFACT_DIR` empty to upload the bundle ZIP; set it to a directory already in the runtime to skip "
                "the dialog (an executor places the files there). An uploaded archive is extracted with "
                "`extract_zip_safely`: every member must be a relative path without `..`, a leading `/` or backslashes that "
                "resolves inside the extraction folder; symlinks are refused; the archive may not expand past 512 MiB; "
                "`extractall` is never used. If the sender gave you the whole-archive SHA-256, paste it into "
                "`EXPECTED_ARTIFACT_ZIP_SHA256` and a substituted archive fails before extraction.\n\n"
                "`verify_artifact_bundle` then runs **before** any state is deserialised: exactly the manifest format the "
                "module writes, every listed file present with its recorded size and SHA-256, no file on disk the manifest "
                "does not list, the required members (`adapter_config.json`, `adapter_model.safetensors`, `provenance.json`, "
                "the tokenizer), `trustRemoteCode` false, a 40-character `baseModelRevision` (a branch name like `main` is "
                "rejected — the Hub can move it), and the base model and revision equal to the ones carried by the module. "
                "A mismatch stops the notebook: an adapter attached to weights it was not trained on produces confident "
                "nonsense rather than an error. The cell prints the provenance a consumer needs: format, base model and "
                "revision, dataset digest and licence, training hyperparameters and the producer's runtime. With `ARTIFACT_DIR` "
                "empty outside Colab, or a path that is not a folder, the cell stops with a message saying what to set.\n\n"
                "**Predict:** a bundle trained on a different base model or revision arrives. At which step does the notebook stop it?\n\n"
                "<details><summary>Check your reasoning</summary>\n\n"
                "At `verify_artifact_bundle`, in this cell, before any weight is deserialised: the provenance's `baseModel` and "
                "`baseModelRevision` must equal the identity carried by the module, otherwise it refuses to attach. Digests "
                "prove the bundle is internally consistent; they do not prove who made it.\n\n"
                "</details>"
            ),
            "code": (
                "ARTIFACT_DIR = ''  # @param {{type:\"string\"}}\n"
                "EXPECTED_ARTIFACT_ZIP_SHA256 = ''  # @param {{type:\"string\"}}\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if ARTIFACT_DIR:\n"
                "    bundle_dir = Path(ARTIFACT_DIR).expanduser()\n"
                "    if not bundle_dir.is_dir():\n"
                "        raise FileNotFoundError(f'ARTIFACT_DIR {{str(bundle_dir)!r}} is not a folder: give the folder that holds the bundle (with {{ARTIFACT_MANIFEST_NAME}}), or leave it empty to upload the ZIP on Colab')\n"
                "    artifact_source = f'directory: {{bundle_dir}}'\n"
                "    archive_sha = None\n"
                "else:\n"
                "    try:\n"
                "        from google.colab import files\n"
                "    except ImportError:\n"
                "        raise RuntimeError('ARTIFACT_DIR is empty and this runtime has no Colab upload dialog: set ARTIFACT_DIR to the folder holding the adapter bundle') from None\n"
                "    uploaded = files.upload() or {{}}\n"
                "    if len(uploaded) != 1:\n"
                "        raise ValueError(f'Upload exactly one adapter bundle ZIP; got {{len(uploaded)}} ({{sorted(uploaded) or \"upload cancelled or empty\"}})')\n"
                "    archive_path = Path('work') / Path(next(iter(uploaded))).name\n"
                "    archive_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    archive_path.write_bytes(next(iter(uploaded.values())))\n"
                "    archive_sha = sha256_of_file(archive_path)\n"
                "    if EXPECTED_ARTIFACT_ZIP_SHA256 and archive_sha.lower() != EXPECTED_ARTIFACT_ZIP_SHA256.strip().lower():\n"
                "        raise ValueError('Whole-ZIP SHA-256 mismatch: this is not the archive you were told to expect')\n"
                "    print(f'archive {{archive_path.name}}: {{archive_path.stat().st_size / 1024**2:.1f}} MB, sha256 {{archive_sha}}')\n"
                "    extraction_root = extract_zip_safely(archive_path, Path('work') / 'external-artifact', size_limit_bytes=512 * 1024**2)\n"
                "    manifests = list(extraction_root.rglob(ARTIFACT_MANIFEST_NAME))\n"
                "    if len(manifests) != 1:\n"
                "        raise ValueError(f'Expected exactly one {{ARTIFACT_MANIFEST_NAME}} in the archive')\n"
                "    bundle_dir = manifests[0].parent\n"
                "    artifact_source = 'upload dialog'\n"
                "artifact_manifest, provenance = verify_artifact_bundle(bundle_dir)\n"
                "print({{'artifact_source': artifact_source, 'format': artifact_manifest['format'], 'formatVersion': artifact_manifest['formatVersion'], 'files': len(artifact_manifest['files']), 'total_bytes': artifact_manifest['totalBytes']}})\n"
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
                "inline manifest. The cell prints the adapter configuration and confirms that the adapter's `B` matrices are "
                "not all zero — an untrained or mis-saved adapter would be indistinguishable from the base model.\n\n"
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
                "if not lora_b_matrices or max(p.abs().max().item() for p in lora_b_matrices) == 0:\n"
                "    raise RuntimeError('Adapter weights are zero or missing')\n"
                "adapter_config = json.loads((Path(bundle_dir) / 'adapter_config.json').read_text(encoding='utf-8'))\n"
                "print({{'adapter_attached': True, 'r': adapter_config.get('r'), 'lora_alpha': adapter_config.get('lora_alpha'), 'target_modules': adapter_config.get('target_modules'), 'device': pipe.device, 'quantized_4bit': pipe.quantized, 'source': pipe.source}})"
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
                "The first comparison uses **greedy decoding** (`do_sample=False`, the module's `DECODING_RULE`) and "
                "non-thinking mode so that the result is deterministic and the only difference between the two columns is "
                "the adapter itself: `model.disable_adapter()` switches it off for the base answer. Expect the answers to be "
                "similar in substance and to differ in language, tone or format — that is what a small adapter trained on "
                "~100 rows changes. Greedy decoding is right for a reproducible check and wrong for a product: small models "
                "fall into repetition loops under it, so the cell re-asks the first prompt twice with the direct-answer "
                "sampling settings (`do_sample=True, temperature=0.7, top_p=0.8, top_k=20`); each sampled answer differs and "
                "none should loop.\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and is produced even here: with no labelled "
                "prompts its verdict is `not-measurable` and it states what would make the task measurable; it is written "
                "to `outputs/{stem}_evaluation_report.json`. The result JSON records the artifact identity and digests, the "
                "provenance the bundle carried, every prompt with its base, adapted and sampled answers, the input "
                "manifest, the notebook's source, the model identity, licence and runtime; the CSV keeps one row per "
                "prompt and answer kind. No credentials are recorded.\n\n"
                "**Predict:** will the two sampled answers be identical? Which verdict will the evaluation report give?\n\n"
                "**What to notice:** language and register in each BASE/ADAPTED pair, and any repeated phrase in the greedy "
                "answers.\n\n"
                "<details><summary>Check your reasoning</summary>\n\n"
                "The sampled answers differ from each other (sampling is random by design), while the greedy pair is "
                "reproducible on the same weights and runtime. The verdict is `not-measurable`: no labelled prompts exist "
                "here, so the answers are qualitative evidence of what the adapter changes, not a score.\n\n"
                "</details>"
            ),
            "code": (
                "import csv\n\n"
                "def base_and_adapted(prompt):\n"
                "    with model.disable_adapter():\n"
                "        base_answer = pipe.generate(prompt, model=model, max_new_tokens=MAX_NEW_TOKENS)\n"
                "    return base_answer, pipe.generate(prompt, model=model, max_new_tokens=MAX_NEW_TOKENS)\n\n"
                "rows = []\n"
                "for prompt in PROMPTS:\n"
                "    base_answer, adapted_answer = base_and_adapted(prompt)\n"
                "    rows.append({{'prompt': prompt, 'base': base_answer, 'adapted': adapted_answer}})\n"
                "    print(f'PROMPT:  {{prompt}}\\nBASE:    {{base_answer}}\\nADAPTED: {{adapted_answer}}\\n')\n"
                "SAMPLING = {{'do_sample': True, 'temperature': 0.7, 'top_p': 0.8, 'top_k': 20}}\n"
                "sampled = [pipe.generate(PROMPTS[0], model=model, max_new_tokens=MAX_NEW_TOKENS, **SAMPLING) for _ in range(2)]\n"
                "for attempt, answer in enumerate(sampled, start=1):\n"
                "    print(f'sampled answer {{attempt}}: {{answer}}\\n')\n"
                "report = evaluation_report(None, sample_kind='BYOD', probes=rows)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "payload = {{\n"
                "    'artifact': {{'source': artifact_source, 'bundle_dir': str(bundle_dir), 'zip_sha256': archive_sha, 'manifest': artifact_manifest, 'provenance': provenance, 'adapter_config': adapter_config}},\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'generations': rows,\n"
                "    'sampled': {{'prompt': PROMPTS[0], 'settings': SAMPLING, 'answers': sampled}},\n"
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
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "A successful run proves that an independently supplied adapter bundle is internally consistent, names the "
        "carried module's pinned base model at its immutable revision, attaches to the digest-verified base without any "
        "network fallback, and changes the model's answers when switched on — without the repository being reachable. It "
        "does **not** authenticate the producer, and it establishes no task quality: the evaluation report says "
        "`not-measurable` because no labelled prompts exist here, the greedy answers are a reproducibility check rather "
        "than a product setting, and the sampled answers are illustrations. Never bypass a failed archive, digest, "
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
        "the session and choose **Run all**. `ARTIFACT_DIR is empty and this runtime has no Colab upload dialog` or `… is "
        "not a folder`: set `ARTIFACT_DIR` to the folder holding the bundle. `Upload exactly one adapter bundle ZIP`: the "
        "upload was cancelled, empty or held several files.\n\n"
        "When a bundle check fails: `Whole-ZIP SHA-256 mismatch` — the archive is not the one whose digest you were given; get "
        "it again, do not \"fix\" the expected hash. `SHA-256 mismatch: <file>` — a file inside the bundle differs from its "
        "manifest entry; the bundle was altered or corrupted in transit. `Manifest/file-set mismatch` — files were added or "
        "removed. `refusing to attach` — the adapter was trained on a different base model or revision than this pipeline "
        "pins; use the matching pipeline. `40-character commit SHA` — the producer recorded a branch name; the bundle is "
        "not reproducibly attributable.\n\n"
        "## Change one thing (next experiments)\n\n"
        "Compare the greedy adapted answer with several sampled ones and note which loops; type a "
        "prompt in English and see whether the adapter still answers in Filipino; hand the same prompts and a labelled "
        "answer key to your own evaluation to obtain a measurable verdict.\n\n"
        "## Glossary\n\n"
        "- **Adapter (PEFT / LoRA):** small low-rank delta weights that only mean something on top of one base model at one revision.\n"
        "- **Adapter bundle:** the adapter weights, config, tokenizer, metrics and provenance plus `artifact-manifest.json`.\n"
        "- **Manifest:** the list of every bundle file with its byte size and SHA-256; any extra, missing or altered file is refused.\n"
        "- **Provenance:** who-made-it-from-what facts carried by the bundle: base model and revision, dataset digest and licence, hyperparameters, producer runtime.\n"
        "- **Trust boundary:** the line between checks that prove consistency (digests, identity) and the trust you place in the producer, which no check provides.\n"
        "- **Greedy decoding:** always the most likely next token; reproducible, used for the off/on comparison.\n"
        "- **Sampled decoding:** drawing the next token from the distribution (`temperature`, `top_p`, `top_k`); what a product would use.\n"
        "- **Input manifest:** the JSON record of which prompts were validated, against which ceilings, with what verdict.\n"
        "- **Isolated environment:** the separate hash-locked Python environment built in Section 1; every later cell runs there.\n\n"
        "## Conclusion (your notes)\n\n"
        "1. Which checks ran before any adapter weight was loaded, and what does each one rule out?\n"
        "2. What did the adapter change in the answers, and how sure can you be without labelled prompts?\n"
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
