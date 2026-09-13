"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier) — E2E.

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline module
(``src/lmpipeline/pipeline.py``) and the base-model pin/stage/verify cell are produced by the generator
from repository sources so they cannot drift from the package. The QLoRA training loop is deliberately
kept as plain PyTorch template cells (every step visible) and consumes only the carried module's API.
The ARTIFACT-INFERENCE companion has its own template, ``tools/notebook_template_artifact_inference.py``.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "language-model-pipeline"
BADGES = [
    (
        "GitHub",
        "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
        f"https://github.com/kurtvalcorza/{REPO}",
    ),
    (
        "Open In Colab",
        "https://colab.research.google.com/assets/colab-badge.svg",
        f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/language_model_finetuning_colab.ipynb",
    ),
    (
        "Hugging Face",
        "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-HuggingFaceTB%2FSmolLM2--360M--Instruct-ffcc4d?style=flat",
        "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct",
    ),
    (
        "Upstream",
        "https://img.shields.io/badge/Upstream-huggingface%2Fsmollm-181717?style=flat&logo=github&logoColor=white",
        "https://github.com/huggingface/smollm",
    ),
    ("arXiv", "https://img.shields.io/badge/arXiv-2502.02737-b31b1b.svg", "https://arxiv.org/abs/2502.02737"),
]

TEMPLATE = {
    "package": "lmpipeline",
    "repo_name": REPO,
    "stem": "language_model_finetuning",
    "notebook_name": "language_model_finetuning_colab.ipynb",
    "profile": "E2E",
    "pipeline_class": "LanguageModelPipeline",
    "weights_key": "smollm2-360m",
    # generator /2: only the tutorial pipeline module is carried. The rest of `lmpipeline` (DIMER
    # worker contract, registry loader, result schema) is not needed by the notebook.
    "modules": ["pipeline.py"],
    "entry_module": "pipeline.py",
    "runtime_imports": ["torch", "transformers"],
    "title": "Language-Model Fine-Tuning — DIMER E2E QLoRA tutorial (standalone)",
    "badges": BADGES,
    "capability": "supervised QLoRA fine-tuning of the pinned `HuggingFaceTB/SmolLM2-360M-Instruct` base model on chat-formatted data, with baseline comparison, adapter export and fresh reload",
    "intro": (
        "You have a few hundred conversations from your own domain (this notebook ships with Filipino Q&A pairs) and an "
        "open, instruction-tuned language model that answers in the wrong language, register or format. Full fine-tuning "
        "would need tens of gigabytes of GPU memory and a multi-gigabyte copy of the weights for every variant you try. With "
        "**QLoRA** you adapt the base model on a free Colab T4, ship a compact **PEFT adapter** instead of another full "
        "base-model copy, and prove the adapter loads cleanly before anyone deploys it.\n\n"
        "**What is trained and what is not.** The base model is frozen in 4-bit `nf4`; only low-rank adapter matrices are "
        "gradient-trained (a fraction of one percent of the parameters). No preprocessing state is fitted: the chat template "
        "and tokenizer come from the base snapshot unchanged. Every base model in the carried registry is already "
        "instruction-tuned, so fine-tuning on ~100 rows shifts *how* it answers, not *whether* it can; expect subtle changes "
        "and some forgetting if you train hard. The carried module supplies the pinned identity, snapshot verification, "
        "dataset normalisation and hygiene, assistant-only loss masking, greedy generation, the adapter-bundle contract and "
        "the `validate_inputs` / `evaluation_report` helpers; the training loop below is plain PyTorch on purpose.\n\n"
        "**Which base model.** This standalone notebook pins **one** base model, `smollm2-360m` (the smallest entry in the "
        "repository's registry, chosen so the download, baseline, training and reload cycle stay short). The other registry "
        "entries (`TUTORIAL_REGISTRY` in the carried module — Qwen3, SmolLM3, Granite, DeepSeek-R1-Distill, "
        "Qwen2.5-Coder, danube3 and the gated `llama-3.2-3b-instruct`) are listed for reference: they are exercised by the "
        "repository's DIMER worker, and switching this tutorial to one of them means regenerating it from a template that "
        "pins that key, not editing a dropdown. No Hugging Face token is needed for the pinned model."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify the "
        "immutable base snapshot, normalise instruction data from three common schemas into chat turns and validate it into "
        "an input manifest (leakage and over-length rows are rejected before training), mask the loss to assistant tokens "
        "using the model's own chat template and say exactly which tokens are supervised, record a deterministic "
        "pre-adaptation baseline, run a one-epoch QLoRA loop you can read line by line, read train/validation loss and "
        "perplexity as optimisation evidence rather than task quality, produce an evaluation report, compare base and adapted "
        "answers on new prompts, export a manifested adapter bundle, and prove it reloads against the verified base snapshot."
    ),
    "exclusions": (
        "full-parameter fine-tuning, preference tuning (RLHF/DPO), merging the adapter into 16-bit weights, task-quality "
        "evaluation (validation loss and perplexity measure format fit on a manufactured split, **not** whether answers are "
        "correct), serving, and any base model other than the pinned `smollm2-360m` snapshot."
    ),
    "prerequisites": [
        "- **Runtime:** a Colab GPU runtime (*Runtime ▸ Change runtime type ▸ T4 GPU*) or another CUDA machine; QLoRA needs `bitsandbytes` 4-bit kernels, so the training cell stops on CPU. Python 3.12 is the executed configuration. The pinned `torch==2.14.0` install is the largest download of the run.",
        "- **Knowledge:** Python functions, dictionaries and list comprehensions; what a language model does at a high level (predicts the next token). Transformer internals are not required.",
        "- **Data:** the default sample is the pinned public `jpaulpoliquit/ph-sft-ai-authored-v1` table (513 AI-authored Filipino/English Q&A rows, Apache-2.0), fetched from the Hugging Face Hub at an immutable dataset revision; `Sample: Dolly` (`databricks-dolly-15k`, CC-BY-SA 3.0 — adapters trained on it inherit the share-alike condition) is the alternative. Optional BYOD upload is gated off by default so the sample path runs top-to-bottom without interaction. Expected BYOD input: `train.jsonl` (plus optional `validation.jsonl`/`val.jsonl` and `test.jsonl`), each line a `messages`, `prompt`/`completion` or `instruction`/`input`/`output` record, uploaded as files or one ZIP. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
        "- **How to use this notebook:** cells with a form on the right (`# @param`) are the knobs; change them and re-run from that cell down. Run the cells in order the first time — each section says what the next cell does before it runs and what to look for after.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Choose the knobs and load the sample or optional BYOD\n\n"
                "The carried module pinned the base model in Section 3 and loaded it in 4-bit **NormalFloat (`nf4`)** with "
                "double quantization: a 360M model drops from ~0.7 GB of 16-bit weights to well under 0.3 GB, and the frozen "
                "weights are dequantized on the fly to the compute dtype (bfloat16 on Ampere and newer, float16 on a T4) for "
                "each matrix multiply. The cell first confirms a CUDA device is present (QLoRA needs it) and prints the "
                "registry ceilings before any data is touched.\n\n"
                "Supervised fine-tuning needs conversations, not rows. Whatever the source schema — chat `messages`, "
                "`prompt`/`completion` pairs, or Alpaca-style `instruction`/`input`/`output` records (the two shipped samples are "
                "Alpaca-shaped) — `canonical` normalises every example to `{{\"messages\": [{{role, content}}, ...]}}`. For a shipped "
                "sample the notebook walks the rows in a deterministic hash order, keeps the first `SAMPLE_LIMIT` that fit "
                "within `MAX_SEQUENCE_LENGTH` **tokens** (measured with the base tokenizer through `pipe.token_length`), and "
                "holds out one fifth as validation with `manufacture_validation`. That validation split is *manufactured from "
                "the training source*: its loss is an optimisation signal, not a task-quality measurement. BYOD files are "
                "read as JSONL; a `validation.jsonl` is used when present, otherwise a fifth of `train.jsonl` is held out. Look "
                "for the split sizes (`{{'train': 96, 'validation': 24}}` with the defaults) and the 16-character dataset "
                "digest prefix — the identity of this exact dataset."
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "DATA_SOURCE = \"Sample: Filipino SFT\"  # @param [\"Sample: Filipino SFT\",\"Sample: Dolly\"]\n"
                "SAMPLE_LIMIT = 120  # @param {{type:\"integer\"}}\n"
                "MAX_SEQUENCE_LENGTH = 512  # @param {{type:\"integer\"}}\n"
                "EPOCHS = 1  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 0.0002  # @param {{type:\"number\"}}\n"
                "LORA_RANK = 8  # @param {{type:\"integer\"}}\n"
                "LORA_ALPHA = 16  # @param {{type:\"integer\"}}\n"
                "SEED = 42  # @param {{type:\"integer\"}}\n"
                "TRAINING_METHOD = 'qlora'\n\n"
                "if not torch.cuda.is_available():\n"
                "    raise RuntimeError('No GPU visible. In Colab: Runtime > Change runtime type > T4 GPU, then rerun from the top.')\n"
                "GPU_NAME = torch.cuda.get_device_name(0)\n"
                "GPU_VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1024**3\n"
                "print({{'ceilings': {{'MAX_SEQUENCE_LENGTH_CEILING': MAX_SEQUENCE_LENGTH_CEILING, 'MAX_TOTAL_TRAIN_TOKENS': MAX_TOTAL_TRAIN_TOKENS, 'MIN_TRAIN_EXAMPLES': MIN_TRAIN_EXAMPLES, 'MAX_NEW_TOKENS_CEILING': MAX_NEW_TOKENS_CEILING}}, 'request': {{'max_sequence_length': MAX_SEQUENCE_LENGTH, 'epochs': EPOCHS, 'learning_rate': LEARNING_RATE, 'lora_rank': LORA_RANK, 'lora_alpha': LORA_ALPHA, 'seed': SEED}}}})\n"
                "print({{'gpu': GPU_NAME, 'vram_gib': round(GPU_VRAM_GB, 2), 'quantized_4bit': pipe.quantized, 'compute_dtype': pipe.compute_dtype, 'source': pipe.source}})\n\n"
                "WORK_DIR = Path('work')\n"
                "shutil.rmtree(WORK_DIR, ignore_errors=True)\n"
                "WORK_DIR.mkdir()\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    byod_root = WORK_DIR / 'byod'\n"
                "    byod_root.mkdir()\n"
                "    if len(uploaded) == 1 and next(iter(uploaded)).lower().endswith('.zip'):\n"
                "        zip_path = WORK_DIR / 'data.zip'\n"
                "        zip_path.write_bytes(next(iter(uploaded.values())))\n"
                "        extract_zip_safely(zip_path, byod_root, size_limit_bytes=2 * 1024**3)\n"
                "    else:\n"
                "        for name, payload in uploaded.items():\n"
                "            (byod_root / Path(name).name).write_bytes(payload)\n\n"
                "    def read_jsonl(path):\n"
                "        return [canonical(json.loads(line)) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]\n\n"
                "    if not (byod_root / 'train.jsonl').exists():\n"
                "        raise ValueError('BYOD requires train.jsonl')\n"
                "    if (byod_root / 'validation.jsonl').exists() and (byod_root / 'val.jsonl').exists():\n"
                "        raise ValueError('Provide either validation.jsonl or val.jsonl, not both')\n"
                "    SPLITS = {{'train': read_jsonl(byod_root / 'train.jsonl')}}\n"
                "    validation_file = byod_root / ('validation.jsonl' if (byod_root / 'validation.jsonl').exists() else 'val.jsonl')\n"
                "    if validation_file.exists():\n"
                "        SPLITS['validation'] = read_jsonl(validation_file)\n"
                "    if (byod_root / 'test.jsonl').exists():\n"
                "        SPLITS['test'] = read_jsonl(byod_root / 'test.jsonl')\n"
                "    if 'validation' not in SPLITS:\n"
                "        SPLITS = {{**SPLITS, **manufacture_validation(SPLITS['train'])}}\n"
                "    DATASET_PROVENANCE = {{'source': 'BYOD', 'usage': 'user-provided'}}\n"
                "    sample_kind = 'BYOD'\n"
                "    data_name = 'byod-jsonl'\n"
                "else:\n"
                "    from datasets import load_dataset\n\n"
                "    sample = SAMPLE_DATASETS[DATA_SOURCE]\n"
                "    all_rows = sorted((canonical(dict(row)) for row in load_dataset(sample['dataset_id'], revision=sample['revision'], split='train')), key=fingerprint)\n"
                "    rows, skipped_long = [], 0\n"
                "    for row in all_rows:\n"
                "        if len(rows) == SAMPLE_LIMIT:\n"
                "            break\n"
                "        if pipe.token_length(row) <= MAX_SEQUENCE_LENGTH:\n"
                "            rows.append(row)\n"
                "        else:\n"
                "            skipped_long += 1\n"
                "    if skipped_long:\n"
                "        print(f'Set aside {{skipped_long}} sample rows longer than {{MAX_SEQUENCE_LENGTH}} tokens while collecting {{len(rows)}}')\n"
                "    SPLITS = manufacture_validation(rows)\n"
                "    DATASET_PROVENANCE = {{'source': sample['dataset_id'], 'revision': sample['revision'], 'license': sample['license'], 'usage': 'tutorial-training-not-benchmark'}}\n"
                "    sample_kind = 'sample'\n"
                "    data_name = sample['dataset_id']\n\n"
                "DATASET_DIGEST = dataset_digest(SPLITS)\n"
                "print({{'sample_kind': sample_kind, 'name': data_name, 'splits': {{name: len(records) for name, records in SPLITS.items()}}, 'dataset_digest': DATASET_DIGEST[:16]}})\n"
                "print('First training example:')\n"
                "print(json.dumps(SPLITS['train'][0], ensure_ascii=False, indent=2)[:600])"
            ),
        },
        {
            "md": (
                "## 5. Validate the splits → input manifest, then render and mask\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks "
                "`pipe.prepare_splits` applies — canonical records with valid roles and a non-empty assistant turn, at least "
                "`MIN_TRAIN_EXAMPLES` training rows, **no identical record in two splits** (leakage is fatal), exact duplicates "
                "inside a split reported as findings but not removed, every rendered example within `MAX_SEQUENCE_LENGTH` "
                "tokens and the training total within `MAX_TOTAL_TRAIN_TOKENS` — and returns an **input manifest** naming the "
                "schema, per-split record and token counts, the dataset digest and the verdict. It is written to "
                "`outputs/{stem}_input_manifest.json`. To show what rejection looks like, the cell also validates a "
                "deliberately leaked probe (a training row copied into validation) and records the pipeline's own error "
                "message as a finding.\n\n"
                "Then `pipe.prepare_splits` renders each conversation with the tokenizer's **chat template** and masks the "
                "loss: the model must not learn to *write the user's questions*, only to *answer them*, so every token outside "
                "an assistant turn gets the label `-100` (`IGNORE_INDEX`, which PyTorch's cross-entropy ignores). This is "
                "**assistant-only loss masking**, the single most common thing to get wrong in SFT. The assistant span starts "
                "exactly where the generation prompt ends — the same prompt the model sees at inference — and runs through the "
                "end-of-turn marker, so the model learns to stop. Over-length rows are rejected, never truncated. The last "
                "print decodes the first training example and wraps every supervised run in `⟦ ⟧`: the answer text and its "
                "end-of-turn marker are inside, the user turn and role markers are not."
            ),
            "code": (
                "os.makedirs('outputs', exist_ok=True)\n"
                "input_manifest = validate_inputs(SPLITS, MAX_SEQUENCE_LENGTH, token_length=pipe.token_length, names=[f'{{data_name}}:{{name}}' for name in SPLITS])\n"
                "# Demonstrate rejection on a probe that leaks a training row into validation; the finding is recorded, not swallowed.\n"
                "probe = {{'train': SPLITS['train'], 'validation': [*SPLITS['validation'], SPLITS['train'][0]]}}\n"
                "try:\n"
                "    validate_inputs(probe, MAX_SEQUENCE_LENGTH)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'leaked-split-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest['inputs'], indent=2))\n"
                "print('findings:', input_manifest['findings'])\n\n"
                "MASKED = pipe.prepare_splits(SPLITS, MAX_SEQUENCE_LENGTH)\n"
                "token_totals = {{name: sum(len(ids) for ids, _ in examples) for name, examples in MASKED.items()}}\n"
                "supervised_totals = {{name: sum(supervised_token_count(labels) for _, labels in examples) for name, examples in MASKED.items()}}\n"
                "longest_train = max((len(ids) for ids, _ in MASKED['train']), default=0)\n"
                "if longest_train > 1024:\n"
                "    print(f'Notice: the longest training example is {{longest_train}} tokens; sequences above ~1024 raise peak VRAM noticeably.')\n"
                "print('tokens per split:', token_totals)\n"
                "print('supervised (assistant) tokens per split:', supervised_totals)\n"
                "print(show_supervision(pipe.tokenizer, *MASKED['train'][0]))"
            ),
        },
        {
            "md": (
                "## 6. Record a deterministic baseline\n\n"
                "You cannot see what training changed unless you record what the model said *before*. The two probe prompts "
                "are asked through `pipe.generate` with **greedy decoding** (`do_sample=False`, the module's `DECODING_RULE`) "
                "and non-thinking mode, so the comparison after training is deterministic given the same weights, device and "
                "library versions. `generate` puts the model in eval mode (no dropout), turns the KV cache on, and restores "
                "whatever state it found, so the same helper is safe to call in the middle of training later. Look for two "
                "short answers; on the unadapted base model expect English or mixed-language replies to the Filipino prompts."
            ),
            "code": (
                "PROMPTS = [\n"
                "    'Ipaliwanag sa simpleng Filipino kung ano ang machine learning.',\n"
                "    'Magbigay ng tatlong paraan para mabawasan ang basura sa opisina.',\n"
                "]\n"
                "print({{'decoding_rule': DECODING_RULE, 'max_new_tokens': 96, 'prompts': len(PROMPTS)}})\n"
                "BASELINE_OUTPUTS = [pipe.generate(prompt) for prompt in PROMPTS]\n"
                "for prompt, answer in zip(PROMPTS, BASELINE_OUTPUTS):\n"
                "    print(f'PROMPT: {{prompt}}\\nBASE:   {{answer}}\\n')"
            ),
        },
        {
            "md": (
                "## 7. Attach LoRA adapters and train\n\n"
                "**LoRA** keeps every original weight matrix $W_0$ frozen and learns a low-rank correction "
                "$W = W_0 + \\frac{{\\alpha}}{{r}} B A$ with $A$ of shape $r \\times d_{{in}}$ and $B$ of shape $d_{{out}} \\times r$. "
                "With rank $r = 8$ that is a few thousand numbers per layer instead of millions; $B$ starts at zero, so at step "
                "0 the adapted model *is* the base model, and $\\alpha / r$ scales the correction (the defaults give 2). "
                "Adapters go on both the attention projections (`q/k/v/o_proj`) and the MLP projections "
                "(`gate/up/down_proj`); attention-only LoRA is cheaper but adapts noticeably less. "
                "`prepare_model_for_kbit_training` does the QLoRA housekeeping: layer norms and the output head stay in "
                "32-bit for stable gradients through the 4-bit layers, and gradient checkpointing trades compute for memory. "
                "Expect **under 1 %** of the parameters to be trainable.\n\n"
                "The loop is deliberately plain PyTorch rather than a `Trainer`, so every moving part is visible: one example "
                "per forward pass and an optimizer step every two (`GRAD_ACCUM = 2`, an effective batch of 2 with no "
                "padding; the loss is scaled by the window size so the gradient matches a true batch), **AdamW at a constant "
                "learning rate** (no warmup or schedule — add one when you train for real), losses averaged **per supervised "
                "token**, and validation loss measured through `pipe.evaluate_loss` in eval mode with gradients off. Seeds "
                "control the shuffle; bitwise determinism across GPU kernels is not promised. Afterwards the two probe "
                "prompts are asked again for the before/after table. A measured Qwen3-0.6B reference run with these settings "
                "on a Colab T4 gave train loss ≈ 3.22, validation loss ≈ 3.19, perplexity ≈ 24, 53 s, 1.52 GiB peak; treat "
                "those as historical comparison data, not expected SmolLM2-360M results."
            ),
            "code": (
                "import gc\n"
                "import random\n"
                "import time\n\n"
                "from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training\n\n"
                "CANDIDATE_TARGETS = {{'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}}\n"
                "LORA_TARGETS = sorted({{name.rsplit('.', 1)[-1] for name, _ in pipe.model.named_modules()}} & CANDIDATE_TARGETS)\n"
                "if not LORA_TARGETS:\n"
                "    raise RuntimeError('No LoRA target modules found; this architecture names its projections differently')\n"
                "lora_config = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM', target_modules=LORA_TARGETS)\n"
                "model = get_peft_model(prepare_model_for_kbit_training(pipe.model), lora_config)\n"
                "trainable_params, all_params = model.get_nb_trainable_parameters()\n"
                "print({{'lora_targets': LORA_TARGETS, 'trainable_parameters': trainable_params, 'all_parameters': all_params, 'trainable_percent': round(100 * trainable_params / all_params, 3)}})\n\n"
                "GRAD_ACCUM = 2  # examples per optimizer step; batch size is 1, so this is the effective batch\n"
                "optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)\n"
                "random.seed(SEED)\n"
                "torch.manual_seed(SEED)\n"
                "torch.cuda.reset_peak_memory_stats()\n"
                "started = time.time()\n"
                "for epoch in range(1, EPOCHS + 1):\n"
                "    model.train()\n"
                "    examples = MASKED['train'][:]\n"
                "    random.shuffle(examples)\n"
                "    optimizer.zero_grad()\n"
                "    epoch_loss, epoch_tokens, pending = 0.0, 0, 0\n"
                "    for step, example in enumerate(examples, start=1):\n"
                "        batch = pipe.to_batch(example, model)\n"
                "        outputs = model(**batch)\n"
                "        window = min(GRAD_ACCUM, len(examples) - (step - 1))  # the last window may hold a single example\n"
                "        (outputs.loss / window).backward()\n"
                "        count = supervised_token_count(example[1])\n"
                "        epoch_loss += outputs.loss.detach().item() * count\n"
                "        epoch_tokens += count\n"
                "        pending += 1\n"
                "        if pending == window:\n"
                "            optimizer.step()\n"
                "            optimizer.zero_grad()\n"
                "            pending = 0\n"
                "    train_loss = epoch_loss / epoch_tokens\n"
                "    validation_loss = pipe.evaluate_loss(MASKED['validation'], model=model)\n"
                "    print(f'epoch {{epoch}}: train loss {{train_loss:.4f}} | validation loss {{validation_loss:.4f}}')\n\n"
                "model.eval()\n"
                "METRICS = {{\n"
                "    'trainLoss': train_loss,\n"
                "    'validationLoss': validation_loss,\n"
                "    'testLoss': pipe.evaluate_loss(MASKED.get('test', []), model=model),\n"
                "    'validationPerplexity': perplexity(validation_loss),\n"
                "    'wallSeconds': time.time() - started,\n"
                "    'peakGpuMemoryBytes': torch.cuda.max_memory_allocated(),\n"
                "    'peakGpuMemoryGiB': torch.cuda.max_memory_allocated() / 1024**3,\n"
                "}}\n"
                "ADAPTED_OUTPUTS = [pipe.generate(prompt, model=model) for prompt in PROMPTS]\n"
                "probes = [{{'prompt': p, 'base': b, 'adapted': a}} for p, b, a in zip(PROMPTS, BASELINE_OUTPUTS, ADAPTED_OUTPUTS)]\n"
                "for row in probes:\n"
                "    print(f\"PROMPT:  {{row['prompt']}}\\nBASE:    {{row['base']}}\\nADAPTED: {{row['adapted']}}\\n\")\n"
                "print(json.dumps(METRICS, indent=2))"
            ),
        },
        {
            "md": (
                "## 8. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. It carries the "
                "metrics the loop produced under the repository's own ids — `trainLoss`, `validationLoss`, `testLoss` when a "
                "test split exists, and `validationPerplexity` ($e^{{\\text{{loss}}}}$: how unsure the model was, on average, at "
                "each answer position) — with the verdict `sample-sanity`: a single seeded run on a validation split "
                "manufactured from the training source, with no dispersion estimate. These are **optimisation** metrics; "
                "they say the adapter is fitting the *format* of the data and nothing about whether the answers are "
                "*correct*. **Task quality** needs a held-out set from your real task, a rubric and ideally human ratings, "
                "which the report names under `needs`; without a scored validation split the verdict is `not-measurable`. "
                "The base/adapted probe answers are recorded as qualitative evidence. The report is written to "
                "`outputs/{stem}_evaluation_report.json`. Validation loss slightly below training loss after one epoch is "
                "normal here (dropout is active during training and the splits share a distribution); with `EPOCHS = 3` the "
                "validation loss usually keeps falling through epoch 2 and turns up around epoch 3–4 — that upturn is "
                "overfitting, and the lowest-validation-loss epoch is the one to keep."
            ),
            "code": (
                "report = evaluation_report(METRICS, sample_kind=sample_kind, n_train=len(SPLITS['train']), n_validation=len(SPLITS['validation']), probes=probes)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps({{k: v for k, v in report.items() if k != 'probes'}}, indent=2))\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No validation split was scored; the loop above produced no held-out loss.')"
            ),
        },
        {
            "md": (
                "## 9. Try new prompts with the adapter off and on\n\n"
                "The probe prompts above are the ones the model saw during evaluation. Real use means unseen instructions. "
                "The cell asks two new questions plus any prompt you type into `CUSTOM_PROMPT`, validated first through "
                "`validate_prompts` (the same checks `generate` applies: non-empty turns, the `MAX_NEW_TOKENS_CEILING` and the "
                "rendered-token ceiling), and shows the base and adapted answers side by side by switching the adapter off "
                "and on (`model.disable_adapter()`) — the cleanest way to see what the adapter alone contributes. Watch for "
                "language and register (did it stay in Filipino?), whether it stops cleanly, and repetition loops, the "
                "classic sign of a small model pushed too hard by greedy decoding. Real products should sample "
                "(`do_sample=True, temperature=0.7, top_p=0.8, top_k=20`); greedy is for reproducible checks."
            ),
            "code": (
                "RUN_NEW_PROMPT_INFERENCE = True  # @param {{type:\"boolean\"}}\n"
                "CUSTOM_PROMPT = ''  # @param {{type:\"string\"}}\n\n"
                "NEW_PROMPTS = [\n"
                "    'Sumulat ng maikling payo para sa isang estudyanteng nagsisimula sa AI.',\n"
                "    'Ipaliwanag ang pagkakaiba ng training data at evaluation data sa dalawang pangungusap.',\n"
                "]\n"
                "if CUSTOM_PROMPT.strip():\n"
                "    NEW_PROMPTS.append(CUSTOM_PROMPT.strip())\n"
                "new_prompt_manifest = validate_prompts(NEW_PROMPTS, 96, token_length=pipe.prompt_token_length)\n\n"
                "def base_and_adapted(prompt):\n"
                "    with model.disable_adapter():\n"
                "        base_answer = pipe.generate(prompt, model=model)\n"
                "    return base_answer, pipe.generate(prompt, model=model)\n\n"
                "new_prompt_rows = []\n"
                "if RUN_NEW_PROMPT_INFERENCE:\n"
                "    for prompt in NEW_PROMPTS:\n"
                "        base_answer, adapted_answer = base_and_adapted(prompt)\n"
                "        new_prompt_rows.append({{'prompt': prompt, 'base': base_answer, 'adapted': adapted_answer}})\n"
                "        print(f'PROMPT:  {{prompt}}\\nBASE:    {{base_answer}}\\nADAPTED: {{adapted_answer}}\\n')\n"
                "print({{'new_prompts': len(NEW_PROMPTS), 'rendered_tokens': [i['rendered_tokens'] for i in new_prompt_manifest['inputs']]}})"
            ),
        },
        {
            "md": (
                "## 10. Export the adapter bundle, prove a fresh reload, and write outputs\n\n"
                "The deliverable is the **adapter bundle**, not a copy of the base model: `pipe.export_adapter_bundle` writes "
                "the LoRA matrices (`adapter_model.safetensors`), the adapter config, the tokenizer with its chat template (so "
                "inference renders prompts exactly as training did), `metrics.json`, `provenance.json` (base model id and "
                "pinned revision, dataset digest and licence, hyperparameters, runtime) and an `artifact-manifest.json` "
                "listing every file with its size and SHA-256. No training rows are written into the bundle. Adapter "
                "artifacts depend on the base model and are **not complete models by themselves**.\n\n"
                "Then the notebook proves the bundle is usable **from disk**: it deletes the in-memory model, loads the base "
                "model again from the digest-verified snapshot of Section 3 (`pipe.reload_base`, no network), and attaches "
                "the saved adapter through `pipe.load_adapter`, which re-verifies the bundle before any state is "
                "deserialised. Three checks follow: the reloaded `B` matrices are not all zero, the adapter-off answer "
                "differs from the adapter-on answer (so the deltas are really applied), and a fresh prompt generates "
                "something. Finally the machine-readable result JSON (metrics, report, input manifest, sample identity and "
                "digest, hyperparameters, the notebook's source, the model identity, licence and runtime) and the "
                "before/after probe table (CSV, one row per prompt) are written, and the bundle is zipped for the companion "
                "artifact-inference notebook. The companion refuses any bundle whose files do not match the manifest."
            ),
            "code": (
                "import csv\n"
                "import zipfile\n\n"
                "ADAPTER_DIR = Path('outputs') / 'adapter-bundle'\n"
                "ARTIFACT_ZIP = Path('outputs') / '{stem}_adapter_bundle.zip'\n"
                "PROVENANCE = {{\n"
                "    'datasetDigest': DATASET_DIGEST,\n"
                "    'dataset': DATASET_PROVENANCE,\n"
                "    'training': {{'method': TRAINING_METHOD, 'epochs': EPOCHS, 'learningRate': LEARNING_RATE, 'loraRank': LORA_RANK, 'loraAlpha': LORA_ALPHA, 'targetModules': LORA_TARGETS, 'maxSequenceLength': MAX_SEQUENCE_LENGTH, 'gradAccum': GRAD_ACCUM, 'seed': SEED}},\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': importlib.metadata.version('peft'), 'gpu': GPU_NAME, 'gpuVramGiB': GPU_VRAM_GB}},\n"
                "    'notebookSource': NOTEBOOK_SOURCE,\n"
                "}}\n"
                "bundle_manifest = pipe.export_adapter_bundle(model, ADAPTER_DIR, metrics=METRICS, provenance=PROVENANCE)\n"
                "print({{'bundle': str(ADAPTER_DIR), 'files': [f['path'] for f in bundle_manifest['files']], 'total_bytes': bundle_manifest['totalBytes']}})\n\n"
                "# Fresh base + adapter reload, from disk, before publication.\n"
                "REPLAY_TOKENS = 16\n"
                "expected_opening = pipe.generate(PROMPTS[0], model=model, max_new_tokens=REPLAY_TOKENS)\n"
                "del model\n"
                "pipe.model = None\n"
                "gc.collect()\n"
                "torch.cuda.empty_cache()\n"
                "pipe.model = pipe.reload_base()\n"
                "reloaded = pipe.load_adapter(ADAPTER_DIR)\n"
                "lora_b_matrices = [param for name, param in reloaded.named_parameters() if 'lora_b' in name.lower()]\n"
                "if not lora_b_matrices or max(p.abs().max().item() for p in lora_b_matrices) == 0:\n"
                "    raise RuntimeError('Reloaded adapter weights are zero or missing')\n"
                "reloaded_opening = pipe.generate(PROMPTS[0], model=reloaded, max_new_tokens=REPLAY_TOKENS)\n"
                "with reloaded.disable_adapter():\n"
                "    adapter_off_opening = pipe.generate(PROMPTS[0], model=reloaded, max_new_tokens=REPLAY_TOKENS)\n"
                "if reloaded_opening == adapter_off_opening:\n"
                "    raise RuntimeError('Reloaded adapter has no effect: adapter-on and adapter-off answers are identical')\n"
                "smoke = pipe.generate('Kumusta! Sagutin sa isang maikling pangungusap.', model=reloaded, max_new_tokens=32)\n"
                "if not smoke:\n"
                "    raise RuntimeError('Fresh reload generated nothing')\n"
                "print('PASS: fresh base + adapter reload from the verified snapshot; adapter weights present and active')\n"
                "print({{'in_memory_adapted_opening': expected_opening, 'reloaded_adapted_opening': reloaded_opening, 'reloaded_adapter_off': adapter_off_opening, 'fresh_prompt': smoke}})\n\n"
                "with zipfile.ZipFile(ARTIFACT_ZIP, 'w', zipfile.ZIP_STORED) as archive:\n"
                "    for path in sorted(ADAPTER_DIR.rglob('*')):\n"
                "        if path.is_file():\n"
                "            archive.write(path, path.relative_to(ADAPTER_DIR).as_posix())\n"
                "payload = {{\n"
                "    'metrics': METRICS,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'new_prompt_manifest': new_prompt_manifest,\n"
                "    'probes': probes,\n"
                "    'new_prompts': new_prompt_rows,\n"
                "    'reload': {{'in_memory_opening': expected_opening, 'reloaded_opening': reloaded_opening, 'adapter_off_opening': adapter_off_opening, 'fresh_prompt': smoke, 'replay_tokens': REPLAY_TOKENS}},\n"
                "    'sample': {{'kind': sample_kind, 'name': data_name, 'splits': {{name: len(records) for name, records in SPLITS.items()}}, 'dataset_digest': DATASET_DIGEST, 'provenance': DATASET_PROVENANCE}},\n"
                "    'artifact': {{'bundle_dir': str(ADAPTER_DIR), 'zip': str(ARTIFACT_ZIP), 'zip_sha256': sha256_of_file(ARTIFACT_ZIP), 'manifest': bundle_manifest}},\n"
                "    'training': PROVENANCE['training'],\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{**PROVENANCE['runtime'], 'device': pipe.device, 'quantized_4bit': pipe.quantized, 'compute_dtype': pipe.compute_dtype}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "with open('outputs/{stem}_probes.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['stage', 'prompt', 'base', 'adapted'])\n"
                "    for row in probes:\n"
                "        writer.writerow(['evaluation-probe', row['prompt'], row['base'], row['adapted']])\n"
                "    for row in new_prompt_rows:\n"
                "        writer.writerow(['new-prompt', row['prompt'], row['base'], row['adapted']])\n"
                "print(f\"Artifact: {{ARTIFACT_ZIP}} ({{ARTIFACT_ZIP.stat().st_size / 1024**2:.1f}} MB) SHA-256 {{payload['artifact']['zip_sha256']}}\")\n"
                "print(sorted(os.listdir('outputs')))\n"
                "from google.colab import files\n"
                "files.download(str(ARTIFACT_ZIP))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The adapter changes *how* the pinned base model answers — language, register, format — on the distribution of the "
        "training rows; it does not add knowledge or make the model correct. `validationLoss` and `validationPerplexity` in "
        "the evaluation report are optimisation evidence on a validation split manufactured from the training source "
        "(verdict `sample-sanity`), not task quality: do not ship on perplexity. The before/after and new-prompt tables are "
        "qualitative; they show a shift, not an improvement. Training on very few rows, raising the epoch count, or a "
        "learning rate above `2e-4` all overfit quickly and degrade general behaviour in ways the metrics above do not "
        "measure; greedy decoding can loop on small models. Adapters trained on `Sample: Dolly` inherit CC-BY-SA 3.0; BYOD "
        "adapters inherit your data's licence and confidentiality obligations, and `provenance.json` records the dataset "
        "digest so that lineage is traceable.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, can "
        "acquire and digest-verify the pinned base snapshot, validate the demonstrated dataset, render and mask it with the "
        "model's own chat template, execute a QLoRA adaptation with per-token train and validation loss, generate with the "
        "adapter off and on, export a manifested adapter bundle, and reconstruct it from disk against the verified base — "
        "without the repository being reachable. It does **not** establish benchmark superiority, task quality on your "
        "domain, safety, fairness, or production fitness.\n\n"
        "**Next experiments:** set `EPOCHS = 3` and watch the validation loss turn up (overfitting; keep the lowest epoch); "
        "raise `LORA_RANK` to 16 with `LORA_ALPHA` 32 and compare loss and answers; switch to `Sample: Dolly` and see the "
        "`set aside` count grow with a 512-token window; enable `USE_BYOD` with a few hundred rows from your own task and "
        "score the adapter against your own held-out set; feed `outputs/{stem}_adapter_bundle.zip` to the companion "
        "artifact-inference notebook in a separate session.\n\n"
        "## References\n\n"
        f"- Repository README: https://github.com/kurtvalcorza/{REPO}/blob/main/README.md\n"
        f"- Repository model card: https://github.com/kurtvalcorza/{REPO}/blob/main/weights/smollm2-360m/MODEL_CARD.md\n"
        f"- Weight provenance: https://github.com/kurtvalcorza/{REPO}/blob/main/weights/README.md\n"
        f"- Artifact-inference companion: https://github.com/kurtvalcorza/{REPO}/blob/main/tutorials/language_model_artifact_inference_colab.ipynb\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/huggingface/smollm\n"
        "- SmolLM2 paper: https://arxiv.org/abs/2502.02737\n"
        "- QLoRA: https://arxiv.org/abs/2305.14314\n"
        "- LoRA: https://arxiv.org/abs/2106.09685"
    ),
}
