"""Public execution API for DIMER language-model tutorial notebooks.

Release-grade notebooks call this module instead of carrying a second implementation of model
resolution, dataset normalization, assistant-only masking, QLoRA setup, training, generation,
and artifact handling. Heavy ML imports stay lazy so the shared contract package remains usable
by the model-agnostic validator.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import random
import re
import shutil
import stat
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .datasets.normalize import Example, detect_family, normalize_record
from .errors import Code, DatasetError, ModelError, ResourceError
from .registry import ModelEntry, ModelRegistry

TUTORIAL_API_VERSION = "1.0"
TUTORIAL_RUNTIME_VERSIONS = {
    "torch": "2.8.0",
    "transformers": "5.16.1",
    "tokenizers": "0.23.2",
    "huggingface-hub": "1.30.0",
    "peft": "0.20.0",
    "accelerate": "1.14.0",
    "bitsandbytes": "0.49.0",
    "safetensors": "0.8.0",
    "datasets": "4.8.5",
    "pandas": "2.3.3",
    "PyYAML": "6.0.3",
    "Jinja2": "3.1.6",
}
IGNORE_INDEX = -100
CANDIDATE_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
)


@dataclass(frozen=True)
class MaskedExample:
    input_ids: list[int]
    labels: list[int]
    line_number: int

    @property
    def supervised_token_count(self) -> int:
        return sum(label != IGNORE_INDEX for label in self.labels)


@dataclass
class TutorialSplits:
    train: list[MaskedExample]
    validation: list[MaskedExample]
    test: list[MaskedExample] = field(default_factory=list)
    validation_was_derived: bool = False

    def counts(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }


@dataclass(frozen=True)
class TutorialTrainingConfig:
    epochs: int = 1
    learning_rate: float = 2e-4
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    seed: int = 42
    weight_decay: float = 0.01

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "qlora",
            "epochs": self.epochs,
            "learningRate": self.learning_rate,
            "loraRank": self.lora_rank,
            "loraAlpha": self.lora_alpha,
            "loraDropout": self.lora_dropout,
            "perDeviceBatchSize": self.per_device_batch_size,
            "gradientAccumulationSteps": self.gradient_accumulation_steps,
            "effectiveBatchSize": self.per_device_batch_size * self.gradient_accumulation_steps,
            "seed": self.seed,
            "weightDecay": self.weight_decay,
        }


@dataclass
class TrainingMetrics:
    train_loss: float | None = None
    validation_loss: float | None = None
    test_loss: float | None = None
    epochs_completed: int = 0
    examples_processed: int = 0
    supervised_tokens: int = 0
    wall_seconds: float = 0.0
    peak_gpu_memory_bytes: int | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def _perplexity(loss: float | None) -> float | None:
        return None if loss is None or loss > 20 else round(math.exp(loss), 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trainLoss": self.train_loss,
            "validationLoss": self.validation_loss,
            "testLoss": self.test_loss,
            "trainPerplexity": self._perplexity(self.train_loss),
            "validationPerplexity": self._perplexity(self.validation_loss),
            "testPerplexity": self._perplexity(self.test_loss),
            "epochsCompleted": self.epochs_completed,
            "examplesProcessed": self.examples_processed,
            "supervisedTokens": self.supervised_tokens,
            "wallSeconds": round(self.wall_seconds, 2),
            "peakGpuMemoryBytes": self.peak_gpu_memory_bytes,
            "history": self.history,
        }


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    target_modules: list[str]
    torch_dtype: str
    quantized: bool


def _installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_identity() -> dict[str, Any]:
    identity: dict[str, Any] = {
        "python": platform.python_version(),
        "tutorialApiVersion": TUTORIAL_API_VERSION,
        "packages": {name: _installed_version(name) for name in TUTORIAL_RUNTIME_VERSIONS},
    }
    try:
        import torch
    except ImportError:
        identity["accelerator"] = {"cudaAvailable": False}
        return identity
    accelerator: dict[str, Any] = {
        "cudaAvailable": torch.cuda.is_available(),
        "torchBuild": torch.__version__,
        "cudaRuntime": torch.version.cuda,
    }
    if torch.cuda.is_available():
        accelerator.update(
            {
                "device": torch.cuda.get_device_name(0),
                "vramGiB": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
                "bf16Supported": bool(torch.cuda.is_bf16_supported()),
            }
        )
    identity["accelerator"] = accelerator
    return identity


def assert_tutorial_runtime() -> dict[str, Any]:
    """Verify the exact tutorial lock without replacing the accelerator-coupled Torch wheel."""
    identity = runtime_identity()
    mismatches: list[str] = []
    for name, expected in TUTORIAL_RUNTIME_VERSIONS.items():
        actual = identity["packages"].get(name)
        comparable = actual.split("+", 1)[0] if actual and name == "torch" else actual
        if comparable != expected:
            mismatches.append(f"{name}: {actual or 'missing'} (expected {expected})")
    if mismatches:
        raise RuntimeError(
            "Tutorial runtime does not match tutorials/requirements-colab.lock:\n- "
            + "\n- ".join(mismatches)
        )
    return identity


def assert_runtime_compatible(provenance: dict[str, Any], current: dict[str, Any]) -> None:
    """Reject an adapter produced by a materially different tutorial software stack."""
    recorded = provenance.get("runtime") or {}
    producer_packages = recorded.get("packages") or {}
    consumer_packages = current.get("packages") or {}
    required = ("torch", "transformers", "tokenizers", "peft", "bitsandbytes", "safetensors")
    missing = [name for name in required if not producer_packages.get(name)]
    if missing:
        raise ValueError("Artifact provenance lacks runtime package versions: " + ", ".join(missing))
    mismatches = [
        f"{name}: artifact={producer_packages[name]} runtime={consumer_packages.get(name)}"
        for name in required
        if producer_packages[name] != consumer_packages.get(name)
    ]
    if mismatches:
        raise ValueError("Artifact/runtime compatibility mismatch:\n- " + "\n- ".join(mismatches))


def seed_everything(seed: int) -> dict[str, Any]:
    random.seed(seed)
    seeded = ["python.random"]
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
        seeded.append("numpy.random")
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        seeded.append("torch.cpu")
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            seeded.append("torch.cuda")
    return {
        "seed": seed,
        "seeded": seeded,
        "remainingVariability": [
            "GPU kernels and reductions may not be bitwise deterministic across devices",
            "quantized kernels may differ across CUDA and bitsandbytes builds",
            "stochastic decoding is nondeterministic unless separately seeded and configured",
        ],
    }


def resolve_tutorial_model(
    model_key: str,
    *,
    method: str = "qlora",
    max_sequence_length: int,
    allow_internal: bool = False,
) -> ModelEntry:
    entry = ModelRegistry.load().resolve(model_key)
    if entry.internal_only and not allow_internal:
        raise ModelError(
            f"Model {model_key!r} is internal-only and not a user-facing DIMER choice.",
            code=Code.MODEL_NOT_APPROVED,
            details={"model_key": model_key},
        )
    entry.require_method(method)
    entry.clamp_sequence_length(max_sequence_length)
    return entry


def resolve_artifact_model(provenance: dict[str, Any]) -> ModelEntry:
    """Resolve artifact provenance back through the canonical registry and require identity parity."""
    key = provenance.get("modelKey")
    if not isinstance(key, str):
        raise ValueError("Artifact provenance is missing modelKey")
    entry = ModelRegistry.load().resolve(key)
    if entry.model_id != provenance.get("baseModel") or entry.revision != provenance.get("baseModelRevision"):
        raise ValueError("Artifact base model identity/revision does not match the canonical registry")
    return entry


def normalize_records(records: Iterable[dict[str, Any]]) -> list[Example]:
    normalized: list[Example] = []
    family: str | None = None
    for line_number, record in enumerate(records, 1):
        detected = detect_family(record, line_number)
        if family is None:
            family = detected
        elif detected != family:
            raise DatasetError(
                f"Line {line_number}: collection mixes schema families ({family!r}, {detected!r}).",
                code=Code.DATASET_SCHEMA_MIXED,
                details={"line": line_number, "expected": family, "found": detected},
            )
        normalized.append(normalize_record(record, detected, line_number))
    return normalized


def derive_validation_split(
    examples: list[Example], *, fraction: float, seed: int
) -> tuple[list[Example], list[Example]]:
    if len(examples) < 2 or fraction <= 0:
        return examples, []
    threshold = int(fraction * 10_000)
    train: list[Example] = []
    validation: list[Example] = []
    for example in examples:
        digest = hashlib.sha256(f"{seed}:{example.fingerprint()}".encode()).digest()
        bucket = int.from_bytes(digest[:4], "big") % 10_000
        (validation if bucket < threshold else train).append(example)
    if not validation:
        validation = [train.pop()]
    elif not train:
        train = [validation.pop()]
    return train, validation


def assert_no_split_leakage(splits: dict[str, list[Example]]) -> None:
    fingerprints = {
        name: {example.fingerprint() for example in examples}
        for name, examples in splits.items()
    }
    names = sorted(fingerprints)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = fingerprints[left] & fingerprints[right]
            if overlap:
                raise DatasetError(
                    f"Split leakage: {len(overlap)} canonical record(s) occur in both "
                    f"{left!r} and {right!r}.",
                    code=Code.DATASET_SPLIT_LEAKAGE,
                    details={"left": left, "right": right, "count": len(overlap)},
                )


def canonical_dataset_digest(splits: dict[str, list[Example]]) -> str:
    digest = hashlib.sha256()
    for split_name in sorted(splits):
        digest.update(split_name.encode())
        for fingerprint in sorted(example.fingerprint() for example in splits[split_name]):
            digest.update(fingerprint.encode())
    return digest.hexdigest()


def render_chat(tokenizer, messages: list[dict[str, str]], *, generation_prompt: bool) -> str:
    if not messages:
        return ""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=generation_prompt
    )


def _encode(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def build_masked_example(
    tokenizer, example: Example, *, max_sequence_length: int
) -> MaskedExample:
    messages = list(example.messages)
    input_ids = _encode(tokenizer, render_chat(tokenizer, messages, generation_prompt=False))
    if len(input_ids) > max_sequence_length:
        raise DatasetError(
            f"Line {example.line_number}: renders to {len(input_ids)} tokens, above the "
            f"configured limit of {max_sequence_length}.",
            code=Code.DATASET_SEQUENCE_TOO_LONG,
            details={"line": example.line_number, "tokens": len(input_ids), "limit": max_sequence_length},
        )
    labels = [IGNORE_INDEX] * len(input_ids)
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix_ids = _encode(
            tokenizer, render_chat(tokenizer, messages[:index], generation_prompt=True)
        )
        upto_ids = _encode(
            tokenizer, render_chat(tokenizer, messages[: index + 1], generation_prompt=False)
        )
        if input_ids[: len(prefix_ids)] != prefix_ids or input_ids[: len(upto_ids)] != upto_ids:
            raise DatasetError(
                f"Line {example.line_number}: chat template is not prefix-stable; "
                "assistant-only masking is unsafe.",
                code=Code.DATASET_CHAT_TEMPLATE_MISSING,
                details={"line": example.line_number, "message": index},
            )
        labels[len(prefix_ids) : len(upto_ids)] = input_ids[len(prefix_ids) : len(upto_ids)]
    if all(label == IGNORE_INDEX for label in labels):
        raise DatasetError(
            f"Line {example.line_number}: no assistant tokens were supervised.",
            code=Code.DATASET_TARGET_EMPTY,
            details={"line": example.line_number},
        )
    return MaskedExample(input_ids, labels, example.line_number)


def tokenize_splits(
    raw: dict[str, list[Example]],
    *,
    tokenizer,
    max_sequence_length: int,
    validation_fraction: float,
    seed: int,
) -> TutorialSplits:
    assert_no_split_leakage(raw)
    train_raw = raw["train"]
    derived = False
    if "validation" in raw:
        validation_raw = raw["validation"]
    else:
        train_raw, validation_raw = derive_validation_split(
            train_raw, fraction=validation_fraction, seed=seed
        )
        derived = True

    def mask_all(items: list[Example]) -> list[MaskedExample]:
        return [
            build_masked_example(tokenizer, item, max_sequence_length=max_sequence_length)
            for item in items
        ]

    return TutorialSplits(
        train=mask_all(train_raw),
        validation=mask_all(validation_raw),
        test=mask_all(raw.get("test", [])),
        validation_was_derived=derived,
    )


def _bf16_supported() -> bool:
    import torch

    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def load_tokenizer(
    entry: ModelEntry,
    *,
    load_ref: str | Path | None = None,
    token: str | None = None,
    local_files_only: bool = False,
):
    from transformers import AutoTokenizer

    source = str(load_ref or entry.model_id)
    kwargs: dict[str, Any] = {"trust_remote_code": False, "local_files_only": local_files_only}
    if not local_files_only:
        kwargs["revision"] = entry.revision
        if token:
            kwargs["token"] = token
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        raise ModelError(
            f"Tokenizer for {entry.key!r} has no chat template.",
            code=Code.MODEL_TOKENIZER_UNAVAILABLE,
        )
    return tokenizer


def resolve_target_modules(model, entry: ModelEntry) -> list[str]:
    if entry.lora_target_modules != "auto":
        return list(entry.lora_target_modules)
    present = {
        name.rsplit(".", 1)[-1]
        for name, _ in model.named_modules()
        if name.rsplit(".", 1)[-1] in CANDIDATE_TARGET_MODULES
    }
    if not present:
        raise ModelError(
            f"No verified LoRA target modules found on {entry.key!r}.",
            code=Code.MODEL_NOT_APPROVED,
            details={"model_key": entry.key},
        )
    return sorted(present)


def load_base_model(
    entry: ModelEntry,
    *,
    tokenizer,
    method: str = "qlora",
    load_ref: str | Path | None = None,
    token: str | None = None,
    local_files_only: bool = False,
    device: str = "cuda",
) -> LoadedModel:
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    entry.require_method(method)
    if method == "qlora" and not torch.cuda.is_available():
        raise ResourceError(
            "QLoRA requires a CUDA GPU; none is visible.", code=Code.RESOURCE_GPU_UNAVAILABLE
        )
    compute_dtype = torch.bfloat16 if _bf16_supported() else torch.float16
    quantization_config = None
    if method == "qlora":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    source = str(load_ref or entry.model_id)
    kwargs: dict[str, Any] = {
        "trust_remote_code": False,
        "dtype": compute_dtype if method == "qlora" else (
            torch.bfloat16 if _bf16_supported() else torch.float32
        ),
        "quantization_config": quantization_config,
        "attn_implementation": "sdpa",
        "local_files_only": local_files_only,
    }
    if not local_files_only:
        kwargs["revision"] = entry.revision
        if token:
            kwargs["token"] = token
    if method == "qlora":
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    if method != "qlora":
        model.to(device)
    loaded_revision = getattr(getattr(model, "config", None), "_commit_hash", None)
    if loaded_revision and not local_files_only and loaded_revision != entry.revision:
        raise ModelError(
            f"Loaded revision {loaded_revision[:12]} does not match pinned "
            f"{entry.revision[:12]} for {entry.key!r}.",
            code=Code.MODEL_REVISION_MISMATCH,
            details={"expected": entry.revision, "loaded": loaded_revision},
        )
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        target_modules=resolve_target_modules(model, entry),
        torch_dtype=str(kwargs["dtype"]).replace("torch.", ""),
        quantized=quantization_config is not None,
    )


def attach_adapter(loaded: LoadedModel, *, rank: int, alpha: int, dropout: float = 0.05):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(loaded.model) if loaded.quantized else loaded.model
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=loaded.target_modules,
    )
    return get_peft_model(model, config)


def trainable_parameter_summary(model) -> dict[str, int | float]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "trainablePercent": round(100.0 * trainable / total, 4) if total else 0.0,
    }


def _collate(batch: list[MaskedExample], *, pad_token_id: int) -> dict[str, Any]:
    import torch

    width = max(len(item.input_ids) for item in batch)
    inputs, labels, masks = [], [], []
    for item in batch:
        padding = width - len(item.input_ids)
        inputs.append(item.input_ids + [pad_token_id] * padding)
        labels.append(item.labels + [IGNORE_INDEX] * padding)
        masks.append([1] * len(item.input_ids) + [0] * padding)
    return {
        "input_ids": torch.tensor(inputs, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
    }


def _batches(items: list[MaskedExample], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def evaluate_loss(
    model,
    items: list[MaskedExample],
    *,
    pad_token_id: int,
    batch_size: int,
    device: str,
) -> float | None:
    import torch

    if not items:
        return None
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for batch in _batches(items, batch_size):
            tensors = {k: v.to(device) for k, v in _collate(batch, pad_token_id=pad_token_id).items()}
            outputs = model(**tensors)
            supervised = int((tensors["labels"] != IGNORE_INDEX).sum().item())
            total_loss += float(outputs.loss.item()) * supervised
            total_tokens += supervised
    if was_training:
        model.train()
    return total_loss / total_tokens if total_tokens else None


def train_adapter(
    model,
    splits: TutorialSplits,
    *,
    config: TutorialTrainingConfig,
    pad_token_id: int,
    device: str = "cuda",
    log=print,
) -> TrainingMetrics:
    import torch

    seed_everything(config.seed)
    metrics = TrainingMetrics()
    started = time.monotonic()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(config.seed)
    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(len(splits.train), generator=generator).tolist()
        shuffled = [splits.train[index] for index in order]
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        epoch_tokens = 0
        micro = 0
        for batch in _batches(shuffled, config.per_device_batch_size):
            tensors = {k: v.to(device) for k, v in _collate(batch, pad_token_id=pad_token_id).items()}
            outputs = model(**tensors)
            (outputs.loss / config.gradient_accumulation_steps).backward()
            supervised = int((tensors["labels"] != IGNORE_INDEX).sum().item())
            epoch_loss += float(outputs.loss.item()) * supervised
            epoch_tokens += supervised
            metrics.examples_processed += len(batch)
            micro += 1
            if micro % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if micro % config.gradient_accumulation_steps:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        metrics.epochs_completed = epoch + 1
        metrics.supervised_tokens += epoch_tokens
        metrics.train_loss = epoch_loss / epoch_tokens if epoch_tokens else None
        metrics.validation_loss = evaluate_loss(
            model,
            splits.validation,
            pad_token_id=pad_token_id,
            batch_size=config.per_device_batch_size,
            device=device,
        )
        metrics.history.append(
            {"epoch": epoch + 1, "trainLoss": metrics.train_loss,
             "validationLoss": metrics.validation_loss}
        )
        log(
            f"epoch {epoch + 1}/{config.epochs} train_loss={metrics.train_loss} "
            f"validation_loss={metrics.validation_loss}"
        )
    metrics.test_loss = evaluate_loss(
        model,
        splits.test,
        pad_token_id=pad_token_id,
        batch_size=config.per_device_batch_size,
        device=device,
    )
    metrics.wall_seconds = time.monotonic() - started
    if torch.cuda.is_available():
        metrics.peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated())
    model.eval()
    return metrics


def generate_reply(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 96,
    decoding: dict[str, Any] | None = None,
) -> str:
    import torch

    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    rendered = render_chat(
        tokenizer, [{"role": "user", "content": prompt.strip()}], generation_prompt=True
    )
    inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    settings = {"do_sample": False, **(decoding or {})}
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            **settings,
        )
    return tokenizer.decode(
        output_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def validate_prompt(tokenizer, prompt: str, *, max_sequence_length: int) -> int:
    """Validate new user input before expensive generation and return prompt-token count."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string")
    rendered = render_chat(
        tokenizer, [{"role": "user", "content": prompt.strip()}], generation_prompt=True
    )
    count = len(_encode(tokenizer, rendered))
    if count >= max_sequence_length:
        raise ValueError(
            f"Prompt renders to {count} tokens; it must be below the {max_sequence_length} "
            "token context ceiling before generation."
        )
    return count


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member_path(root: Path, member_name: str) -> Path:
    if "\\" in member_name:
        raise ValueError(f"Unsafe archive path (backslash): {member_name!r}")
    relative = PurePosixPath(member_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe archive path: {member_name!r}")
    target = (root.resolve() / Path(*relative.parts)).resolve()
    if target != root.resolve() and root.resolve() not in target.parents:
        raise ValueError(f"Archive member escapes extraction root: {member_name!r}")
    return target


def safe_extract_zip(
    zip_path: str | Path, root: str | Path, *, size_limit_bytes: int
) -> Path:
    root = Path(root).resolve()
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    expanded = 0
    seen: set[str] = set()
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            canonical = PurePosixPath(info.filename).as_posix()
            if canonical in seen and not info.is_dir():
                raise ValueError(f"Duplicate archive member: {canonical}")
            seen.add(canonical)
            if stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF):
                raise ValueError("Symlinks are not allowed in the archive")
            expanded += info.file_size
            if expanded > size_limit_bytes:
                raise ValueError("Archive expands beyond the allowed size")
            target = _safe_member_path(root, info.filename)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, open(target, "wb") as destination:
                shutil.copyfileobj(source, destination)
    return root


def verify_manifested_directory(
    artifact_root: str | Path,
    *,
    manifest_name: str = "artifact-manifest.json",
    expected_format: str = "peft_adapter",
    expected_version: int = 1,
) -> dict[str, Any]:
    root = Path(artifact_root).resolve()
    manifest = json.loads((root / manifest_name).read_text())
    if manifest.get("format") != expected_format or manifest.get("formatVersion") != expected_version:
        raise ValueError("Unsupported artifact format")
    listed: set[str] = set()
    listed_bytes = 0
    for record in manifest.get("files", []):
        path = _safe_member_path(root, record["path"])
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"SHA-256 or size mismatch: {record['path']}")
        listed.add(record["path"])
        listed_bytes += record["bytes"]
    on_disk = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != manifest_name
    }
    if on_disk != listed or listed_bytes != manifest.get("totalBytes"):
        raise ValueError("Manifest/file-set mismatch")
    required = {"adapter_config.json", "adapter_model.safetensors", "provenance.json"}
    if not required.issubset(listed):
        raise ValueError("Artifact manifest is missing required adapter/provenance files")
    return manifest


def consume_adapter_archive(
    archive_path: str | Path,
    *,
    extraction_root: str | Path,
    expected_archive_sha256: str = "",
    size_limit_bytes: int = 512 * 1024**2,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    archive_path = Path(archive_path)
    actual_sha = sha256_file(archive_path)
    expected = expected_archive_sha256.strip().lower()
    if expected and actual_sha.lower() != expected:
        raise ValueError("Whole-ZIP SHA-256 mismatch")
    extracted = safe_extract_zip(
        archive_path, extraction_root, size_limit_bytes=size_limit_bytes
    )
    manifests = list(extracted.rglob("artifact-manifest.json"))
    if len(manifests) != 1:
        raise ValueError("Expected exactly one artifact-manifest.json")
    root = manifests[0].parent
    manifest = verify_manifested_directory(root)
    provenance = json.loads((root / "provenance.json").read_text())
    revision = provenance.get("baseModelRevision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Invalid baseModelRevision provenance")
    if provenance.get("trustRemoteCode") is not False:
        raise ValueError("trustRemoteCode must be false")
    return root, manifest, provenance


def export_adapter_bundle(
    model,
    tokenizer,
    *,
    destination: str | Path,
    provenance: dict[str, Any],
    metrics: dict[str, Any],
) -> Path:
    destination = Path(destination)
    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True)
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination / "tokenizer")
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (destination / "MODEL_CARD.md").write_text(
        f"# PEFT adapter for {provenance['baseModel']}\n\n"
        f"Base revision: `{provenance['baseModelRevision']}`.\n\n"
        "Optimization metrics in metrics.json are tutorial evidence, not task-quality validation.\n"
    )
    records = [
        {
            "path": path.relative_to(destination).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(destination.rglob("*"))
        if path.is_file() and path.name != "artifact-manifest.json"
    ]
    manifest = {
        "format": "peft_adapter",
        "formatVersion": 1,
        "files": records,
        "totalBytes": sum(record["bytes"] for record in records),
    }
    (destination / "artifact-manifest.json").write_text(json.dumps(manifest, indent=2))
    verify_manifested_directory(destination)
    return destination


def zip_directory(root: str | Path, destination: str | Path) -> Path:
    root = Path(root)
    destination = Path(destination)
    if destination.exists():
        destination.unlink()
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_STORED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(root).as_posix())
    return destination


def load_adapter_for_inference(
    entry: ModelEntry,
    *,
    artifact_root: str | Path,
    token: str | None = None,
    load_ref: str | Path | None = None,
    local_files_only: bool = False,
):
    from peft import PeftModel

    artifact_root = Path(artifact_root)
    tokenizer = load_tokenizer(
        entry, load_ref=artifact_root / "tokenizer", local_files_only=True
    )
    loaded = load_base_model(
        entry,
        tokenizer=tokenizer,
        method="qlora",
        load_ref=load_ref,
        token=token,
        local_files_only=local_files_only,
    )
    model = PeftModel.from_pretrained(loaded.model, artifact_root, is_trainable=False)
    model.eval()
    return model, tokenizer
