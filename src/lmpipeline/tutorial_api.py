"""Notebook support around the production language-model finetuner.

This module deliberately does **not** implement model loading, assistant masking, training,
artifact publication, or generation. Those operations are owned by the deployable
``language-model-finetuner`` package. Tutorial notebooks load an immutable checkout of that
repository and call its production modules; the helpers here cover only notebook/runtime
provenance, canonical registry checks, and hostile external-ZIP validation.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import random
import re
import shutil
import stat
import subprocess
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

from .datasets.normalize import Example, detect_family, normalize_record
from .errors import Code, DatasetError, ModelError
from .registry import ModelEntry, ModelRegistry

TUTORIAL_API_VERSION = "1.1"
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


def _installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_identity() -> dict[str, Any]:
    packages = {name: _installed_version(name) for name in TUTORIAL_RUNTIME_VERSIONS}
    identity: dict[str, Any] = {
        "python": platform.python_version(),
        "tutorialApiVersion": TUTORIAL_API_VERSION,
        "packages": packages,
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
        props = torch.cuda.get_device_properties(0)
        accelerator.update(
            {
                "device": torch.cuda.get_device_name(0),
                "vramGiB": round(props.total_memory / 1024**3, 2),
                "bf16Supported": bool(torch.cuda.is_bf16_supported()),
            }
        )
    identity["accelerator"] = accelerator
    return identity


def assert_tutorial_runtime() -> dict[str, Any]:
    """Fail unless the notebook runtime matches the committed compatibility lock."""
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
    """Require the critical producer and consumer ML stacks to match exactly."""
    producer = provenance.get("packageVersions") or provenance.get("runtime", {}).get("packages") or {}
    consumer = current.get("packages") or {}
    required = ("torch", "transformers", "tokenizers", "peft", "bitsandbytes", "safetensors")
    missing = [name for name in required if not producer.get(name)]
    if missing:
        raise ValueError(
            "Artifact provenance lacks runtime package versions: " + ", ".join(missing)
        )
    mismatches = [
        f"{name}: artifact={producer[name]} runtime={consumer.get(name)}"
        for name in required
        if producer[name].split("+", 1)[0] != (consumer.get(name) or "").split("+", 1)[0]
    ]
    if mismatches:
        raise ValueError("Artifact/runtime compatibility mismatch:\n- " + "\n- ".join(mismatches))


def seed_everything(seed: int) -> dict[str, Any]:
    """Seed notebook-visible RNGs before production model/adapter construction."""
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
            "stochastic decoding is nondeterministic unless separately seeded",
        ],
    }


def assert_finetuner_checkout(root: str | Path, expected_revision: str) -> str:
    """Prove the production source imported by a notebook is the immutable requested SHA."""
    if re.fullmatch(r"[0-9a-f]{40}", expected_revision) is None:
        raise ValueError("expected finetuner revision must be a 40-character commit SHA")
    root = Path(root).resolve()
    try:
        actual = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Could not establish production finetuner checkout identity") from exc
    if actual != expected_revision:
        raise RuntimeError(
            f"Production finetuner checkout mismatch: {actual} != {expected_revision}"
        )
    return actual


def resolve_tutorial_model(
    model_key: str,
    *,
    method: str,
    max_sequence_length: int,
    allow_internal: bool = False,
) -> ModelEntry:
    """Resolve the canonical pipeline registry; never carry a notebook-side registry."""
    entry = ModelRegistry.load().resolve(model_key)
    if entry.internal_only and not allow_internal:
        raise ModelError(
            f"Model {model_key!r} is internal-only and not user-facing.",
            code=Code.MODEL_NOT_APPROVED,
            details={"model_key": model_key},
        )
    entry.require_method(method)
    entry.clamp_sequence_length(max_sequence_length)
    return entry


def resolve_artifact_model(provenance: dict[str, Any]) -> ModelEntry:
    """Resolve artifact identity back through the current canonical registry."""
    key = provenance.get("modelKey")
    if not isinstance(key, str):
        raise ValueError("Artifact provenance is missing modelKey")
    entry = ModelRegistry.load().resolve(key)
    if entry.model_id != provenance.get("baseModel"):
        raise ValueError("Artifact base model does not match the canonical registry")
    if entry.revision != provenance.get("baseModelRevision"):
        raise ValueError("Artifact base revision does not match the canonical registry")
    if provenance.get("trustRemoteCode") is not False:
        raise ValueError("trustRemoteCode must be false")
    return entry


def normalize_records(records: Iterable[dict[str, Any]]) -> list[Example]:
    """Normalize an in-memory public sample through the canonical dataset contract."""
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


def assert_no_split_leakage(splits: dict[str, list[Example]]) -> None:
    fingerprints = {
        name: {example.fingerprint() for example in examples} for name, examples in splits.items()
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
    """Stable digest for in-memory tutorial sample records."""
    digest = hashlib.sha256()
    for split_name in sorted(splits):
        digest.update(split_name.encode())
        for identity in sorted(example.fingerprint() for example in splits[split_name]):
            digest.update(identity.encode())
    return digest.hexdigest()


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
    resolved_root = root.resolve()
    target = (resolved_root / Path(*relative.parts)).resolve()
    if target != resolved_root and resolved_root not in target.parents:
        raise ValueError(f"Archive member escapes extraction root: {member_name!r}")
    return target


def safe_extract_zip(
    zip_path: str | Path,
    root: str | Path,
    *,
    size_limit_bytes: int,
) -> Path:
    """Extract an untrusted ZIP after path, link, duplicate, and size checks."""
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
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
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


def verify_manifested_directory(artifact_root: str | Path) -> dict[str, Any]:
    """Strictly verify the production artifact manifest plus its expected file set."""
    root = Path(artifact_root).resolve()
    manifest_path = root / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("Artifact manifest has no file records")

    listed: set[str] = set()
    listed_bytes = 0
    for record in records:
        member = record.get("path")
        if not isinstance(member, str) or member in listed:
            raise ValueError("Artifact manifest contains an invalid or duplicate path")
        path = _safe_member_path(root, member)
        if not path.is_file():
            raise ValueError(f"Manifested file is missing: {member}")
        expected_bytes = record.get("bytes")
        expected_sha = record.get("sha256")
        if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha:
            raise ValueError(f"SHA-256 or size mismatch: {member}")
        listed.add(member)
        listed_bytes += int(expected_bytes)

    on_disk = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact-manifest.json"
    }
    if on_disk != listed or listed_bytes != manifest.get("totalBytes"):
        raise ValueError("Manifest/file-set mismatch")
    required = {"adapter_config.json", "adapter_model.safetensors", "provenance.json"}
    if not required.issubset(listed):
        raise ValueError("Artifact manifest is missing required adapter/provenance files")
    if not any(member.startswith("tokenizer/") for member in listed):
        raise ValueError("Artifact manifest is missing tokenizer assets")
    return manifest


def consume_adapter_archive(
    archive_path: str | Path,
    *,
    extraction_root: str | Path,
    expected_archive_sha256: str = "",
    size_limit_bytes: int = 512 * 1024**2,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Safely ingest a production PEFT artifact supplied from outside the notebook run."""
    archive_path = Path(archive_path)
    actual_sha = sha256_file(archive_path)
    expected = expected_archive_sha256.strip().lower()
    if expected and actual_sha.lower() != expected:
        raise ValueError("Whole-ZIP SHA-256 mismatch")
    extracted = safe_extract_zip(
        archive_path,
        extraction_root,
        size_limit_bytes=size_limit_bytes,
    )
    manifests = list(extracted.rglob("artifact-manifest.json"))
    if len(manifests) != 1:
        raise ValueError("Expected exactly one artifact-manifest.json")
    root = manifests[0].parent
    manifest = verify_manifested_directory(root)
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    revision = provenance.get("baseModelRevision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Invalid baseModelRevision provenance")
    if provenance.get("trustRemoteCode") is not False:
        raise ValueError("trustRemoteCode must be false")
    return root, manifest, provenance


def validate_prompt(tokenizer, prompt: str, *, max_sequence_length: int) -> int:
    """Validate new user text using the production inference renderer."""
    from finetuner.inference import render_prompt

    rendered = render_prompt(tokenizer, prompt)
    count = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
    if count >= max_sequence_length:
        raise ValueError(
            f"Prompt renders to {count} tokens; it must be below the "
            f"{max_sequence_length}-token context ceiling before generation."
        )
    return count


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
