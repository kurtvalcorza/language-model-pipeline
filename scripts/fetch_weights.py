#!/usr/bin/env python
"""Download and verify base-model weights for the language-model pipeline.

Downloads registered Hugging Face causal LM weights at pinned immutable revisions,
following the DIMER base-model snapshot schema (dimer-base-manifest.json).

Usage:
    # List all registered base models:
    python scripts/fetch_weights.py --list

    # Download default model (qwen3-0.6b) to weights/qwen3-0.6b/:
    python scripts/fetch_weights.py

    # Download a specific model to weights/<model-key>/:
    python scripts/fetch_weights.py --model smollm3-3b

    # Download to an explicit custom destination folder:
    python scripts/fetch_weights.py --model qwen3-0.6b --dest custom/path

    # Dry-run (check files and total download size without downloading):
    python scripts/fetch_weights.py --dry-run

    # Verify existing weights against dimer-base-manifest.json:
    python scripts/fetch_weights.py --verify-only

    # Download and package into a DIMER ZIP archive:
    python scripts/fetch_weights.py --model qwen3-0.6b --zip dimer-base-model.zip
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

# Add src/ to path so lmpipeline can be imported directly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lmpipeline.registry import DEFAULT_REGISTRY_PATH, ModelEntry, ModelRegistry  # noqa: E402

DEFAULT_MODEL_KEY = "qwen3-0.6b"
DEFAULT_DEST_DIR = ROOT / "weights"


MANIFEST_NAME = "dimer-base-manifest.json"
MANIFEST_FORMAT = "dimer_hf_snapshot"
MANIFEST_FORMAT_VERSION = 1

# Allowed patterns for base-model causal LM snapshots
ALLOW_PATTERNS = [
    "*.safetensors*",
    "*.json",
    "*.jinja",
    "*.txt",
    "*.model",
    "LICENSE*",
    "README*",
]

# Forbidden or redundant formats (pickle weights, framework binaries, git metadata)
IGNORE_PATTERNS = [
    "*.bin",
    "*.pt",
    "*.pth",
    "*.onnx",
    "*.h5",
    "*.msgpack",
    "*.ipynb",
    ".git*",
]


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file efficiently in 1MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def format_bytes(num_bytes: int) -> str:
    """Format bytes into human-readable string (e.g., 3.44 GiB)."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PiB"


def load_raw_registry(path: Path | None = None) -> dict[str, Any]:
    """Load raw registry YAML dict to access extra annotations like 'gated'."""
    reg_path = path or DEFAULT_REGISTRY_PATH
    if not reg_path.is_file():
        raise FileNotFoundError(f"Registry file not found: {reg_path}")
    return yaml.safe_load(reg_path.read_text(encoding="utf-8")) or {}


def get_model_specs(
    registry_path: Path | None = None,
) -> tuple[ModelRegistry, dict[str, Any]]:
    """Load both structured ModelRegistry and raw spec dict."""
    reg_path = registry_path or DEFAULT_REGISTRY_PATH
    registry = ModelRegistry.load(reg_path)
    raw = load_raw_registry(reg_path)
    specs = raw.get("models") or {}
    return registry, specs


def list_registered_models(registry_path: Path | None = None) -> list[dict[str, Any]]:
    """Return summary information for all registered models."""
    registry, specs = get_model_specs(registry_path)
    rows: list[dict[str, Any]] = []

    for key, entry in registry._entries.items():
        spec = specs.get(key) or {}
        gated = bool(spec.get("gated") or key == "llama-3.2-3b-instruct")
        status = "user-facing"
        if not entry.enabled:
            status = f"blocked ({spec.get('blocked_reason', entry.approval_state)})"
        elif entry.internal_only:
            status = "internal-only"

        rows.append(
            {
                "key": key,
                "model_id": entry.model_id,
                "revision": entry.revision or "none",
                "revision_short": (entry.revision[:8] if entry.revision else "none"),
                "license": entry.license or "unknown",
                "status": status,
                "enabled": entry.enabled,
                "gated": gated,
                "default": key == DEFAULT_MODEL_KEY,
            }
        )
    return rows


def print_model_table(registry_path: Path | None = None) -> None:
    """Print an aligned ASCII table of registered models."""
    models = list_registered_models(registry_path)
    print("\nRegistered Base Models:")
    header = (
        f"{'':2} {'Key':<22} {'Model ID':<36} {'Revision':<10} {'License':<14} "
        f"{'Status':<28} {'Gated':<6}"
    )
    print(header)
    print("-" * len(header))
    for m in models:
        prefix = "* " if m["default"] else "  "
        gated_str = "Yes" if m["gated"] else "No"
        print(
            f"{prefix}{m['key']:<22} {m['model_id']:<36} {m['revision_short']:<10} "
            f"{m['license']:<14} {m['status']:<28} {gated_str:<6}"
        )
    print("\n* Default model key for CLI fetch\n")


def matches_patterns(
    filename: str,
    allow: list[str] = ALLOW_PATTERNS,
    ignore: list[str] = IGNORE_PATTERNS,
) -> bool:
    """Check if a filename matches allow patterns and does not match ignore patterns."""
    for pattern in ignore:
        if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(
            Path(filename).name, pattern
        ):
            return False
    for pattern in allow:
        if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(
            Path(filename).name, pattern
        ):
            return True
    return False


def generate_manifest(
    model_dir: Path,
    model_key: str,
    model_id: str,
    revision: str,
) -> dict[str, Any]:
    """Generate a dimer-base-manifest.json dictionary from files in model_dir."""
    records: list[dict[str, Any]] = []
    total_bytes = 0

    for p in sorted(model_dir.rglob("*")):
        if p.is_file() and p.name != MANIFEST_NAME:
            if p.is_symlink():
                raise ValueError(f"Symlinks not allowed in manifest generation: {p}")
            rel_path = p.relative_to(model_dir).as_posix()
            size = p.stat().st_size
            digest = sha256_file(p)
            records.append(
                {
                    "path": rel_path,
                    "bytes": size,
                    "sha256": digest,
                }
            )
            total_bytes += size

    return {
        "format": MANIFEST_FORMAT,
        "formatVersion": MANIFEST_FORMAT_VERSION,
        "modelKey": model_key,
        "modelId": model_id,
        "revision": revision,
        "files": records,
        "totalBytes": total_bytes,
    }


def verify_snapshot(
    model_dir: Path,
    expected_model_key: str | None = None,
    expected_model_id: str | None = None,
    expected_revision: str | None = None,
) -> tuple[bool, list[str]]:
    """Verify a snapshot directory against its dimer-base-manifest.json.

    Returns (is_valid, list_of_error_strings).
    """
    errors: list[str] = []
    manifest_path = model_dir / MANIFEST_NAME

    if not manifest_path.is_file():
        return False, [f"Manifest not found: {manifest_path}"]

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, [f"Failed to parse {MANIFEST_NAME}: {exc}"]

    if manifest.get("format") != MANIFEST_FORMAT:
        errors.append(
            f"Invalid format {manifest.get('format')!r}; expected {MANIFEST_FORMAT!r}"
        )
    if manifest.get("formatVersion") != MANIFEST_FORMAT_VERSION:
        errors.append(
            f"Invalid formatVersion {manifest.get('formatVersion')!r}; "
            f"expected {MANIFEST_FORMAT_VERSION}"
        )

    model_key = manifest.get("modelKey")
    model_id = manifest.get("modelId")
    revision = manifest.get("revision")

    if expected_model_key and model_key != expected_model_key:
        errors.append(f"Model key mismatch: manifest={model_key}, expected={expected_model_key}")
    if expected_model_id and model_id != expected_model_id:
        errors.append(f"Model ID mismatch: manifest={model_id}, expected={expected_model_id}")
    if expected_revision and revision != expected_revision:
        errors.append(f"Revision mismatch: manifest={revision}, expected={expected_revision}")

    listed_files: set[str] = set()
    calculated_size = 0

    resolved_root = model_dir.resolve()
    for item in manifest.get("files", []):
        rel_posix = item.get("path", "")
        if not isinstance(rel_posix, str) or not rel_posix:
            errors.append(f"Invalid or empty path in manifest: {rel_posix!r}")
            continue

        # Reject backslashes, drive prefixes, and UNC paths
        if "\\" in rel_posix or ":" in rel_posix:
            errors.append(
                f"Unsafe path in manifest (contains backslash or drive prefix): {rel_posix!r}"
            )
            continue

        pure = PurePosixPath(rel_posix)
        if pure.is_absolute() or ".." in pure.parts:
            errors.append(f"Unsafe path in manifest (absolute or traversal): {rel_posix!r}")
            continue

        raw_candidate = resolved_root / Path(*pure.parts)
        try:
            target_file = raw_candidate.resolve()
        except Exception as exc:
            errors.append(f"Failed to resolve path {rel_posix!r}: {exc}")
            continue

        # Strict containment check: resolved target must be strictly inside resolved_root
        try:
            target_file.relative_to(resolved_root)
        except ValueError:
            errors.append(f"Path escape in manifest: {rel_posix!r}")
            continue

        if target_file == resolved_root:
            errors.append(f"Path references snapshot root: {rel_posix!r}")
            continue

        # Reject symlinks
        if raw_candidate.is_symlink() or target_file.is_symlink():
            errors.append(f"Symlinks not allowed in manifest: {rel_posix!r}")
            continue

        if not target_file.is_file():
            errors.append(f"Missing file listed in manifest: {rel_posix}")
            continue

        actual_size = target_file.stat().st_size
        expected_size = item.get("bytes")
        if actual_size != expected_size:
            errors.append(
                f"Size mismatch for {rel_posix}: on-disk={actual_size}, manifest={expected_size}"
            )

        actual_sha = sha256_file(target_file)
        expected_sha = item.get("sha256")
        if actual_sha != expected_sha:
            errors.append(
                f"SHA-256 mismatch for {rel_posix}: on-disk={actual_sha}, manifest={expected_sha}"
            )

        listed_files.add(rel_posix)
        calculated_size += actual_size

    # Check for unlisted files on disk
    actual_files = {
        p.relative_to(resolved_root).as_posix()
        for p in resolved_root.rglob("*")
        if p.is_file() and p.name != MANIFEST_NAME
    }

    unlisted = actual_files - listed_files
    if unlisted:
        errors.append(f"Unlisted files found on disk: {sorted(unlisted)}")

    if manifest.get("totalBytes") != calculated_size:
        errors.append(
            f"Manifest totalBytes mismatch: manifest={manifest.get('totalBytes')}, "
            f"calculated={calculated_size}"
        )

    if not any(f.endswith(".safetensors") for f in listed_files):
        errors.append("No .safetensors files listed in manifest")

    return len(errors) == 0, errors


def package_dimer_zip(model_dir: Path, output_zip: Path) -> str:
    """Package model snapshot directory into a DIMER ZIP archive and return outer SHA-256."""
    output_zip = output_zip.resolve()
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_STORED) as z:
        for p in sorted(model_dir.rglob("*")):
            if p.is_file() and p.resolve() != output_zip:
                if p.is_symlink():
                    raise ValueError(f"Symlinks not allowed in ZIP archive: {p}")
                z.write(p, p.relative_to(model_dir).as_posix())

    return sha256_file(output_zip)


def dry_run_model(
    entry: ModelEntry,
    token: str | None = None,
) -> None:
    """Query Hugging Face Hub metadata to display download files and sizes."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    print(f"\n[Dry Run] Querying Hugging Face for {entry.model_id} (rev: {entry.revision[:10]})...")
    info = api.model_info(entry.model_id, revision=entry.revision, files_metadata=True)

    candidates = [
        s for s in info.siblings if matches_patterns(s.rfilename)
    ]

    total_size = sum(s.size or 0 for s in candidates)
    print(f"\nTarget files ({len(candidates)} files, {format_bytes(total_size)} total):")
    for s in sorted(candidates, key=lambda x: x.rfilename):
        size_str = format_bytes(s.size) if s.size is not None else "unknown"
        print(f"  - {s.rfilename:<42} ({size_str})")
    print(f"\nTotal expected download: {format_bytes(total_size)}\n")


def fetch_model_weights(
    model_key: str = DEFAULT_MODEL_KEY,
    dest_dir: Path | None = None,
    *,
    token: str | None = None,
    write_manifest: bool = True,
    verify_after_download: bool = True,
    create_zip: Path | None = None,
    dry_run: bool = False,
    force: bool = False,
    registry_path: Path | None = None,
) -> int:
    """Download base model weights and generate snapshot manifest."""
    registry, specs = get_model_specs(registry_path)

    if model_key not in registry._entries:
        valid_keys = ", ".join(sorted(registry._entries.keys()))
        print(
            f"Error: Unknown model_key '{model_key}'.\nAvailable keys: {valid_keys}",
            file=sys.stderr,
        )
        return 1

    entry = registry._entries[model_key]
    spec = specs.get(model_key) or {}

    if not entry.revision:
        print(f"Error: Model '{model_key}' has no pinned immutable revision.", file=sys.stderr)
        return 1

    if entry.requires_trust_remote_code and not force:
        print(
            f"Error: Model '{model_key}' requires trust_remote_code=True, which is forbidden "
            f"under SECURITY.md. Pass --force to override for testing.",
            file=sys.stderr,
        )
        return 1

    if not entry.enabled and not force:
        blocked_reason = spec.get("blocked_reason", entry.approval_state)
        print(
            f"Error: Model '{model_key}' is disabled in the registry "
            f"(reason: '{blocked_reason}').\n"
            f"Pass --force to override for testing/evaluation.",
            file=sys.stderr,
        )
        return 1

    # Check for gated token requirement
    is_gated = bool(spec.get("gated") or model_key == "llama-3.2-3b-instruct")
    hf_token = token or os.environ.get("HF_TOKEN")
    if is_gated and not hf_token:
        print(
            f"Error: Model '{model_key}' ({entry.model_id}) is gated on Hugging Face.\n"
            f"Please provide an access token via --token or the HF_TOKEN environment variable.",
            file=sys.stderr,
        )
        return 1

    if dest_dir is None:
        dest_dir = DEFAULT_DEST_DIR / model_key
    dest_dir = dest_dir.resolve()

    if dry_run:
        try:
            from huggingface_hub import HfApi  # noqa: F401
        except ImportError:
            print(
                "Error: 'huggingface_hub' is required for dry-run inspection.\n"
                "Install it via: pip install huggingface-hub",
                file=sys.stderr,
            )
            return 1
        try:
            dry_run_model(entry, token=hf_token)
            return 0
        except Exception as exc:
            print(f"Dry-run failed: {exc}", file=sys.stderr)
            return 1

    dest_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Fetching Base Model Weights: {model_key}")
    print(f"Repository ID:  {entry.model_id}")
    print(f"Revision:       {entry.revision}")
    print(f"License:        {entry.license}")
    print(f"Destination:    {dest_dir}")
    print("=" * 70)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "Error: 'huggingface_hub' is required to download model weights.\n"
            "Install it via: pip install huggingface-hub",
            file=sys.stderr,
        )
        return 1

    try:
        snapshot_download(
            repo_id=entry.model_id,
            revision=entry.revision,
            local_dir=str(dest_dir),
            token=hf_token,
            allow_patterns=ALLOW_PATTERNS,
            ignore_patterns=IGNORE_PATTERNS,
        )
    except Exception as exc:
        print(f"\nDownload failed: {exc}", file=sys.stderr)
        return 1

    # Clean up internal Hugging Face cache metadata directory if created
    cache_dir = dest_dir / ".cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)

    # Check that safetensors weights exist
    safetensors_files = list(dest_dir.glob("*.safetensors"))
    if not safetensors_files:
        print("Error: No .safetensors weights found in downloaded files.", file=sys.stderr)
        return 1

    # Write dimer-base-manifest.json
    if write_manifest:
        print(f"\nGenerating {MANIFEST_NAME}...")
        manifest = generate_manifest(
            model_dir=dest_dir,
            model_key=entry.key,
            model_id=entry.model_id,
            revision=entry.revision,
        )
        manifest_path = dest_dir / MANIFEST_NAME
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        size_str = format_bytes(manifest["totalBytes"])
        print(f"[OK] Manifest written ({len(manifest['files'])} files, {size_str})")

    # Verify snapshot integrity
    if verify_after_download:
        print("\nVerifying snapshot integrity...")
        valid, errors = verify_snapshot(
            dest_dir,
            expected_model_key=entry.key,
            expected_model_id=entry.model_id,
            expected_revision=entry.revision,
        )
        if not valid:
            print("Verification FAILED:", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            return 1
        print("[OK] All snapshot files, sizes, and SHA-256 hashes verified.")

    # Package into ZIP if requested
    if create_zip:
        zip_path = create_zip.resolve()
        print(f"\nPackaging DIMER ZIP archive to {zip_path}...")
        outer_sha = package_dimer_zip(dest_dir, zip_path)
        zip_size = format_bytes(zip_path.stat().st_size)
        print(f"[OK] DIMER ZIP created: {zip_path} ({zip_size})")
        print(f"  Outer SHA-256: {outer_sha}")

    print(f"\nSuccessfully prepared weights in: {dest_dir}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download and verify base-model weights for the language-model pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List all registered base models and their metadata, then exit.",
    )
    parser.add_argument(
        "-m", "--model",
        default=DEFAULT_MODEL_KEY,
        help=f"Model key from model-registry.yaml (default: {DEFAULT_MODEL_KEY}).",
    )
    parser.add_argument(
        "-d", "--dest",
        type=Path,
        default=None,
        help="Destination directory (default: weights/<model_key>).",
    )
    parser.add_argument(
        "--subfolder",
        action="store_true",
        help="Download into a subfolder named after the model key (<dest>/<model_key>).",
    )
    parser.add_argument(
        "-t", "--token",
        default=None,
        help="Hugging Face access token for gated models (defaults to HF_TOKEN env var).",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Do not write dimer-base-manifest.json after downloading.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an existing snapshot directory without downloading.",
    )
    parser.add_argument(
        "--zip",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="Package the downloaded snapshot into a DIMER ZIP archive.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check files and total size on Hugging Face without downloading.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow fetching models that are disabled or require trust_remote_code.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Path to custom model registry YAML.",
    )

    args = parser.parse_args(argv)

    if args.list:
        print_model_table(args.registry)
        return 0

    if args.dest is None:
        dest_dir = DEFAULT_DEST_DIR / args.model
    else:
        dest_dir = args.dest
        if args.subfolder:
            dest_dir = dest_dir / args.model

    if args.verify_only:
        target_dir = dest_dir
        if not (target_dir / MANIFEST_NAME).is_file():
            # Check model subfolder if target_dir is root weights/
            candidate = target_dir / args.model
            if (candidate / MANIFEST_NAME).is_file():
                target_dir = candidate
            elif target_dir.is_dir():
                model_subdirs = [
                    p for p in target_dir.iterdir()
                    if p.is_dir() and (p / MANIFEST_NAME).is_file()
                ]
                if len(model_subdirs) == 1:
                    target_dir = model_subdirs[0]

        print(f"Verifying snapshot in {target_dir}...")
        valid, errors = verify_snapshot(target_dir)
        if not valid:
            print("Verification FAILED:", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            return 1
        print(f"[OK] Snapshot in {target_dir} is VALID.")
        return 0

    zip_path = None
    if args.zip is not None:
        if args.zip == "":
            zip_path = dest_dir.parent / f"{dest_dir.name}.zip"
        else:
            zip_path = Path(args.zip)

    return fetch_model_weights(
        model_key=args.model,
        dest_dir=dest_dir,
        token=args.token,
        write_manifest=not args.no_manifest,
        verify_after_download=True,
        create_zip=zip_path,
        dry_run=args.dry_run,
        force=args.force,
        registry_path=args.registry,
    )


if __name__ == "__main__":
    sys.exit(main())
