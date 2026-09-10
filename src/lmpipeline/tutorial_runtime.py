"""Secure source/bootstrap helpers for release-grade tutorial notebooks.

These helpers are notebook orchestration only. They do not implement training, inference,
model loading, masking, or artifact publication.
"""

from __future__ import annotations

import base64
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .tutorial_api import consume_adapter_archive as _consume_adapter_archive

_FINETUNER_URL = "https://github.com/kurtvalcorza/language-model-finetuner.git"


def github_token_from_runtime() -> str:
    """Read an authorized GitHub token without printing or persisting it.

    Plain Jupyter users may provide ``GITHUB_TOKEN`` in the process environment. Colab users
    may instead create a private secret named ``GITHUB_TOKEN`` and grant notebook access.
    """
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        return token

    try:
        from google.colab import userdata
    except ImportError:
        userdata = None

    if userdata is not None:
        try:
            token = (userdata.get("GITHUB_TOKEN") or "").strip()
        except Exception:
            token = ""
        if token:
            return token

    raise RuntimeError(
        "language-model-finetuner is private. Provide a GitHub token with read access as "
        "Colab Secret GITHUB_TOKEN (with notebook access enabled) or as the GITHUB_TOKEN "
        "environment variable. The token is used only through an ephemeral Git auth header."
    )


def _git_auth_environment(token: str) -> dict[str, str]:
    if not token or any(character.isspace() for character in token):
        raise ValueError("GitHub token must be a non-empty single-line secret")
    credential = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    env = os.environ.copy()
    env.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.https://github.com/.extraHeader",
            "GIT_CONFIG_VALUE_0": f"Authorization: Basic {credential}",
        }
    )
    return env


def checkout_private_finetuner(
    destination: str | Path,
    *,
    revision: str,
    token: str,
) -> Path:
    """Clone the private production repo and detach at an immutable commit.

    Authentication is supplied through process environment, not command arguments or the
    remote URL, so the secret is not written into notebook output or repository config.
    """
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("finetuner revision must be a 40-character commit SHA")
    destination = Path(destination).resolve()
    shutil.rmtree(destination, ignore_errors=True)
    env = _git_auth_environment(token)

    subprocess.run(
        ["git", "clone", "--quiet", "--no-checkout", _FINETUNER_URL, str(destination)],
        check=True,
        env=env,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--quiet", "--detach", revision],
        check=True,
        env=env,
    )
    return destination


def consume_adapter_archive(
    archive_path: str | Path,
    *,
    extraction_root: str | Path,
    expected_archive_sha256: str = "",
    size_limit_bytes: int = 512 * 1024**2,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Consume an adapter ZIP and require the manifest at the archive root.

    The underlying support helper validates traversal, links, duplicates, sizes, hashes and
    the manifested subtree. Requiring the manifest at the extraction root closes the remaining
    ambiguity where an attacker could place an otherwise valid artifact in a subdirectory and
    add an unmanifested sibling outside that subtree.
    """
    extraction_root = Path(extraction_root).resolve()
    root, manifest, provenance = _consume_adapter_archive(
        archive_path,
        extraction_root=extraction_root,
        expected_archive_sha256=expected_archive_sha256,
        size_limit_bytes=size_limit_bytes,
    )
    if root.resolve() != extraction_root:
        raise ValueError("artifact-manifest.json must be at the ZIP archive root")
    return root, manifest, provenance
