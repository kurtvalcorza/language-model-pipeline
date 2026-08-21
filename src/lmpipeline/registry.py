"""Model registry loading and Base Model resolution.

The registry is the runtime authority for which models may be fine-tuned. Both the
validator and the finetuner resolve through this module so they cannot disagree about
which model a run refers to.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import Code, ModelError

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

# The registry ships inside the package so a vendored copy cannot drift from the code
# that reads it. See scripts/vendor_sync.py.
DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parent / "data" / "model-registry.yaml"


@dataclass(frozen=True)
class MethodProfile:
    """Measured requirement for one (model, training method) pair.

    Per method, not per model: a 4-bit QLoRA run and a bf16 LoRA run of the same model can
    differ by more than 2x, so a single per-model number would either block QLoRA on a GPU
    where it fits, or admit a LoRA run that cannot.
    """

    min_vram_gb: float | None = None
    measured_on: str | None = None

    @property
    def is_measured(self) -> bool:
        return self.min_vram_gb is not None and self.measured_on is not None


@dataclass(frozen=True)
class ResourceProfile:
    """Per-method measured profiles for one model."""

    methods: dict[str, MethodProfile] = field(default_factory=dict)

    def for_method(self, method: str) -> MethodProfile:
        return self.methods.get(method, MethodProfile())

    @property
    def is_measured(self) -> bool:
        """True only when every declared method carries a real measurement."""
        return bool(self.methods) and all(p.is_measured for p in self.methods.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            method: {"minVramGb": p.min_vram_gb, "measuredOn": p.measured_on}
            for method, p in sorted(self.methods.items())
        }


@dataclass(frozen=True)
class ModelEntry:
    key: str
    model_id: str
    revision: str | None
    provider: str
    license: str
    architecture: str
    backend: str
    requires_trust_remote_code: bool
    training_methods: tuple[str, ...]
    max_sequence_length: int | None
    chat_template: str
    lora_target_modules: Any
    serving_profile: str
    approval_state: str
    enabled: bool
    internal_only: bool = False
    resource_profile: ResourceProfile = field(default_factory=ResourceProfile)

    def require_method(self, method: str) -> None:
        if method not in self.training_methods:
            raise ModelError(
                f"Training method {method!r} is not supported for model {self.key!r}. "
                f"Supported: {', '.join(self.training_methods) or 'none'}.",
                code=Code.CONFIG_METHOD_UNSUPPORTED,
                details={"model_key": self.key, "method": method},
            )

    def clamp_sequence_length(self, requested: int) -> int:
        """Narrow a requested length to the model ceiling, or fail if it exceeds it.

        The DIMER manifest advertises one envelope shared by every registration; this
        registry holds the authoritative per-model ceiling. Requests above it fail loudly
        rather than being silently truncated (DATASET_SPEC.md: no silent truncation).
        """
        if self.max_sequence_length is None:
            raise ModelError(
                f"Model {self.key!r} has no verified max_sequence_length.",
                code=Code.MODEL_NOT_APPROVED,
                details={"model_key": self.key},
            )
        if requested > self.max_sequence_length:
            raise ModelError(
                f"Requested max_sequence_length {requested} exceeds the ceiling "
                f"{self.max_sequence_length} for model {self.key!r}.",
                code=Code.CONFIG_OUT_OF_BOUNDS,
                details={
                    "model_key": self.key,
                    "requested": requested,
                    "ceiling": self.max_sequence_length,
                },
            )
        return requested


class ModelRegistry:
    def __init__(self, entries: dict[str, ModelEntry], *, schema_version: str):
        self._entries = entries
        self.schema_version = schema_version

    @classmethod
    def load(cls, path: Path | str | None = None) -> ModelRegistry:
        path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        models = raw.get("models") or {}
        entries: dict[str, ModelEntry] = {}
        for key, spec in models.items():
            rp = spec.get("resource_profile") or {}
            entries[key] = ModelEntry(
                key=key,
                model_id=spec["model_id"],
                revision=spec.get("revision"),
                provider=spec.get("provider", ""),
                license=spec.get("license", ""),
                architecture=spec.get("architecture", "causal_lm"),
                backend=spec.get("backend", "generic_causal_lm"),
                requires_trust_remote_code=bool(spec.get("requires_trust_remote_code", False)),
                training_methods=tuple(spec.get("training_methods") or ()),
                max_sequence_length=spec.get("max_sequence_length"),
                chat_template=spec.get("chat_template", "tokenizer"),
                lora_target_modules=spec.get("lora_target_modules", "auto"),
                serving_profile=spec.get("serving_profile", "peft_adapter"),
                approval_state=spec.get("approval_state", "experimental"),
                enabled=bool(spec.get("enabled", False)),
                internal_only=bool(spec.get("internal_only", False)),
                resource_profile=ResourceProfile(
                    methods={
                        method: MethodProfile(
                            min_vram_gb=(spec or {}).get("min_vram_gb"),
                            measured_on=(spec or {}).get("measured_on"),
                        )
                        for method, spec in (rp or {}).items()
                    }
                ),
            )
        return cls(entries, schema_version=str(raw.get("schema_version", "1.0")))

    # -- lookup ----------------------------------------------------------------

    def keys(self, *, user_facing: bool = False) -> list[str]:
        return sorted(
            k
            for k, e in self._entries.items()
            if e.enabled and (not user_facing or not e.internal_only)
        )

    def resolve(self, model_key: str | None) -> ModelEntry:
        """Resolve a model_key to an approved, loadable entry, or fail structurally."""
        if not model_key:
            raise ModelError(
                "No model_key was supplied. It is required and must be one of: "
                + ", ".join(self.keys(user_facing=True)),
                code=Code.MODEL_KEY_MISSING,
            )

        entry = self._entries.get(model_key)
        if entry is None:
            raise ModelError(
                f"Unknown model_key {model_key!r}. Approved keys: "
                + ", ".join(self.keys(user_facing=True)),
                code=Code.MODEL_NOT_APPROVED,
                details={"model_key": model_key},
            )
        if not entry.enabled:
            raise ModelError(
                f"Model {model_key!r} is present in the registry but disabled "
                f"(approval_state={entry.approval_state!r}).",
                code=Code.MODEL_DISABLED,
                details={"model_key": model_key, "approval_state": entry.approval_state},
            )
        if entry.requires_trust_remote_code:
            raise ModelError(
                f"Model {model_key!r} requires remote code execution, which is blocked "
                "pending security review (SECURITY.md).",
                code=Code.MODEL_REMOTE_CODE_BLOCKED,
                details={"model_key": model_key},
            )
        if not entry.revision or not _SHA_RE.match(entry.revision):
            raise ModelError(
                f"Model {model_key!r} has no immutable pinned revision "
                f"(got {entry.revision!r}). A 40-character commit SHA is required.",
                code=Code.MODEL_REVISION_UNPINNED,
                details={"model_key": model_key, "revision": entry.revision},
            )
        return entry

    def reconcile_base_model(self, entry: ModelEntry, base_model: str | None) -> str | None:
        """Cross-check DIMER's Base Model field against the resolved registry entry.

        DIMER's Base Model is display/provenance metadata, not the runtime selector — see
        DEPLOYMENT.md. When the platform does supply it, a disagreement means the
        registration and the user's selection describe different models, which must fail
        rather than train the wrong thing silently.
        """
        if not base_model:
            return None
        if base_model.strip() != entry.model_id:
            raise ModelError(
                f"DIMER Base Model {base_model!r} does not match the resolved registry "
                f"entry {entry.key!r} ({entry.model_id}).",
                code=Code.MODEL_BASE_MODEL_CONFLICT,
                details={"base_model": base_model, "model_key": entry.key,
                         "model_id": entry.model_id},
            )
        return base_model.strip()
