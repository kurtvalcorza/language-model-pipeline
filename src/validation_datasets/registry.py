"""Dataset source registry: what may be fetched, under what terms, at which revision."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent.parent / "validation-datasets" / "registry.yaml"
)

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

# Licences under which converted rows may be committed to this repository. Everything else
# means: commit the recipe and the digest, never the data.
REDISTRIBUTABLE = {"yes", "share_alike"}


class RegistryError(RuntimeError):
    """Raised for any registry or policy violation. Always fails the build."""


@dataclass(frozen=True)
class Profile:
    name: str
    count: int | None
    selection: str
    # Languages this profile deliberately removes before selection. Tier 3 excludes `tl`
    # so it is independent of Tier 2 by construction rather than by luck — see the
    # acceptance ladder in validation-datasets/README.md.
    exclude_languages: tuple[str, ...] = ()

    @property
    def excludes_languages(self) -> bool:
        return bool(self.exclude_languages)


@dataclass(frozen=True)
class DatasetSource:
    id: str
    source_id: str
    revision: str
    config: str | None
    split: str | None
    license: str
    gated: bool
    redistribution: str
    intended_use: tuple[str, ...]
    languages: tuple[str, ...]
    canonical_schema: str
    conversion_version: int
    approval_state: str
    enabled: bool
    overlaps_with: tuple[str, ...] = ()
    profiles: dict[str, Profile] = field(default_factory=dict)
    # Acceptance ladder position: 1 English, 2 Filipino, 3 multilingual-excluding-tl,
    # 4 held-out Filipino evaluation. Tiers 2 and 3 are independent acceptance runs;
    # tier 4 is the only evaluation tier.
    tier: int | None = None
    # How THIS suite uses the source, which is not always what upstream called it.
    # uner-tagalog publishes only a `test` split and we deliberately repurpose it as SFT
    # acceptance training material. Both facts are recorded so a later reader cannot mistake
    # the run for an independent Tagalog benchmark.
    pipeline_usage: str = "training"
    # Row field carrying the language code. Required before any language exclusion can be
    # applied; `None` means unknown, and an exclusion request then fails closed.
    language_field: str | None = None

    @property
    def is_evaluation_only(self) -> bool:
        return "evaluation_only" in self.intended_use

    @property
    def may_commit_rows(self) -> bool:
        return self.redistribution in REDISTRIBUTABLE and not self.gated

    def profile(self, name: str) -> Profile:
        if name not in self.profiles:
            available = ", ".join(sorted(self.profiles)) or "none"
            raise RegistryError(
                f"{self.id!r} has no profile {name!r}. Available: {available}."
            )
        return self.profiles[name]

    def require_usable_for_training(self) -> None:
        """Fail closed before anything is fetched or converted."""
        if not self.enabled:
            raise RegistryError(
                f"{self.id!r} is disabled (approval_state={self.approval_state!r})."
                + (" It is gated and requires accepting upstream terms." if self.gated
                   else "")
            )
        if self.is_evaluation_only:
            raise RegistryError(
                f"{self.id!r} is evaluation-only and must never be used as training data."
            )
        if not _SHA_RE.match(self.revision or ""):
            raise RegistryError(
                f"{self.id!r} has no immutable pinned revision (got {self.revision!r}). "
                "An acceptance dataset that moves when upstream main moves is not "
                "acceptance data."
            )


class DatasetRegistry:
    def __init__(self, sources: dict[str, DatasetSource], *, schema_version: str):
        self._sources = sources
        self.schema_version = schema_version

    @classmethod
    def load(cls, path: Path | str | None = None) -> DatasetRegistry:
        path = Path(path) if path is not None else REGISTRY_PATH
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        sources = {}
        for key, spec in (raw.get("sources") or {}).items():
            profiles = {
                name: Profile(
                    name=name,
                    count=(p or {}).get("count"),
                    selection=(p or {}).get("selection", "stable_hash"),
                    exclude_languages=tuple((p or {}).get("exclude_languages") or ()),
                )
                for name, p in (spec.get("profiles") or {}).items()
            }
            sources[key] = DatasetSource(
                id=key,
                source_id=spec["source_id"],
                revision=spec.get("revision") or "",
                config=spec.get("config"),
                split=spec.get("split"),
                license=spec.get("license", ""),
                gated=bool(spec.get("gated", False)),
                redistribution=spec.get("redistribution", "unclear"),
                intended_use=tuple(spec.get("intended_use") or ()),
                languages=tuple(spec.get("languages") or ()),
                canonical_schema=spec.get("canonical_schema", "messages"),
                conversion_version=int(spec.get("conversion_version", 0)),
                approval_state=spec.get("approval_state", "experimental"),
                # Sources default to enabled; blocked ones set it explicitly.
                enabled=bool(spec.get("enabled", True)),
                overlaps_with=tuple(spec.get("overlaps_with") or ()),
                profiles=profiles,
                tier=spec.get("tier"),
                pipeline_usage=spec.get("pipeline_usage", "training"),
                language_field=spec.get("language_field"),
            )
        return cls(sources, schema_version=str(raw.get("schema_version", "1.0")))

    def ids(self, *, enabled_only: bool = False) -> list[str]:
        return sorted(
            k for k, s in self._sources.items() if not enabled_only or s.enabled
        )

    def get(self, source_id: str) -> DatasetSource:
        if source_id not in self._sources:
            raise RegistryError(
                f"Unknown dataset {source_id!r}. Known: {', '.join(self.ids())}."
            )
        return self._sources[source_id]

    def overlap_partners(self, source_id: str) -> list[str]:
        """Sources declared to share rows with this one.

        Recorded because two acceptance tiers drawn from the same upstream corpus are not
        independent, and a model accepted on one after training on the other is being
        evaluated on its own training data.
        """
        return sorted(self.get(source_id).overlaps_with)
