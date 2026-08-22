"""Decide which vulnerability findings block a release, and which are merely reported.

A scanner that emits every finding as a failure gets muted within a week; one that emits
everything as a warning was never a gate. This module is the difference: it applies one
committed policy to both scanners' output, so `validator` and `finetuner` cannot drift on
what counts as release-blocking.

Two scanners, deliberately different strictness, because strictness tracks CONTROLLABILITY:

  * `pip-audit` covers the Python dependencies we pin ourselves. We can bump those, so any
    finding with a fix available blocks.
  * Trivy covers the built images, whose OS packages come from an upstream base pinned for
    correctness (the CUDA/torch base is why sm_120 works at all). Only a CRITICAL with a fix
    available blocks; the rest is reported.

Exceptions are time-bounded and an EXPIRED exception blocks. That is the point of the
expiry: renewal must be a deliberate act carrying a fresh reachability argument, not
something that lapses into permanence by being forgotten.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

POLICY_PATH = Path(__file__).resolve().parent / "data" / "vulnerability-policy.yaml"

REQUIRED_EXCEPTION_FIELDS = ("id", "package", "rationale", "owner", "expires", "scope")


class PolicyError(RuntimeError):
    """The policy file itself is unusable. Never treated as 'no findings'."""


@dataclass(frozen=True)
class Finding:
    source: str          # "pip-audit" | "trivy"
    identifier: str
    package: str
    installed: str
    severity: str        # UNKNOWN for pip-audit, which does not carry one
    fix: str | None

    def describe(self) -> str:
        fix = f"fix={self.fix}" if self.fix else "no fix available"
        return (f"[{self.source}] {self.identifier} {self.package} {self.installed} "
                f"severity={self.severity} {fix}")


@dataclass
class Decision:
    blocking: list[Finding] = field(default_factory=list)
    excepted: list[tuple[Finding, str]] = field(default_factory=list)
    reported: list[Finding] = field(default_factory=list)
    expired: list[tuple[Finding, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.blocking and not self.expired


def load_policy(path: Path | str | None = None) -> dict[str, Any]:
    """Load and validate the policy. A malformed policy is an error, never an empty pass."""
    path = Path(path) if path is not None else POLICY_PATH
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PolicyError(f"No vulnerability policy at {path}") from exc
    except yaml.YAMLError as exc:
        raise PolicyError(f"Vulnerability policy at {path} is not valid YAML: {exc}") from exc

    if not isinstance(doc, dict) or "dependencies" not in doc or "images" not in doc:
        raise PolicyError(f"Vulnerability policy at {path} is missing required sections")

    for entry in doc.get("exceptions") or []:
        missing = [f for f in REQUIRED_EXCEPTION_FIELDS if not entry.get(f)]
        if missing:
            raise PolicyError(
                f"Exception {entry.get('id', '<no id>')!r} is missing {', '.join(missing)}. "
                "Every exception must record what, why, who, until when, and where."
            )
        try:
            _dt.date.fromisoformat(str(entry["expires"]))
        except ValueError as exc:
            raise PolicyError(
                f"Exception {entry['id']!r} has an unparseable expires "
                f"{entry['expires']!r}; use YYYY-MM-DD."
            ) from exc
    return doc


def parse_pip_audit(document: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    for dep in document.get("dependencies") or []:
        for vuln in dep.get("vulns") or []:
            fixes = vuln.get("fix_versions") or []
            findings.append(Finding(
                source="pip-audit",
                identifier=vuln.get("id", "?"),
                package=dep.get("name", "?"),
                installed=dep.get("version", "?"),
                severity="UNKNOWN",
                fix=",".join(fixes) if fixes else None,
            ))
    return findings


def parse_trivy(document: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    for result in document.get("Results") or []:
        for vuln in result.get("Vulnerabilities") or []:
            findings.append(Finding(
                source="trivy",
                identifier=vuln.get("VulnerabilityID", "?"),
                package=vuln.get("PkgName", "?"),
                installed=vuln.get("InstalledVersion", "?"),
                severity=vuln.get("Severity", "UNKNOWN"),
                fix=vuln.get("FixedVersion") or None,
            ))
    return findings


def _matching_exception(
    finding: Finding, policy: dict[str, Any], *, scope: str
) -> dict[str, Any] | None:
    for entry in policy.get("exceptions") or []:
        identifiers = {str(entry["id"])} | {str(a) for a in (entry.get("aliases") or [])}
        if finding.identifier not in identifiers:
            continue
        if scope not in (entry.get("scope") or []):
            continue
        return entry
    return None


def _is_blocking(finding: Finding, policy: dict[str, Any]) -> bool:
    if finding.source == "pip-audit":
        rules = policy["dependencies"]
        if finding.fix:
            return bool(rules.get("block_when_fix_available", True))
        return bool(rules.get("block_when_no_fix", False))

    rules = policy["images"]
    if finding.severity not in (rules.get("block_severities") or []):
        return False
    if rules.get("require_fix_available", True) and not finding.fix:
        return False
    return True


def evaluate(
    findings: list[Finding],
    *,
    policy: dict[str, Any],
    scope: str,
    today: _dt.date | None = None,
) -> Decision:
    """Sort findings into blocking, excepted, expired and reported.

    `today` is injectable so the expiry rule is testable without waiting for a date to pass.
    """
    today = today or _dt.date.today()
    decision = Decision()

    for finding in findings:
        if not _is_blocking(finding, policy):
            decision.reported.append(finding)
            continue

        entry = _matching_exception(finding, policy, scope=scope)
        if entry is None:
            decision.blocking.append(finding)
            continue

        expires = _dt.date.fromisoformat(str(entry["expires"]))
        if expires < today:
            decision.expired.append((
                finding,
                f"exception expired {expires.isoformat()} (owner {entry['owner']}); "
                "renew with a fresh reachability argument or fix the finding",
            ))
        else:
            decision.excepted.append((finding, f"{entry['owner']} until {expires.isoformat()}"))

    return decision


def render(decision: Decision) -> str:
    lines: list[str] = []
    if decision.blocking:
        lines.append(f"BLOCKING ({len(decision.blocking)}):")
        lines += [f"  {f.describe()}" for f in decision.blocking]
    if decision.expired:
        lines.append(f"EXPIRED EXCEPTIONS ({len(decision.expired)}):")
        lines += [f"  {f.describe()} -- {why}" for f, why in decision.expired]
    if decision.excepted:
        lines.append(f"excepted ({len(decision.excepted)}):")
        lines += [f"  {f.describe()} -- {why}" for f, why in decision.excepted]
    if decision.reported:
        lines.append(f"reported, not blocking ({len(decision.reported)})")
    if not lines:
        lines.append("no findings")
    return "\n".join(lines)
