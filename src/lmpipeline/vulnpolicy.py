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


class ScanError(RuntimeError):
    """Scanner output is missing, truncated or not a scan result at all.

    Distinct from "the scan found nothing", and that distinction is the entire point. An
    empty document is what a CRASHED scanner leaves behind, and treating it as zero
    findings turns a broken gate into a green one -- the same defect as a cross-repo check
    that reports success when its credential is absent.

    Found the hard way: a probe that swallowed stderr reported `raw: {}` and PASS for a
    base image that in fact carries 21 blocking findings.
    """


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
        mode = str(entry.get("match", "id"))
        if mode not in ("id", "package"):
            raise PolicyError(
                f"Exception {entry['id']!r} has match={mode!r}; expected 'id' or 'package'."
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
    """Parse pip-audit JSON, refusing anything that is not a completed audit.

    Same reasoning as parse_trivy: `dependencies: []` is a real audit of nothing, but a
    document with no `dependencies` key at all is a failed run, and must not read as clean.
    """
    if not isinstance(document, dict) or "dependencies" not in document:
        raise ScanError(
            "pip-audit output is not a completed audit document: no dependencies field. "
            "This is missing or failed scanner output, not a clean result."
        )

    dependencies = document.get("dependencies")
    if not isinstance(dependencies, list):
        raise ScanError(
            f"pip-audit output has a malformed dependencies field of type "
            f"{type(dependencies).__name__}; expected a list."
        )

    findings: list[Finding] = []
    for dep in dependencies:
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
    """Parse Trivy JSON, refusing anything that is not a completed scan.

    The invariant is NOT "Results must be non-empty" -- a genuinely clean image legitimately
    yields no findings, and rejecting that would make the gate lie in the other direction.
    It is that a completed Trivy document must be *distinguishable* from missing or failed
    scanner output.

    `SchemaVersion` is the marker: Trivy emits it on every successful run, and no partial
    write or crash artefact carries it. So `{"SchemaVersion": 2, "Results": []}` is a clean
    scan and passes, while `{}` -- what a crashed scanner leaves -- is refused.
    """
    if not isinstance(document, dict) or "SchemaVersion" not in document:
        raise ScanError(
            "Trivy output is not a completed scan document: no SchemaVersion field. "
            "This is missing or failed scanner output, not a clean result. Check the "
            "scanner's exit code and stderr rather than treating this as zero findings."
        )

    results = document.get("Results")
    if results is None:
        # Trivy omits Results when an image has nothing it can analyse. That is a real,
        # completed scan of a genuinely empty target, so it is zero findings, not an error.
        results = []
    if not isinstance(results, list):
        raise ScanError(
            f"Trivy output has a malformed Results field of type {type(results).__name__}; "
            "expected a list. Refusing to interpret it as zero findings."
        )

    findings: list[Finding] = []
    for result in results:
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
    """Find the exception covering this finding, if any.

    Two match modes, and the broader one is OPT-IN so it can never be reached by accident:

      match: id       (default) -- this advisory, by id or alias. Use when the argument is
                      about a specific vulnerability.
      match: package  -- every finding in one package. Use ONLY when the reachability
                      argument is about the package rather than any individual CVE, e.g.
                      "nothing in this image can consume these files at all". It
                      deliberately also covers advisories not yet published, which is
                      exactly what makes it broader and why it must be stated explicitly,
                      scoped to one image, and bounded by an expiry.
    """
    for entry in policy.get("exceptions") or []:
        if scope not in (entry.get("scope") or []):
            continue

        mode = str(entry.get("match", "id"))
        if mode == "package":
            if finding.package == str(entry["package"]):
                return entry
            continue

        identifiers = {str(entry["id"])} | {str(a) for a in (entry.get("aliases") or [])}
        if finding.identifier in identifiers:
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
