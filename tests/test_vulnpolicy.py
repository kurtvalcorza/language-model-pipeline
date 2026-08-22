"""The vulnerability policy must gate, not decorate.

Issue #22 asks for a documented severity policy, an exception process that records what/why/
who/until-when, and proof that a known vulnerability actually fails the gate. Each of those
is asserted here, and each rule is shown BLOCKING as well as passing -- a gate that has only
ever passed is indistinguishable from one that never fires.

The severity split is not arbitrary and the tests say so: strictness tracks controllability.
We pin our Python dependencies and can bump them, so any fixable finding blocks. Base-image
OS packages come from an upstream image pinned so that sm_120 works at all, so only a
CRITICAL with a fix blocks. Measured 2026-08-22 on python:3.12-slim: "any HIGH with a fix"
would have blocked 36 unactionable Debian findings on day one.
"""

from __future__ import annotations

import datetime as dt

import pytest
import yaml

from lmpipeline.vulnpolicy import (  # noqa: I001
    POLICY_PATH,
    Finding,
    PolicyError,
    ScanError,
    evaluate,
    load_policy,
    parse_pip_audit,
    parse_trivy,
    render,
)

TODAY = dt.date(2026, 8, 22)


def _policy(**overrides):
    doc = {
        "schema_version": "1.0",
        "dependencies": {"block_when_fix_available": True, "block_when_no_fix": False},
        "images": {"block_severities": ["CRITICAL"], "require_fix_available": True},
        "exceptions": [],
    }
    doc.update(overrides)
    return doc


def _dep(identifier="PYSEC-1", fix="9.9.9"):
    return Finding("pip-audit", identifier, "transformers", "4.57.1", "UNKNOWN", fix)


def _img(identifier="CVE-1", severity="CRITICAL", fix="1.2.3"):
    return Finding("trivy", identifier, "libssl", "1.0.0", severity, fix)


# -- the committed policy itself ---------------------------------------------------


def test_the_committed_policy_loads_and_is_complete():
    policy = load_policy()
    assert policy["dependencies"]["block_when_fix_available"] is True
    assert policy["images"]["block_severities"] == ["CRITICAL"]


def test_every_committed_exception_records_what_why_who_and_until_when():
    """The audit trail IS the exception process; a bare suppression is not one."""
    policy = load_policy()
    for entry in policy["exceptions"]:
        for required in ("id", "package", "rationale", "owner", "expires", "scope"):
            assert entry.get(required), f"{entry.get('id')} lacks {required}"
        dt.date.fromisoformat(str(entry["expires"]))
        assert len(entry["rationale"].split()) >= 20, (
            f"{entry['id']}: a rationale must explain the MECHANISM by which the code path "
            "is unreachable or mitigated, not merely assert that it is"
        )


def test_an_exception_missing_a_field_is_rejected(tmp_path):
    doc = _policy(exceptions=[{"id": "X", "package": "p", "owner": "o",
                               "expires": "2099-01-01", "scope": ["validator"]}])
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(PolicyError, match="rationale"):
        load_policy(path)


def test_an_unparseable_expiry_is_rejected(tmp_path):
    doc = _policy(exceptions=[{"id": "X", "package": "p", "rationale": "r", "owner": "o",
                               "expires": "soon", "scope": ["validator"]}])
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(PolicyError, match="expires"):
        load_policy(path)


def test_a_missing_policy_is_an_error_not_an_empty_pass(tmp_path):
    """The failure mode that would quietly disable the gate entirely."""
    with pytest.raises(PolicyError):
        load_policy(tmp_path / "absent.yaml")


# -- dependency findings -----------------------------------------------------------


def test_a_fixable_dependency_finding_blocks():
    """The acceptance criterion: a known vulnerability fails the gate."""
    decision = evaluate([_dep()], policy=_policy(), scope="validator", today=TODAY)
    assert not decision.ok
    assert decision.blocking


def test_an_unfixable_dependency_finding_is_reported_not_blocking():
    decision = evaluate([_dep(fix=None)], policy=_policy(), scope="validator", today=TODAY)
    assert decision.ok
    assert decision.reported


# -- image findings ----------------------------------------------------------------


def test_a_fixable_critical_image_finding_blocks():
    decision = evaluate([_img()], policy=_policy(), scope="finetuner", today=TODAY)
    assert not decision.ok


def test_an_unfixed_critical_is_reported_because_it_is_not_actionable_here():
    decision = evaluate([_img(fix=None)], policy=_policy(), scope="finetuner", today=TODAY)
    assert decision.ok


def test_a_fixable_high_is_reported_not_blocking():
    """Blocking these would have meant 36 unactionable Debian findings on day one."""
    decision = evaluate([_img(severity="HIGH")], policy=_policy(), scope="finetuner",
                        today=TODAY)
    assert decision.ok
    assert decision.reported


# -- exceptions --------------------------------------------------------------------


def _with_exception(expires, scope=("validator", "finetuner"), identifier="PYSEC-1"):
    return _policy(exceptions=[{
        "id": identifier, "package": "transformers", "rationale": "unreachable: no Trainer",
        "owner": "kurtvalcorza", "expires": expires, "scope": list(scope),
    }])


def test_a_live_exception_suppresses_the_block():
    decision = evaluate([_dep()], policy=_with_exception("2099-01-01"),
                        scope="validator", today=TODAY)
    assert decision.ok
    assert decision.excepted


def test_an_expired_exception_blocks_again():
    """The point of time-bounding: silence must lapse, not become permanent."""
    decision = evaluate([_dep()], policy=_with_exception("2026-08-21"),
                        scope="validator", today=TODAY)
    assert not decision.ok
    assert decision.expired
    assert "renew" in decision.expired[0][1]


def test_an_exception_expiring_today_is_still_live():
    """Off-by-one on the boundary would silently re-block a fresh renewal."""
    decision = evaluate([_dep()], policy=_with_exception("2026-08-22"),
                        scope="validator", today=TODAY)
    assert decision.ok


def test_an_exception_does_not_leak_across_scopes():
    """An argument about the finetuner says nothing about the validator."""
    decision = evaluate([_dep()], policy=_with_exception("2099-01-01", scope=("finetuner",)),
                        scope="validator", today=TODAY)
    assert not decision.ok


def test_an_exception_matches_on_alias_too():
    """Scanners disagree on identifiers: PYSEC-… here, CVE-… or GHSA-… elsewhere."""
    policy = _policy(exceptions=[{
        "id": "PYSEC-2026-2288", "aliases": ["CVE-2026-1839"], "package": "transformers",
        "rationale": "unreachable", "owner": "k", "expires": "2099-01-01",
        "scope": ["finetuner"],
    }])
    decision = evaluate([_dep(identifier="CVE-2026-1839")], policy=policy,
                        scope="finetuner", today=TODAY)
    assert decision.ok
    assert decision.excepted


# -- parsers -----------------------------------------------------------------------


def test_pip_audit_output_is_parsed():
    doc = {"dependencies": [
        {"name": "transformers", "version": "4.57.1",
         "vulns": [{"id": "PYSEC-2026-2288", "fix_versions": ["5.0.0"]},
                   {"id": "PYSEC-2025-217", "fix_versions": []}]},
        {"name": "PyYAML", "version": "6.0.3", "vulns": []},
    ]}
    findings = parse_pip_audit(doc)
    assert len(findings) == 2
    assert findings[0].fix == "5.0.0"
    assert findings[1].fix is None


def test_trivy_output_is_parsed():
    doc = {"SchemaVersion": 2, "Results": [{"Vulnerabilities": [
        {"VulnerabilityID": "CVE-1", "PkgName": "libssl", "InstalledVersion": "1.0",
         "Severity": "CRITICAL", "FixedVersion": "1.1"},
        {"VulnerabilityID": "CVE-2", "PkgName": "zlib", "InstalledVersion": "1.0",
         "Severity": "LOW"},
    ]}]}
    findings = parse_trivy(doc)
    assert [f.severity for f in findings] == ["CRITICAL", "LOW"]
    assert findings[1].fix is None


def test_empty_scanner_output_is_now_refused_rather_than_read_as_clean():
    """This test previously asserted the DEFECT.

    It required `parse_*({}) == []`, i.e. that a crashed scanner's empty output be read as
    zero findings -- which is precisely how a broken gate turns green. The behaviour is
    deliberately reversed: an incomplete document is now a ScanError.
    """
    with pytest.raises(ScanError):
        parse_pip_audit({})
    with pytest.raises(ScanError):
        parse_trivy({})


def test_render_names_the_blocking_findings():
    decision = evaluate([_dep()], policy=_policy(), scope="validator", today=TODAY)
    text = render(decision)
    assert "BLOCKING" in text and "PYSEC-1" in text


# -- the real advisories this repository actually carries ---------------------------


def test_the_two_transformers_advisories_are_excepted_today():
    """Grounded in the live pip-audit result, not a hypothetical.

    transformers 4.57.1 carries five advisories; three have no fix anywhere and are
    reported, two are fixed only in 5.0.0 / 5.3.0 -- major versions whose chat-template
    rendering would change token counts and therefore which datasets validate. Both are
    excepted with a reachability argument rather than silently ignored.
    """
    policy = load_policy()
    findings = [
        Finding("pip-audit", "PYSEC-2026-2288", "transformers", "4.57.1", "UNKNOWN", "5.0.0"),
        Finding("pip-audit", "PYSEC-2026-2289", "transformers", "4.57.1", "UNKNOWN", "5.3.0"),
        Finding("pip-audit", "PYSEC-2025-217", "transformers", "4.57.1", "UNKNOWN", None),
    ]
    for scope in ("validator", "finetuner"):
        decision = evaluate(findings, policy=policy, scope=scope, today=TODAY)
        assert decision.ok, render(decision)
        assert len(decision.excepted) == 2
        assert len(decision.reported) == 1


def test_those_exceptions_expire_and_then_block():
    """Proof the expiry is real: the same input, a later date, and the gate closes."""
    policy = load_policy()
    finding = Finding("pip-audit", "PYSEC-2026-2288", "transformers", "4.57.1",
                      "UNKNOWN", "5.0.0")
    expiry = dt.date.fromisoformat(
        str(next(e for e in policy["exceptions"] if e["id"] == "PYSEC-2026-2288")["expires"])
    )
    after = expiry + dt.timedelta(days=1)
    decision = evaluate([finding], policy=policy, scope="finetuner", today=after)
    assert not decision.ok
    assert decision.expired


def test_the_policy_file_ships_inside_the_package():
    """It must reach the containers, which vendor the package rather than the repo."""
    assert POLICY_PATH.is_file()
    assert POLICY_PATH.parent.name == "data"


# -- scan ingestion must fail closed ------------------------------------------------
#
# An empty document is what a CRASHED scanner leaves behind. Reading it as zero findings
# turns a broken gate into a green one -- the same defect as a cross-repo check reporting
# success when its credential is absent.
#
# Found by falling into it: a probe that swallowed stderr reported `raw: {}` and PASS for a
# base image that in fact carries 21 blocking findings.
#
# The invariant is NOT "Results must be non-empty". A genuinely clean image legitimately has
# none, and rejecting that would make the gate lie in the other direction. It is that a
# COMPLETED document must be distinguishable from missing output.


def test_a_clean_trivy_scan_is_accepted():
    """{"SchemaVersion": 2, "Results": []} is a real scan of a clean image."""
    assert parse_trivy({"SchemaVersion": 2, "Results": []}) == []


def test_trivy_omitting_results_entirely_is_still_a_clean_scan():
    """Trivy omits Results when the target has nothing analysable. Completed, not failed."""
    assert parse_trivy({"SchemaVersion": 2, "ArtifactName": "scratch"}) == []


def test_an_empty_trivy_document_is_refused():
    """The regression: `{}` used to parse as zero findings and pass the gate."""
    with pytest.raises(ScanError, match="SchemaVersion"):
        parse_trivy({})


def test_malformed_trivy_results_are_refused():
    with pytest.raises(ScanError, match="malformed"):
        parse_trivy({"SchemaVersion": 2, "Results": "not a list"})


def test_non_dict_trivy_output_is_refused():
    with pytest.raises(ScanError):
        parse_trivy([])


def test_a_clean_pip_audit_is_accepted():
    assert parse_pip_audit({"dependencies": []}) == []


def test_an_empty_pip_audit_document_is_refused():
    with pytest.raises(ScanError, match="dependencies"):
        parse_pip_audit({})


def test_malformed_pip_audit_dependencies_are_refused():
    with pytest.raises(ScanError, match="malformed"):
        parse_pip_audit({"dependencies": {}})


def test_the_real_failure_this_prevents():
    """End to end: a crashed scanner must not be able to produce a passing gate.

    Before this change the sequence below returned an empty finding list, which evaluate()
    scored as ok -- a green gate over an image nobody had scanned.
    """
    with pytest.raises(ScanError):
        evaluate(parse_trivy({}), policy=_policy(), scope="finetuner", today=TODAY)
