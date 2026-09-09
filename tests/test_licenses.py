from __future__ import annotations

from lmpipeline.licenses import (
    INCLUDE_APPLICABLE_LICENSE_NOTICE,
    PRESERVE_LLAMA_ATTRIBUTION,
    license_notice_markdown,
    obligations_for_license,
)


def test_llama_32_obligations_are_explicit_and_stable():
    assert obligations_for_license("llama3.2") == (
        PRESERVE_LLAMA_ATTRIBUTION,
        INCLUDE_APPLICABLE_LICENSE_NOTICE,
    )


def test_permissive_and_unknown_licenses_do_not_gain_guessed_obligations():
    assert obligations_for_license("apache-2.0") == ()
    assert obligations_for_license("mit") == ()
    assert obligations_for_license("future-license") == ()
    assert obligations_for_license(None) == ()


def test_llama_notice_names_the_pinned_base_and_required_actions():
    notice = license_notice_markdown(
        "llama3.2", "meta-llama/Llama-3.2-3B-Instruct"
    )
    normalized = " ".join(notice.split())
    assert "meta-llama/Llama-3.2-3B-Instruct" in normalized
    assert "Meta Llama 3.2 Community" in normalized
    assert "preserve the applicable Meta Llama attribution" in normalized
    assert "include the applicable Llama 3.2 license notice" in normalized


def test_no_extra_notice_is_invented_for_apache_models():
    assert license_notice_markdown("apache-2.0", "Qwen/Qwen3-1.7B") == ""
