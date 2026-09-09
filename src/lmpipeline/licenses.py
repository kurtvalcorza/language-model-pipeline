"""Machine-readable derivative-license obligations for model artifacts.

The model registry records the upstream license identifier. This module turns that identifier
into the small set of downstream actions the artifact contract can enforce or surface. Keep
these identifiers stable: they are provenance data consumed after a training run, not display
copy.
"""

from __future__ import annotations

from typing import Final

PRESERVE_LLAMA_ATTRIBUTION: Final = "preserve_llama_attribution"
INCLUDE_APPLICABLE_LICENSE_NOTICE: Final = "include_applicable_license_notice"

_LICENSE_OBLIGATIONS: Final[dict[str, tuple[str, ...]]] = {
    "llama3.2": (
        PRESERVE_LLAMA_ATTRIBUTION,
        INCLUDE_APPLICABLE_LICENSE_NOTICE,
    ),
}


def obligations_for_license(license_id: str | None) -> tuple[str, ...]:
    """Return stable downstream-obligation identifiers for an upstream license.

    Unknown or ordinary permissive licenses intentionally resolve to an empty tuple. Adding a
    new non-empty mapping is a policy change and should be backed by an explicit license review;
    the function never guesses obligations from a license name.
    """
    if not license_id:
        return ()
    return _LICENSE_OBLIGATIONS.get(license_id.strip().lower(), ())


def license_notice_markdown(license_id: str | None, model_id: str) -> str:
    """Render the artifact notice required by the known license policy, if any.

    This is not a substitute for the upstream license text. It makes the obligations carried in
    provenance visible in the generated model card so an exported adapter does not lose the
    attribution/notice requirement at the point of distribution.
    """
    if (license_id or "").strip().lower() != "llama3.2":
        return ""

    return f"""## License and attribution obligations

This adapter is a derivative of `{model_id}`, licensed under the Meta Llama 3.2 Community
License. When distributing or deploying this adapter, preserve the applicable Meta Llama
attribution and include the applicable Llama 3.2 license notice. The upstream license terms for
the pinned base model remain controlling; this notice does not replace them.
"""
