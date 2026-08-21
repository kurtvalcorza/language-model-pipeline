from __future__ import annotations

import pytest

from lmpipeline.errors import Code, ModelError
from lmpipeline.registry import ModelRegistry


@pytest.fixture(scope="module")
def registry() -> ModelRegistry:
    return ModelRegistry.load()


def test_shipped_registry_loads(registry):
    assert registry.schema_version == "1.0"
    assert "qwen3-1.7b" in registry.keys()


def test_user_facing_keys_exclude_internal_smoke_model(registry):
    assert "qwen3-0.6b" in registry.keys()
    assert "qwen3-0.6b" not in registry.keys(user_facing=True)


def test_every_enabled_entry_has_an_immutable_revision(registry):
    """The core supply-chain rule: no enabled model may float on a branch."""
    for key in registry.keys():
        entry = registry.resolve(key)
        assert len(entry.revision) == 40
        assert entry.revision != "main"


def test_no_enabled_entry_requires_remote_code(registry):
    for key in registry.keys():
        assert registry.resolve(key).requires_trust_remote_code is False


def test_missing_model_key_fails_structurally(registry):
    with pytest.raises(ModelError) as exc:
        registry.resolve(None)
    assert exc.value.code == Code.MODEL_KEY_MISSING


def test_unknown_model_key_is_rejected(registry):
    with pytest.raises(ModelError) as exc:
        registry.resolve("llama-3-70b")
    assert exc.value.code == Code.MODEL_NOT_APPROVED


def test_remote_code_model_is_blocked_even_though_present(registry):
    """Phi is in the registry so its status is auditable, but must never resolve."""
    with pytest.raises(ModelError) as exc:
        registry.resolve("phi-4-mini-instruct")
    # Disabled is checked before remote-code; either is a correct refusal.
    assert exc.value.code in (Code.MODEL_DISABLED, Code.MODEL_REMOTE_CODE_BLOCKED)


def test_sequence_length_may_narrow_but_not_widen(registry):
    entry = registry.resolve("qwen3-1.7b")
    assert entry.clamp_sequence_length(2048) == 2048
    with pytest.raises(ModelError) as exc:
        entry.clamp_sequence_length(8192)
    assert exc.value.code == Code.CONFIG_OUT_OF_BOUNDS


def test_unsupported_training_method_is_rejected(registry):
    entry = registry.resolve("qwen3-1.7b")
    entry.require_method("lora")
    with pytest.raises(ModelError) as exc:
        entry.require_method("full")
    assert exc.value.code == Code.CONFIG_METHOD_UNSUPPORTED


def test_base_model_mismatch_fails_rather_than_training_wrong_model(registry):
    entry = registry.resolve("qwen3-1.7b")
    assert registry.reconcile_base_model(entry, "Qwen/Qwen3-1.7B") == "Qwen/Qwen3-1.7B"
    assert registry.reconcile_base_model(entry, None) is None
    with pytest.raises(ModelError) as exc:
        registry.reconcile_base_model(entry, "Qwen/Qwen3-4B")
    assert exc.value.code == Code.MODEL_BASE_MODEL_CONFLICT


def test_resource_profiles_are_per_method(registry):
    """A single per-model number cannot express "qlora fits, lora does not"."""
    entry = registry.resolve("qwen3-1.7b")
    assert entry.resource_profile.for_method("qlora").min_vram_gb == 9.3
    # LoRA exceeded the measurement GPU and spilled, so it carries no number.
    assert entry.resource_profile.for_method("lora").min_vram_gb is None
    # An undeclared method resolves to an empty profile rather than raising.
    assert entry.resource_profile.for_method("full").min_vram_gb is None


def test_every_measured_number_names_its_hardware(registry):
    """A profile without provenance is indistinguishable from a guess."""
    for key in registry.keys():
        profile = registry.resolve(key).resource_profile
        for method, method_profile in profile.methods.items():
            if method_profile.min_vram_gb is not None:
                assert method_profile.measured_on, (
                    f"{key}/{method} has a VRAM number but no measured_on"
                )
            else:
                assert method_profile.measured_on is None, (
                    f"{key}/{method} claims hardware but records no measurement"
                )


def test_unmeasured_combinations_stay_null(registry):
    """Spilled and OOM runs must not become numbers.

    qwen3-4b LoRA ran out of memory and granite LoRA spilled to host RAM; neither is a
    valid measurement, so both stay null until measured on hardware that fits them.
    """
    assert registry.resolve("qwen3-4b").resource_profile.for_method(
        "lora"
    ).min_vram_gb is None
    assert registry.resolve("granite-4.1-3b").resource_profile.for_method(
        "lora"
    ).min_vram_gb is None
