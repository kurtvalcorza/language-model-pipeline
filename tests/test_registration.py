"""The DIMER registration block must stay resolvable as the registry changes.

C-2's failure mode is not a bad value, it is a *stale* one: `model_id` resolves against the
pipeline's registered `fineTunableModels`, and an id the block does not carry raises
`ValueError: Unsupported fine-tunable model id` **before the container is created** — no
result document, no callback, nothing for a user to read. Approving a new model in the
registry without re-registering it produces exactly that, and nothing else in this
repository would notice.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from lmpipeline.registry import ModelRegistry

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "build_registration", ROOT / "scripts" / "build_registration.py"
)
build_registration = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_registration)


@pytest.fixture(scope="module")
def registry() -> ModelRegistry:
    return ModelRegistry.load()


@pytest.fixture(scope="module")
def block(registry) -> list[dict]:
    return build_registration.build(registry)


def test_the_generated_registration_passes_the_backends_own_rules(block, registry):
    assert build_registration.check(block, registry) == []


def test_every_user_facing_model_is_registered_and_nothing_else_is(block, registry):
    """Both directions matter.

    A missing id fails at launch. An *extra* id offers the user a model the container will
    refuse with MODEL_KEY_MISSING — same dead end, opposite cause.
    """
    assert sorted(e["id"] for e in block) == sorted(registry.keys(user_facing=True))


def test_internal_only_and_disabled_models_are_never_offered(block, registry):
    offered = {e["id"] for e in block}
    for key in registry.keys():
        entry = registry.resolve(key)
        if entry.internal_only:
            assert key not in offered, f"{key} is internal-only and must not reach the dropdown"

    # phi-4-mini-instruct requires trust_remote_code and is held at enabled: false. The
    # container refuses it too, but a disabled model reaching the Builder means a user can
    # pick something guaranteed to fail.
    assert "phi-4-mini-instruct" not in offered


def test_the_default_model_id_is_one_of_the_registered_ids(block):
    assert build_registration.DEFAULT_MODEL_ID in {e["id"] for e in block}


def test_no_field_is_left_for_the_backend_to_substitute_a_yolo_default_into(block):
    """Omitting these does not fail — it silently yields ultralytics/yolo values.

    `get_pipeline_fine_tunable_models` fills `framework` and `provider` with "ultralytics",
    `supportedDatasetFormat` with "yolo", and `defaultTrainingParams` with YOLO's
    COMMON_TRAINING_DEFAULTS, whose learning rate is 50x ours and which carry `mosaic`,
    `degrees` and `fliplr`. Those values then travel to the container in
    DIMER_MODEL_CONFIG_JSON as though we had chosen them.
    """
    for entry in block:
        assert entry["framework"] == "transformers-peft"
        assert entry["provider"] not in ("", "ultralytics")
        assert entry["supportedDatasetFormat"] == "jsonl_messages"
        params = entry["defaultTrainingParams"]
        assert params["method"] in ("lora", "qlora")
        assert not {"mosaic", "degrees", "fliplr"} & set(params)


def test_base_weights_carry_the_pinned_revision_and_are_unique(block, registry):
    """`baseWeights` is opaque to the backend but is the key of its legacy lookup map.

    Two entries sharing one value would collide in `model_ids_by_base_weights` and silently
    remap one model's legacy selector onto another's id.
    """
    seen = set()
    for entry in block:
        model_id, _, revision = entry["baseWeights"].partition("@")
        registry_entry = registry.resolve(entry["id"])
        assert model_id == registry_entry.model_id
        assert revision == registry_entry.revision
        assert len(revision) == 40, "a pinned revision is a full commit sha, never a branch"
        assert entry["baseWeights"] not in seen
        seen.add(entry["baseWeights"])


def test_sequence_length_defaults_never_exceed_the_model_ceiling(block, registry):
    for entry in block:
        requested = entry["defaultTrainingParams"]["max_sequence_length"]
        ceiling = registry.resolve(entry["id"]).max_sequence_length
        assert ceiling is None or requested <= ceiling


def test_check_catches_a_registration_that_drifted_from_the_registry(block, registry):
    """The gate has to fail on the realistic mistake, not only pass on the good input."""
    stale = [dict(entry) for entry in block]
    stale[0]["id"] = "qwen3-42b-that-was-never-approved"
    problems = build_registration.check(stale, registry)
    assert any("not an approved registry key" in p for p in problems)

    dropped = [dict(entry) for entry in block[1:]]
    problems = build_registration.check(dropped, registry)
    assert any("absent from the registration" in p for p in problems)

    empty_weights = [dict(entry) for entry in block]
    empty_weights[0]["baseWeights"] = ""
    assert any("empty baseWeights" in p for p in build_registration.check(empty_weights, registry))
