"""The JSON Schemas, checked against real documents and shown to reject bad ones.

Issue #9. Two rules shape every test here:

  * **Validate what the code actually emits.** A schema checked only against hand-authored
    examples proves nothing — the examples get written until they pass, and the emitter is
    never consulted. So the registry test reads the real YAML, and the result tests build
    documents through `Result.to_dict()` exactly as the containers do.
  * **Show each schema rejecting.** A schema that has only ever accepted is
    indistinguishable from one that accepts everything.

The consumer repositories carry the other half: the validator validates output from real
`validate()` runs, and the finetuner validates a real `Job.to_dict()` and a real training
result.
"""

from __future__ import annotations

import json

import pytest
import yaml
from jsonschema import Draft202012Validator

from lmpipeline.errors import Code, Stage
from lmpipeline.registry import DEFAULT_REGISTRY_PATH
from lmpipeline.result import Result
from lmpipeline.schema import (
    MODEL_REGISTRY,
    SCHEMA_NAMES,
    TRAINING_RESULT,
    VALIDATION_RESULT,
    SchemaError,
    load_schema,
    schema_path,
    validate_document,
)

# -- the schemas themselves ----------------------------------------------------


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_every_schema_is_itself_a_valid_schema(name):
    """An invalid schema silently accepts everything in some validators."""
    Draft202012Validator.check_schema(load_schema(name))


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_every_schema_is_committed_and_parseable(name):
    assert schema_path(name).is_file()
    assert json.loads(schema_path(name).read_text(encoding="utf-8"))


def test_an_unknown_schema_name_fails_loudly():
    with pytest.raises(SchemaError):
        load_schema("no-such-thing")


# -- the real registry ---------------------------------------------------------


def _registry_doc() -> dict:
    return yaml.safe_load(DEFAULT_REGISTRY_PATH.read_text(encoding="utf-8"))


def test_the_committed_registry_validates():
    validate_document(MODEL_REGISTRY, _registry_doc())


def test_an_enabled_entry_may_not_pin_a_branch():
    """The invariant that was previously only a test: enabled means immutably pinned."""
    doc = _registry_doc()
    doc["models"]["qwen3-1.7b"]["revision"] = "main"
    with pytest.raises(SchemaError) as exc:
        validate_document(MODEL_REGISTRY, doc)
    assert "revision" in str(exc.value)


def test_a_measured_vram_figure_must_name_its_configuration():
    """A number without its provenance is an unfalsifiable claim.

    A profile measured at 2048x1 says nothing about a 512x2 job, so a bare figure is not
    just undocumented — it is unusable without inventing the context it was taken in.
    """
    doc = _registry_doc()
    doc["models"]["qwen3-1.7b"]["resource_profile"]["qlora"].pop("measured_at")
    with pytest.raises(SchemaError) as exc:
        validate_document(MODEL_REGISTRY, doc)
    assert "measured_at" in str(exc.value)


def test_an_unmeasured_profile_is_allowed_to_be_null():
    """null means UNMEASURED and must stay expressible; it is not the same as unlimited."""
    doc = _registry_doc()
    doc["models"]["qwen3-1.7b"]["resource_profile"]["qlora"] = {
        "min_vram_gb": None, "measured_on": None, "measured_at": None,
    }
    validate_document(MODEL_REGISTRY, doc)


def test_a_remote_code_model_may_not_be_enabled():
    doc = _registry_doc()
    doc["models"]["qwen3-1.7b"]["requires_trust_remote_code"] = True
    with pytest.raises(SchemaError):
        validate_document(MODEL_REGISTRY, doc)


def test_a_disabled_placeholder_may_be_unpinned():
    """phi-4-mini-instruct is real and legitimately null-filled; the schema must allow it."""
    doc = _registry_doc()
    assert doc["models"]["phi-4-mini-instruct"]["revision"] is None
    validate_document(MODEL_REGISTRY, doc)


def test_an_unknown_registry_field_is_rejected():
    """A typo in a key would otherwise be silently ignored at load time."""
    doc = _registry_doc()
    doc["models"]["qwen3-1.7b"]["revsion"] = "typo"
    with pytest.raises(SchemaError):
        validate_document(MODEL_REGISTRY, doc)


# -- results, built the way the containers build them --------------------------


def _validation_success() -> dict:
    result = Result.success(
        stage=Stage.VALIDATION, message="Dataset accepted.",
        code=Code.VALIDATION_SUCCEEDED,
        dataset_summary={"source": "directory", "fileCount": 2},
        provenance={
            "modelScope": "model-agnostic",
            "tokenizerChecks": "finetuner",
        },
    )
    result.add_check("splits_resolved", True, "Found train, validation.")
    return result.to_dict()


def _training_success() -> dict:
    result = Result.success(
        stage=Stage.TRAINING, message="Training complete.",
        code=Code.TRAINING_SUCCEEDED,
        artifact={
            "format": "peft_adapter", "path": "/data/output/model",
            "fileCount": 1, "totalBytes": 1234,
            "files": [{"path": "adapter_model.safetensors", "bytes": 1234,
                       "sha256": "a" * 64}],
        },
        provenance={
            "baseModel": "Qwen/Qwen3-1.7B",
            "baseModelRevision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
            "modelKey": "qwen3-1.7b", "quantized": True,
            "datasetDigest": "b" * 64, "trustRemoteCode": False,
        },
    )
    return result.to_dict()


def test_a_real_validation_success_validates():
    validate_document(VALIDATION_RESULT, _validation_success())


def test_a_validation_success_must_state_its_model_agnostic_scope():
    """C-1: the validator cannot know the model, so its pass must not imply one."""
    document = _validation_success()
    del document["metadata"]["languageModelPipeline"]["provenance"]["modelScope"]
    with pytest.raises(SchemaError):
        validate_document(VALIDATION_RESULT, document)


def test_a_validation_success_naming_a_model_is_rejected():
    """A model claim here would be a claim the container cannot verify — the exact
    false-provenance shape the C-1 audit caught. The schema refuses it outright rather
    than letting it ride along as an ignored extra field."""
    document = _validation_success()
    document["metadata"]["languageModelPipeline"]["provenance"]["baseModel"] = (
        "Qwen/Qwen3-1.7B"
    )
    with pytest.raises(SchemaError):
        validate_document(VALIDATION_RESULT, document)


def test_a_real_training_success_validates():
    validate_document(TRAINING_RESULT, _training_success())


@pytest.mark.parametrize("stage, schema", [
    (Stage.VALIDATION, VALIDATION_RESULT),
    (Stage.TRAINING, TRAINING_RESULT),
])
def test_a_real_failure_result_validates(stage, schema):
    """Failures are the common case in production and must satisfy the contract too."""
    document = Result.from_exception(
        ValueError("boom"), stage=stage,
    ).to_dict()
    validate_document(schema, document)


def test_classnames_is_mandatory_and_its_absence_is_caught():
    """The portal checklist makes this mandatory and gives it its own failure mode.

    It is the single field most likely to be dropped by a future refactor, because an SFT
    run has no classes and an author may reasonably think it does not apply.
    """
    document = _validation_success()
    del document["metadata"]["classNames"]
    with pytest.raises(SchemaError) as exc:
        validate_document(VALIDATION_RESULT, document)
    assert "classNames" in str(exc.value)


def test_the_pipeline_extensions_must_stay_nested():
    """The boundary this schema exists to protect.

    DIMER's documented shape is the OUTER object. Promoting an extension field to the top
    level is exactly the change that would displace something Workbench renders.
    """
    document = _validation_success()
    document["metrics"] = document["metadata"]["languageModelPipeline"]["metrics"]
    with pytest.raises(SchemaError):
        validate_document(VALIDATION_RESULT, document)


def test_a_failure_may_not_claim_the_success_code():
    document = _validation_success()
    document["successful"] = False
    with pytest.raises(SchemaError):
        validate_document(VALIDATION_RESULT, document)


def test_a_successful_training_run_must_publish_hashed_files():
    document = _training_success()
    document["metadata"]["languageModelPipeline"]["artifact"]["files"] = []
    with pytest.raises(SchemaError):
        validate_document(TRAINING_RESULT, document)


def test_a_successful_training_run_must_pin_the_revision_it_loaded():
    document = _training_success()
    document["metadata"]["languageModelPipeline"]["provenance"]["baseModelRevision"] = "main"
    with pytest.raises(SchemaError):
        validate_document(TRAINING_RESULT, document)


def test_a_validation_result_is_not_a_training_result():
    """The stage discriminator must actually discriminate."""
    with pytest.raises(SchemaError):
        validate_document(TRAINING_RESULT, _validation_success())
    with pytest.raises(SchemaError):
        validate_document(VALIDATION_RESULT, _training_success())


def test_every_problem_is_reported_not_only_the_first():
    """One error at a time turns a shape change into a rebuild-and-retry loop."""
    document = _validation_success()
    del document["metadata"]["classNames"]
    del document["metadata"]["taskType"]
    with pytest.raises(SchemaError) as exc:
        validate_document(VALIDATION_RESULT, document)
    message = str(exc.value)
    assert "classNames" in message and "taskType" in message


# -- blocked_reason / gated ------------------------------------------------------
#
# These encode policy state, so they are enforced structurally rather than by convention.
# Each is shown REFUSING, because a rule that has only ever passed is indistinguishable from
# one that is never evaluated.


def test_a_blocked_entry_must_say_why():
    """`blocked` alone forces every reader to reconstruct the reason from prose or history."""
    doc = _registry_doc()
    del doc["models"]["phi-4-mini-instruct"]["blocked_reason"]
    with pytest.raises(SchemaError) as exc:
        validate_document(MODEL_REGISTRY, doc)
    assert "blocked_reason" in str(exc.value)


def test_blocked_reason_is_a_closed_vocabulary():
    """Free text cannot be filtered, compared, or acted on by CI."""
    doc = _registry_doc()
    doc["models"]["phi-4-mini-instruct"]["blocked_reason"] = "needs more thought"
    with pytest.raises(SchemaError):
        validate_document(MODEL_REGISTRY, doc)


def test_a_credential_blocked_model_may_not_be_user_facing():
    """The invariant that keeps llama-3.2-3b-instruct out of the Workbench enum.

    Offering a model whose credentials DIMER cannot yet deliver would fail at run time, and
    would also break the finetuner's manifest/registry agreement test.
    """
    doc = _registry_doc()
    doc["models"]["llama-3.2-3b-instruct"]["internal_only"] = False
    with pytest.raises(SchemaError) as exc:
        validate_document(MODEL_REGISTRY, doc)
    assert "internal_only" in str(exc.value)


def test_gated_mirrors_the_hub_vocabulary_not_a_boolean():
    """`auto` and `manual` differ operationally; collapsing them loses that distinction."""
    doc = _registry_doc()
    doc["models"]["llama-3.2-3b-instruct"]["gated"] = True
    with pytest.raises(SchemaError):
        validate_document(MODEL_REGISTRY, doc)

    for value in (False, "auto", "manual"):
        doc["models"]["llama-3.2-3b-instruct"]["gated"] = value
        validate_document(MODEL_REGISTRY, doc)


def test_a_gated_model_is_not_baked_into_the_validator_image():
    """The reason the entry is disabled rather than merely blocked.

    fetch_tokenizers.py bakes one tokenizer per ENABLED key and fails the build if any cannot
    be fetched. A gated key in that set would make every validator image build require an HF
    credential, which DEPLOYMENT.md §4 says DIMER builds do not have.
    """
    from lmpipeline.registry import ModelRegistry

    registry = ModelRegistry.load()
    assert "llama-3.2-3b-instruct" not in registry.keys()
    assert "llama-3.2-3b-instruct" not in registry.keys(user_facing=True)
