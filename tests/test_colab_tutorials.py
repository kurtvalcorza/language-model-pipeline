from pathlib import Path
from runpy import run_path

import pytest

VALIDATOR = run_path(
    Path(__file__).resolve().parents[1] / "scripts" / "validate_colab_tutorials.py"
)
INFERENCE = VALIDATOR["INFERENCE"]
MAIN = VALIDATOR["MAIN"]
code_text = VALIDATOR["code_text"]
load_notebook = VALIDATOR["load_notebook"]
markdown_text = VALIDATOR["markdown_text"]
repository_pin = VALIDATOR["repository_pin"]
validate_member_path = VALIDATOR["validate_member_path"]
validate_notebooks = VALIDATOR["validate_notebooks"]


def test_notebooks_exist_and_validate():
    assert MAIN.is_file()
    assert INFERENCE.is_file()
    validate_notebooks()


def test_notebooks_are_unexecuted():
    for path in (MAIN, INFERENCE):
        notebook = load_notebook(path)
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                assert cell.get("execution_count") is None
                assert cell.get("outputs") == []


@pytest.mark.parametrize(
    "member",
    [
        "/etc/passwd",
        "../../etc/passwd",
        "checkpoints/../../escape",
        "../model.safetensors",
        "..\\evil.safetensors",
    ],
)
def test_member_path_rejects_traversal(member):
    with pytest.raises(ValueError):
        validate_member_path(member)


@pytest.mark.parametrize(
    "member",
    [
        "adapter_model.safetensors",
        "tokenizer/tokenizer.json",
        "nested/ok/file.json",
        "model-00001-of-00002.safetensors",
    ],
)
def test_member_path_accepts_contained_members(member):
    assert str(validate_member_path(member)) == member


def test_normative_profiles_are_declared_in_metadata_and_markdown():
    main = load_notebook(MAIN)
    inference = load_notebook(INFERENCE)
    assert main["metadata"]["dimer"] == {
        "notebook_profile": "E2E",
        "notebook_spec_version": "1.0",
    }
    assert inference["metadata"]["dimer"] == {
        "notebook_profile": "ARTIFACT-INFERENCE",
        "notebook_spec_version": "1.0",
    }
    assert "**Profile:** `E2E`" in markdown_text(main)
    assert "**Profile:** `ARTIFACT-INFERENCE`" in markdown_text(inference)


def test_notebooks_use_repository_api_not_parallel_core_implementations():
    for path in (MAIN, INFERENCE):
        code = code_text(load_notebook(path))
        assert "from lmpipeline.tutorial_api import" in code
        assert "TUTORIAL_REGISTRY" not in code
        assert "AutoModelForCausalLM" not in code
        assert "get_peft_model" not in code
        assert "PeftModel.from_pretrained" not in code


def test_both_notebooks_install_same_immutable_repository_revision():
    main_pin = repository_pin(load_notebook(MAIN), label="main")
    inference_pin = repository_pin(load_notebook(INFERENCE), label="inference")
    assert len(main_pin) == 40
    assert main_pin == inference_pin


def test_finetuning_uses_user_facing_default_and_real_byod_path():
    notebook = load_notebook(MAIN)
    code = code_text(notebook)
    markdown = markdown_text(notebook)
    assert 'BASE_MODEL_KEY = "qwen3-1.7b"' in code
    assert 'DATA_SOURCE = "Sample: Filipino SFT"' in code
    assert '"Bring Your Own Dataset"' in code
    assert "resolve_dataset" in code
    assert "load_examples" in code
    assert "BYOD privacy boundary" in markdown
    assert "Do not upload confidential" in markdown


def test_finetuning_seeds_before_model_construction_and_exports_outputs():
    code = code_text(load_notebook(MAIN))
    assert code.index("seed_everything(SEED)") < code.index("load_base_model(")
    assert "tutorial_predictions.jsonl" in code
    assert "tutorial_metrics.json" in code
    assert "export_adapter_bundle" in code
    assert "load_adapter_for_inference" in code


def test_artifact_inference_is_external_and_has_real_new_input_and_export():
    notebook = load_notebook(INFERENCE)
    code = code_text(notebook)
    markdown = markdown_text(notebook)
    assert "files.upload()" in code
    assert "consume_adapter_archive" in code
    assert "CUSTOM_PROMPT" in code
    assert "validate_prompt" in code
    assert "artifact_inference_predictions.jsonl" in code
    assert "artifact_inference_provenance.json" in code
    assert "externally supplied PEFT adapter ZIP" in markdown
    assert "sender authenticity" in markdown


def test_artifact_inference_requires_runtime_and_registry_parity():
    code = code_text(load_notebook(INFERENCE))
    assert "assert_runtime_compatible(PROVENANCE, RUNTIME)" in code
    assert "resolve_artifact_model(PROVENANCE)" in code


def test_notebooks_do_not_claim_static_validation_is_runtime_evidence():
    for path in (MAIN, INFERENCE):
        markdown = markdown_text(load_notebook(path)).lower()
        assert "static checks prove execution" not in markdown
