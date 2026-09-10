from pathlib import Path
from runpy import run_path

import pytest

VALIDATOR = run_path(
    Path(__file__).resolve().parents[1] / "scripts" / "validate_colab_tutorials.py"
)
INFERENCE = VALIDATOR["INFERENCE"]
MAIN = VALIDATOR["MAIN"]
code_text = VALIDATOR["code_text"]
finetuner_pin = VALIDATOR["finetuner_pin"]
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


def test_e2e_notebook_imports_real_finetuner_components():
    code = code_text(load_notebook(MAIN))
    for marker in (
        "from finetuner.artifacts import",
        "from finetuner.backends import",
        "from finetuner.config import TrainingConfig",
        "from finetuner.data import",
        "from finetuner.inference import",
        "from finetuner.masking import build_masked_example",
        "from finetuner.training import train",
    ):
        assert marker in code
    for forbidden in (
        "AutoModelForCausalLM",
        "get_peft_model",
        "PeftModel.from_pretrained",
        "def train(",
        "def generate_reply(",
    ):
        assert forbidden not in code


def test_both_notebooks_pin_same_pipeline_and_finetuner_revisions():
    main = load_notebook(MAIN)
    inference = load_notebook(INFERENCE)
    main_pipeline = repository_pin(main, label="main")
    inference_pipeline = repository_pin(inference, label="inference")
    main_finetuner = finetuner_pin(main, label="main")
    inference_finetuner = finetuner_pin(inference, label="inference")
    assert len(main_pipeline) == 40
    assert len(main_finetuner) == 40
    assert main_pipeline == inference_pipeline
    assert main_finetuner == inference_finetuner
    assert main_finetuner == "3772f0ca4e0f7130ffd5f826ea40f43b7212339e"


def test_private_finetuner_bootstrap_is_explicit_and_secret_safe():
    direct_clone = (
        "!git clone -q "
        "https://github.com/kurtvalcorza/language-model-finetuner.git"
    )
    for path in (MAIN, INFERENCE):
        notebook = load_notebook(path)
        code = code_text(notebook)
        markdown = markdown_text(notebook)
        assert "github_token_from_runtime()" in code
        assert "checkout_private_finetuner(" in code
        assert "del _GITHUB_TOKEN" in code
        assert direct_clone not in code
        assert "https://x-access-token:" not in code
        assert "GITHUB_TOKEN" in markdown
        assert "read access" in markdown


def test_finetuning_uses_user_facing_default_and_real_byod_path():
    notebook = load_notebook(MAIN)
    code = code_text(notebook)
    markdown = markdown_text(notebook)
    assert 'BASE_MODEL_KEY = "qwen3-1.7b"' in code
    assert 'DATA_SOURCE = "Sample: Filipino SFT"' in code
    assert '"Bring Your Own Dataset"' in code
    assert "load_normalized_splits" in code
    assert "dataset_digest" in code
    assert "Private production source" in markdown
    assert "do not upload confidential" in markdown.lower()


def test_finetuning_seeds_before_model_construction_and_exports_outputs():
    code = code_text(load_notebook(MAIN))
    assert code.index("seed_everything(SEED)") < code.index("load_base_model(")
    assert "tutorial_predictions.jsonl" in code
    assert "tutorial_predictions.csv" in code
    assert "tutorial_metrics.json" in code
    assert "stage_artifact" in code
    assert "load_adapter_for_inference" in code


def test_fresh_reconstruction_proves_adapter_activity():
    code = code_text(load_notebook(MAIN))
    markdown = markdown_text(load_notebook(MAIN)).lower()
    assert "verify_adapter_active(" in code
    assert "lora b matrices" in markdown
    assert "adapter-on/off logit" in markdown


def test_e2e_asserts_consumer_required_package_provenance():
    code = code_text(load_notebook(MAIN))
    assert '"safetensors"' in code
    assert 'PROVENANCE["packageVersions"]' in code
    assert "Producer provenance missing package versions" in code


def test_e2e_explains_principal_metrics_and_adapter_dependency():
    markdown = markdown_text(load_notebook(MAIN))
    for marker in (
        "trainLoss",
        "validationLoss",
        "testLoss",
        "trainPerplexity",
        "examplesPerSecond",
        "peakGpuMemoryBytes",
        "optimization evidence",
        "PEFT adapter, not a complete model",
        "exact base model and immutable revision",
    ):
        assert marker in markdown


def test_versioned_manifest_is_produced_and_consumed():
    main = code_text(load_notebook(MAIN))
    inference = code_text(load_notebook(INFERENCE))
    assert 'MANIFEST.get("format")' in main
    assert 'MANIFEST.get("formatVersion")' in main
    assert '("peft_adapter", 1)' in main
    assert 'MANIFEST.get("format")' in inference
    assert 'MANIFEST.get("formatVersion")' in inference
    assert '("peft_adapter", 1)' in inference


def test_artifact_inference_is_external_and_has_real_new_input_and_export():
    notebook = load_notebook(INFERENCE)
    code = code_text(notebook)
    markdown = markdown_text(notebook)
    assert "files.upload()" in code
    assert "from lmpipeline.tutorial_runtime import" in code
    assert "consume_adapter_archive" in code
    assert "CUSTOM_PROMPT" in code
    assert "validate_prompt" in code
    assert "artifact_inference_predictions.jsonl" in code
    assert "artifact_inference_predictions.csv" in code
    assert "artifact_inference_provenance.json" in code
    assert "externally supplied PEFT adapter ZIP" in markdown
    assert "sender authenticity" in markdown
    assert "root-level manifest" in markdown


def test_artifact_inference_requires_registry_package_and_source_parity():
    code = code_text(load_notebook(INFERENCE))
    assert "assert_runtime_compatible(PROVENANCE, RUNTIME)" in code
    assert "resolve_artifact_model(PROVENANCE)" in code
    assert "RECORDED_RUNTIME_REVISIONS" in code
    assert "PIPELINE_RUNTIME_REVISION" in code
    assert "FINETUNER_RUNTIME_REVISION" in code
    assert "if RECORDED_RUNTIME_REVISIONS:" in code


def test_notebooks_include_troubleshooting_next_experiments_and_source_links():
    for path in (MAIN, INFERENCE):
        markdown = markdown_text(load_notebook(path)).lower()
        assert "common failures" in markdown
        assert "next experiments" in markdown
        assert "https://github.com/kurtvalcorza/language-model-pipeline" in markdown
        assert "https://huggingface.co" in markdown


def test_notebooks_do_not_claim_static_validation_is_runtime_evidence():
    for path in (MAIN, INFERENCE):
        markdown = markdown_text(load_notebook(path)).lower()
        assert "static checks prove execution" not in markdown
        assert "static ci is not rel1/rel5 execution evidence" in markdown
