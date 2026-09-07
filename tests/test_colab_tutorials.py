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


def test_validator_is_code_cell_scoped():
    notebook = load_notebook(MAIN)
    code = code_text(notebook)
    assert "build_masked_example" in code
    assert "RUN_NEW_PROMPT_INFERENCE" in code


def test_qwen06b_is_default_and_llama_is_selectable():
    code = code_text(load_notebook(MAIN))
    assert 'BASE_MODEL_KEY = "qwen3-0.6b"' in code
    assert '"qwen3-0.6b"' in code
    assert '"smollm3-3b"' in code
    assert '"llama-3.2-3b-instruct"' in code
    assert '"meta-llama/Llama-3.2-3B-Instruct"' in code


def test_hf_secret_is_read_but_never_printed():
    code = code_text(load_notebook(MAIN))
    assert 'userdata.get("HF_TOKEN")' in code
    assert 'print(HF_TOKEN)' not in code
    assert '"HF_TOKEN": HF_TOKEN' not in code


def test_dimer_zip_requires_identity_manifest():
    code = code_text(load_notebook(MAIN))
    assert '"dimer-base-manifest.json"' in code
    assert '"dimer_hf_snapshot"' in code
    assert "modelId" in code
    assert "revision" in code
    assert "sha256" in code


def test_llama_dimer_zip_is_not_assumed_redistributable():
    code = code_text(load_notebook(MAIN))
    llama_start = code.index('"llama-3.2-3b-instruct"')
    llama_block = code[llama_start : llama_start + 1200]
    assert '"dimer_zip":False' in llama_block


def test_training_method_form_is_not_integer_widget():
    code = code_text(load_notebook(MAIN))
    training_lines = [
        line for line in code.splitlines()
        if line.strip().startswith("TRAINING_METHOD")
    ]
    assert len(training_lines) == 1
    line = training_lines[0]
    assert '# @param ["qlora"]' in line
    assert 'type:"integer"' not in line


def test_tutorial_markdown_is_substantive():
    markdown = markdown_text(load_notebook(MAIN))
    assert "What you will learn" in markdown
    assert "For **Llama 3.2 3B Instruct**" in markdown
    assert "What a successful run proves" in markdown
    assert len(markdown.split()) > 900


def test_inference_notebook_supports_hf_secret_and_dimer_zip():
    code = code_text(load_notebook(INFERENCE))
    assert 'userdata.get("HF_TOKEN")' in code
    assert 'BASE_MODEL_SOURCE = "Pinned Hugging Face"' in code
    assert '"dimer-base-manifest.json"' in code
    assert "PeftModel.from_pretrained" in code


def test_no_training_rows_written_to_artifact_code():
    code = code_text(load_notebook(MAIN))
    # Artifact publication should serialize model/tokenizer/metrics/provenance, not SPLITS.
    artifact_section = code[code.index('Path("/content/dimer-lm-adapter.staging")') :]
    assert 'write_text(json.dumps(SPLITS' not in artifact_section
    assert 'json.dump(SPLITS' not in artifact_section
