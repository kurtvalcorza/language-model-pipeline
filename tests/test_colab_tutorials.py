from pathlib import Path

import pytest

from scripts.validate_colab_tutorials import (
    INFERENCE,
    MAIN,
    load_notebook,
    validate_member_path,
    validate_notebooks,
)


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
    ],
)
def test_member_path_accepts_contained_members(member):
    assert validate_member_path(member) == Path(member)


def test_validator_is_code_cell_scoped():
    notebook = load_notebook(MAIN)
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "build_masked_example" in code
    assert "RUN_NEW_PROMPT_INFERENCE" in code
