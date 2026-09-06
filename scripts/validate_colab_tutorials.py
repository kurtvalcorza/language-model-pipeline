"""Static validation for the standalone language-model Colab tutorials."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "tutorials"

MAIN = TUTORIALS / "language_model_finetuning_colab.ipynb"
INFERENCE = TUTORIALS / "language_model_artifact_inference_colab.ipynb"

MAIN_MARKERS = (
    "TUTORIAL_REGISTRY",
    '"smollm3-3b"',
    '"llama-3.2-3b-instruct"',
    'BASE_MODEL_KEY = "smollm3-3b"',
    'userdata.get("HF_TOKEN")',
    'MODEL_SOURCE = "Pinned Hugging Face"',
    '"dimer-base-manifest.json"',
    '"dimer_hf_snapshot"',
    "trust_remote_code",
    "MAX_TOTAL_TRAIN_TOKENS = 50_000_000",
    "build_masked_example",
    "prepare_model_for_kbit_training",
    "BASELINE_OUTPUTS",
    "ADAPTED_OUTPUTS",
    "RUN_NEW_PROMPT_INFERENCE",
    "artifact-manifest.json",
    "DATASET_DIGEST",
    "Fresh base + adapter reload",
    "GPT-5.6 Sol High",
)

INFERENCE_MARKERS = (
    "EXPECTED_ARTIFACT_ZIP_SHA256",
    'userdata.get("HF_TOKEN")',
    'BASE_MODEL_SOURCE = "Pinned Hugging Face"',
    '"dimer-base-manifest.json"',
    '"dimer_hf_snapshot"',
    "manifest_member_path",
    "artifact-manifest.json",
    "trustRemoteCode",
    "trust_remote_code",
    "baseModelRevision",
    "PeftModel.from_pretrained",
    "SHA-256 mismatch",
    "GPT-5.6 Sol High",
)

FORBIDDEN_CODE = (
    "trust_remote_code=True",
    "pickle.load",
    "pickle.loads",
    "torch.load(",
)

EDUCATIONAL_MARKERS = (
    "What you will learn",
    "assistant-only",
    "optimization",
    "task quality",
    "base model",
    "tokenizer",
    "adapter",
)


def load_notebook(path: Path) -> dict:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        raise AssertionError(f"{path.name}: expected nbformat 4")
    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        raise AssertionError(f"{path.name}: notebook has no cells")
    return notebook


def cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def code_text(notebook: dict) -> str:
    return "\n".join(
        cell_source(cell)
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def markdown_text(notebook: dict) -> str:
    return "\n".join(
        cell_source(cell)
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )


def compile_code_cells(notebook: dict, *, label: str) -> None:
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = cell_source(cell)
        if source.lstrip().startswith("%pip"):
            continue
        try:
            ast.parse(source)
        except SyntaxError as exc:
            raise AssertionError(
                f"{label}: code cell {index} does not parse: {exc}"
            ) from exc


def assert_clean_notebook(notebook: dict, *, label: str) -> None:
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None:
            raise AssertionError(
                f"{label}: cell {index} has execution_count"
            )
        if cell.get("outputs"):
            raise AssertionError(f"{label}: cell {index} has outputs")


def validate_member_path(member: str) -> PurePosixPath:
    if "\\" in member:
        raise ValueError(f"unsafe member path: {member!r}")
    path = PurePosixPath(member)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe member path: {member!r}")
    return path


def assert_markers(text: str, markers: tuple[str, ...], *, label: str) -> None:
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise AssertionError(f"{label}: missing code markers: {missing}")


def assert_colab_param_annotations(notebook: dict) -> None:
    """Every # @param line must annotate exactly one assignment.

    This guards the regression where TRAINING_METHOD and MAX_SEQUENCE_LENGTH shared a line,
    causing Colab to apply an integer widget to the string training-method value.
    """
    for cell_index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        for line_number, line in enumerate(cell_source(cell).splitlines(), 1):
            if "# @param" not in line:
                continue
            code_part = line.split("# @param", 1)[0].strip()
            try:
                tree = ast.parse(code_part)
            except SyntaxError as exc:
                raise AssertionError(
                    f"main: malformed @param line at cell {cell_index}, "
                    f"line {line_number}: {line!r}"
                ) from exc
            if len(tree.body) != 1 or not isinstance(
                tree.body[0], (ast.Assign, ast.AnnAssign)
            ):
                raise AssertionError(
                    f"main: @param must annotate exactly one assignment at "
                    f"cell {cell_index}, line {line_number}: {line!r}"
                )


def assert_educational_markdown(notebook: dict) -> None:
    markdown = markdown_text(notebook)
    lower = markdown.lower()
    missing = [
        marker
        for marker in EDUCATIONAL_MARKERS
        if marker.lower() not in lower
    ]
    if missing:
        raise AssertionError(
            f"main: missing educational explanations: {missing}"
        )

    markdown_cells = sum(
        cell.get("cell_type") == "markdown"
        for cell in notebook["cells"]
    )
    if markdown_cells < 10:
        raise AssertionError(
            f"main: expected tutorial-style markdown coverage; found {markdown_cells} cells"
        )

    # A very terse notebook can satisfy marker checks accidentally. Require meaningful prose.
    word_count = len(re.findall(r"\b[\w’-]+\b", markdown))
    if word_count < 900:
        raise AssertionError(
            f"main: tutorial markdown is too terse ({word_count} words)"
        )


def validate_notebooks() -> None:
    main = load_notebook(MAIN)
    inference = load_notebook(INFERENCE)

    for label, notebook in (("main", main), ("inference", inference)):
        assert_clean_notebook(notebook, label=label)
        compile_code_cells(notebook, label=label)
        text = code_text(notebook)
        for forbidden in FORBIDDEN_CODE:
            if forbidden in text:
                raise AssertionError(
                    f"{label}: forbidden executable marker {forbidden!r}"
                )

    assert_markers(code_text(main), MAIN_MARKERS, label="main")
    assert_markers(
        code_text(inference),
        INFERENCE_MARKERS,
        label="inference",
    )
    assert_colab_param_annotations(main)
    assert_educational_markdown(main)


if __name__ == "__main__":
    validate_notebooks()
    print("Standalone language-model Colab tutorials validate.")
