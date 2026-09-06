"""Static validation for the standalone language-model Colab tutorials."""

from __future__ import annotations

import ast
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "tutorials"

MAIN = TUTORIALS / "language_model_finetuning_colab.ipynb"
INFERENCE = TUTORIALS / "language_model_artifact_inference_colab.ipynb"

MAIN_MARKERS = (
    "TUTORIAL_REGISTRY",
    "trust_remote_code=False",
    "MAX_TOTAL_TRAIN_TOKENS = 50_000_000",
    "safe_extract_zip",
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
    "manifest_member_path",
    "artifact-manifest.json",
    "trustRemoteCode",
    "trust_remote_code=False",
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


def load_notebook(path: Path) -> dict:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        raise AssertionError(f"{path.name}: expected nbformat 4")
    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        raise AssertionError(f"{path.name}: notebook has no cells")
    return notebook


def code_text(notebook: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def compile_code_cells(notebook: dict, *, label: str) -> None:
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
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
    path = PurePosixPath(member)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe member path: {member!r}")
    return path


def assert_markers(text: str, markers: tuple[str, ...], *, label: str) -> None:
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise AssertionError(f"{label}: missing code markers: {missing}")


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


if __name__ == "__main__":
    validate_notebooks()
    print("Standalone language-model Colab tutorials validate.")
