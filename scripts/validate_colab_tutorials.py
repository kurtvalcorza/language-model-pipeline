"""Static conformance checks for release-grade language-model tutorial notebooks.

This proves source structure only. It deliberately does not claim clean-runtime execution;
that evidence is recorded separately under tutorials/RELEASE_VERIFICATION.md.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "tutorials"
MAIN = TUTORIALS / "language_model_finetuning_colab.ipynb"
INFERENCE = TUTORIALS / "language_model_artifact_inference_colab.ipynb"
LOCK = TUTORIALS / "requirements-colab.lock"
SPEC_VERSION = "1.0"

MAIN_MARKERS = (
    "from lmpipeline.tutorial_api import",
    "assert_tutorial_runtime",
    "seed_everything",
    "resolve_tutorial_model",
    "normalize_records",
    "resolve_dataset",
    "load_examples",
    "tokenize_splits",
    "attach_adapter",
    "train_adapter",
    "generate_reply",
    "CUSTOM_PROMPT",
    "tutorial_predictions.jsonl",
    "export_adapter_bundle",
    "load_adapter_for_inference",
)

INFERENCE_MARKERS = (
    "from lmpipeline.tutorial_api import",
    "assert_tutorial_runtime",
    "consume_adapter_archive",
    "resolve_artifact_model",
    "assert_runtime_compatible",
    "load_adapter_for_inference",
    "CUSTOM_PROMPT",
    "validate_prompt",
    "generate_reply",
    "artifact_inference_predictions.jsonl",
    "artifact_inference_provenance.json",
)

FORBIDDEN_NOTEBOOK_CORE = (
    "TUTORIAL_REGISTRY",
    "AutoModelForCausalLM",
    "get_peft_model",
    "prepare_model_for_kbit_training",
    "PeftModel.from_pretrained",
    "def train_adapter(",
    "def build_masked_example(",
    "def resolve_target_modules(",
)

FORBIDDEN_EXECUTION = (
    "trust_remote_code=True",
    "pickle.load",
    "pickle.loads",
    "torch.load(",
)

PLACEHOLDERS = re.compile(r"\b(?:TODO|TBD|FIXME)\b", re.IGNORECASE)
REPO_PIN = re.compile(
    r"git\+https://github\.com/kurtvalcorza/language-model-pipeline\.git@([0-9a-f]{40})"
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
        cell_source(cell) for cell in notebook["cells"] if cell.get("cell_type") == "code"
    )


def markdown_text(notebook: dict) -> str:
    return "\n".join(
        cell_source(cell)
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )


def all_text(notebook: dict) -> str:
    return "\n".join(cell_source(cell) for cell in notebook["cells"])


def compile_code_cells(notebook: dict, *, label: str) -> None:
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = cell_source(cell)
        python_lines = [
            line for line in source.splitlines()
            if not line.lstrip().startswith(("%", "!"))
        ]
        try:
            ast.parse("\n".join(python_lines))
        except SyntaxError as exc:
            raise AssertionError(f"{label}: code cell {index} does not parse: {exc}") from exc


def assert_clean_notebook(notebook: dict, *, label: str) -> None:
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None:
            raise AssertionError(f"{label}: cell {index} has execution_count")
        if cell.get("outputs"):
            raise AssertionError(f"{label}: cell {index} has persisted outputs")
    match = PLACEHOLDERS.search(all_text(notebook))
    if match:
        raise AssertionError(f"{label}: unresolved placeholder {match.group(0)!r}")


def validate_member_path(member: str) -> PurePosixPath:
    if "\\" in member:
        raise ValueError(f"unsafe member path: {member!r}")
    path = PurePosixPath(member)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe member path: {member!r}")
    return path


def assert_profile(notebook: dict, expected: str, *, label: str) -> None:
    dimer = notebook.get("metadata", {}).get("dimer", {})
    if dimer.get("notebook_profile") != expected:
        raise AssertionError(f"{label}: expected dimer.notebook_profile={expected!r}")
    if dimer.get("notebook_spec_version") != SPEC_VERSION:
        raise AssertionError(f"{label}: expected notebook spec {SPEC_VERSION}")
    markdown = markdown_text(notebook)
    if f"**Profile:** `{expected}`" not in markdown:
        raise AssertionError(f"{label}: profile must also be visible to the learner")


def assert_markers(text: str, markers: tuple[str, ...], *, label: str) -> None:
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise AssertionError(f"{label}: missing required markers: {missing}")


def assert_no_parallel_implementation(notebook: dict, *, label: str) -> None:
    code = code_text(notebook)
    present = [marker for marker in FORBIDDEN_NOTEBOOK_CORE if marker in code]
    if present:
        raise AssertionError(
            f"{label}: notebook reimplements core pipeline behavior instead of using "
            f"lmpipeline.tutorial_api: {present}"
        )
    dangerous = [marker for marker in FORBIDDEN_EXECUTION if marker in code]
    if dangerous:
        raise AssertionError(f"{label}: forbidden executable/deserialization markers: {dangerous}")


def assert_colab_param_annotations(notebook: dict, *, label: str) -> None:
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
                    f"{label}: malformed @param at cell {cell_index}, line {line_number}"
                ) from exc
            if len(tree.body) != 1 or not isinstance(tree.body[0], (ast.Assign, ast.AnnAssign)):
                raise AssertionError(
                    f"{label}: @param must annotate exactly one assignment at cell "
                    f"{cell_index}, line {line_number}"
                )


def assert_lock_is_exact() -> None:
    lines = [
        line.strip() for line in LOCK.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise AssertionError("requirements-colab.lock is empty")
    bad = [line for line in lines if not re.fullmatch(r"[A-Za-z0-9_.-]+==[^=<>!~\s]+", line)]
    if bad:
        raise AssertionError(f"requirements-colab.lock contains non-exact requirements: {bad}")


def repository_pin(notebook: dict, *, label: str) -> str:
    matches = REPO_PIN.findall(code_text(notebook))
    if len(matches) != 1:
        raise AssertionError(f"{label}: expected exactly one immutable repository install pin")
    return matches[0]


def assert_learning_contract(main: dict, inference: dict) -> None:
    main_md = markdown_text(main)
    inference_md = markdown_text(inference)
    required_main = (
        "Adaptation semantics",
        "assistant-only",
        "optimization evidence",
        "It does **not** establish",
        "BYOD privacy boundary",
        "reconstruct",
    )
    required_inference = (
        "No training or fine-tuning occurs",
        "Trust boundary",
        "sender authenticity",
        "externally supplied",
        "new user input",
        "It does **not** establish",
    )
    assert_markers(main_md, required_main, label="main markdown")
    assert_markers(inference_md, required_inference, label="inference markdown")


def validate_notebooks() -> None:
    main = load_notebook(MAIN)
    inference = load_notebook(INFERENCE)
    assert_profile(main, "E2E", label="main")
    assert_profile(inference, "ARTIFACT-INFERENCE", label="inference")
    assert_lock_is_exact()

    for label, notebook in (("main", main), ("inference", inference)):
        assert_clean_notebook(notebook, label=label)
        compile_code_cells(notebook, label=label)
        assert_no_parallel_implementation(notebook, label=label)
        assert_colab_param_annotations(notebook, label=label)

    assert_markers(code_text(main), MAIN_MARKERS, label="main")
    assert_markers(code_text(inference), INFERENCE_MARKERS, label="inference")
    assert_learning_contract(main, inference)

    main_pin = repository_pin(main, label="main")
    inference_pin = repository_pin(inference, label="inference")
    if main_pin != inference_pin:
        raise AssertionError(
            "tutorial notebooks must install the same immutable language-model-pipeline revision: "
            f"main={main_pin}, inference={inference_pin}"
        )


if __name__ == "__main__":
    validate_notebooks()
    print("Static DIMER Notebook Specification 1.0 checks pass.")
