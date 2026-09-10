"""Static conformance checks for release-grade language-model tutorial notebooks.

These checks prove source structure only. Clean-runtime execution evidence is recorded
separately under tutorials/RELEASE_VERIFICATION.md.
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
TUTORIAL_API = ROOT / "src" / "lmpipeline" / "tutorial_api.py"
SPEC_VERSION = "1.0"

PIPELINE_PIN = re.compile(
    r"git\+https://github\.com/kurtvalcorza/language-model-pipeline\.git@([0-9a-f]{40})"
)
FINETUNER_PIN = re.compile(
    r"git -C /content/language-model-finetuner checkout -q ([0-9a-f]{40})"
)
PLACEHOLDERS = re.compile(r"\b(?:TODO|TBD|FIXME)\b", re.IGNORECASE)

MAIN_MARKERS = (
    "from finetuner.artifacts import build_provenance, stage_artifact, verify_manifest",
    "from finetuner.backends import (",
    "from finetuner.config import TrainingConfig",
    "from finetuner.data import (",
    "from finetuner.inference import (",
    "from finetuner.masking import build_masked_example",
    "from finetuner.training import train",
    "load_normalized_splits",
    "tokenize_splits",
    "load_base_model(",
    "attach_adapter(",
    "train(",
    "stage_artifact(",
    "generate_reply(",
    "verify_adapter_active(",
    "CUSTOM_PROMPT",
    "tutorial_predictions.jsonl",
    'PROVENANCE["runtimeRevisions"]',
)

INFERENCE_MARKERS = (
    "from finetuner.inference import (",
    "load_adapter_for_inference(",
    "generate_reply(",
    "verify_adapter_active(",
    "consume_adapter_archive(",
    "resolve_artifact_model(PROVENANCE)",
    "assert_runtime_compatible(PROVENANCE, RUNTIME)",
    "CUSTOM_PROMPT",
    "validate_prompt(",
    "artifact_inference_predictions.jsonl",
    "artifact_inference_provenance.json",
    "RECORDED_RUNTIME_REVISIONS",
)

FORBIDDEN_NOTEBOOK_CORE = (
    "TUTORIAL_REGISTRY",
    "AutoModelForCausalLM",
    "get_peft_model",
    "prepare_model_for_kbit_training",
    "PeftModel.from_pretrained",
    "def train(",
    "def train_adapter(",
    "def build_masked_example(",
    "def load_base_model(",
    "def load_adapter_for_inference(",
    "def generate_reply(",
    "def stage_artifact(",
)

FORBIDDEN_EXECUTION = (
    "trust_remote_code=True",
    "pickle.load",
    "pickle.loads",
    "torch.load(",
)

FORBIDDEN_TUTORIAL_API_CORE_DEFS = (
    "def train_adapter(",
    "def load_base_model(",
    "def attach_adapter(",
    "def build_masked_example(",
    "def tokenize_splits(",
    "def generate_reply(",
    "def load_adapter_for_inference(",
    "def stage_artifact(",
    "def export_adapter_bundle(",
)


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def code_text(notebook: dict) -> str:
    return "\n".join(
        cell_source(cell) for cell in notebook["cells"] if cell.get("cell_type") == "code"
    )


def markdown_text(notebook: dict) -> str:
    return "\n".join(
        cell_source(cell) for cell in notebook["cells"] if cell.get("cell_type") == "markdown"
    )


def all_text(notebook: dict) -> str:
    return "\n".join(cell_source(cell) for cell in notebook["cells"])


def compile_code_cells(notebook: dict, *, label: str) -> None:
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = cell_source(cell)
        python_lines = [
            line for line in source.splitlines() if not line.lstrip().startswith(("%", "!"))
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
        raise ValueError("backslash")
    path = PurePosixPath(member)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("unsafe")
    return path


def assert_profile(notebook: dict, expected: str, *, label: str) -> None:
    dimer = notebook.get("metadata", {}).get("dimer", {})
    if dimer.get("notebook_profile") != expected:
        raise AssertionError(f"{label}: expected dimer.notebook_profile={expected!r}")
    if dimer.get("notebook_spec_version") != SPEC_VERSION:
        raise AssertionError(f"{label}: expected notebook spec {SPEC_VERSION}")
    if f"**Profile:** `{expected}`" not in markdown_text(notebook):
        raise AssertionError(f"{label}: profile must also be visible to the learner")


def assert_markers(text: str, markers: tuple[str, ...], *, label: str) -> None:
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise AssertionError(f"{label}: missing required markers: {missing}")


def assert_no_parallel_implementation(notebook: dict, *, label: str) -> None:
    code = code_text(notebook)
    present = [marker for marker in FORBIDDEN_NOTEBOOK_CORE if marker in code]
    if present:
        raise AssertionError(f"{label}: notebook reimplements production behavior: {present}")
    dangerous = [marker for marker in FORBIDDEN_EXECUTION if marker in code]
    if dangerous:
        raise AssertionError(f"{label}: forbidden executable/deserialization markers: {dangerous}")


def assert_tutorial_api_is_support_only() -> None:
    text = TUTORIAL_API.read_text(encoding="utf-8")
    present = [marker for marker in FORBIDDEN_TUTORIAL_API_CORE_DEFS if marker in text]
    if present:
        raise AssertionError(
            "lmpipeline.tutorial_api must remain notebook support, not a parallel trainer: "
            f"{present}"
        )


def assert_colab_param_annotations(notebook: dict, *, label: str) -> None:
    for cell_index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        for line_number, line in enumerate(cell_source(cell).splitlines(), 1):
            if "# @param" not in line:
                continue
            code_part = line.split("# @param", 1)[0].rstrip()
            try:
                tree = ast.parse(code_part)
            except SyntaxError as exc:
                raise AssertionError(
                    f"{label}: malformed @param at cell {cell_index}, line {line_number}"
                ) from exc
            if len(tree.body) != 1 or not isinstance(tree.body[0], (ast.Assign, ast.AnnAssign)):
                raise AssertionError(
                    f"{label}: @param must annotate one assignment at cell "
                    f"{cell_index}, line {line_number}"
                )


def assert_lock_is_exact() -> None:
    lines = [
        line.strip()
        for line in LOCK.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise AssertionError("requirements-colab.lock is empty")
    bad = [line for line in lines if not re.fullmatch(r"[A-Za-z0-9_.-]+==[^=<>!~\s]+", line)]
    if bad:
        raise AssertionError(f"requirements-colab.lock contains non-exact requirements: {bad}")


def _single_pin(pattern: re.Pattern[str], notebook: dict, *, label: str, kind: str) -> str:
    matches = pattern.findall(code_text(notebook))
    if len(matches) != 1:
        raise AssertionError(f"{label}: expected exactly one immutable {kind} revision pin")
    return matches[0]


def repository_pin(notebook: dict, *, label: str) -> str:
    return _single_pin(PIPELINE_PIN, notebook, label=label, kind="pipeline")


def finetuner_pin(notebook: dict, *, label: str) -> str:
    return _single_pin(FINETUNER_PIN, notebook, label=label, kind="finetuner")


def assert_learning_contract(main: dict, inference: dict) -> None:
    assert_markers(
        markdown_text(main),
        (
            "production `finetuner.data`",
            "production masking implementation",
            "optimization evidence",
            "BYOD privacy boundary",
            "runtime-source revisions",
            "does **not** establish",
        ),
        label="main markdown",
    )
    assert_markers(
        markdown_text(inference),
        (
            "externally supplied PEFT adapter ZIP",
            "No training or fine-tuning occurs",
            "Trust boundary",
            "sender authenticity",
            "production inference surface",
            "runtime-source revisions",
            "does **not** establish",
        ),
        label="inference markdown",
    )


def validate_notebooks() -> None:
    main = load_notebook(MAIN)
    inference = load_notebook(INFERENCE)
    assert_profile(main, "E2E", label="main")
    assert_profile(inference, "ARTIFACT-INFERENCE", label="inference")
    assert_lock_is_exact()
    assert_tutorial_api_is_support_only()

    for label, notebook in (("main", main), ("inference", inference)):
        assert_clean_notebook(notebook, label=label)
        compile_code_cells(notebook, label=label)
        assert_no_parallel_implementation(notebook, label=label)
        assert_colab_param_annotations(notebook, label=label)

    assert_markers(code_text(main), MAIN_MARKERS, label="main")
    assert_markers(code_text(inference), INFERENCE_MARKERS, label="inference")
    assert_learning_contract(main, inference)

    main_pipeline = repository_pin(main, label="main")
    inference_pipeline = repository_pin(inference, label="inference")
    main_finetuner = finetuner_pin(main, label="main")
    inference_finetuner = finetuner_pin(inference, label="inference")
    if main_pipeline != inference_pipeline:
        raise AssertionError("both notebooks must use the same pipeline runtime revision")
    if main_finetuner != inference_finetuner:
        raise AssertionError("both notebooks must use the same finetuner runtime revision")

    for label, notebook, pipeline, finetuner in (
        ("main", main, main_pipeline, main_finetuner),
        ("inference", inference, inference_pipeline, inference_finetuner),
    ):
        code = code_text(notebook)
        if f'PIPELINE_RUNTIME_REVISION = "{pipeline}"' not in code:
            raise AssertionError(f"{label}: pipeline runtime provenance does not match install pin")
        if f'FINETUNER_RUNTIME_REVISION = "{finetuner}"' not in code:
            raise AssertionError(f"{label}: finetuner runtime provenance does not match checkout pin")


if __name__ == "__main__":
    validate_notebooks()
    print("Static DIMER Notebook Specification 1.0 checks pass.")
