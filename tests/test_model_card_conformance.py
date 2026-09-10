from pathlib import Path

import pytest


MODEL_CARDS = (
    Path("weights/qwen3-0.6b/MODEL_CARD.md"),
    Path("weights/smollm2-360m/MODEL_CARD.md"),
)

REQUIRED_HEADINGS = (
    "###### Description",
    "#### Intended Use and Limitations",
    "###### Primary Intended Uses",
    "###### Primary Intended Users",
    "###### Out-of-scope use cases",
    "#### Factors",
    "###### Groups",
    "###### Instrumentation",
    "###### Environment",
    "#### Metrics",
    "###### Performance Measures",
    "###### Decision thresholds",
    "###### Approaches to uncertainty and variability",
    "#### Ethical considerations and biases",
    "###### Data",
    "###### Human Life",
    "###### Mitigations",
    "###### Risks and harms",
    "###### Use cases",
)


@pytest.mark.parametrize("path", MODEL_CARDS)
def test_dimer_model_card_v1_structure(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    assert text.startswith("---\n")
    assert 'model_card_spec: "1.0"' in text
    assert "base_model_revision:" in text
    assert "approval_state:" in text

    positions = [text.index(heading) for heading in REQUIRED_HEADINGS]
    assert positions == sorted(positions)
    assert len(positions) == len(set(positions))

    h1_lines = [line for line in text.splitlines() if line.startswith("# ")]
    assert len(h1_lines) == 1

    lowered = text.lower()
    for forbidden in ("<placeholder>", "todo: replace", "lorem ipsum"):
        assert forbidden not in lowered


def test_dimer_model_cards_state_accelerator_contract() -> None:
    for path in MODEL_CARDS:
        text = path.read_text(encoding="utf-8")
        environment = text.split("###### Environment", 1)[1].split("#### Metrics", 1)[0]
        assert "LoRA can execute on CPU or CUDA" in environment
        assert "QLoRA requires an NVIDIA CUDA device" in environment
        assert "RESOURCE_GPU_UNAVAILABLE" in environment
        assert "not qualified" in environment
