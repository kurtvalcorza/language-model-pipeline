"""Final prose pass (Kurt's ruling, 2026-10-26): the flagged expectation sentences are removed or
rewritten as records of named runs. language-model: Section 9 and Section 12 worked answers.
Fails on PR head 7fdc50c.
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / 'tutorials/language_model_finetuning_colab.ipynb'


def _markdown():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


GONE = [
    'and loop less',
    'The greedy answer, if any',
    'Often, but not always',
    'so expect some prompts',
]
PRESENT = [
    'a sampled answer can loop as much or more',
    'Read the BASE/ADAPTED pairs printed above',
]


def test_flagged_expectations_are_gone():
    md = _markdown()
    for text in GONE:
        assert text not in md, text


def test_replacement_records_are_present():
    md = _markdown()
    for text in PRESENT:
        assert text in md, text
