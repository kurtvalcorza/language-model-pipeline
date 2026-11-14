"""#65: the Section 8 accumulation loop must also step the optimizer on the last, partial window.

The loop is plain PyTorch in a template cell, so there is no package function to call. These tests
lift the epoch body out of the committed notebook's training cell and execute it against stubs that
record every scaled backward pass and every optimizer step. The old window arithmetic computed the
window from the current step's remaining count, so the final window shrank while it was being
filled and never reached `pending == window`: 8 examples at accumulation 2 took 3 steps, 10 at
accumulation 4 took 1.
"""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "tutorials" / "language_model_finetuning_colab.ipynb"


def _training_cell() -> str:
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    for cell in cells:
        src = cell["source"]
        src = "".join(src) if isinstance(src, list) else src
        if cell["cell_type"] == "code" and "GRAD_ACCUM = " in src and "optimizer.step()" in src:
            return src
    raise AssertionError("training cell with GRAD_ACCUM not found in the notebook")


def _epoch_body() -> str:
    """The inner per-epoch statements: from the accumulator reset up to the train-loss line."""
    lines = _training_cell().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("    epoch_loss, epoch_tokens"))
    end = next(i for i, ln in enumerate(lines) if ln.startswith("    train_loss ="))
    return textwrap.dedent("\n".join(lines[start:end]))


class _Recorder:
    def __init__(self) -> None:
        self.window_scales: list[int] = []  # divisor applied to each example's loss
        self.pending: list[int] = []  # divisors accumulated since the last step
        self.steps: list[list[int]] = []  # the divisors each optimizer step consumed

    def model(self, **_batch):
        recorder = self

        class _Loss:
            def __truediv__(self, divisor):
                return SimpleNamespace(backward=lambda: recorder._backward(divisor))

            def detach(self):
                return SimpleNamespace(item=lambda: 1.0)

        return SimpleNamespace(loss=_Loss())

    def _backward(self, divisor: int) -> None:
        self.window_scales.append(divisor)
        self.pending.append(divisor)

    def step(self) -> None:
        self.steps.append(self.pending)
        self.pending = []

    def zero_grad(self) -> None:
        self.pending = []


def _run_epoch(n_examples: int, grad_accum: int) -> _Recorder:
    rec = _Recorder()
    namespace = {
        "examples": [(f"prompt {i}", f"answer {i}") for i in range(n_examples)],
        "GRAD_ACCUM": grad_accum,
        "epoch": 1,
        "model": rec.model,
        "optimizer": SimpleNamespace(step=rec.step, zero_grad=rec.zero_grad),
        "pipe": SimpleNamespace(to_batch=lambda example, model: {}),
        "supervised_token_count": lambda answer: 1,
    }
    exec(compile(_epoch_body(), "<notebook training loop>", "exec"), namespace)  # noqa: S102
    return rec


CASES = [(8, 2), (96, 2), (100, 2), (5, 3), (10, 4), (1, 4), (7, 1), (4, 4), (9, 4)]


@pytest.mark.parametrize(("n_examples", "grad_accum"), CASES)
def test_every_epoch_takes_ceil_n_over_accum_optimizer_steps(
    n_examples: int, grad_accum: int
) -> None:
    rec = _run_epoch(n_examples, grad_accum)
    assert len(rec.steps) == math.ceil(n_examples / grad_accum)
    assert rec.pending == [], "gradients were left accumulated and unstepped at epoch end"
    assert sum(len(window) for window in rec.steps) == n_examples


@pytest.mark.parametrize(("n_examples", "grad_accum"), CASES)
def test_each_window_is_scaled_by_its_own_size(n_examples: int, grad_accum: int) -> None:
    # A step's gradient is the mean over its window: every example in a window of k is divided
    # by k, including the last, partial window (not by a count that shrinks as it fills).
    rec = _run_epoch(n_examples, grad_accum)
    for window in rec.steps:
        assert window == [len(window)] * len(window)
    assert [len(w) for w in rec.steps[:-1]] == [grad_accum] * (len(rec.steps) - 1)


def test_the_cell_raises_when_the_step_count_is_wrong() -> None:
    # The cell guards its own invariant so a regression surfaces in the run log instead of
    # silently discarding gradients: a step that never fires must raise at epoch end.
    rec = _Recorder()
    namespace = {
        "examples": [("p", "a")] * 4,
        "GRAD_ACCUM": 2,
        "epoch": 1,
        "model": rec.model,
        "optimizer": SimpleNamespace(step=lambda: None, zero_grad=lambda: None),
        "pipe": SimpleNamespace(to_batch=lambda example, model: {}),
        "supervised_token_count": lambda answer: 1,
    }
    body = _epoch_body().replace("optimizer_steps += 1", "optimizer_steps += 0")
    with pytest.raises(RuntimeError, match="optimizer steps, expected 2"):
        exec(compile(body, "<mutated training loop>", "exec"), namespace)  # noqa: S102
