from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from third_party.textgrad import BlackboxLLM, TextualGradientDescent, Variable
from third_party.textgrad.autograd.functional import sum as tg_sum
from third_party.textgrad_utils.prompt_complexity import calculate_text_complexity

from synapse.textgrad_support import TextGradSettings


@dataclass
class BatchTrainingResult:
    """
    Lightweight record capturing the outcome of a single TextGrad batch update.
    """

    batch_loss: float
    updated_loss: float
    accepted: bool
    complexity: float


class TextGradPromptTrainer:
    """
    Helper that mirrors FedTextGrad's prompt optimisation loop for a SYNAPSE client.
    """

    def __init__(self, settings: TextGradSettings) -> None:
        self.settings = settings

    def train_batches(
        self,
        dataloader: Iterable[Tuple[Sequence[str], Sequence[int]]],
        model: BlackboxLLM,
        optimizer: TextualGradientDescent,
        eval_fn,
        system_prompt: Variable,
    ) -> List[BatchTrainingResult]:
        """
        Execute a lightweight TextGrad training loop using proximal updates.
        """
        results: List[BatchTrainingResult] = []
        max_steps = self.settings.max_steps
        step_counter = 0

        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()

            batch_loss = 0.0
            losses = []
            for x, y in zip(batch_x, batch_y):
                x_var = Variable(x, requires_grad=False, role_description="client query")
                y_val = self._coerce_label(y)
                y_var = Variable(y_val, requires_grad=False, role_description="ground truth answer")
                response = model(x_var)
                try:
                    eval_output = eval_fn(inputs={"prediction": response, "ground_truth_answer": y_var})
                except Exception:
                    eval_output = eval_fn([x_var, y_var, response])

                losses.append(eval_output)
                batch_loss += self._extract_accuracy(eval_output)

            batch_loss /= max(len(batch_x), 1)

            baseline_prompt = system_prompt.get_value()
            total_loss = tg_sum(losses)
            total_loss.backward(engine=optimizer.engine)
            optimizer.step()

            updated_loss = 0.0
            for x, y in zip(batch_x, batch_y):
                x_var = Variable(x, requires_grad=False, role_description="client query")
                y_val = self._coerce_label(y)
                y_var = Variable(y_val, requires_grad=False, role_description="ground truth answer")
                response = model(x_var)
                try:
                    eval_output = eval_fn(inputs={"prediction": response, "ground_truth_answer": y_var})
                except Exception:
                    eval_output = eval_fn([x_var, y_var, response])
                updated_loss += self._extract_accuracy(eval_output)

            updated_loss /= max(len(batch_x), 1)

            accepted_update = self._accept_update(batch_loss, updated_loss)
            if not accepted_update:
                system_prompt.set_value(baseline_prompt)

            complexity = calculate_text_complexity(system_prompt.get_value())
            results.append(
                BatchTrainingResult(
                    batch_loss=batch_loss,
                    updated_loss=updated_loss,
                    accepted=accepted_update,
                    complexity=complexity,
                )
            )

            step_counter += 1
            if max_steps is not None and step_counter >= max_steps:
                break

        return results

    def _accept_update(self, loss: float, updated_loss: float) -> bool:
        """
        Decide whether to keep the candidate prompt update following the proximal rule.
        """
        if not self.settings.proximal_update:
            return updated_loss >= loss

        if updated_loss <= loss and updated_loss != 1.0:
            return False
        return True

    @staticmethod
    def _extract_accuracy(eval_variable: Variable) -> float:
        """
        Pull the numerical score out of the evaluation variable.
        """
        value = eval_variable.get_value()
        match = re.search(r"<ACCURACY>\s*(\d+)\s*</ACCURACY>", value)
        if match:
            return float(match.group(1))
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _coerce_label(label) -> int:
        """
        Map various dataset label formats (raw ints, numpy scalars, strings) to an integer.
        """
        if isinstance(label, (int, np.integer)):
            return int(label)
        if isinstance(label, str):
            matches = re.findall(r"-?\d+", label)
            if matches:
                return int(matches[-1])
        raise ValueError(f"Unable to parse label from value: {label!r}")
