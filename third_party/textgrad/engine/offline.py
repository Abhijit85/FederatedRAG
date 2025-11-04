import re
from typing import List, Union

from .base import EngineLM


class OfflineMockEngine(EngineLM):
    """
    Minimal EngineLM implementation that keeps the TextGrad pipelines running
    when no networked model is available. It performs simple heuristics so the
    optimiser always receives well-formed updates and math tasks get a
    deterministic answer.
    """

    DEFAULT_SYSTEM_PROMPT = "You are an offline TextGrad mock model."

    def __init__(self, model_string: str = "offline-mock", system_prompt: str | None = None, **kwargs) -> None:
        self.model_string = model_string
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self._call_count = 0

    def __call__(self, prompt: Union[str, List[Union[str, bytes]]], **kwargs):
        return self.generate(prompt, **kwargs)

    def generate(self, prompt: Union[str, List[Union[str, bytes]]], system_prompt: str | None = None, **kwargs) -> str:
        self._call_count += 1
        if isinstance(prompt, list):
            prompt_text = " ".join(map(str, prompt))
        else:
            prompt_text = prompt

        improved = self._maybe_improve_variable(prompt_text)
        if improved:
            return improved

        numeric_summary = self._simple_numeric_answer(prompt_text)
        sys_prompt = (system_prompt or self.system_prompt).strip()
        return f"[offline-textgrad #{self._call_count}] Using '{sys_prompt[:32]}...' Answer: {numeric_summary}"

    @staticmethod
    def _maybe_improve_variable(prompt: str) -> str | None:
        match = re.search(r"<VARIABLE>\s*(.*?)\s*</VARIABLE>", prompt, flags=re.DOTALL)
        if not match:
            return None
        variable_text = match.group(1).strip()
        if not variable_text:
            variable_text = "placeholder variable"
        improved = f"{variable_text}\n\n(offline refinement)"
        return f"<IMPROVED_VARIABLE>{improved}</IMPROVED_VARIABLE>"

    @staticmethod
    def _simple_numeric_answer(prompt: str) -> int:
        numbers = re.findall(r"-?\d+", prompt)
        if not numbers:
            return 0
        return sum(int(value) for value in numbers) % 100
