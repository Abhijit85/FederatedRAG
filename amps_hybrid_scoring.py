from __future__ import annotations

import re
import warnings
from typing import Any, Optional

import sympy
from sympy.parsing.latex import parse_latex

from openrouter_client import chat_completion
from third_party.textgrad.tasks.livebenchmath import (
    last_boxed_only_string,
    normalize_final_answer,
    remove_boxed,
)


AMBIGUOUS_REASONS = {
    "parse_failed",
    "subtract_failed",
    "simplify_error",
    "exception",
    "timeout",
}


def _sanitize_prediction(prediction: str) -> str:
    text = str(prediction or "")
    text = text.replace("+C", "")
    text = text.replace("+ C", "")
    text = text.replace("\\\\fbox{", "\\\\boxed{")
    return text


def exact_amps_equiv(gold_answer: str, parsed_prediction: str) -> tuple[bool, str]:
    try:
        try:
            parsed_gold = parse_latex(gold_answer)
            parsed_pred = parse_latex(parsed_prediction)
        except (
            sympy.parsing.latex.errors.LaTeXParsingError,
            sympy.SympifyError,
            TypeError,
        ):
            warnings.warn(f"couldn't parse one of {gold_answer} or {parsed_prediction}")
            return False, "parse_failed"

        try:
            diff = parsed_gold - parsed_pred
        except TypeError:
            warnings.warn(f"couldn't subtract {gold_answer} and {parsed_prediction}")
            return False, "subtract_failed"

        try:
            if sympy.Abs(sympy.simplify(diff)) < 0.001:
                return True, "exact_equiv"
            return False, "simplify_false"
        except ValueError:
            warnings.warn(f"Had some trouble simplifying when comparing {gold_answer} and {parsed_prediction}")
            return False, "simplify_error"
    except TimeoutError:
        warnings.warn(f"Timed out comparing {gold_answer} and {parsed_prediction}")
        return False, "timeout"
    except Exception as exc:
        warnings.warn(f"Failed comparing {gold_answer} and {parsed_prediction} with {exc}")
        return False, "exception"


def _extract_boxed_prediction(prediction: str) -> tuple[Optional[str], Optional[str]]:
    boxed = last_boxed_only_string(prediction)
    if not boxed:
        return None, None
    parsed = normalize_final_answer(remove_boxed(boxed))
    return boxed, parsed


def llm_judge_amps_equivalence(
    *,
    question: str,
    gold_answer: str,
    prediction_text: str,
    parsed_prediction: str | None,
    judge_model: str,
    max_tokens: int = 256,
) -> dict[str, Any]:
    system = (
        "You are a strict mathematical equivalence judge for benchmark scoring. "
        "Decide whether the predicted final answer should receive credit against the gold answer. "
        "Treat algebraically equivalent exact forms as correct. Treat malformed answers and merely approximate decimals as incorrect when the gold answer is exact. "
        "Output only tags in this exact format: <EQUIV>0 or 1</EQUIV><REASON>short reason</REASON>."
    )
    user = (
        f"Question:\n{question}\n\n"
        f"Gold answer:\n{gold_answer}\n\n"
        f"Predicted final answer (normalized if available):\n{parsed_prediction or 'N/A'}\n\n"
        f"Raw prediction text:\n{prediction_text}\n\n"
        "Is the predicted final answer mathematically equivalent to the gold answer for exact benchmark scoring?"
    )
    response = chat_completion(
        model=judge_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0,
    )
    text = response.choices[0].message.content or ""
    verdict_match = re.search(r"<EQUIV>\s*([01])\s*</EQUIV>", text)
    reason_match = re.search(r"<REASON>\s*(.*?)\s*</REASON>", text, flags=re.DOTALL)
    verdict = int(verdict_match.group(1)) if verdict_match else 0
    reason = reason_match.group(1).strip() if reason_match else "unparsed_judge_response"
    return {
        "judge_model": judge_model,
        "judge_raw_response": text,
        "judge_score": verdict,
        "judge_reason": reason,
    }


def hybrid_amps_score(
    *,
    question: str,
    prediction_text: str,
    gold_answer: str | list[str],
    judge_model: str | None = None,
    judge_max_tokens: int = 256,
) -> dict[str, Any]:
    gold = gold_answer[-1] if isinstance(gold_answer, list) else gold_answer
    sanitized = _sanitize_prediction(prediction_text)
    boxed_prediction, parsed_prediction = _extract_boxed_prediction(sanitized)
    if parsed_prediction is None:
        result = {
            "final_score": 0,
            "final_mode": "exact",
            "exact_score": 0,
            "exact_reason": "no_boxed_final",
            "boxed_prediction": boxed_prediction,
            "parsed_prediction": parsed_prediction,
            "judge_used": False,
        }
        if judge_model:
            judged = llm_judge_amps_equivalence(
                question=question,
                gold_answer=gold,
                prediction_text=sanitized,
                parsed_prediction=parsed_prediction,
                judge_model=judge_model,
                max_tokens=judge_max_tokens,
            )
            result.update(judged)
            result["judge_used"] = True
            result["final_score"] = judged["judge_score"]
            result["final_mode"] = "judge"
        return result

    exact_hit, exact_reason = exact_amps_equiv(gold, parsed_prediction)
    result = {
        "final_score": int(exact_hit),
        "final_mode": "exact",
        "exact_score": int(exact_hit),
        "exact_reason": exact_reason,
        "boxed_prediction": boxed_prediction,
        "parsed_prediction": parsed_prediction,
        "judge_used": False,
    }
    if exact_hit or not judge_model:
        return result

    judged = llm_judge_amps_equivalence(
        question=question,
        gold_answer=gold,
        prediction_text=sanitized,
        parsed_prediction=parsed_prediction,
        judge_model=judge_model,
        max_tokens=judge_max_tokens,
    )
    result.update(judged)
    result["judge_used"] = True
    result["final_score"] = judged["judge_score"]
    result["final_mode"] = "judge"
    return result
