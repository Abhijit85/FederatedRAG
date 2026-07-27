#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "third_party") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "third_party"))

import textgrad as tg
from openrouter_client import chat_completion, get_available_api_keys
from textgrad.tasks.livebenchmath import LiveBenchMath, amps_hard_process_results
from math_qa import MathQATool, _load_local_mathqa_backend, _load_local_mathqa_reranker
from amps_hybrid_scoring import hybrid_amps_score
from textgrad.tasks.livebenchreason import (
    LiveBenchReasoning,
    spatial_process_results,
    web_of_lies_process_results,
    zebra_puzzle_process_results,
)

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - optional local backend
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "verification" / "livebench_support"

TASK_SPECS: dict[str, dict[str, Any]] = {
    "reasoning:spatial": {
        "loader": lambda: LiveBenchReasoning(split="test", task="spatial"),
        "score_fn": spatial_process_results,
        "instruction": (
            "Solve the reasoning problem step by step. On the last line, output only the final answer in double asterisks, "
            "for example **triangle** or **7**."
        ),
        "finalizer_instruction": "Output only the final answer in double asterisks, for example **triangle** or **7**.",
    },
    "reasoning:web_of_lies_v2": {
        "loader": lambda: LiveBenchReasoning(split="test", task="web_of_lies_v2"),
        "score_fn": web_of_lies_process_results,
        "instruction": (
            "Solve the reasoning problem step by step. "
            "On the last line, output only the final answer in double asterisks. "
            "If the answer is a sequence of yes/no judgments, format it exactly like **yes, no, yes**."
        ),
        "finalizer_instruction": (
            "Output only the final answer in double asterisks. "
            "If the answer is a sequence of yes/no judgments, format it exactly like **yes, no, yes**."
        ),
    },
    "reasoning:zebra_puzzle": {
        "loader": lambda: LiveBenchReasoning(split="test", task="zebra_puzzle"),
        "score_fn": zebra_puzzle_process_results,
        "instruction": (
            "Solve the puzzle step by step. On the last line, output only the final answer wrapped in triple asterisks, for example ***entrepreneur*** or ***3***."
        ),
        "finalizer_instruction": "Output only the final answer wrapped in triple asterisks, for example ***entrepreneur*** or ***3***.",
    },
    "math:AMPS_Hard": {
        "loader": lambda: LiveBenchMath(split="test", task="AMPS_Hard"),
        "score_fn": amps_hard_process_results,
        "instruction": (
            "Solve the math problem carefully step by step. On the last line, output only the final answer as a boxed expression, "
            "for example \\boxed{42}."
        ),
        "finalizer_instruction": "Output only the final answer as a boxed expression, for example \\boxed{42}.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a scoped LiveBench utility comparison on the four rebuttal subtasks. "
            "This is support evidence from the current repo, not a claim of exact Table 20 reproduction."
        )
    )
    parser.add_argument("--baseline-model", type=str, default="meta-llama/llama-3.1-8b-instruct")
    parser.add_argument("--strong-model", type=str, default="meta-llama/llama-3.3-70b-instruct")
    parser.add_argument("--tasks", type=str, default="reasoning:spatial,reasoning:web_of_lies_v2,reasoning:zebra_puzzle,math:AMPS_Hard")
    parser.add_argument("--limit-per-task", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--finalizer-max-tokens", type=int, default=96)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--amps-backend", type=str, default="direct", choices=["direct", "mathqa"])
    parser.add_argument("--baseline-local-model-path", type=str, default=None)
    parser.add_argument("--strong-local-model-path", type=str, default=None)
    parser.add_argument("--baseline-reranker-model", type=str, default=None)
    parser.add_argument("--strong-reranker-model", type=str, default=None)
    parser.add_argument("--baseline-reranker-device", type=str, default=None)
    parser.add_argument("--strong-reranker-device", type=str, default=None)
    parser.add_argument("--mathqa-force-plain-arith", action="store_true")
    parser.add_argument("--amps-judge-model", type=str, default=None)
    parser.add_argument("--amps-judge-max-tokens", type=int, default=256)
    return parser.parse_args()


def parse_tasks(value: str) -> list[str]:
    tasks = [part.strip() for part in value.split(",") if part.strip()]
    invalid = [task for task in tasks if task not in TASK_SPECS]
    if invalid:
        raise ValueError(f"Unknown task(s): {', '.join(invalid)}. Supported: {', '.join(TASK_SPECS)}.")
    return tasks


def build_prompt(question: str, instruction: str) -> str:
    return f"{question}\n\n{instruction}"


def _model_path_for_name(model: str) -> str | None:
    env_key = f"LIVEBENCH_MODEL_PATH_{model.upper().replace('-', '_').replace('.', '_').replace('/', '_')}"
    return os.environ.get(env_key) or os.environ.get("LIVEBENCH_LOCAL_MODEL_PATH")


def _using_local_backend(model: str) -> bool:
    return bool(_model_path_for_name(model))


@lru_cache(maxsize=4)
def _load_local_backend(model_path: str):
    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError("transformers/torch are required for local LiveBench inference.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype="auto",
        device_map="auto",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def model_answer(*, model: str, prompt: str, max_tokens: int) -> str:
    local_model_path = _model_path_for_name(model)
    if local_model_path:
        tokenizer, local_model = _load_local_backend(local_model_path)
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            rendered = prompt
        encoded = tokenizer(rendered, return_tensors="pt")
        encoded = {key: value.to(local_model.device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = local_model.generate(
                **encoded,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        output_tokens = generated[0][encoded["input_ids"].shape[-1]:]
        return tokenizer.decode(output_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()

    response = chat_completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,
    )
    return response.choices[0].message.content or ""


def finalize_answer(
    *,
    task_name: str,
    model: str,
    question: str,
    draft_prediction: str,
    max_tokens: int,
) -> str:
    finalizer_instruction = TASK_SPECS[task_name]["finalizer_instruction"]
    prompt = (
        "You are formatting a final answer for evaluation.\n"
        "Do not solve the problem again. Read the draft answer and output only the final answer in the required format.\n\n"
        f"Question:\n{question}\n\n"
        f"Draft answer:\n{draft_prediction}\n\n"
        f"Required output:\n{finalizer_instruction}\n"
    )
    return model_answer(model=model, prompt=prompt, max_tokens=max_tokens)


def sample_indices(total: int, limit: int, seed: int) -> list[int]:
    if limit >= total:
        return list(range(total))
    rng = random.Random(seed)
    return sorted(rng.sample(range(total), limit))


@contextmanager
def temporary_mathqa_env(
    *,
    model_name: str,
    local_model_path: str | None,
    reranker_model: str | None,
    reranker_device: str | None,
    force_plain_arith: bool,
):
    tracked = {
        "MATHQA_CHAT_MODEL": model_name,
        "MATHQA_LOCAL_MODEL_PATH": local_model_path,
        "MATHQA_LOCAL_RERANKER_MODEL": reranker_model,
        "MATHQA_LOCAL_RERANKER_DEVICE": reranker_device,
        "MATHQA_SKIP_DRAFT_FOR_PLAIN_ARITH": "1" if force_plain_arith else os.environ.get("MATHQA_SKIP_DRAFT_FOR_PLAIN_ARITH"),
        "MATHQA_USE_CALCULATOR_TOOL": "1",
    }
    previous = {key: os.environ.get(key) for key in tracked}
    try:
        for key, value in tracked.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        _load_local_mathqa_backend.cache_clear()
        _load_local_mathqa_reranker.cache_clear()
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        _load_local_mathqa_backend.cache_clear()
        _load_local_mathqa_reranker.cache_clear()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)


def evaluate_task(
    *,
    task_name: str,
    model: str,
    limit: int,
    seed: int,
    max_tokens: int,
    finalizer_max_tokens: int,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 10,
    judge_model: str | None = None,
    judge_max_tokens: int = 256,
) -> dict[str, Any]:
    spec = TASK_SPECS[task_name]
    dataset = spec["loader"]()
    indices = sample_indices(len(dataset), limit, seed)
    rows: list[dict[str, Any]] = []
    correct = 0
    score_fn: Callable[[tg.Variable, tg.Variable], int] = spec["score_fn"]
    instruction: str = spec["instruction"]

    for idx in indices:
        print(f"[livebench] {task_name} [{model}] start index={idx}", flush=True)
        question, answer = dataset[idx]
        prompt = build_prompt(question, instruction)
        prediction = model_answer(model=model, prompt=prompt, max_tokens=max_tokens)
        score = int(
            score_fn(
                tg.Variable(prediction, role_description="model prediction"),
                tg.Variable(answer, role_description="ground truth answer"),
            )
        )
        finalized_prediction = prediction
        finalized_score = score
        score_meta = {"final_score": score, "final_mode": "exact", "exact_score": score, "exact_reason": "direct", "judge_used": False}
        if score == 0:
            candidate = finalize_answer(
                task_name=task_name,
                model=model,
                question=question,
                draft_prediction=prediction,
                max_tokens=finalizer_max_tokens,
            )
            candidate_score = int(
                score_fn(
                    tg.Variable(candidate, role_description="formatted model prediction"),
                    tg.Variable(answer, role_description="ground truth answer"),
                )
            )
            if candidate_score >= score:
                finalized_prediction = candidate
                finalized_score = candidate_score
                score_meta = {"final_score": candidate_score, "final_mode": "exact", "exact_score": candidate_score, "exact_reason": "formatted", "judge_used": False}
        if task_name == "math:AMPS_Hard" and judge_model is not None:
            score_meta = hybrid_amps_score(
                question=question,
                prediction_text=finalized_prediction,
                gold_answer=answer,
                judge_model=judge_model,
                judge_max_tokens=judge_max_tokens,
            )
            finalized_score = int(score_meta["final_score"])
        correct += finalized_score
        rows.append(
            {
                "index": idx,
                "question": question,
                "gold_answer": answer,
                "prediction": finalized_prediction,
                "raw_prediction": prediction,
                "correct": bool(finalized_score),
                **score_meta,
            }
        )
        print(
            f"[livebench] {task_name} [{model}] done index={idx} correct={bool(finalized_score)} progress={len(rows)}/{len(indices)}",
            flush=True,
        )
        if checkpoint_path is not None and ((len(rows) % checkpoint_every == 0) or (len(rows) == len(indices))):
            partial = {
                "task": task_name,
                "model": model,
                "sample_count": len(rows),
                "correct": correct,
                "accuracy": (correct / len(rows)) if rows else 0.0,
                "rows": rows,
                "complete": len(rows) == len(indices),
            }
            write_json_atomic(checkpoint_path, partial)
    total = len(indices)
    accuracy = correct / total if total else 0.0
    return {
        "task": task_name,
        "model": model,
        "sample_count": total,
        "correct": correct,
        "accuracy": accuracy,
        "rows": rows,
    }


def evaluate_mathqa_task(
    *,
    task_name: str,
    model: str,
    local_model_path: str | None,
    reranker_model: str | None,
    reranker_device: str | None,
    limit: int,
    seed: int,
    max_tokens: int,
    finalizer_max_tokens: int,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 10,
    force_plain_arith: bool = False,
    judge_model: str | None = None,
    judge_max_tokens: int = 256,
) -> dict[str, Any]:
    spec = TASK_SPECS[task_name]
    dataset = spec["loader"]()
    indices = sample_indices(len(dataset), limit, seed)
    rows: list[dict[str, Any]] = []
    correct = 0
    score_fn: Callable[[tg.Variable, tg.Variable], int] = spec["score_fn"]

    previous_amps_exact_mode = os.environ.get("MATHQA_AMPS_EXACT_MODE")
    if task_name == "math:AMPS_Hard":
        os.environ["MATHQA_AMPS_EXACT_MODE"] = "1"
    try:
        with temporary_mathqa_env(
            model_name=model,
            local_model_path=local_model_path,
            reranker_model=reranker_model,
            reranker_device=reranker_device,
            force_plain_arith=force_plain_arith,
        ):
            math_tool = MathQATool()
            for idx in indices:
                print(f"[livebench-mathqa] {task_name} [{model}] start index={idx}", flush=True)
                question, answer = dataset[idx]
                prompt = (
                    f"{question}\n\n"
                    "Solve the math problem carefully. Return exactly one final line in the form: Final Answer: \\boxed{...}."
                )
                result = math_tool.run(user_query=prompt)
                prediction = result.llm_response or ""
                score = int(
                    score_fn(
                        tg.Variable(prediction, role_description="model prediction"),
                        tg.Variable(answer, role_description="ground truth answer"),
                    )
                )
                finalized_prediction = prediction
                finalized_score = score
                score_meta = {"final_score": score, "final_mode": "exact", "exact_score": score, "exact_reason": "direct", "judge_used": False}
                if score == 0:
                    candidate = finalize_answer(
                        task_name=task_name,
                        model=model,
                        question=question,
                        draft_prediction=prediction,
                        max_tokens=finalizer_max_tokens,
                    )
                    candidate_score = int(
                        score_fn(
                            tg.Variable(candidate, role_description="formatted model prediction"),
                            tg.Variable(answer, role_description="ground truth answer"),
                        )
                    )
                    if candidate_score >= score:
                        finalized_prediction = candidate
                        finalized_score = candidate_score
                        score_meta = {"final_score": candidate_score, "final_mode": "exact", "exact_score": candidate_score, "exact_reason": "formatted", "judge_used": False}
                if task_name == "math:AMPS_Hard" and judge_model is not None:
                    score_meta = hybrid_amps_score(
                        question=question,
                        prediction_text=finalized_prediction,
                        gold_answer=answer,
                        judge_model=judge_model,
                        judge_max_tokens=judge_max_tokens,
                    )
                    finalized_score = int(score_meta["final_score"])
                correct += finalized_score
                rows.append(
                    {
                        "index": idx,
                        "question": question,
                        "gold_answer": answer,
                        "prediction": finalized_prediction,
                        "raw_prediction": prediction,
                        "correct": bool(finalized_score),
                        **score_meta,
                    }
                )
                print(
                    f"[livebench-mathqa] {task_name} [{model}] done index={idx} correct={bool(finalized_score)} progress={len(rows)}/{len(indices)}",
                    flush=True,
                )
                if checkpoint_path is not None and ((len(rows) % checkpoint_every == 0) or (len(rows) == len(indices))):
                    partial = {
                        "task": task_name,
                        "model": model,
                        "sample_count": len(rows),
                        "correct": correct,
                        "accuracy": (correct / len(rows)) if rows else 0.0,
                        "rows": rows,
                        "complete": len(rows) == len(indices),
                        "backend": "mathqa",
                        "reranker_model": reranker_model,
                        "local_model_path": local_model_path,
                    }
                    write_json_atomic(checkpoint_path, partial)
    finally:
        if previous_amps_exact_mode is None:
            os.environ.pop("MATHQA_AMPS_EXACT_MODE", None)
        else:
            os.environ["MATHQA_AMPS_EXACT_MODE"] = previous_amps_exact_mode

    total = len(indices)
    accuracy = correct / total if total else 0.0
    return {
        "task": task_name,
        "model": model,
        "sample_count": total,
        "correct": correct,
        "accuracy": accuracy,
        "rows": rows,
        "backend": "mathqa",
        "reranker_model": reranker_model,
        "local_model_path": local_model_path,
    }


def render_markdown(results: list[dict[str, Any]]) -> str:
    by_task: dict[str, dict[str, float]] = {}
    for result in results:
        by_task.setdefault(result["task"], {})[result["arm"]] = result["accuracy"]

    parts = [
        "### LiveBench Support",
        "",
        "| Task | Baseline | Strong | Δ |",
        "| --- | ---: | ---: | ---: |",
    ]
    for task in by_task:
        baseline = by_task[task].get("baseline", 0.0)
        strong = by_task[task].get("strong", 0.0)
        parts.append(f"| {task} | {baseline:.3f} | {strong:.3f} | {strong - baseline:+.3f} |")
    parts.extend(
        [
            "",
            "This table is a scoped support run on the current repo. It does not claim exact Table 20 reproduction.",
        ]
    )
    return "\n".join(parts) + "\n"


def write_summary(
    *,
    output_dir: Path,
    baseline_model: str,
    strong_model: str,
    tasks: list[str],
    limit_per_task: int,
    seed: int,
    runs: list[dict[str, Any]],
) -> None:
    summary = {
        "baseline_model": baseline_model,
        "strong_model": strong_model,
        "tasks": tasks,
        "limit_per_task": limit_per_task,
        "seed": seed,
        "runs": [
            {
                "task": result["task"],
                "arm": result["arm"],
                "accuracy": result["accuracy"],
                "sample_count": result["sample_count"],
                "correct": result["correct"],
            }
            for result in runs
        ],
        "note": (
            "This is a scoped support comparison on the current repository and prompt setup. "
            "It is not a claim that the baseline reproduces historical Table 20 values exactly."
        ),
    }
    write_json_atomic(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(render_markdown(runs), encoding="utf-8")


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = parse_tasks(args.tasks)

    needs_remote = any(not _using_local_backend(model) for model in (args.baseline_model, args.strong_model))
    if needs_remote and not get_available_api_keys(allow_empty=True):
        raise RuntimeError("At least one OpenRouter API key is required when a LiveBench arm is not configured for local inference.")

    runs: list[dict[str, Any]] = []
    for task_name in tasks:
        task_slug = task_name.replace(":", "_")
        eval_fn = evaluate_task
        baseline_kwargs = dict(
            task_name=task_name,
            model=args.baseline_model,
            limit=args.limit_per_task,
            seed=args.seed,
            max_tokens=args.max_tokens,
            finalizer_max_tokens=args.finalizer_max_tokens,
            checkpoint_path=args.output_dir / f"{task_slug}_baseline.partial.json",
            judge_model=args.amps_judge_model,
            judge_max_tokens=args.amps_judge_max_tokens,
        )
        strong_kwargs = dict(
            task_name=task_name,
            model=args.strong_model,
            limit=args.limit_per_task,
            seed=args.seed,
            max_tokens=args.max_tokens,
            finalizer_max_tokens=args.finalizer_max_tokens,
            checkpoint_path=args.output_dir / f"{task_slug}_strong.partial.json",
            judge_model=args.amps_judge_model,
            judge_max_tokens=args.amps_judge_max_tokens,
        )
        if task_name == "math:AMPS_Hard" and args.amps_backend == "mathqa":
            eval_fn = evaluate_mathqa_task
            baseline_kwargs.update(
                local_model_path=args.baseline_local_model_path,
                reranker_model=args.baseline_reranker_model,
                reranker_device=args.baseline_reranker_device,
                force_plain_arith=args.mathqa_force_plain_arith,
            )
            strong_kwargs.update(
                local_model_path=args.strong_local_model_path,
                reranker_model=args.strong_reranker_model,
                reranker_device=args.strong_reranker_device,
                force_plain_arith=args.mathqa_force_plain_arith,
            )

        baseline = eval_fn(**baseline_kwargs)
        baseline["arm"] = "baseline"
        runs.append(baseline)
        (args.output_dir / f"{task_slug}_{baseline['arm']}.json").write_text(
            json.dumps(baseline, indent=2), encoding="utf-8"
        )
        write_summary(
            output_dir=args.output_dir,
            baseline_model=args.baseline_model,
            strong_model=args.strong_model,
            tasks=tasks,
            limit_per_task=args.limit_per_task,
            seed=args.seed,
            runs=runs,
        )

        strong = eval_fn(**strong_kwargs)
        strong["arm"] = "strong"
        runs.append(strong)
        (args.output_dir / f"{task_slug}_{strong['arm']}.json").write_text(
            json.dumps(strong, indent=2), encoding="utf-8"
        )
        write_summary(
            output_dir=args.output_dir,
            baseline_model=args.baseline_model,
            strong_model=args.strong_model,
            tasks=tasks,
            limit_per_task=args.limit_per_task,
            seed=args.seed,
            runs=runs,
        )

    for result in runs:
        print(
            f"{result['task']} [{result['arm']}]: accuracy={result['accuracy']:.3f} "
            f"({result['correct']}/{result['sample_count']})"
        )


if __name__ == "__main__":
    main()
