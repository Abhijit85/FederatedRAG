#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / 'third_party') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'third_party'))

from synapse.clients.textgrad_trainer import TextGradPromptTrainer
from synapse.textgrad_support import TextGradSettings
from third_party.textgrad import BlackboxLLM, TextualGradientDescent, Variable
from third_party.textgrad.tasks import load_task
from third_party.textgrad.tasks.base import DataLoader
from amps_hybrid_scoring import hybrid_amps_score

DEFAULT_OUTPUT_DIR = Path("artifacts") / "verification" / "amps_textgrad_opt"
DEFAULT_TASK = "livebench_math__AMPS_Hard"
DEFAULT_PROMPT = (
    "You are an exact mathematics solver for olympiad-style algebra, statistics, and linear algebra. "
    "Solve the problem carefully and symbolically when needed. Preserve exact forms: fractions, radicals, "
    "factorizations, and algebraic expressions. Do not replace exact answers with decimal approximations unless "
    "the question explicitly asks for approximation. On the final line, output only the final answer in LaTeX, "
    "wrapped in \\boxed{} with no extra commentary."
)
DEFAULT_CONSTRAINTS = [
    "Keep answers mathematically exact whenever the problem asks for an exact form.",
    "Preserve algebraic equivalence and prefer simplified symbolic forms over decimals.",
    "The final line must contain only the final answer wrapped in \\boxed{}.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small TextGrad optimization loop on LiveBench AMPS_Hard.")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--model", type=str, default="gpt-4o-2024-05-13", help="Forward model used to answer AMPS questions.")
    parser.add_argument("--evaluation-engine", type=str, default="gpt-4o-2024-05-13", help="Backward/optimizer engine for TextGrad.")
    parser.add_argument("--train-limit", type=int, default=12)
    parser.add_argument("--val-limit", type=int, default=8)
    parser.add_argument("--test-limit", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--gradient-memory", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--constraint", action="append", default=[])
    parser.add_argument("--judge-model", type=str, default=None)
    parser.add_argument("--judge-max-tokens", type=int, default=256)
    return parser.parse_args()


def slice_dataset(dataset, limit: int):
    limit = min(limit, len(dataset))
    return [dataset[i] for i in range(limit)]


def extract_accuracy(eval_variable: Variable) -> float:
    value = eval_variable.get_value()
    text = str(value)
    if "<ACCURACY>" in text:
        import re
        match = re.search(r"<ACCURACY>\s*(\d+)\s*</ACCURACY>", text)
        if match:
            return float(match.group(1))
    try:
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def evaluate_samples(model: BlackboxLLM, eval_fn, samples: list[tuple[str, str]], *, judge_model: str | None = None, judge_max_tokens: int = 256) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    correct = 0
    exact_correct = 0
    judged_correct = 0
    judge_used = 0
    for idx, (question, answer) in enumerate(samples, start=1):
        x_var = Variable(question, requires_grad=False, role_description="benchmark query")
        y_var = Variable(answer, requires_grad=False, role_description="ground truth answer")
        response = model(x_var)
        prediction_text = response.get_value()
        try:
            eval_output = eval_fn(inputs={"prediction": response, "ground_truth_answer": y_var})
        except Exception:
            eval_output = eval_fn([x_var, y_var, response])
        score = extract_accuracy(eval_output)
        meta = {"final_score": int(score >= 1.0), "final_mode": "exact", "exact_score": int(score >= 1.0), "exact_reason": "eval_fn", "judge_used": False}
        if judge_model is not None:
            meta = hybrid_amps_score(
                question=question,
                prediction_text=prediction_text,
                gold_answer=answer,
                judge_model=judge_model,
                judge_max_tokens=judge_max_tokens,
            )
        hit = int(meta["final_score"] >= 1)
        correct += hit
        exact_correct += int(meta.get("exact_score", 0))
        judged_correct += int(meta.get("judge_score", 0))
        judge_used += int(bool(meta.get("judge_used", False)))
        rows.append(
            {
                "index": idx,
                "question": question,
                "gold_answer": answer,
                "prediction": prediction_text,
                "score": score,
                "correct": bool(hit),
                **meta,
            }
        )
    total = len(rows)
    return {
        "sample_count": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "exact_correct": exact_correct,
        "exact_accuracy": (exact_correct / total) if total else 0.0,
        "judged_correct": judged_correct,
        "judge_used_count": judge_used,
        "rows": rows,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    load_dotenv()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    settings = TextGradSettings(
        enabled=True,
        evaluation_engine_name=args.evaluation_engine,
        test_engine_name=args.model,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        proximal_update=True,
    )
    settings.ensure_engines()

    train_set, val_set, test_set, eval_fn = load_task(args.task, evaluation_api=settings.evaluation_engine)
    train_samples = slice_dataset(train_set, args.train_limit)
    val_samples = slice_dataset(val_set, args.val_limit)
    test_samples = slice_dataset(test_set, args.test_limit)

    constraints = list(DEFAULT_CONSTRAINTS)
    constraints.extend(args.constraint)

    system_prompt = Variable(
        args.system_prompt,
        requires_grad=True,
        role_description="system prompt for AMPS_Hard exact math solver",
    )
    model = BlackboxLLM(settings.test_engine, system_prompt)
    optimizer = TextualGradientDescent(
        engine=settings.evaluation_engine,
        parameters=[system_prompt],
        constraints=constraints,
        gradient_memory=args.gradient_memory,
    )
    trainer = TextGradPromptTrainer(settings)

    baseline_val = evaluate_samples(model, eval_fn, val_samples, judge_model=args.judge_model, judge_max_tokens=args.judge_max_tokens)
    baseline_test = evaluate_samples(model, eval_fn, test_samples, judge_model=args.judge_model, judge_max_tokens=args.judge_max_tokens)

    dataloader = DataLoader(train_samples, batch_size=args.batch_size, shuffle=False)
    results = trainer.train_batches(
        dataloader,
        model,
        optimizer,
        eval_fn,
        system_prompt,
        validation_samples=val_samples,
        total_questions=len(train_samples),
    )

    final_val = evaluate_samples(model, eval_fn, val_samples, judge_model=args.judge_model, judge_max_tokens=args.judge_max_tokens)
    final_test = evaluate_samples(model, eval_fn, test_samples, judge_model=args.judge_model, judge_max_tokens=args.judge_max_tokens)

    payload = {
        "task": args.task,
        "model": args.model,
        "evaluation_engine": args.evaluation_engine,
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "test_limit": args.test_limit,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "gradient_memory": args.gradient_memory,
        "constraints": constraints,
        "initial_system_prompt": args.system_prompt,
        "optimized_system_prompt": system_prompt.get_value(),
        "baseline": {
            "val": baseline_val,
            "test": baseline_test,
        },
        "optimized": {
            "val": final_val,
            "test": final_test,
        },
        "train_batches": [
            {
                "batch_loss": item.batch_loss,
                "updated_loss": item.updated_loss,
                "accepted": item.accepted,
                "complexity": item.complexity,
            }
            for item in results
        ],
    }

    write_json(args.output_dir / "summary.json", payload)
    print(json.dumps({
        "baseline_val": baseline_val["accuracy"],
        "baseline_test": baseline_test["accuracy"],
        "optimized_val": final_val["accuracy"],
        "optimized_test": final_test["accuracy"],
        "output": str(args.output_dir / "summary.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
