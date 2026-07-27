#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openrouter_client import chat_completion
from synapse.retrieval.learned_router import load_routing_examples

DEFAULT_RUNLOG = REPO_ROOT / "GSM8K_500_rebuttal_run" / "GSM8K_500_runlog.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "verification" / "gsm8k_openrouter_router"
DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct"
LABELS = [
    "Financial and Banking Calculator",
    "Percentage and Proportion Solver",
    "Algebraic Word Problem Solver",
    "Work, Rate, and Time Analyzer",
    "General Logic and Counting",
    "Geometry and Measurement",
]
LABEL_RULES = {
    "Financial and Banking Calculator": (
        "Choose only for money, prices, profit, loss, cost, spending, interest, "
        "discounts, wages, revenue, or explicit financial quantities."
    ),
    "Percentage and Proportion Solver": (
        "Choose for percentages, fractions, ratios, proportions, shares of a whole, "
        "or repeated remaining-percentage calculations."
    ),
    "Algebraic Word Problem Solver": (
        "Choose for unknown variables, comparative relations like twice/half/more than/"
        "fewer than, or problems best solved by setting up an equation."
    ),
    "Work, Rate, and Time Analyzer": (
        "Choose for speed, distance, time, schedules, work rate, per-minute/per-hour "
        "processes, or coordinated activity over time."
    ),
    "General Logic and Counting": (
        "Choose for plain arithmetic story problems, counting, totals, add/subtract/"
        "multiply/divide discrete quantities, or when no other specialist clearly dominates."
    ),
    "Geometry and Measurement": (
        "Choose for area, perimeter, circumference, volume, angles, dimensions, lengths, "
        "unit conversion tied to physical measurement, or geometric objects."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an OpenRouter-hosted replacement GSM8K router on seeded held-out subsets. "
            "This is a new scorer, not a reconstruction of the historical paper router."
        )
    )
    parser.add_argument("--runlog", type=Path, default=DEFAULT_RUNLOG)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--examples-per-label", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def parse_seed_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def sample_indices(total: int, seed: int, sample_count: int) -> list[int]:
    if sample_count > total:
        raise ValueError(f"Requested {sample_count} rows, but only {total} rows are available.")
    rng = random.Random(seed)
    return sorted(rng.sample(range(total), sample_count))


def build_example_bank(
    rows: list[dict[str, Any]],
    *,
    exclude_indices: set[int],
    examples_per_label: int,
) -> dict[str, list[str]]:
    bank = {label: [] for label in LABELS}
    for idx, row in enumerate(rows):
        if idx in exclude_indices:
            continue
        label = row["label"]
        if label not in bank or len(bank[label]) >= examples_per_label:
            continue
        bank[label].append(row["query_text"])
        if all(len(bank[label_name]) >= examples_per_label for label_name in LABELS if label_name in bank):
            break
    return bank


def build_prompt(query_text: str, example_bank: dict[str, list[str]]) -> str:
    examples: list[str] = []
    for label in LABELS:
        for example in example_bank.get(label, []):
            examples.append(f"Label: {label}\nQuery: {example}")
    return (
        "You are a strict router for GSM8K math word problems. "
        "Pick exactly one label from the allowed set. Use the decision rules first and the examples second. "
        "Prefer General Logic and Counting when the problem is simple arithmetic and no specialist is clearly necessary. "
        "Return only the exact label.\n\n"
        "Allowed labels and rules:\n"
        + "\n".join(f"- {label}: {LABEL_RULES[label]}" for label in LABELS)
        + "\n\nLabeled examples:\n"
        + "\n\n".join(examples)
        + "\n\nQuery:\n"
        + query_text
        + "\n"
    )


def normalize_prediction(text: str) -> str:
    cleaned = text.strip().splitlines()[0].strip()
    for label in LABELS:
        if cleaned == label:
            return label
    lowered = cleaned.lower()
    for label in LABELS:
        if label.lower() in lowered:
            return label
    return cleaned


def classify(query_text: str, *, model: str, example_bank: dict[str, list[str]]) -> str:
    response = chat_completion(
        model=model,
        messages=[{"role": "user", "content": build_prompt(query_text, example_bank)}],
        max_tokens=24,
        temperature=0,
    )
    content = response.choices[0].message.content or ""
    return normalize_prediction(content)


def summarize_seed(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    sample_count: int,
    model: str,
    examples_per_label: int,
) -> dict[str, Any]:
    indices = sample_indices(len(rows), seed=seed, sample_count=sample_count)
    example_bank = build_example_bank(rows, exclude_indices=set(indices), examples_per_label=examples_per_label)
    result_rows: list[dict[str, Any]] = []
    correct = 0
    for idx in indices:
        row = rows[idx]
        predicted = classify(row["query_text"], model=model, example_bank=example_bank)
        hit = predicted == row["label"]
        correct += int(hit)
        result_rows.append(
            {
                "query_id": row["query_id"],
                "sample_id": row["sample_id"],
                "query_text": row["query_text"],
                "ground_truth_domain": row["label"],
                "predicted_domain": predicted,
                "routed_correctly": hit,
            }
        )
    accuracy = correct / sample_count if sample_count else 0.0
    return {
        "seed": seed,
        "sample_count": sample_count,
        "correct": correct,
        "accuracy": accuracy,
        "rows": result_rows,
    }


def mean_sd(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values) if values else 0.0
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, sd


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_routing_examples(args.runlog)
    rows = [
        {
            "query_id": example.query_id,
            "sample_id": example.sample_id,
            "query_text": example.query_text,
            "label": example.label,
        }
        for example in examples
        if example.label in LABELS
    ]
    seeds = parse_seed_list(args.seeds)
    per_seed = [
        summarize_seed(
            rows,
            seed=seed,
            sample_count=args.sample_count,
            model=args.model,
            examples_per_label=args.examples_per_label,
        )
        for seed in seeds
    ]

    for result in per_seed:
        out_path = args.output_dir / f"routing_seed_{result['seed']}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    accuracies = [float(result["accuracy"]) for result in per_seed]
    mean_accuracy, sd_accuracy = mean_sd(accuracies)
    summary = {
        "runlog": str(args.runlog),
        "record_count": len(rows),
        "labels": LABELS,
        "model": args.model,
        "examples_per_label": args.examples_per_label,
        "sample_count": args.sample_count,
        "seeds": seeds,
        "mean_accuracy": mean_accuracy,
        "sd_accuracy": sd_accuracy,
        "per_seed_accuracy": {str(result["seed"]): result["accuracy"] for result in per_seed},
        "output_dir": str(args.output_dir),
        "note": (
            "This is a replacement OpenRouter-hosted label classifier evaluated on held-out seeded subsets. "
            "It is not a reconstruction of the historical paper router."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        "### GSM8K OpenRouter Router",
        "",
        f"- Model: {args.model}",
        f"- Held-out seeded accuracy: {mean_accuracy:.3f} ± {sd_accuracy:.3f}",
        f"- Per-seed: {', '.join(f'{seed}={summary['per_seed_accuracy'][str(seed)]:.3f}' for seed in seeds)}",
        f"- Examples per label: {args.examples_per_label}",
        "",
        "This is a replacement scorer, not a reconstruction of the historical paper router.",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"record_count={len(rows)}")
    print(f"mean_accuracy={mean_accuracy:.3f}")
    print(f"sd_accuracy={sd_accuracy:.3f}")
    print(f"per_seed={summary['per_seed_accuracy']}")


if __name__ == "__main__":
    main()
