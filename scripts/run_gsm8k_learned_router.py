#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synapse.retrieval import LearnedTextRouter, cross_validated_predictions, load_routing_examples

DEFAULT_RUNLOG = REPO_ROOT / "GSM8K_500_rebuttal_run" / "GSM8K_500_runlog.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "verification" / "gsm8k_learned_router"
DEFAULT_MODEL_PATH = DEFAULT_OUTPUT_DIR / "learned_router.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate a replacement learned GSM8K router on the April 3, 2026 "
            "routing corpus. This is a new scorer, not a reproduction of the historical paper router."
        )
    )
    parser.add_argument("--runlog", type=Path, default=DEFAULT_RUNLOG)
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def parse_seed_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def mean_sd(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values) if values else 0.0
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, sd


def sample_rows(rows: list[dict[str, Any]], seed: int, sample_count: int) -> list[dict[str, Any]]:
    if sample_count > len(rows):
        raise ValueError(f"Requested {sample_count} rows, but only {len(rows)} rows are available.")
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), sample_count))
    return [rows[idx] for idx in indices]


def summarize_seed(rows: list[dict[str, Any]], seed: int, sample_count: int) -> dict[str, Any]:
    subset = sample_rows(rows, seed=seed, sample_count=sample_count)
    correct = sum(1 for row in subset if row["routed_correctly"])
    accuracy = correct / sample_count if sample_count else 0.0
    return {
        "seed": seed,
        "sample_count": sample_count,
        "correct": correct,
        "accuracy": accuracy,
        "rows": subset,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_routing_examples(args.runlog)
    oof_rows = cross_validated_predictions(
        examples,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )
    for row in oof_rows:
        row["source"] = "out_of_fold"

    seeds = parse_seed_list(args.seeds)
    per_seed = [summarize_seed(oof_rows, seed=seed, sample_count=args.sample_count) for seed in seeds]
    for result in per_seed:
        out_path = args.output_dir / f"routing_seed_{result['seed']}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    accuracies = [float(result["accuracy"]) for result in per_seed]
    mean_accuracy, sd_accuracy = mean_sd(accuracies)

    router = LearnedTextRouter().fit(examples)
    router.save(args.model_out)

    summary = {
        "runlog": str(args.runlog),
        "record_count": len(examples),
        "labels": sorted({example.label for example in examples}),
        "n_splits": args.n_splits,
        "random_state": args.random_state,
        "sample_count": args.sample_count,
        "seeds": seeds,
        "mean_accuracy": mean_accuracy,
        "sd_accuracy": sd_accuracy,
        "per_seed_accuracy": {str(result["seed"]): result["accuracy"] for result in per_seed},
        "model_out": str(args.model_out),
        "output_dir": str(args.output_dir),
        "note": (
            "This is a replacement learned router trained on the April 3, 2026 GSM8K routing corpus. "
            "Seeded accuracies are computed from out-of-fold predictions to avoid train/test leakage."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        "### GSM8K Learned Router",
        "",
        f"- Record count: {len(examples)}",
        f"- Labels: {', '.join(summary['labels'])}",
        f"- Cross-validated seeded accuracy: {mean_accuracy:.3f} ± {sd_accuracy:.3f}",
        f"- Per-seed: {', '.join(f'{seed}={summary['per_seed_accuracy'][str(seed)]:.3f}' for seed in seeds)}",
        f"- Saved full-fit model: {args.model_out}",
        "",
        "This is a replacement scorer, not a reconstruction of the historical paper router.",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"record_count={len(examples)}")
    print(f"mean_accuracy={mean_accuracy:.3f}")
    print(f"sd_accuracy={sd_accuracy:.3f}")
    print(f"per_seed={summary['per_seed_accuracy']}")
    print(f"model_out={args.model_out}")


if __name__ == "__main__":
    main()
