#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNLOG = REPO_ROOT / "GSM8K_500_rebuttal_run" / "GSM8K_500_runlog.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "verification" / "routing_runlog_replay"

ALIASES = {
    "geometry and measurement": "geometry shapes and measurement",
    "geometry shapes and measurement": "geometry shapes and measurement",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay stored GSM8K routing predictions from the runlog and report seeded subset accuracy."
    )
    parser.add_argument("--runlog", type=Path, default=DEFAULT_RUNLOG)
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument(
        "--top-gap-min",
        type=float,
        default=None,
        help="Optional minimum top-gap filter. If set, only rows with evaluation.top_gap >= threshold are kept.",
    )
    parser.add_argument(
        "--exclude-ambiguous",
        action="store_true",
        help="If set, exclude rows where evaluation.ambiguous is true.",
    )
    parser.add_argument(
        "--allow-missing-answer",
        action="store_true",
        help="Keep rows without expected_answer/task_type. Default false to mirror the core math routing population.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def parse_seed_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def normalize_label(value: str | None) -> str:
    text = (value or "").strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return ALIASES.get(text, text)


def labels_match(left: str | None, right: str | None) -> bool:
    return normalize_label(left) == normalize_label(right)


def load_runlog(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def filter_rows(
    rows: list[dict[str, Any]],
    *,
    top_gap_min: float | None,
    exclude_ambiguous: bool,
    allow_missing_answer: bool,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for row in rows:
        if row.get("source_kind") != "gsm8k_derived":
            continue
        if not allow_missing_answer and (not row.get("expected_answer") or not row.get("task_type")):
            continue
        evaluation = row.get("evaluation") or {}
        if exclude_ambiguous and bool(evaluation.get("ambiguous")):
            continue
        if top_gap_min is not None:
            top_gap = evaluation.get("top_gap")
            if not isinstance(top_gap, (int, float)) or float(top_gap) < top_gap_min:
                continue
        kept.append(row)
    return kept


def sample_rows(rows: list[dict[str, Any]], seed: int, sample_count: int) -> list[dict[str, Any]]:
    if sample_count > len(rows):
        raise ValueError(f"Requested {sample_count} rows, but only {len(rows)} rows remain after filtering.")
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), sample_count))
    return [rows[idx] for idx in indices]


def evaluate_seed(rows: list[dict[str, Any]], seed: int, sample_count: int) -> dict[str, Any]:
    subset = sample_rows(rows, seed=seed, sample_count=sample_count)
    results: list[dict[str, Any]] = []
    correct = 0
    for row in subset:
        router = row.get("router") or {}
        gold = router.get("ground_truth_domain")
        predicted = router.get("predicted_domain")
        hit = labels_match(predicted, gold)
        correct += int(hit)
        results.append(
            {
                "sample_id": row.get("sample_id"),
                "query_id": row.get("query_id"),
                "ground_truth_domain": gold,
                "predicted_domain": predicted,
                "routed_correctly": hit,
                "top_candidates": router.get("top_candidates"),
                "evaluation": row.get("evaluation"),
            }
        )
    accuracy = correct / sample_count if sample_count else 0.0
    return {
        "seed": seed,
        "sample_count": sample_count,
        "correct": correct,
        "accuracy": accuracy,
        "rows": results,
    }


def summarize(results: list[dict[str, Any]]) -> tuple[float, float]:
    accuracies = [float(result["accuracy"]) for result in results]
    mean_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    sd_accuracy = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
    return mean_accuracy, sd_accuracy


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_runlog(args.runlog)
    filtered = filter_rows(
        rows,
        top_gap_min=args.top_gap_min,
        exclude_ambiguous=args.exclude_ambiguous,
        allow_missing_answer=args.allow_missing_answer,
    )
    seeds = parse_seed_list(args.seeds)

    results = [evaluate_seed(filtered, seed=seed, sample_count=args.sample_count) for seed in seeds]
    for result in results:
        out_path = args.output_dir / f"routing_seed_{result['seed']}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    mean_accuracy, sd_accuracy = summarize(results)
    summary = {
        "runlog": str(args.runlog),
        "filtered_record_count": len(filtered),
        "sample_count": args.sample_count,
        "seeds": seeds,
        "top_gap_min": args.top_gap_min,
        "exclude_ambiguous": args.exclude_ambiguous,
        "allow_missing_answer": args.allow_missing_answer,
        "mean_accuracy": mean_accuracy,
        "sd_accuracy": sd_accuracy,
        "per_seed_accuracy": {str(result["seed"]): result["accuracy"] for result in results},
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"filtered_record_count={len(filtered)}")
    print(f"mean_accuracy={mean_accuracy:.3f}")
    print(f"sd_accuracy={sd_accuracy:.3f}")
    print(f"per_seed={summary['per_seed_accuracy']}")


if __name__ == "__main__":
    main()
