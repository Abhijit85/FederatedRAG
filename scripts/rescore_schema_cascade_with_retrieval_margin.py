#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rescore a finished schema-cascade paired run using retrieval top-1/top-2 cosine margin "
            "as the deferral signal and retrieval top-1 as the cheap routing tier."
        )
    )
    parser.add_argument(
        "--paired-summary",
        type=Path,
        required=True,
        help="Path to summary.json from scripts/run_gsm8k_schema_cascade_paired.py",
    )
    parser.add_argument(
        "--sample-file",
        type=Path,
        default=REPO_ROOT / "GSM8K_500_rebuttal_run" / "GSM8K_500_samples.json",
    )
    parser.add_argument("--thresholds", type=str, default="0.02,0.05,0.1,0.15,0.2")
    parser.add_argument("--small-model-billions", type=float, default=0.0)
    parser.add_argument("--large-model-billions", type=float, default=8.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_thresholds(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_label(value: str | None) -> str:
    text = (value or "").strip().lower()
    text = text.replace("&", " and ")
    for ch in ":,.-_/()[]{}":
        text = text.replace(ch, " ")
    text = " ".join(text.split())
    aliases = {
        "geometry and measurement": "geometry shapes and measurement",
        "geometry shapes and measurement": "geometry shapes and measurement",
    }
    return aliases.get(text, text)


def labels_match(left: str | None, right: str | None) -> bool:
    return normalize_label(left) == normalize_label(right)


def effective_cost_ratio(deferral_rate: float, small_model_billions: float, large_model_billions: float) -> float:
    if large_model_billions <= 0:
        return 1.0
    return (small_model_billions + deferral_rate * large_model_billions) / large_model_billions


def build_query_lookup(sample_file: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(sample_file)
    records = payload["records"] if isinstance(payload, dict) else payload
    lookup: dict[str, dict[str, Any]] = {}
    for record in records:
        query_id = record.get("query_id") or record.get("sample_id")
        if not query_id:
            continue
        router = record.get("router") or {}
        top_candidates = router.get("top_candidates") or []
        top1 = top_candidates[0]["domain"] if len(top_candidates) >= 1 else ""
        top1_score = float(top_candidates[0]["cosine_score"]) if len(top_candidates) >= 1 else None
        top2_score = float(top_candidates[1]["cosine_score"]) if len(top_candidates) >= 2 else None
        gap = None
        if top1_score is not None and top2_score is not None:
            gap = top1_score - top2_score
        lookup[str(query_id)] = {
            "retrieval_top1_domain": top1,
            "retrieval_top1_score": top1_score,
            "retrieval_top2_score": top2_score,
            "retrieval_margin": gap,
        }
    return lookup


def summarize_retrieval_only(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "label": "retrieval_top1_only",
        "mean_accuracy": statistics.mean(item["accuracy"] for item in seed_results) if seed_results else 0.0,
        "sd_accuracy": statistics.stdev(item["accuracy"] for item in seed_results) if len(seed_results) > 1 else 0.0,
        "per_seed_accuracy": {str(item["seed"]): item["accuracy"] for item in seed_results},
        "seed_results": seed_results,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paired_summary = load_json(args.paired_summary)
    query_lookup = build_query_lookup(args.sample_file)
    thresholds = parse_thresholds(args.thresholds)

    large_seed_results = paired_summary["baselines"]["large_only"]["seed_results"]
    large_rows_by_seed = {int(item["seed"]): item["rows"] for item in large_seed_results}
    large_latency_by_seed = {int(item["seed"]): float(item["mean_latency_seconds"]) for item in large_seed_results}
    sample_count = int(paired_summary["sample_count"])

    retrieval_seed_results: list[dict[str, Any]] = []
    cached_rows: dict[int, list[dict[str, Any]]] = {}
    for seed, rows in large_rows_by_seed.items():
        out_rows = []
        correct = 0
        for row in rows:
            extra = query_lookup[row["query_id"]]
            predicted = extra["retrieval_top1_domain"]
            hit = labels_match(predicted, row["ground_truth_domain"])
            correct += int(hit)
            out_rows.append(
                {
                    "query_id": row["query_id"],
                    "ground_truth_domain": row["ground_truth_domain"],
                    "retrieval_top1_domain": predicted,
                    "retrieval_margin": extra["retrieval_margin"],
                    "large_predicted_domain": row["predicted_domain"],
                    "routed_correctly": hit,
                    "top_candidates": row["top_candidates"],
                }
            )
        cached_rows[seed] = out_rows
        retrieval_seed_results.append(
            {
                "seed": seed,
                "sample_count": sample_count,
                "accuracy": correct / sample_count if sample_count else 0.0,
                "correct": correct,
                "rows": out_rows,
            }
        )

    retrieval_only = summarize_retrieval_only(retrieval_seed_results)
    threshold_summaries: list[dict[str, Any]] = []
    for threshold in thresholds:
        seed_results: list[dict[str, Any]] = []
        for seed, rows in cached_rows.items():
            correct = 0
            kept = 0
            kept_correct = 0
            deferrals = 0
            out_rows = []
            for row in rows:
                margin = row["retrieval_margin"]
                keep_retrieval = isinstance(margin, (int, float)) and float(margin) >= threshold
                predicted = row["retrieval_top1_domain"] if keep_retrieval else row["large_predicted_domain"]
                hit = labels_match(predicted, row["ground_truth_domain"])
                correct += int(hit)
                if keep_retrieval:
                    kept += 1
                    kept_correct += int(hit)
                    tier = "retrieval_top1"
                else:
                    deferrals += 1
                    tier = "large"
                out_rows.append(
                    {
                        "query_id": row["query_id"],
                        "ground_truth_domain": row["ground_truth_domain"],
                        "predicted_domain": predicted,
                        "routed_correctly": hit,
                        "tier": tier,
                        "deferred_to_large": not keep_retrieval,
                        "retrieval_margin": margin,
                        "retrieval_top1_domain": row["retrieval_top1_domain"],
                        "large_predicted_domain": row["large_predicted_domain"],
                        "top_candidates": row["top_candidates"],
                    }
                )
            accuracy = correct / sample_count if sample_count else 0.0
            kept_rate = kept / sample_count if sample_count else 0.0
            deferral_rate = deferrals / sample_count if sample_count else 0.0
            kept_accuracy = kept_correct / kept if kept else None
            compute_ratio = effective_cost_ratio(deferral_rate, args.small_model_billions, args.large_model_billions)
            latency_ratio = deferral_rate if large_latency_by_seed[seed] else None
            seed_results.append(
                {
                    "seed": seed,
                    "threshold": threshold,
                    "sample_count": sample_count,
                    "accuracy": accuracy,
                    "correct": correct,
                    "kept": kept,
                    "kept_rate": kept_rate,
                    "kept_accuracy": kept_accuracy,
                    "deferrals": deferrals,
                    "deferral_rate": deferral_rate,
                    "latency_ratio_vs_large_only": latency_ratio,
                    "effective_compute_ratio_vs_large_only": compute_ratio,
                    "effective_compute_reduction_vs_large_only": 1.0 - compute_ratio,
                    "rows": out_rows,
                }
            )
        summary = {
            "threshold": threshold,
            "mean_accuracy": statistics.mean(item["accuracy"] for item in seed_results) if seed_results else 0.0,
            "sd_accuracy": statistics.stdev(item["accuracy"] for item in seed_results) if len(seed_results) > 1 else 0.0,
            "mean_kept_rate": statistics.mean(item["kept_rate"] for item in seed_results) if seed_results else 0.0,
            "sd_kept_rate": statistics.stdev(item["kept_rate"] for item in seed_results) if len(seed_results) > 1 else 0.0,
            "mean_deferral_rate": statistics.mean(item["deferral_rate"] for item in seed_results) if seed_results else 0.0,
            "sd_deferral_rate": statistics.stdev(item["deferral_rate"] for item in seed_results) if len(seed_results) > 1 else 0.0,
            "mean_kept_accuracy": statistics.mean(item["kept_accuracy"] for item in seed_results if item["kept_accuracy"] is not None) if seed_results else None,
            "sd_kept_accuracy": statistics.stdev(item["kept_accuracy"] for item in seed_results if item["kept_accuracy"] is not None) if sum(item["kept_accuracy"] is not None for item in seed_results) > 1 else 0.0,
            "mean_latency_ratio_vs_large_only": statistics.mean(item["latency_ratio_vs_large_only"] for item in seed_results if item["latency_ratio_vs_large_only"] is not None) if seed_results else None,
            "mean_effective_compute_ratio_vs_large_only": statistics.mean(item["effective_compute_ratio_vs_large_only"] for item in seed_results) if seed_results else 1.0,
            "mean_effective_compute_reduction_vs_large_only": statistics.mean(item["effective_compute_reduction_vs_large_only"] for item in seed_results) if seed_results else 0.0,
            "per_seed_accuracy": {str(item["seed"]): item["accuracy"] for item in seed_results},
            "per_seed_deferral_rate": {str(item["seed"]): item["deferral_rate"] for item in seed_results},
            "per_seed_kept_accuracy": {str(item["seed"]): item["kept_accuracy"] for item in seed_results},
            "seed_results": seed_results,
        }
        threshold_summaries.append(summary)
        threshold_slug = str(threshold).replace("-", "neg_").replace(".", "p")
        out_dir = args.output_dir / f"threshold_{threshold_slug}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for seed_result in seed_results:
            (out_dir / f"routing_seed_{seed_result['seed']}.json").write_text(json.dumps(seed_result, indent=2), encoding="utf-8")

    out = {
        "paired_summary": str(args.paired_summary),
        "sample_file": str(args.sample_file),
        "thresholds": thresholds,
        "small_model_billions": args.small_model_billions,
        "large_model_billions": args.large_model_billions,
        "baselines": {
            "retrieval_top1_only": retrieval_only,
            "large_only": paired_summary["baselines"]["large_only"],
        },
        "results": threshold_summaries,
        "note": "Retrieval-margin cascade: accept retrieval top-1 when top-1/top-2 cosine margin exceeds threshold; otherwise defer to paired large-only reranker prediction.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(
        {
            "output": str(args.output_dir / "summary.json"),
            "retrieval_top1_only_accuracy": retrieval_only["mean_accuracy"],
            "thresholds": [
                {
                    "threshold": row["threshold"],
                    "accuracy": row["mean_accuracy"],
                    "deferral_rate": row["mean_deferral_rate"],
                    "kept_accuracy": row["mean_kept_accuracy"],
                }
                for row in threshold_summaries
            ],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
