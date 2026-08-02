#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_gsm8k_router_cascade import (  # noqa: E402
    RouterDecision,
    RouterSpec,
    ScenarioDoc,
    effective_cost_ratio,
    labels_match,
    local_router_decision,
    resolve_router_spec,
)
from scripts.run_gsm8k_schema_control import temporary_structured_payload_mode  # noqa: E402
from scripts.run_routing_verification import (  # noqa: E402
    artifact_route_label,
    build_runtime,
    gold_route_label,
    load_records,
    query_text,
    sample_records,
    selector_expanded_max_items,
    temporary_routing_alignment_profile,
)


DEFAULT_SAMPLE_FILE = REPO_ROOT / "GSM8K_500_rebuttal_run" / "GSM8K_500_samples.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "verification" / "gsm8k_schema_cascade_paired"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a paper-aligned GSM8K cascade with schema-control retrieval held fixed, "
            "logging per-query tier attribution and paired large-only baselines on the same sampled queries."
        )
    )
    parser.add_argument("--sample-file", type=Path, default=DEFAULT_SAMPLE_FILE)
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--seeds", type=str, default="42,123,456,789,1024")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--client-count", type=int, default=5)
    parser.add_argument("--max-items", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--thresholds", type=str, default="0.5,1.0,1.5")
    parser.add_argument("--small-model-billions", type=float, default=1.0)
    parser.add_argument("--large-model-billions", type=float, default=8.0)
    parser.add_argument(
        "--small-model-path",
        type=str,
        default=str(REPO_ROOT / "artifacts" / "models" / "Llama-3.2-1B-Instruct"),
    )
    parser.add_argument(
        "--large-model-path",
        type=str,
        default="/mnt/shared/shared_hf_home/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_seed_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_threshold_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def scenario_doc_from_artifact(artifact: Any) -> ScenarioDoc:
    label = artifact_route_label(artifact)
    payload = artifact.structured_payload or {}
    text = ""
    if isinstance(payload, dict):
        for key in ("scenario_context", "context", "scenario", "domain"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                text = value.strip()
                break
    if not text:
        text = str(getattr(artifact, "text", "") or "").strip()
    if not text:
        text = f"Routing scenario label: {label}."
    return ScenarioDoc(label=label, text=text, embedding=[])


def build_router_specs(args: argparse.Namespace) -> tuple[RouterSpec, RouterSpec]:
    small = resolve_router_spec(
        RouterSpec(
            label="llama32_1b_schema_cascade",
            backend="local",
            model=args.small_model_path,
            local_model_path=args.small_model_path,
        )
    )
    large = resolve_router_spec(
        RouterSpec(
            label="llama31_8b_schema_cascade",
            backend="local",
            model=args.large_model_path,
            local_model_path=args.large_model_path,
        )
    )
    return small, large


def model_decision(query: str, docs: list[ScenarioDoc], spec: RouterSpec, max_tokens: int) -> RouterDecision:
    return local_router_decision(query, docs, spec, max_tokens=max_tokens) if docs else RouterDecision("", 0.0, None, None, None, "")


def prepare_cached_rows(
    *,
    records: list[dict[str, Any]],
    seeds: list[int],
    sample_count: int,
    rounds: int,
    client_count: int,
    max_items: int,
    max_tokens: int,
    small_spec: RouterSpec,
    large_spec: RouterSpec,
) -> dict[int, list[dict[str, Any]]]:
    cached: dict[int, list[dict[str, Any]]] = {}
    with temporary_structured_payload_mode("typed"), temporary_routing_alignment_profile():
        runtime = build_runtime(rounds=rounds, client_count=client_count)
        for seed in seeds:
            subset = sample_records(records, seed=seed, sample_count=sample_count)
            rows: list[dict[str, Any]] = []
            for record in subset:
                query = query_text(record)
                gold = gold_route_label(record)
                artifacts = runtime.get_context_for_query(query, max_items=selector_expanded_max_items(max_items))
                docs = [scenario_doc_from_artifact(artifact) for artifact in artifacts][:max_items]
                small = model_decision(query, docs, small_spec, max_tokens=max_tokens)
                large = model_decision(query, docs, large_spec, max_tokens=max_tokens)
                rows.append(
                    {
                        "query_id": record.get("query_id") or record.get("sample_id"),
                        "query_text": query,
                        "ground_truth_domain": gold,
                        "top_candidates": [doc.label for doc in docs],
                        "small": {
                            "predicted_domain": small.predicted_label,
                            "latency_seconds": small.latency_seconds,
                            "answer_avg_logprob": small.answer_avg_logprob,
                            "option_margin_logprob": small.option_margin_logprob,
                            "option_scores_logprob": small.option_scores_logprob,
                            "raw_response": small.raw_response,
                        },
                        "large": {
                            "predicted_domain": large.predicted_label,
                            "latency_seconds": large.latency_seconds,
                            "answer_avg_logprob": large.answer_avg_logprob,
                            "option_margin_logprob": large.option_margin_logprob,
                            "option_scores_logprob": large.option_scores_logprob,
                            "raw_response": large.raw_response,
                        },
                    }
                )
            cached[seed] = rows
    return cached


def summarize_large_only(cached_rows: dict[int, list[dict[str, Any]]], sample_count: int) -> dict[str, Any]:
    seed_results: list[dict[str, Any]] = []
    for seed, rows in cached_rows.items():
        correct = 0
        latency_sum = 0.0
        out_rows = []
        for row in rows:
            predicted = row["large"]["predicted_domain"]
            hit = labels_match(predicted, row["ground_truth_domain"])
            correct += int(hit)
            latency_sum += float(row["large"]["latency_seconds"])
            out_rows.append(
                {
                    "query_id": row["query_id"],
                    "ground_truth_domain": row["ground_truth_domain"],
                    "predicted_domain": predicted,
                    "routed_correctly": hit,
                    "tier": "large_only",
                    "top_candidates": row["top_candidates"],
                }
            )
        seed_results.append(
            {
                "seed": seed,
                "sample_count": sample_count,
                "accuracy": correct / sample_count if sample_count else 0.0,
                "correct": correct,
                "mean_latency_seconds": latency_sum / sample_count if sample_count else 0.0,
                "rows": out_rows,
            }
        )
    return {
        "label": "large_only",
        "mean_accuracy": statistics.mean(item["accuracy"] for item in seed_results) if seed_results else 0.0,
        "sd_accuracy": statistics.stdev(item["accuracy"] for item in seed_results) if len(seed_results) > 1 else 0.0,
        "mean_latency_seconds": statistics.mean(item["mean_latency_seconds"] for item in seed_results) if seed_results else 0.0,
        "sd_latency_seconds": statistics.stdev(item["mean_latency_seconds"] for item in seed_results) if len(seed_results) > 1 else 0.0,
        "per_seed_accuracy": {str(item["seed"]): item["accuracy"] for item in seed_results},
        "seed_results": seed_results,
    }


def summarize_small_only(cached_rows: dict[int, list[dict[str, Any]]], sample_count: int) -> dict[str, Any]:
    seed_results: list[dict[str, Any]] = []
    for seed, rows in cached_rows.items():
        correct = 0
        latency_sum = 0.0
        out_rows = []
        for row in rows:
            predicted = row["small"]["predicted_domain"]
            hit = labels_match(predicted, row["ground_truth_domain"])
            correct += int(hit)
            latency_sum += float(row["small"]["latency_seconds"])
            out_rows.append(
                {
                    "query_id": row["query_id"],
                    "ground_truth_domain": row["ground_truth_domain"],
                    "predicted_domain": predicted,
                    "routed_correctly": hit,
                    "tier": "small_only",
                    "confidence_value": row["small"]["option_margin_logprob"],
                    "top_candidates": row["top_candidates"],
                }
            )
        seed_results.append(
            {
                "seed": seed,
                "sample_count": sample_count,
                "accuracy": correct / sample_count if sample_count else 0.0,
                "correct": correct,
                "mean_latency_seconds": latency_sum / sample_count if sample_count else 0.0,
                "rows": out_rows,
            }
        )
    return {
        "label": "small_only",
        "mean_accuracy": statistics.mean(item["accuracy"] for item in seed_results) if seed_results else 0.0,
        "sd_accuracy": statistics.stdev(item["accuracy"] for item in seed_results) if len(seed_results) > 1 else 0.0,
        "mean_latency_seconds": statistics.mean(item["mean_latency_seconds"] for item in seed_results) if seed_results else 0.0,
        "sd_latency_seconds": statistics.stdev(item["mean_latency_seconds"] for item in seed_results) if len(seed_results) > 1 else 0.0,
        "per_seed_accuracy": {str(item["seed"]): item["accuracy"] for item in seed_results},
        "seed_results": seed_results,
    }


def summarize_threshold(
    *,
    cached_rows: dict[int, list[dict[str, Any]]],
    threshold: float,
    sample_count: int,
    small_model_billions: float,
    large_model_billions: float,
    large_only_mean_latency_seconds: float,
) -> dict[str, Any]:
    seed_results: list[dict[str, Any]] = []
    for seed, rows in cached_rows.items():
        correct = 0
        kept = 0
        deferrals = 0
        kept_correct = 0
        latency_sum = 0.0
        out_rows = []
        for row in rows:
            confidence_value = row["small"]["option_margin_logprob"]
            keep_small = isinstance(confidence_value, (int, float)) and float(confidence_value) >= threshold
            predicted = row["small"]["predicted_domain"] if keep_small else row["large"]["predicted_domain"]
            hit = labels_match(predicted, row["ground_truth_domain"])
            correct += int(hit)
            latency_sum += float(row["small"]["latency_seconds"])
            if keep_small:
                kept += 1
                kept_correct += int(hit)
                chosen_tier = "small"
            else:
                deferrals += 1
                latency_sum += float(row["large"]["latency_seconds"])
                chosen_tier = "large"
            out_rows.append(
                {
                    "query_id": row["query_id"],
                    "query_text": row["query_text"],
                    "ground_truth_domain": row["ground_truth_domain"],
                    "predicted_domain": predicted,
                    "routed_correctly": hit,
                    "tier": chosen_tier,
                    "deferred_to_large": not keep_small,
                    "small_margin": confidence_value,
                    "small_predicted_domain": row["small"]["predicted_domain"],
                    "large_predicted_domain": row["large"]["predicted_domain"],
                    "small_raw_response": row["small"]["raw_response"],
                    "large_raw_response": row["large"]["raw_response"],
                    "top_candidates": row["top_candidates"],
                }
            )
        accuracy = correct / sample_count if sample_count else 0.0
        deferral_rate = deferrals / sample_count if sample_count else 0.0
        kept_rate = kept / sample_count if sample_count else 0.0
        kept_accuracy = kept_correct / kept if kept else None
        mean_latency_seconds = latency_sum / sample_count if sample_count else 0.0
        latency_ratio = mean_latency_seconds / large_only_mean_latency_seconds if large_only_mean_latency_seconds else None
        compute_ratio = effective_cost_ratio(deferral_rate, small_model_billions, large_model_billions)
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
                "mean_latency_seconds": mean_latency_seconds,
                "latency_ratio_vs_large_only": latency_ratio,
                "effective_compute_ratio_vs_large_only": compute_ratio,
                "effective_compute_reduction_vs_large_only": 1.0 - compute_ratio,
                "rows": out_rows,
            }
        )
    return {
        "threshold": threshold,
        "mean_accuracy": statistics.mean(item["accuracy"] for item in seed_results) if seed_results else 0.0,
        "sd_accuracy": statistics.stdev(item["accuracy"] for item in seed_results) if len(seed_results) > 1 else 0.0,
        "mean_kept_rate": statistics.mean(item["kept_rate"] for item in seed_results) if seed_results else 0.0,
        "sd_kept_rate": statistics.stdev(item["kept_rate"] for item in seed_results) if len(seed_results) > 1 else 0.0,
        "mean_deferral_rate": statistics.mean(item["deferral_rate"] for item in seed_results) if seed_results else 0.0,
        "sd_deferral_rate": statistics.stdev(item["deferral_rate"] for item in seed_results) if len(seed_results) > 1 else 0.0,
        "mean_kept_accuracy": statistics.mean(item["kept_accuracy"] for item in seed_results if item["kept_accuracy"] is not None) if seed_results else None,
        "sd_kept_accuracy": (
            statistics.stdev(item["kept_accuracy"] for item in seed_results if item["kept_accuracy"] is not None)
            if sum(item["kept_accuracy"] is not None for item in seed_results) > 1
            else 0.0
        ),
        "mean_latency_seconds": statistics.mean(item["mean_latency_seconds"] for item in seed_results) if seed_results else 0.0,
        "sd_latency_seconds": statistics.stdev(item["mean_latency_seconds"] for item in seed_results) if len(seed_results) > 1 else 0.0,
        "mean_latency_ratio_vs_large_only": statistics.mean(item["latency_ratio_vs_large_only"] for item in seed_results if item["latency_ratio_vs_large_only"] is not None) if seed_results else None,
        "mean_effective_compute_ratio_vs_large_only": statistics.mean(item["effective_compute_ratio_vs_large_only"] for item in seed_results) if seed_results else 1.0,
        "mean_effective_compute_reduction_vs_large_only": statistics.mean(item["effective_compute_reduction_vs_large_only"] for item in seed_results) if seed_results else 0.0,
        "per_seed_accuracy": {str(item["seed"]): item["accuracy"] for item in seed_results},
        "per_seed_deferral_rate": {str(item["seed"]): item["deferral_rate"] for item in seed_results},
        "per_seed_kept_accuracy": {str(item["seed"]): item["kept_accuracy"] for item in seed_results},
        "seed_results": seed_results,
    }


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.sample_file)
    seeds = parse_seed_list(args.seeds)
    thresholds = parse_threshold_list(args.thresholds)
    small_spec, large_spec = build_router_specs(args)

    cached_rows = prepare_cached_rows(
        records=records,
        seeds=seeds,
        sample_count=args.sample_count,
        rounds=args.rounds,
        client_count=args.client_count,
        max_items=args.max_items,
        max_tokens=args.max_tokens,
        small_spec=small_spec,
        large_spec=large_spec,
    )

    large_only = summarize_large_only(cached_rows, args.sample_count)
    small_only = summarize_small_only(cached_rows, args.sample_count)

    threshold_results = []
    for threshold in thresholds:
        summary = summarize_threshold(
            cached_rows=cached_rows,
            threshold=threshold,
            sample_count=args.sample_count,
            small_model_billions=args.small_model_billions,
            large_model_billions=args.large_model_billions,
            large_only_mean_latency_seconds=large_only["mean_latency_seconds"],
        )
        threshold_results.append(summary)
        threshold_slug = str(threshold).replace("-", "neg_").replace(".", "p")
        out_dir = args.output_dir / f"threshold_{threshold_slug}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for seed_result in summary["seed_results"]:
            (out_dir / f"routing_seed_{seed_result['seed']}.json").write_text(json.dumps(seed_result, indent=2), encoding="utf-8")

    for baseline in (small_only, large_only):
        out_dir = args.output_dir / baseline["label"]
        out_dir.mkdir(parents=True, exist_ok=True)
        for seed_result in baseline["seed_results"]:
            (out_dir / f"routing_seed_{seed_result['seed']}.json").write_text(json.dumps(seed_result, indent=2), encoding="utf-8")

    summary = {
        "sample_file": str(args.sample_file),
        "sample_count": args.sample_count,
        "seeds": seeds,
        "rounds": args.rounds,
        "client_count": args.client_count,
        "max_items": args.max_items,
        "max_tokens": args.max_tokens,
        "thresholds": thresholds,
        "small_spec": {
            "label": small_spec.label,
            "model": small_spec.model,
            "local_model_path": small_spec.local_model_path,
        },
        "large_spec": {
            "label": large_spec.label,
            "model": large_spec.model,
            "local_model_path": large_spec.local_model_path,
        },
        "small_model_billions": args.small_model_billions,
        "large_model_billions": args.large_model_billions,
        "baselines": {
            "small_only": small_only,
            "large_only": large_only,
        },
        "results": threshold_results,
        "note": "Schema-control retrieval held fixed; per-query rows include chosen tier and paired large-only baseline on the exact same sampled queries.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(
        {
            "output": str(args.output_dir / "summary.json"),
            "large_only_accuracy": large_only["mean_accuracy"],
            "small_only_accuracy": small_only["mean_accuracy"],
            "thresholds": [
                {
                    "threshold": row["threshold"],
                    "accuracy": row["mean_accuracy"],
                    "deferral_rate": row["mean_deferral_rate"],
                    "kept_accuracy": row["mean_kept_accuracy"],
                }
                for row in threshold_results
            ],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
