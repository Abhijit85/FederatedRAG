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

from math_qa import JinaAIClient
from scripts.run_gsm8k_router_cascade import (
    RouterSpec,
    collect_scenario_docs,
    cosine_similarity,
    gold_route_label,
    labels_match,
    load_records,
    parse_arm_specs,
    parse_seed_list,
    query_text,
    resolve_router_spec,
    router_decision,
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "verification" / "gsm8k_endpoint_validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clean endpoint baselines through the patched cascade router path.")
    parser.add_argument("--sample-file", type=Path, default=REPO_ROOT / "GSM8K_500_rebuttal_run" / "GSM8K_500_samples.json")
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--client-count", type=int, default=5)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--embed-model", type=str, default="jina-embeddings-v2-base-en")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--arm", action="append", required=True, help="Repeatable arm spec: label|backend|model_or_path")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sample_records(records: list[dict[str, Any]], seed: int, sample_count: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(records)), sample_count))
    return [records[idx] for idx in indices]


def infer_parse_ok(raw_response: str, candidates: list[str]) -> bool:
    for candidate in candidates:
        if labels_match(raw_response, candidate):
            return True
    normalized_raw = raw_response.strip().lower()
    return any(candidate.lower() in normalized_raw or normalized_raw in candidate.lower() for candidate in candidates if normalized_raw)


def top1_pick_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    hits = 0
    for row in rows:
        candidates = row.get("top_candidates") or []
        if candidates and labels_match(row.get("predicted_domain"), candidates[0]):
            hits += 1
    return hits / len(rows)


def parse_fail_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    fails = sum(1 for row in rows if not row.get("parse_ok", False))
    return fails / len(rows)


def evaluate_arm(
    *,
    spec: RouterSpec,
    records: list[dict[str, Any]],
    scenario_docs: list[Any],
    jina_client: JinaAIClient,
    seeds: list[int],
    sample_count: int,
    k: int,
    embed_model: str,
    max_tokens: int,
    output_dir: Path,
) -> dict[str, Any]:
    arm_dir = output_dir / spec.label
    arm_dir.mkdir(parents=True, exist_ok=True)
    seed_results: list[dict[str, Any]] = []

    previous_model = os.environ.get("JINA_EMBED_MODEL")
    for seed in seeds:
        subset = sample_records(records, seed=seed, sample_count=sample_count)
        os.environ["JINA_EMBED_MODEL"] = embed_model
        try:
            query_embeddings = jina_client.get_embeddings([query_text(record) for record in subset])
        finally:
            if previous_model is None:
                os.environ.pop("JINA_EMBED_MODEL", None)
            else:
                os.environ["JINA_EMBED_MODEL"] = previous_model

        rows: list[dict[str, Any]] = []
        correct = 0
        total_latency = 0.0
        for record, query_embedding in zip(subset, query_embeddings):
            query = query_text(record)
            gold = gold_route_label(record)
            ranked = sorted(
                scenario_docs,
                key=lambda doc: cosine_similarity(query_embedding, doc.embedding),
                reverse=True,
            )[:k]
            decision = router_decision(query, ranked, spec, max_tokens=max_tokens) if ranked else None
            predicted = decision.predicted_label if decision else ""
            hit = labels_match(predicted, gold)
            total_latency += decision.latency_seconds if decision else 0.0
            correct += int(hit)
            candidates = [candidate.label for candidate in ranked]
            raw_response = decision.raw_response if decision else ""
            rows.append(
                {
                    "query_id": record.get("query_id") or record.get("sample_id"),
                    "query_text": query,
                    "ground_truth_domain": gold,
                    "predicted_domain": predicted,
                    "routed_correctly": hit,
                    "latency_seconds": decision.latency_seconds if decision else 0.0,
                    "raw_response": raw_response,
                    "parse_ok": infer_parse_ok(raw_response, candidates),
                    "option_scores_logprob": decision.option_scores_logprob if decision else None,
                    "option_margin_logprob": decision.option_margin_logprob if decision else None,
                    "answer_avg_logprob": decision.answer_avg_logprob if decision else None,
                    "top_candidates": candidates,
                }
            )
        result = {
            "seed": seed,
            "sample_count": sample_count,
            "correct": correct,
            "accuracy": correct / sample_count if sample_count else 0.0,
            "mean_latency_seconds": total_latency / sample_count if sample_count else 0.0,
            "top1_pick_rate": top1_pick_rate(rows),
            "parse_fail_rate": parse_fail_rate(rows),
            "rows": rows,
        }
        (arm_dir / f"routing_seed_{seed}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        seed_results.append(result)

    accuracies = [float(item["accuracy"]) for item in seed_results]
    latencies = [float(item["mean_latency_seconds"]) for item in seed_results]
    top1_rates = [float(item["top1_pick_rate"]) for item in seed_results]
    parse_rates = [float(item["parse_fail_rate"]) for item in seed_results]
    summary = {
        "label": spec.label,
        "backend": spec.backend,
        "model": spec.model,
        "local_model_path": spec.local_model_path,
        "sample_count": sample_count,
        "k": k,
        "seeds": seeds,
        "mean_accuracy": sum(accuracies) / len(accuracies),
        "sd_accuracy": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
        "mean_latency_seconds": sum(latencies) / len(latencies),
        "sd_latency_seconds": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "mean_top1_pick_rate": sum(top1_rates) / len(top1_rates),
        "mean_parse_fail_rate": sum(parse_rates) / len(parse_rates),
        "per_seed_accuracy": {str(item["seed"]): item["accuracy"] for item in seed_results},
        "per_seed_latency_seconds": {str(item["seed"]): item["mean_latency_seconds"] for item in seed_results},
        "per_seed_top1_pick_rate": {str(item["seed"]): item["top1_pick_rate"] for item in seed_results},
        "per_seed_parse_fail_rate": {str(item["seed"]): item["parse_fail_rate"] for item in seed_results},
        "output_dir": str(arm_dir),
    }
    (arm_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.sample_file)
    seeds = parse_seed_list(args.seeds)
    jina_client = JinaAIClient(api_keys=os.environ.get("JINA_API_KEY") and [os.environ["JINA_API_KEY"]] or [])
    scenario_docs = collect_scenario_docs(
        rounds=args.rounds,
        client_count=args.client_count,
        jina_client=jina_client,
        embed_model=args.embed_model,
    )
    arm_specs = [resolve_router_spec(spec) for spec in parse_arm_specs(args.arm)]
    summaries = [
        evaluate_arm(
            spec=spec,
            records=records,
            scenario_docs=scenario_docs,
            jina_client=jina_client,
            seeds=seeds,
            sample_count=args.sample_count,
            k=args.k,
            embed_model=args.embed_model,
            max_tokens=args.max_tokens,
            output_dir=args.output_dir,
        )
        for spec in arm_specs
    ]
    combined = {
        "sample_file": str(args.sample_file),
        "sample_count": args.sample_count,
        "rounds": args.rounds,
        "client_count": args.client_count,
        "k": args.k,
        "embed_model": args.embed_model,
        "seeds": seeds,
        "max_tokens": args.max_tokens,
        "arms": summaries,
        "note": "Clean endpoint-only validation run through the patched cascade router path.",
    }
    (args.output_dir / "combined_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
