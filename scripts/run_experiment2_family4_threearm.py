#!/usr/bin/env python3
"""
run_experiment2_family4_threearm.py
===================================
Three-arm extension of run_experiment2_family4.py that isolates what typing buys.
Same records, queries, conflict construction, and success metric as Result 4.
Only the surfaced representative changes across arms.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_experiment2_family4 import (
    build_records,
    build_queries,
    build_neighbors,
    build_condition_artifacts,
    cluster_artifacts,
    merge_cluster,
    typed_surface,
    retrieval_score,
    answer_field_hit,
    predicted_label,
    MergedArtifact,
    CALIBRATION,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "verification" / "experiment2_family4_threearm"
RATES = [0, 20, 40, 60]
SEEDS = [1, 2, 3, 4, 5]
ARMS = ["full", "structured_untyped", "typed_generic"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Three-arm typing-isolation extension of family4.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--arms", nargs="+", default=ARMS)
    parser.add_argument("--rates", nargs="+", type=int, default=RATES)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    return parser.parse_args()


def _member_surface(member, cluster):
    return MergedArtifact(
        member.signature,
        member.text,
        dict(member.metadata),
        dict(member.payload),
        list(cluster.members),
    )


def resolve_full(cluster: MergedArtifact, query_case):
    return typed_surface(cluster, query_case)


def resolve_structured_untyped(cluster: MergedArtifact, query_case):
    # Keep structured fields, but surface the member that wins under generic weights.
    best_member = max(
        cluster.members,
        key=lambda member: retrieval_score(query_case, _member_surface(member, cluster), "flat"),
    )
    return _member_surface(best_member, cluster)


def resolve_typed_generic(cluster: MergedArtifact, query_case):
    # Preserve typed fields but remove query-time conflict-log selection.
    counts: dict[str, int] = {}
    for member in cluster.members:
        decisive = str(member.payload.get("decisive_cue") or "")
        counts[decisive] = counts.get(decisive, 0) + 1
    winner = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
    candidates = [member for member in cluster.members if str(member.payload.get("decisive_cue") or "") == winner]
    best_member = max(
        candidates,
        key=lambda member: retrieval_score(query_case, _member_surface(member, cluster), "typed"),
    )
    return _member_surface(best_member, cluster)


RESOLVERS = {
    "full": resolve_full,
    "structured_untyped": resolve_structured_untyped,
    "typed_generic": resolve_typed_generic,
}


def evaluate_arm(records, queries, neighbors, rate: int, seed: int, arm: str) -> dict[str, object]:
    # Use the same conflict construction as Result 4's typed run.
    artifacts, conflicts = build_condition_artifacts(records, neighbors, rate, seed, "typed")
    merged = [merge_cluster(cluster, "typed") for cluster in cluster_artifacts(artifacts)]
    resolver = RESOLVERS[arm]
    rows = []
    correct = 0
    for query_case in queries:
        ranked = sorted(
            merged,
            key=lambda cluster: retrieval_score(query_case, resolver(cluster, query_case), "typed" if arm != "structured_untyped" else "flat"),
            reverse=True,
        )
        top = resolver(ranked[0], query_case)
        predicted = predicted_label(top)
        hit = predicted == query_case.target_scenario and answer_field_hit(query_case, top)
        correct += int(hit)
        rows.append({
            "query_id": query_case.query_id,
            "query": query_case.query,
            "target_scenario": query_case.target_scenario,
            "predicted_scenario": predicted,
            "predicted_decisive_cue": str(top.payload.get("decisive_cue") or ""),
            "predicted_exemplar": str(top.payload.get("exemplar") or ""),
            "correct": hit,
            "top_signature": top.signature,
            "top_cluster_size": len(top.members),
            "top_score": retrieval_score(query_case, top, "typed" if arm != "structured_untyped" else "flat"),
        })
    accuracy = correct / len(queries) if queries else 0.0
    return {
        "arm": arm,
        "seed": seed,
        "rate": rate,
        "sample_count": len(queries),
        "correct": correct,
        "score": accuracy,
        "cluster_count": len(merged),
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
        "rows": rows,
        "calibration": dict(CALIBRATION),
    }


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = build_records()
    queries = build_queries(records)
    neighbors = build_neighbors(records)

    for arm in args.arms:
        for rate in args.rates:
            for seed in args.seeds:
                result = evaluate_arm(records, queries, neighbors, rate, seed, arm)
                out_path = args.out / f"{arm}_{rate}_{seed}.json"
                out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                print(json.dumps({
                    "arm": arm,
                    "rate": rate,
                    "seed": seed,
                    "score": round(float(result["score"]), 3),
                }))


if __name__ == "__main__":
    main()
