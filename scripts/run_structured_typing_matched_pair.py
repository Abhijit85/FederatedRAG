#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_experiment2_family4 import (  # noqa: E402
    MergedArtifact,
    answer_field_hit,
    build_condition_artifacts,
    build_neighbors,
    build_queries,
    build_records,
    cluster_artifacts,
    flatten_cluster,
    predicted_label,
    retrieval_score,
)
from run_experiment2_family4_threearm import (  # noqa: E402
    RATES,
    SEEDS,
    resolve_full,
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "verification" / "structured_typing_matched_pair"
ARMS = ["full", "typed_same_merge", "untyped_same_merge"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a clean matched-pair typing ablation on the family-4 contradiction sweep. "
            "The typed and untyped controls share the same generic field-wise merge and differ "
            "only in whether retrieval scoring uses the structured field roles."
        )
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--arms", nargs="+", default=ARMS)
    parser.add_argument("--rates", nargs="+", type=int, default=RATES)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    return parser.parse_args()


LABEL = {
    "full": "Full SYNAPSE",
    "typed_same_merge": "Typed, same merge",
    "untyped_same_merge": "Untyped, same merge",
}


def mean_sd(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values) if values else 0.0
    sd = st.stdev(values) if len(values) > 1 else 0.0
    return mean, sd


def paired_delta(left: dict[int, float], right: dict[int, float]) -> dict[str, Any]:
    common = sorted(set(left) & set(right))
    diffs = [left[seed] - right[seed] for seed in common]
    mean, sd = mean_sd(diffs)
    return {
        "n": len(common),
        "per_seed": {str(seed): left[seed] - right[seed] for seed in common},
        "mean": mean,
        "sd": sd,
        "all_zero": all(abs(value) < 1e-12 for value in diffs),
    }


def generic_merge(cluster: list[Any]) -> MergedArtifact:
    rep = max(cluster, key=lambda artifact: (len(artifact.text), len(artifact.payload)))
    text, metadata, payload = flatten_cluster(cluster)
    return MergedArtifact(rep.signature, text, metadata, payload, list(cluster))


def resolve_typed_same_merge(cluster: MergedArtifact, query_case) -> MergedArtifact:
    return cluster


def resolve_untyped_same_merge(cluster: MergedArtifact, query_case) -> MergedArtifact:
    return cluster


RESOLVERS = {
    "full": resolve_full,
    "typed_same_merge": resolve_typed_same_merge,
    "untyped_same_merge": resolve_untyped_same_merge,
}

SCORING_MODE = {
    "full": "typed",
    "typed_same_merge": "typed",
    "untyped_same_merge": "flat",
}


def evaluate_arm(records, queries, neighbors, rate: int, seed: int, arm: str) -> dict[str, Any]:
    artifacts, conflicts = build_condition_artifacts(records, neighbors, rate, seed, "typed")
    raw_clusters = cluster_artifacts(artifacts)
    merged = [generic_merge(cluster) for cluster in raw_clusters]
    resolver = RESOLVERS[arm]
    scoring_mode = SCORING_MODE[arm]
    rows = []
    correct = 0
    for query_case in queries:
        ranked = sorted(
            merged,
            key=lambda cluster: retrieval_score(query_case, resolver(cluster, query_case), scoring_mode),
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
            "top_score": retrieval_score(query_case, top, scoring_mode),
            "scoring_mode": scoring_mode,
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
        "note": (
            "typed_same_merge and untyped_same_merge use the same generic field-wise merge. "
            "They differ only in query-time scoring: typed uses field-role-aware weights, "
            "untyped uses flat weights on the same merged payload."
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    parts = [
        "### Clean Matched-Pair Typing Ablation",
        "",
        "The `typed_same_merge` and `untyped_same_merge` arms share the same generic field-wise merge. The only changed mechanism is whether query-time scoring uses the structured field roles.",
        "",
        "| Arm | 0% | 20% | 40% | 60% |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for arm in summary["arms"]:
        row = summary["per_arm"][arm]
        cells = []
        for rate in summary["rates"]:
            rate_row = row[str(rate)]
            cells.append(f"{rate_row['mean']:.3f} ± {rate_row['sd']:.3f}")
        parts.append(f"| {LABEL.get(arm, arm)} | " + " | ".join(cells) + " |")

    parts.extend(
        [
            "",
            "| Contrast | 0% | 20% | 40% | 60% |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, rows in summary["contrasts"].items():
        cells = []
        for rate in summary["rates"]:
            rate_row = rows[str(rate)]
            suffix = " (all-zero)" if rate_row["all_zero"] else ""
            cells.append(f"{rate_row['mean']:+.3f}{suffix}")
        parts.append(f"| {name} | " + " | ".join(cells) + " |")

    return "\n".join(parts) + "\n"


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    records = build_records()
    queries = build_queries(records)
    neighbors = build_neighbors(records)

    raw_runs: dict[str, dict[int, dict[int, dict[str, Any]]]] = {arm: {} for arm in args.arms}
    per_arm_summary: dict[str, dict[str, Any]] = {}

    for arm in args.arms:
        per_arm_summary[arm] = {}
        for rate in args.rates:
            per_seed_scores: dict[int, float] = {}
            raw_runs[arm][rate] = {}
            for seed in args.seeds:
                result = evaluate_arm(records, queries, neighbors, rate, seed, arm)
                raw_runs[arm][rate][seed] = result
                per_seed_scores[seed] = float(result["score"])
                out_path = args.out / arm / f"conflict_{rate:02d}" / f"seed_{seed}.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            mean, sd = mean_sd(list(per_seed_scores.values()))
            per_arm_summary[arm][str(rate)] = {
                "mean": mean,
                "sd": sd,
                "per_seed": {str(seed): value for seed, value in per_seed_scores.items()},
            }

    contrasts: dict[str, dict[str, Any]] = {}
    contrast_pairs = [
        ("typed_same_merge", "untyped_same_merge"),
        ("full", "typed_same_merge"),
    ]
    for left_arm, right_arm in contrast_pairs:
        if left_arm not in args.arms or right_arm not in args.arms:
            continue
        name = f"{left_arm} - {right_arm}"
        contrasts[name] = {}
        for rate in args.rates:
            left = {seed: float(raw_runs[left_arm][rate][seed]["score"]) for seed in args.seeds}
            right = {seed: float(raw_runs[right_arm][rate][seed]["score"]) for seed in args.seeds}
            contrasts[name][str(rate)] = paired_delta(left, right)

    summary = {
        "arms": args.arms,
        "rates": args.rates,
        "seeds": args.seeds,
        "per_arm": per_arm_summary,
        "contrasts": contrasts,
        "note": (
            "Clean family-4 matched pair. typed_same_merge and untyped_same_merge share the "
            "same generic merge; only query-time field-role semantics differ."
        ),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out / "summary.md").write_text(render_markdown(summary), encoding="utf-8")

    for arm in args.arms:
        print(
            f"{arm}: "
            + "  ".join(
                f"{rate}%={per_arm_summary[arm][str(rate)]['mean']:.3f}±{per_arm_summary[arm][str(rate)]['sd']:.3f}"
                for rate in args.rates
            )
        )
    for name, rows in contrasts.items():
        print(
            f"{name}: "
            + "  ".join(
                f"{rate}%={rows[str(rate)]['mean']:+.3f}"
                + (" (all-zero)" if rows[str(rate)]["all_zero"] else "")
                for rate in args.rates
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
