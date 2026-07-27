#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_gsm8k_paper_recovery_sweep import (  # noqa: E402
    cross_validated_predictions,
    cross_validated_predictions_aux,
    evaluate_seed as evaluate_historical_seed,
    historical_rows,
    label_prototypes,
    load_compendium_text,
    load_paper_labels,
    load_runlog,
    predict_nearest_neighbor,
    predict_pipeline,
    predict_prototype,
)
from scripts.run_routing_verification import (  # noqa: E402
    _historical_label_selector,
    DEFAULT_SAMPLE_FILE,
    artifact_route_label,
    build_credentials,
    evaluate_seed as evaluate_runtime_seed,
    gold_route_label,
    load_records,
    normalize_label,
    query_text,
    sample_records,
    selector_expanded_max_items,
    selector_mode,
    temporary_routing_alignment_profile,
)
from scripts.run_table27_fresh_compare import (  # noqa: E402
    build_centralized_runtime,
    build_federated_runtime,
    paired_stats,
    summarize,
)
from synapse.retrieval import RetrievalConfig, RetrievalPlanner  # noqa: E402

DEFAULT_RUNLOG = REPO_ROOT / "GSM8K_500_rebuttal_run" / "GSM8K_500_runlog.jsonl"
DEFAULT_COMPENDIUM = REPO_ROOT / "mathqa_tools_compendium.json"
DEFAULT_EVOLUTION = REPO_ROOT / "GSM8K_500_rebuttal_run" / "GSM8K_compendium_evolution.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "verification" / "table27_strict_compare"

RUNTIME_ARM_CHOICES = (
    "runtime_federated",
    "runtime_centralized_direct",
    "runtime_centralized_direct_sourceaware",
    "runtime_centralized_direct_examples_sourceaware",
    "runtime_centralized_direct_sourcecap2",
    "runtime_centralized_direct_sourcecap2_pooled",
    "runtime_centralized_clustered",
)
HISTORICAL_ARM_CHOICES = (
    "historical_cv_svm",
    "historical_cv_logreg",
    "historical_cv_cnb",
    "historical_cv_aux_collapse",
    "historical_query_bank_nn",
    "historical_prototype_12",
    "historical_prototype_30",
    "historical_prototype_60",
)
ARM_CHOICES = RUNTIME_ARM_CHOICES + HISTORICAL_ARM_CHOICES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a stricter paired Table 27-style comparison on shared seeded subsets. "
            "Supports both current runtime arms and preserved historical-paper-space arms."
        )
    )
    parser.add_argument("--sample-file", type=Path, default=DEFAULT_SAMPLE_FILE)
    parser.add_argument("--runlog", type=Path, default=DEFAULT_RUNLOG)
    parser.add_argument("--compendium", type=Path, default=DEFAULT_COMPENDIUM)
    parser.add_argument("--evolution", type=Path, default=DEFAULT_EVOLUTION)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--client-count", type=int, default=5)
    parser.add_argument("--max-items", type=int, default=5)
    parser.add_argument("--synapse-arm", choices=ARM_CHOICES, default="runtime_federated")
    parser.add_argument("--centralized-arm", choices=ARM_CHOICES, default="historical_cv_svm")
    parser.add_argument("--runtime-include-training-artifacts", action="store_true")
    parser.add_argument("--runtime-training-sample-limit", type=int, default=0)
    parser.add_argument("--runtime-included-tools", type=str, default="mathqa")
    parser.add_argument("--runtime-label-selector", type=str, default="historical_cv_svm")
    parser.add_argument("--runtime-synapse-label-selector", type=str, default="")
    parser.add_argument("--runtime-centralized-label-selector", type=str, default="")
    parser.add_argument("--runtime-training-shard-mode", type=str, default="")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def parse_seed_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def write_seed_csv(path: Path, seeds: list[int], values: list[float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "acc"])
        for seed, value in zip(seeds, values):
            writer.writerow([seed, f"{value:.6f}"])


def render_markdown(
    *,
    args: argparse.Namespace,
    synapse_summary: dict[str, Any],
    centralized_summary: dict[str, Any],
    pair: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "### Strict Table 27 Comparison",
            "",
            f"- sample_file: `{args.sample_file}`",
            f"- sample_count: `{args.sample_count}`",
            f"- seeds: `{args.seeds}`",
            f"- synapse_arm: `{args.synapse_arm}`",
            f"- centralized_arm: `{args.centralized_arm}`",
            f"- rounds: `{args.rounds}`",
            f"- client_count: `{args.client_count}`",
            f"- max_items: `{args.max_items}`",
            f"- runtime_include_training_artifacts: `{args.runtime_include_training_artifacts}`",
            f"- runtime_training_sample_limit: `{args.runtime_training_sample_limit}`",
            f"- runtime_included_tools: `{args.runtime_included_tools}`",
            f"- runtime_label_selector: `{args.runtime_label_selector}`",
            f"- runtime_training_shard_mode: `{args.runtime_training_shard_mode}`",
            "",
            "| Arm | Mean acc. | SD | Seeds |",
            "| --- | ---: | ---: | --- |",
            (
                f"| {args.synapse_arm} | {synapse_summary['mean_accuracy']:.3f} | {synapse_summary['sd_accuracy']:.3f} | "
                + ", ".join(f"{seed}={value:.3f}" for seed, value in synapse_summary["per_seed_accuracy"].items())
                + " |"
            ),
            (
                f"| {args.centralized_arm} | {centralized_summary['mean_accuracy']:.3f} | {centralized_summary['sd_accuracy']:.3f} | "
                + ", ".join(f"{seed}={value:.3f}" for seed, value in centralized_summary["per_seed_accuracy"].items())
                + " |"
            ),
            "",
            "| Paired quantity | Value |",
            "| --- | ---: |",
            f"| Mean diff ({args.synapse_arm} - {args.centralized_arm}) | {pair['mean_diff']:+.3f} |",
            f"| SD diff | {pair['sd_diff']:.3f} |",
            f"| SE diff | {pair['se_diff']:.3f} |",
            f"| t statistic | {pair['t_value']:+.3f} |",
            "",
            "This is a current-repo paired comparator with stricter historical-arm options. It is not automatically paper provenance.",
            "",
        ]
    )


def build_eval_rows(sample_file: Path, runlog: Path, evolution: Path) -> list[dict[str, str]]:
    sample_records_payload = load_records(sample_file)
    runlog_rows = load_runlog(runlog)
    paper_labels = load_paper_labels(evolution)
    rows = historical_rows(runlog_rows, paper_labels)
    row_by_qid = {row["query_id"]: row for row in rows}
    eval_rows = [
        row_by_qid[str(record.get("query_id") or record.get("sample_id") or "")]
        for record in sample_records_payload
        if str(record.get("query_id") or record.get("sample_id") or "") in row_by_qid
    ]
    return eval_rows


def build_historical_predictions(
    *,
    arm: str,
    eval_rows: list[dict[str, str]],
    seeds: list[int],
    sample_count: int,
    compendium_text: dict[str, str],
    paper_labels: list[str],
) -> list[str]:
    if arm == "historical_cv_svm":
        return cross_validated_predictions(eval_rows, classifier="svm")
    if arm == "historical_cv_logreg":
        return cross_validated_predictions(eval_rows, classifier="logreg")
    if arm == "historical_cv_cnb":
        return cross_validated_predictions(eval_rows, classifier="cnb")
    if arm == "historical_cv_aux_collapse":
        return cross_validated_predictions_aux(eval_rows)

    if arm == "historical_query_bank_nn":
        all_preds = [""] * len(eval_rows)
        for seed in seeds:
            idx = set(sorted(__import__("random").Random(seed).sample(range(len(eval_rows)), sample_count)))
            train_rows = [row for i, row in enumerate(eval_rows) if i not in idx]
            test_rows = [row for i, row in enumerate(eval_rows) if i in idx]
            preds = predict_nearest_neighbor(train_rows, test_rows)
            for target_idx, pred in zip(sorted(idx), preds):
                all_preds[target_idx] = pred
        return all_preds

    if arm.startswith("historical_prototype_"):
        examples_per_label = int(arm.rsplit("_", 1)[1])
        all_preds = [""] * len(eval_rows)
        for seed in seeds:
            idx = set(sorted(__import__("random").Random(seed).sample(range(len(eval_rows)), sample_count)))
            train_rows = [row for i, row in enumerate(eval_rows) if i not in idx]
            test_rows = [row for i, row in enumerate(eval_rows) if i in idx]
            preds = predict_prototype(
                train_rows=train_rows,
                eval_rows=test_rows,
                labels=paper_labels,
                compendium_text=compendium_text,
                examples_per_label=examples_per_label,
            )
            for target_idx, pred in zip(sorted(idx), preds):
                all_preds[target_idx] = pred
        return all_preds

    raise ValueError(f"Unsupported historical arm: {arm}")


def evaluate_historical_arm(
    *,
    arm: str,
    eval_rows: list[dict[str, str]],
    seeds: list[int],
    sample_count: int,
    compendium_text: dict[str, str],
    paper_labels: list[str],
) -> list[dict[str, Any]]:
    predictions = build_historical_predictions(
        arm=arm,
        eval_rows=eval_rows,
        seeds=seeds,
        sample_count=sample_count,
        compendium_text=compendium_text,
        paper_labels=paper_labels,
    )
    return [
        evaluate_historical_seed(predictions, eval_rows, seed=seed, sample_count=sample_count)
        for seed in seeds
    ]


def evaluate_runtime_arm(
    *,
    arm: str,
    sample_file: Path,
    sample_count: int,
    seeds: list[int],
    rounds: int,
    client_count: int,
    max_items: int,
    include_training_artifacts: bool,
    training_sample_limit: int,
    included_tools: str,
    runtime_label_selector: str,
    training_shard_mode: str,
) -> list[dict[str, Any]]:
    records = load_records(sample_file)
    with temporary_routing_alignment_profile(
        include_training_artifacts=include_training_artifacts,
        training_sample_limit=training_sample_limit,
        included_tools=included_tools,
        runtime_label_selector=runtime_label_selector,
        training_shard_mode=training_shard_mode,
    ):
        if arm == "runtime_federated":
            runtime = build_federated_runtime(rounds, client_count)
        elif arm == "runtime_centralized_direct":
            runtime = build_centralized_runtime(rounds, client_count, "direct")
        elif arm == "runtime_centralized_direct_sourceaware":
            runtime = build_centralized_runtime(rounds, client_count, "direct_sourceaware")
        elif arm == "runtime_centralized_direct_examples_sourceaware":
            runtime = build_centralized_runtime(rounds, client_count, "direct_examples_sourceaware")
        elif arm == "runtime_centralized_direct_sourcecap2":
            runtime = build_centralized_runtime(rounds, client_count, "direct_sourcecap2")
        elif arm == "runtime_centralized_direct_sourcecap2_pooled":
            runtime = build_centralized_runtime(rounds, client_count, "direct_sourcecap2")
        elif arm == "runtime_centralized_clustered":
            runtime = build_centralized_runtime(rounds, client_count, "clustered")
        else:
            raise ValueError(f"Unsupported runtime arm: {arm}")
        if arm == "runtime_centralized_direct_sourcecap2_pooled":
            return [
                evaluate_runtime_seed_pooled(
                    runtime=runtime,
                    records=records,
                    seed=seed,
                    sample_count=sample_count,
                    max_items=max_items,
                )
                for seed in seeds
            ]
        return [
            evaluate_runtime_seed(
                runtime=runtime,
                records=records,
                seed=seed,
                sample_count=sample_count,
                max_items=max_items,
            )
            for seed in seeds
        ]


def pooled_label_prediction(runtime, query: str, max_items: int) -> tuple[str, list[str], list[list[Any]]]:
    compendium = runtime.server.compendium
    artifacts = compendium.build_snapshot().artifacts
    expanded_max_items = selector_expanded_max_items(max_items)
    planner = runtime.retrieval_planner or RetrievalPlanner(RetrievalConfig(max_artifacts=expanded_max_items))
    planner.config.max_artifacts = expanded_max_items
    selected = planner.select(query, artifacts)

    label_scores: dict[str, float] = {}
    label_counts: dict[str, int] = {}
    top_candidates: list[str] = []
    ranked_labels: list[list[Any]] = []
    svm_pipe, svm_classes = _historical_label_selector()
    decision = svm_pipe.decision_function([query])
    scores = decision[0] if getattr(decision, "ndim", 1) > 1 else decision
    class_scores = {normalize_label(label): float(score) for label, score in zip(svm_classes, scores)}
    pooled_weight = float(os.environ.get("SYNAPSE_POOLED_EVIDENCE_WEIGHT", "0.15"))
    for idx, artifact in enumerate(selected):
        label = artifact_route_label(artifact)
        norm = normalize_label(label)
        if not norm:
            continue
        top_candidates.append(label)
        score = planner._score_artifact(query, artifact)
        # Preserve pooled multiplicity and reward earlier evidence.
        score += 1.0 / (idx + 1)
        label_scores[norm] = label_scores.get(norm, 0.0) + score
        label_counts[norm] = label_counts.get(norm, 0) + 1

    if not label_scores:
        return "", top_candidates, ranked_labels

    label_surface: dict[str, str] = {}
    for label in top_candidates:
        norm = normalize_label(label)
        label_surface.setdefault(norm, label)

    blended_scores = {
        norm: class_scores.get(norm, float("-inf")) + pooled_weight * total
        for norm, total in label_scores.items()
    }
    best_norm = max(blended_scores, key=lambda norm: (blended_scores[norm], label_counts.get(norm, 0)))
    for norm, total in sorted(blended_scores.items(), key=lambda item: item[1], reverse=True):
        ranked_labels.append([label_surface.get(norm, norm), round(total, 6), label_counts.get(norm, 0)])
    return label_surface.get(best_norm, best_norm), top_candidates, ranked_labels


def evaluate_runtime_seed_pooled(
    *,
    runtime,
    records: list[dict[str, Any]],
    seed: int,
    sample_count: int,
    max_items: int,
) -> dict[str, Any]:
    subset = sample_records(records, seed=seed, sample_count=sample_count)
    rows: list[dict[str, Any]] = []
    correct = 0

    for record in subset:
        query = query_text(record)
        gold = gold_route_label(record)
        predicted, top_candidates, ranked_labels = pooled_label_prediction(runtime, query, max_items)
        hit = normalize_label(predicted) == normalize_label(gold)
        correct += int(hit)
        rows.append(
            {
                "query_id": record.get("query_id") or record.get("sample_id"),
                "query_text": query,
                "ground_truth_domain": gold,
                "predicted_domain": predicted,
                "routed_correctly": hit,
                "top_candidates": top_candidates,
                "ranked_labels": ranked_labels,
            }
        )

    accuracy = correct / sample_count if sample_count else 0.0
    top_counts = {}
    for row in rows:
        label = row["predicted_domain"]
        top_counts[label] = top_counts.get(label, 0) + 1
    top_predicted = sorted(top_counts.items(), key=lambda item: item[1], reverse=True)[:5]
    return {
        "seed": seed,
        "sample_count": sample_count,
        "correct": correct,
        "accuracy": accuracy,
        "top_predicted_domains": [[label, count] for label, count in top_predicted],
        "rows": rows,
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    mean_accuracy, sd_accuracy = summarize(results)
    return {
        "mean_accuracy": mean_accuracy,
        "sd_accuracy": sd_accuracy,
        "per_seed_accuracy": {result["seed"]: float(result["accuracy"]) for result in results},
    }


def write_arm_dir(path: Path, results: list[dict[str, Any]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for result in results:
        (path / f"routing_seed_{result['seed']}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seed_list(args.seeds)

    paper_labels = load_paper_labels(args.evolution)
    compendium_text = load_compendium_text(args.compendium)
    eval_rows = build_eval_rows(args.sample_file, args.runlog, args.evolution)

    def run_arm(arm: str, *, runtime_label_selector: str | None = None) -> list[dict[str, Any]]:
        if arm in RUNTIME_ARM_CHOICES:
            selector = runtime_label_selector or args.runtime_label_selector
            return evaluate_runtime_arm(
                arm=arm,
                sample_file=args.sample_file,
                sample_count=args.sample_count,
                seeds=seeds,
                rounds=args.rounds,
                client_count=args.client_count,
                max_items=args.max_items,
                include_training_artifacts=args.runtime_include_training_artifacts,
                training_sample_limit=args.runtime_training_sample_limit,
                included_tools=args.runtime_included_tools,
                runtime_label_selector=selector,
                training_shard_mode=args.runtime_training_shard_mode,
            )
        return evaluate_historical_arm(
            arm=arm,
            eval_rows=eval_rows,
            seeds=seeds,
            sample_count=args.sample_count,
            compendium_text=compendium_text,
            paper_labels=paper_labels,
        )

    synapse_selector = args.runtime_synapse_label_selector or args.runtime_label_selector
    centralized_selector = args.runtime_centralized_label_selector or args.runtime_label_selector

    synapse_results = run_arm(args.synapse_arm, runtime_label_selector=synapse_selector)
    centralized_results = run_arm(args.centralized_arm, runtime_label_selector=centralized_selector)

    synapse_dir = args.output_dir / "synapse"
    centralized_dir = args.output_dir / "centralized"
    write_arm_dir(synapse_dir, synapse_results)
    write_arm_dir(centralized_dir, centralized_results)

    synapse_seed_acc = [float(result["accuracy"]) for result in synapse_results]
    centralized_seed_acc = [float(result["accuracy"]) for result in centralized_results]
    synapse_summary = summarize_results(synapse_results)
    centralized_summary = summarize_results(centralized_results)
    pair = paired_stats(synapse_seed_acc, centralized_seed_acc)

    write_seed_csv(args.output_dir / "synapse_seed_values.csv", seeds, synapse_seed_acc)
    write_seed_csv(args.output_dir / "centralized_seed_values.csv", seeds, centralized_seed_acc)

    summary = {
        "sample_file": str(args.sample_file),
        "runlog": str(args.runlog),
        "compendium": str(args.compendium),
        "evolution": str(args.evolution),
        "sample_count": args.sample_count,
        "seeds": seeds,
        "rounds": args.rounds,
        "client_count": args.client_count,
        "max_items": args.max_items,
        "runtime_include_training_artifacts": args.runtime_include_training_artifacts,
        "runtime_training_sample_limit": args.runtime_training_sample_limit,
        "runtime_included_tools": args.runtime_included_tools,
        "runtime_label_selector": args.runtime_label_selector,
        "runtime_synapse_label_selector": synapse_selector,
        "runtime_centralized_label_selector": centralized_selector,
        "runtime_training_shard_mode": args.runtime_training_shard_mode,
        "synapse_arm": args.synapse_arm,
        "centralized_arm": args.centralized_arm,
        "paper_labels": paper_labels,
        "synapse": synapse_summary,
        "centralized": centralized_summary,
        "paired": pair,
        "artifacts": {
            "synapse_seed_csv": str(args.output_dir / "synapse_seed_values.csv"),
            "centralized_seed_csv": str(args.output_dir / "centralized_seed_values.csv"),
        },
        "note": (
            "Strict paired comparator over shared seeded subsets. Runtime arms use the current repository runtime; "
            "historical arms reuse the preserved six-scenario paper-time reconstruction assets."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "summary.md").write_text(
        render_markdown(
            args=args,
            synapse_summary=synapse_summary,
            centralized_summary=centralized_summary,
            pair=pair,
        ),
        encoding="utf-8",
    )

    print(
        f"{args.synapse_arm}: mean={synapse_summary['mean_accuracy']:.3f}, "
        f"sd={synapse_summary['sd_accuracy']:.3f}, seeds={synapse_summary['per_seed_accuracy']}"
    )
    print(
        f"{args.centralized_arm}: mean={centralized_summary['mean_accuracy']:.3f}, "
        f"sd={centralized_summary['sd_accuracy']:.3f}, seeds={centralized_summary['per_seed_accuracy']}"
    )
    print(
        f"paired: mean_diff={pair['mean_diff']:+.3f}, sd_diff={pair['sd_diff']:.3f}, "
        f"se_diff={pair['se_diff']:.3f}, t={pair['t_value']:+.3f}"
    )


if __name__ == "__main__":
    main()
