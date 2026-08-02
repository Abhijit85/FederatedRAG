#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_table27_fresh_compare import paired_stats  # noqa: E402
from scripts.run_table27_strict_compare import evaluate_runtime_arm, parse_seed_list  # noqa: E402
from scripts.run_table27_tost import paired_tost  # noqa: E402

DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "verification" / "table27_runtime_tost"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fresh current-runtime Table 27-style paired TOST on GSM8K routing. "
            "This targets the near-paper current runtime path rather than the historical provenance reconstruction."
        )
    )
    parser.add_argument("--sample-file", type=Path, default=REPO_ROOT / "GSM8K_500_rebuttal_run" / "GSM8K_500_samples.json")
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--client-count", type=int, default=5)
    parser.add_argument("--max-items", type=int, default=5)
    parser.add_argument("--runtime-included-tools", type=str, default="mathqa")
    parser.add_argument("--runtime-label-selector", type=str, default="historical_cv_svm")
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-mean", type=float, default=0.92)
    parser.add_argument("--target-sd", type=float, default=0.02)
    return parser


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    accuracies = [float(result["accuracy"]) for result in results]
    mean_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    if len(accuracies) > 1:
        mean_value = mean_accuracy
        variance = sum((value - mean_value) ** 2 for value in accuracies) / (len(accuracies) - 1)
        sd_accuracy = variance ** 0.5
    else:
        sd_accuracy = 0.0
    return {
        "mean_accuracy": mean_accuracy,
        "sd_accuracy": sd_accuracy,
        "per_seed_accuracy": {result["seed"]: float(result["accuracy"]) for result in results},
    }


def render_markdown(summary: dict[str, Any]) -> str:
    syn = summary["synapse"]
    cen = summary["centralized"]
    pair = summary["paired"]
    tost = summary["tost"]
    return "\n".join(
        [
            "### Table 27 Runtime TOST",
            "",
            f"- sample_file: `{summary['sample_file']}`",
            f"- sample_count: `{summary['sample_count']}`",
            f"- seeds: `{','.join(str(seed) for seed in summary['seeds'])}`",
            f"- rounds: `{summary['rounds']}`",
            f"- client_count: `{summary['client_count']}`",
            f"- max_items: `{summary['max_items']}`",
            f"- runtime_included_tools: `{summary['runtime_included_tools']}`",
            f"- runtime_label_selector: `{summary['runtime_label_selector']}`",
            "",
            "| Arm | Mean acc. | SD | Seeds |",
            "| --- | ---: | ---: | --- |",
            (
                "| runtime_federated | "
                f"{syn['mean_accuracy']:.3f} | {syn['sd_accuracy']:.3f} | "
                + ", ".join(f"{seed}={value:.3f}" for seed, value in syn["per_seed_accuracy"].items())
                + " |"
            ),
            (
                "| runtime_centralized_direct | "
                f"{cen['mean_accuracy']:.3f} | {cen['sd_accuracy']:.3f} | "
                + ", ".join(f"{seed}={value:.3f}" for seed, value in cen["per_seed_accuracy"].items())
                + " |"
            ),
            "",
            "| Paired quantity | Value |",
            "| --- | ---: |",
            f"| Mean diff (federated - centralized) | {pair['mean_diff']:+.3f} |",
            f"| SD diff | {pair['sd_diff']:.3f} |",
            f"| SE diff | {pair['se_diff']:.3f} |",
            f"| t statistic | {pair['t_value']:+.3f} |",
            "",
            f"TOST margin: `±{summary['margin']:.3f}`",
            f"90% CI: `[{tost['ci_lo']:+.4f}, {tost['ci_hi']:+.4f}]`",
            f"One-sided p-values: `p(> -m)={tost['p_lower']:.4f}`, `p(< +m)={tost['p_upper']:.4f}`",
            f"Verdict: `{'EQUIVALENT' if tost['equivalent'] else 'NOT EQUIVALENT'}`",
            "",
            "This is a fresh current-runtime TOST on the near-paper runtime path.",
            "",
        ]
    )


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seed_list(args.seeds)

    shared_kwargs = dict(
        sample_file=args.sample_file,
        sample_count=args.sample_count,
        seeds=seeds,
        rounds=args.rounds,
        client_count=args.client_count,
        max_items=args.max_items,
        include_training_artifacts=False,
        training_sample_limit=0,
        included_tools=args.runtime_included_tools,
        runtime_label_selector=args.runtime_label_selector,
        training_shard_mode="",
    )

    synapse_results = evaluate_runtime_arm(arm="runtime_federated", **shared_kwargs)
    centralized_results = evaluate_runtime_arm(arm="runtime_centralized_direct", **shared_kwargs)

    synapse_seed_acc = [float(result["accuracy"]) for result in synapse_results]
    centralized_seed_acc = [float(result["accuracy"]) for result in centralized_results]
    synapse_summary = summarize_results(synapse_results)
    centralized_summary = summarize_results(centralized_results)
    pair = paired_stats(synapse_seed_acc, centralized_seed_acc)
    tost = paired_tost(synapse_seed_acc, centralized_seed_acc, args.margin, args.alpha)

    summary = {
        "sample_file": str(args.sample_file),
        "sample_count": args.sample_count,
        "seeds": seeds,
        "rounds": args.rounds,
        "client_count": args.client_count,
        "max_items": args.max_items,
        "runtime_included_tools": args.runtime_included_tools,
        "runtime_label_selector": args.runtime_label_selector,
        "target_mean": args.target_mean,
        "target_sd": args.target_sd,
        "margin": args.margin,
        "alpha": args.alpha,
        "synapse": synapse_summary,
        "centralized": centralized_summary,
        "paired": pair,
        "tost": tost,
        "note": "Fresh current-runtime paired TOST on the near-paper Table 27 path.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")

    print(
        f"runtime_federated: mean={synapse_summary['mean_accuracy']:.3f}, "
        f"sd={synapse_summary['sd_accuracy']:.3f}, seeds={synapse_summary['per_seed_accuracy']}"
    )
    print(
        f"runtime_centralized_direct: mean={centralized_summary['mean_accuracy']:.3f}, "
        f"sd={centralized_summary['sd_accuracy']:.3f}, seeds={centralized_summary['per_seed_accuracy']}"
    )
    print(
        f"paired: mean_diff={pair['mean_diff']:+.3f}, sd_diff={pair['sd_diff']:.3f}, "
        f"se_diff={pair['se_diff']:.3f}, t={pair['t_value']:+.3f}"
    )
    print(
        f"tost: margin=±{args.margin:.3f}, ci=[{tost['ci_lo']:+.4f}, {tost['ci_hi']:+.4f}], "
        f"p_lower={tost['p_lower']:.4f}, p_upper={tost['p_upper']:.4f}, "
        f"equivalent={bool(tost['equivalent'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
