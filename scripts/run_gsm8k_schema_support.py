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

from scripts.run_gsm8k_schema_control import (
    DEFAULT_OUTPUT_DIR as DEFAULT_SCHEMA_OUTPUT_DIR,
    SUPPORTED_CONDITIONS,
    evaluate_answer_seed,
    parse_seed_list,
    summarize,
    temporary_structured_payload_mode,
)
from scripts.run_routing_verification import (
    DEFAULT_SAMPLE_FILE,
    build_credentials,
    evaluate_seed,
    load_records,
    temporary_routing_alignment_profile,
)
from synapse.runtime import SynapseRuntime

DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "verification" / "gsm8k_schema_support"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a support-grade GSM8K schema ablation table on the current branch. "
            "This uses the same reconstructed routing profile as the rebuttal verification scripts, "
            "but reports support evidence from the present code rather than claiming historical-paper reproduction."
        )
    )
    parser.add_argument("--sample-file", type=Path, default=DEFAULT_SAMPLE_FILE)
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--client-count", type=int, default=5)
    parser.add_argument("--max-items", type=int, default=5)
    parser.add_argument("--conditions", type=str, default="full,merge_up,drop_annex")
    parser.add_argument(
        "--routing-only",
        action="store_true",
        help="Skip final-answer evaluation and report routing metrics only.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def build_runtime(rounds: int, client_count: int) -> SynapseRuntime:
    runtime = SynapseRuntime.build_local_runtime(REPO_ROOT, build_credentials(), client_count=client_count)
    for _ in range(max(1, rounds)):
        runtime.run_round()
    return runtime


def parse_conditions(value: str) -> list[str]:
    conditions = [part.strip() for part in value.split(",") if part.strip()]
    invalid = [condition for condition in conditions if condition not in SUPPORTED_CONDITIONS]
    if invalid:
        raise ValueError(
            f"Unknown condition(s): {', '.join(sorted(invalid))}. "
            f"Supported: {', '.join(sorted(SUPPORTED_CONDITIONS))}."
        )
    return conditions


def mean_sd(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values) if values else 0.0
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, sd


def run_condition(
    *,
    condition: str,
    records: list[dict[str, Any]],
    seeds: list[int],
    sample_count: int,
    rounds: int,
    client_count: int,
    max_items: int,
    routing_only: bool,
    output_dir: Path,
) -> dict[str, Any]:
    print(f"[c3] start condition={condition} seeds={seeds} sample_count={sample_count}", flush=True)
    mode = SUPPORTED_CONDITIONS[condition]
    condition_dir = output_dir / condition
    condition_dir.mkdir(parents=True, exist_ok=True)

    with temporary_structured_payload_mode(mode), temporary_routing_alignment_profile():
        runtime = build_runtime(rounds=rounds, client_count=client_count)
        routing_results = []
        for seed in seeds:
            print(f"[c3] condition={condition} routing seed={seed} start", flush=True)
            result = evaluate_seed(
                runtime=runtime,
                records=records,
                seed=seed,
                sample_count=sample_count,
                max_items=max_items,
            )
            routing_results.append(result)
            print(f"[c3] condition={condition} routing seed={seed} done acc={result['accuracy']:.3f}", flush=True)
        answer_results = []
        if not routing_only:
            for seed in seeds:
                print(f"[c3] condition={condition} answer seed={seed} start", flush=True)
                result = evaluate_answer_seed(
                    runtime=runtime,
                    records=records,
                    seed=seed,
                    sample_count=sample_count,
                )
                answer_results.append(result)
                print(f"[c3] condition={condition} answer seed={seed} done acc={result['accuracy']:.3f}", flush=True)

    for result in routing_results:
        (condition_dir / f"routing_seed_{result['seed']}.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
    for result in answer_results:
        (condition_dir / f"answer_seed_{result['seed']}.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )

    mean_routing, sd_routing = summarize(routing_results)
    summary: dict[str, Any] = {
        "condition": condition,
        "structured_payload_mode": mode,
        "sample_count": sample_count,
        "rounds": rounds,
        "client_count": client_count,
        "seeds": seeds,
        "mean_routing_accuracy": mean_routing,
        "sd_routing_accuracy": sd_routing,
        "per_seed_routing_accuracy": {str(result["seed"]): result["accuracy"] for result in routing_results},
        "routing_only": routing_only,
        "output_dir": str(condition_dir),
        "note": (
            "Support-only schema ablation on the current runtime; not a claim that these values reproduce "
            "the historical paper table."
        ),
    }
    if answer_results:
        mean_answer, sd_answer = summarize(answer_results)
        summary["mean_answer_accuracy"] = mean_answer
        summary["sd_answer_accuracy"] = sd_answer
        summary["per_seed_answer_accuracy"] = {str(result["seed"]): result["accuracy"] for result in answer_results}

    (condition_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[c3] done condition={condition} routing={mean_routing:.3f} answer={summary.get('mean_answer_accuracy', float('nan')):.3f}", flush=True)
    return summary


def render_markdown(rows: list[dict[str, Any]]) -> str:
    has_answers = any("mean_answer_accuracy" in row for row in rows)
    parts = [
        "### GSM8K Schema Support",
        "",
    ]
    if has_answers:
        parts.extend(
            [
                "| Condition | Routing acc. | Routing SD | Answer acc. | Answer SD | Δ answer vs full |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
    else:
        parts.extend(
            [
                "| Condition | Routing acc. | Routing SD | Δ routing vs full |",
                "| --- | ---: | ---: | ---: |",
            ]
        )

    full_answer = next((row.get("mean_answer_accuracy") for row in rows if row["condition"] == "full"), None)
    full_routing = next((row["mean_routing_accuracy"] for row in rows if row["condition"] == "full"), None)
    for row in rows:
        if has_answers:
            answer = row.get("mean_answer_accuracy")
            delta_answer = (answer - full_answer) if isinstance(answer, float) and isinstance(full_answer, float) else None
            parts.append(
                f"| {row['condition']} | {row['mean_routing_accuracy']:.3f} | {row['sd_routing_accuracy']:.3f} | "
                f"{(answer if isinstance(answer, float) else 0.0):.3f} | "
                f"{(row.get('sd_answer_accuracy', 0.0)):.3f} | "
                f"{(delta_answer if delta_answer is not None else 0.0):+.3f} |"
            )
        else:
            delta_routing = row["mean_routing_accuracy"] - (full_routing or 0.0)
            parts.append(
                f"| {row['condition']} | {row['mean_routing_accuracy']:.3f} | {row['sd_routing_accuracy']:.3f} | "
                f"{delta_routing:+.3f} |"
            )
    parts.extend(
        [
            "",
            "Modes on this branch:",
            "- `full`: typed payload with separate scenario, precaution, and annex channels",
            "- `merge_up`: merges scenario context and precautions into a single notes channel",
            "- `drop_annex`: removes annex fields while keeping scenario and precaution channels",
            "- `untyped`: removes only the top-level type label",
            "- `no_payload`: removes the structured payload entirely",
            "",
            "This report is support evidence from the current branch. It is not a claim that the numbers match the paper's historical table.",
        ]
    )
    return "\n".join(parts) + "\n"


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.sample_file)
    seeds = parse_seed_list(args.seeds)
    conditions = parse_conditions(args.conditions)

    rows = [
        run_condition(
            condition=condition,
            records=records,
            seeds=seeds,
            sample_count=args.sample_count,
            rounds=args.rounds,
            client_count=args.client_count,
            max_items=args.max_items,
            routing_only=args.routing_only,
            output_dir=args.output_dir,
        )
        for condition in conditions
    ]

    combined = {
        "sample_file": str(args.sample_file),
        "sample_count": args.sample_count,
        "rounds": args.rounds,
        "client_count": args.client_count,
        "max_items": args.max_items,
        "seeds": seeds,
        "conditions": rows,
        "routing_only": args.routing_only,
        "base_runner": str(DEFAULT_SCHEMA_OUTPUT_DIR),
    }
    (args.output_dir / "combined_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    (args.output_dir / "summary.md").write_text(render_markdown(rows), encoding="utf-8")

    for row in rows:
        line = (
            f"{row['condition']}: routing={row['mean_routing_accuracy']:.3f}±{row['sd_routing_accuracy']:.3f}"
        )
        if "mean_answer_accuracy" in row:
            line += f", answer={row['mean_answer_accuracy']:.3f}±{row['sd_answer_accuracy']:.3f}"
        print(line)


if __name__ == "__main__":
    main()
