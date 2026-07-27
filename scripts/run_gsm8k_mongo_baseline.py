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

from math_qa import MathQATool
from scripts.run_gsm8k_schema_control import answer_matches, extract_final_answer
from scripts.run_routing_verification import (
    DEFAULT_SAMPLE_FILE,
    build_credentials,
    evaluate_seed,
    load_records,
    query_text,
    sample_records,
    temporary_routing_alignment_profile,
)
from synapse.runtime import SynapseRuntime

DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "verification" / "gsm8k_mongo_baseline"
SCENARIO_MODES = {"none", "gold", "runtime"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run GSM8K answer evaluation through the Mongo-backed MathQATool. "
            "This isolates the paper-time solver path from the local-compendium routing verifier."
        )
    )
    parser.add_argument("--sample-file", type=Path, default=DEFAULT_SAMPLE_FILE)
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--client-count", type=int, default=5)
    parser.add_argument("--max-items", type=int, default=5)
    parser.add_argument(
        "--scenario-mode",
        type=str,
        default="gold",
        help="One of: none, gold, runtime.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def parse_seed_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_runtime(rounds: int, client_count: int) -> SynapseRuntime:
    runtime = SynapseRuntime.build_local_runtime(REPO_ROOT, build_credentials(), client_count=client_count)
    for _ in range(max(1, rounds)):
        runtime.run_round()
    return runtime


def runtime_scenario_label(runtime: SynapseRuntime, query: str, max_items: int) -> str | None:
    artifacts = runtime.get_context_for_query(query, max_items=max_items)
    for artifact in artifacts:
        if artifact.metadata.get("tool") != "mathqa":
            continue
        scenario = artifact.metadata.get("scenario")
        if isinstance(scenario, str) and scenario.strip():
            return scenario.strip()
    return None


def scenario_for_record(
    *,
    runtime: SynapseRuntime | None,
    record: dict[str, Any],
    scenario_mode: str,
    max_items: int,
) -> str | None:
    if scenario_mode == "none":
        return None
    if scenario_mode == "gold":
        router = record.get("router")
        if isinstance(router, dict):
            value = router.get("ground_truth_domain")
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None
    if scenario_mode == "runtime":
        if runtime is None:
            return None
        return runtime_scenario_label(runtime, query_text(record), max_items=max_items)
    raise ValueError(f"Unsupported scenario mode: {scenario_mode}")


def evaluate_answer_seed(
    *,
    runtime: SynapseRuntime | None,
    math_tool: MathQATool,
    records: list[dict[str, Any]],
    seed: int,
    sample_count: int,
    scenario_mode: str,
    max_items: int,
) -> dict[str, Any]:
    subset = sample_records(records, seed=seed, sample_count=sample_count)
    rows: list[dict[str, Any]] = []
    correct = 0

    for record in subset:
        query = query_text(record)
        scenario = scenario_for_record(
            runtime=runtime,
            record=record,
            scenario_mode=scenario_mode,
            max_items=max_items,
        )
        prompt = (
            f"{query}\n"
            "Return only one line in exactly this format: Final Answer: <answer>. Do not include any other text."
        )
        result = math_tool.run(
            user_query=prompt,
            data_item={
                **record,
                "task_type": "math",
                "dataset": "gsm8k",
            },
            recommended_scenario=scenario,
        )
        llm_output = result.llm_response or ""
        prediction = extract_final_answer(llm_output)
        gold = str(record.get("expected_answer") or "")
        hit = answer_matches(gold, prediction)
        correct += int(hit)
        rows.append(
            {
                "query_id": record.get("query_id") or record.get("sample_id"),
                "scenario_mode": scenario_mode,
                "recommended_scenario": scenario,
                "gold_answer": gold,
                "prediction": prediction,
                "correct": hit,
            }
        )

    accuracy = correct / sample_count if sample_count else 0.0
    return {
        "seed": seed,
        "sample_count": sample_count,
        "correct": correct,
        "accuracy": accuracy,
        "rows": rows,
    }


def summarize(results: list[dict[str, Any]]) -> tuple[float, float]:
    accuracies = [float(result["accuracy"]) for result in results]
    mean_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    sd_accuracy = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
    return mean_accuracy, sd_accuracy


def render_markdown(summary: dict[str, Any]) -> str:
    parts = [
        "### GSM8K Mongo-Backed Math Baseline",
        "",
        f"- `scenario_mode`: `{summary['scenario_mode']}`",
        f"- `mongo_collection`: `{summary['mongo_collection']}`",
        f"- `sample_count`: `{summary['sample_count']}` per seed",
        f"- `mean_answer_accuracy`: `{summary['mean_answer_accuracy']:.3f}`",
        f"- `sd_answer_accuracy`: `{summary['sd_answer_accuracy']:.3f}`",
        "",
        "| Seed | Accuracy |",
        "| --- | ---: |",
    ]
    for seed, value in summary["per_seed_answer_accuracy"].items():
        parts.append(f"| {seed} | {value:.3f} |")
    return "\n".join(parts) + "\n"


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    random.seed(0)
    args = parse_args()
    if args.scenario_mode not in SCENARIO_MODES:
        raise ValueError(f"--scenario-mode must be one of: {', '.join(sorted(SCENARIO_MODES))}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.sample_file)
    seeds = parse_seed_list(args.seeds)

    os.environ["SYNAPSE_ENABLE_DP"] = "0"
    os.environ["PROMPT_ATTACK"] = "0"
    os.environ["MATHQA_FORCE_OPTION_ONLY"] = "0"
    os.environ["SYNAPSE_INCLUDED_TOOLS"] = "mathqa"
    os.environ["SYNAPSE_INCLUDE_TRAINING_ARTIFACTS"] = "0"
    os.environ["SYNAPSE_TRAINING_SAMPLE_LIMIT"] = "0"

    runtime = None
    routing_results: list[dict[str, Any]] = []
    if args.scenario_mode == "runtime":
        with temporary_routing_alignment_profile():
            runtime = build_runtime(rounds=args.rounds, client_count=args.client_count)
            routing_results = [
                evaluate_seed(
                    runtime=runtime,
                    records=records,
                    seed=seed,
                    sample_count=args.sample_count,
                    max_items=args.max_items,
                )
                for seed in seeds
            ]

    math_tool = MathQATool()
    if not math_tool.rag_system:
        raise RuntimeError("MathQATool failed to initialize its Mongo-backed RAG system.")

    answer_results = [
        evaluate_answer_seed(
            runtime=runtime,
            math_tool=math_tool,
            records=records,
            seed=seed,
            sample_count=args.sample_count,
            scenario_mode=args.scenario_mode,
            max_items=args.max_items,
        )
        for seed in seeds
    ]

    for result in answer_results:
        out_path = args.output_dir / f"answer_seed_{result['seed']}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for result in routing_results:
        out_path = args.output_dir / f"routing_seed_{result['seed']}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    mean_answer, sd_answer = summarize(answer_results)
    summary: dict[str, Any] = {
        "sample_file": str(args.sample_file),
        "sample_count": args.sample_count,
        "seeds": seeds,
        "scenario_mode": args.scenario_mode,
        "mongo_uri_present": bool(os.environ.get("MONGO_URI")),
        "mongo_collection": os.environ.get("MATHQA_COLLECTION", "math_problems"),
        "mean_answer_accuracy": mean_answer,
        "sd_answer_accuracy": sd_answer,
        "per_seed_answer_accuracy": {str(result["seed"]): result["accuracy"] for result in answer_results},
        "output_dir": str(args.output_dir),
        "note": "Answer accuracy is computed via the Mongo-backed MathQATool path rather than the local-compendium-only verifier.",
    }
    if routing_results:
        mean_routing, sd_routing = summarize(routing_results)
        summary["mean_routing_accuracy"] = mean_routing
        summary["sd_routing_accuracy"] = sd_routing
        summary["per_seed_routing_accuracy"] = {str(result["seed"]): result["accuracy"] for result in routing_results}

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")

    print(
        f"scenario_mode={args.scenario_mode} "
        f"answer={mean_answer:.3f}±{sd_answer:.3f} "
        f"seeds={summary['per_seed_answer_accuracy']}"
    )


if __name__ == "__main__":
    main()
