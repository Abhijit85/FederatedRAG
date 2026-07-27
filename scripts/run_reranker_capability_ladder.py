#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_routing_verification import (
    DEFAULT_SAMPLE_FILE,
    build_credentials,
    evaluate_seed,
    load_records,
    temporary_routing_alignment_profile,
)
from synapse.runtime import SynapseRuntime

DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "verification" / "reranker_capability_ladder"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the GSM8K routing reranker capability ladder over shared seeds."
    )
    parser.add_argument("--sample-file", type=Path, default=DEFAULT_SAMPLE_FILE)
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--client-count", type=int, default=5)
    parser.add_argument("--max-items", type=int, default=5)
    parser.add_argument("--reranker-device", type=str, default=None)
    parser.add_argument("--arm", action="append", default=[], help="Repeatable 'label=model_id'. Use NONE for no local reranker.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def parse_seed_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_arms(values: list[str]) -> list[tuple[str, str | None]]:
    if not values:
        return [
            ("baseline", None),
        ]
    arms: list[tuple[str, str | None]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected --arm label=model_id, got: {value}")
        label, model = value.split("=", 1)
        label = label.strip()
        model = model.strip()
        if not label:
            raise ValueError(f"Arm label cannot be empty: {value}")
        arms.append((label, None if model.upper() == "NONE" else model))
    return arms


@contextmanager
def temporary_reranker(model_name: str | None, device: str | None) -> Iterator[None]:
    keys = [
        "SYNAPSE_LOCAL_RERANK_MODEL",
        "SYNAPSE_RERANK_MODEL",
        "SYNAPSE_LOCAL_RERANK_DEVICE",
        "SYNAPSE_RERANK_DEVICE",
    ]
    previous = {key: os.environ.get(key) for key in keys}
    try:
        if model_name:
            os.environ["SYNAPSE_LOCAL_RERANK_MODEL"] = model_name
            os.environ["SYNAPSE_RERANK_MODEL"] = model_name
        else:
            os.environ.pop("SYNAPSE_LOCAL_RERANK_MODEL", None)
            os.environ.pop("SYNAPSE_RERANK_MODEL", None)
        if device:
            os.environ["SYNAPSE_LOCAL_RERANK_DEVICE"] = device
            os.environ["SYNAPSE_RERANK_DEVICE"] = device
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def build_runtime(rounds: int, client_count: int) -> SynapseRuntime:
    runtime = SynapseRuntime.build_local_runtime(REPO_ROOT, build_credentials(), client_count=client_count)
    for _ in range(max(1, rounds)):
        runtime.run_round()
    return runtime


def summarize_results(results: list[dict[str, object]]) -> tuple[float, float]:
    accuracies = [float(result["accuracy"]) for result in results]
    mean_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    sd_accuracy = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
    return mean_accuracy, sd_accuracy


def run_arm(
    *,
    label: str,
    model_name: str | None,
    device: str | None,
    records: list[dict[str, object]],
    seeds: list[int],
    sample_count: int,
    rounds: int,
    client_count: int,
    max_items: int,
    output_dir: Path,
) -> dict[str, object]:
    arm_dir = output_dir / label
    arm_dir.mkdir(parents=True, exist_ok=True)

    with temporary_reranker(model_name, device), temporary_routing_alignment_profile():
        runtime = build_runtime(rounds=rounds, client_count=client_count)
        results = [
            evaluate_seed(
                runtime=runtime,
                records=records,
                seed=seed,
                sample_count=sample_count,
                max_items=max_items,
            )
            for seed in seeds
        ]

    for result in results:
        out_path = arm_dir / f"routing_seed_{result['seed']}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    mean_accuracy, sd_accuracy = summarize_results(results)
    summary = {
        "label": label,
        "reranker_model": model_name,
        "sample_count": sample_count,
        "rounds": rounds,
        "client_count": client_count,
        "seeds": seeds,
        "mean_accuracy": mean_accuracy,
        "sd_accuracy": sd_accuracy,
        "per_seed_accuracy": {str(result["seed"]): result["accuracy"] for result in results},
        "output_dir": str(arm_dir),
    }
    (arm_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def render_markdown(rows: list[dict[str, object]]) -> str:
    parts = [
        "### Reranker Capability Ladder",
        "",
        "| Reranker | Seeds | Mean routing acc. | SD |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in rows:
        seed_values = ", ".join(f"{int(seed)}={value:.3f}" for seed, value in row["per_seed_accuracy"].items())
        parts.append(
            f"| {row['label']} | {seed_values} | {row['mean_accuracy']:.3f} | {row['sd_accuracy']:.3f} |"
        )
    return "\n".join(parts) + "\n"


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.sample_file)
    seeds = parse_seed_list(args.seeds)
    arms = parse_arms(args.arm)

    rows = [
        run_arm(
            label=label,
            model_name=model_name,
            device=args.reranker_device,
            records=records,
            seeds=seeds,
            sample_count=args.sample_count,
            rounds=args.rounds,
            client_count=args.client_count,
            max_items=args.max_items,
            output_dir=args.output_dir,
        )
        for label, model_name in arms
    ]

    combined = {
        "sample_file": str(args.sample_file),
        "sample_count": args.sample_count,
        "rounds": args.rounds,
        "client_count": args.client_count,
        "seeds": seeds,
        "arms": rows,
    }
    (args.output_dir / "combined_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    (args.output_dir / "summary.md").write_text(render_markdown(rows), encoding="utf-8")

    for row in rows:
        print(
            f"{row['label']}: mean={row['mean_accuracy']:.3f}, sd={row['sd_accuracy']:.3f}, "
            f"seeds={row['per_seed_accuracy']}, model={row['reranker_model']}"
        )


if __name__ == "__main__":
    main()
