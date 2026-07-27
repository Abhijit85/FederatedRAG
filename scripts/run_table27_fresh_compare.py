#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_routing_verification import (  # noqa: E402
    DEFAULT_SAMPLE_FILE,
    build_credentials,
    evaluate_seed,
    load_records,
    temporary_routing_alignment_profile,
)
from synapse.knowledge.compendium import KnowledgeArtifact, KnowledgePackage  # noqa: E402
from synapse.runtime import SynapseRuntime  # noqa: E402

DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "verification" / "table27_fresh_compare"


@dataclass
class ArmResult:
    name: str
    runtime: SynapseRuntime
    summaries: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fresh paired federated-vs-centralized GSM8K routing comparison on the current codebase. "
            "This is a current-runtime comparator, not a claim that the submitted paper path has been recovered."
        )
    )
    parser.add_argument("--sample-file", type=Path, default=DEFAULT_SAMPLE_FILE)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--client-count", type=int, default=5)
    parser.add_argument("--max-items", type=int, default=5)
    parser.add_argument(
        "--centralized-mode",
        choices=("direct", "direct_sourceaware", "direct_examples_sourceaware", "direct_sourcecap2", "clustered"),
        default="direct",
        help=(
            "`direct` pools client-selected artifacts into one central compendium without edge cosine clustering; "
            "`direct_sourceaware` preserves per-client raw artifacts by source-salting every signature before central ingest; "
            "`direct_examples_sourceaware` salts only training-example signatures while keeping shared scenario artifacts deduped; "
            "`direct_sourcecap2` keeps at most two centralized copies per base signature, preserving some pooled multiplicity without full duplication; "
            "`clustered` reuses the current edge merge path and is expected to match the federated arm more closely."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def parse_seed_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _annotate_package(package: KnowledgePackage) -> KnowledgePackage:
    annotated: list[KnowledgeArtifact] = []
    for artifact in package.artifacts:
        metadata = dict(artifact.metadata or {})
        metadata.setdefault("source_id", package.source_id)
        annotated.append(
            KnowledgeArtifact(
                signature=artifact.signature,
                text=artifact.text,
                structured_payload=artifact.structured_payload,
                metadata=metadata,
                textgrad_variable=artifact.textgrad_variable,
            )
        )
    return KnowledgePackage(
        source_id=package.source_id,
        artifacts=annotated,
        created_at=package.created_at,
        metadata=dict(package.metadata or {}),
    )


def _annotate_package_sourceaware(package: KnowledgePackage) -> KnowledgePackage:
    annotated: list[KnowledgeArtifact] = []
    for artifact in package.artifacts:
        metadata = dict(artifact.metadata or {})
        metadata.setdefault("source_id", package.source_id)
        source_id = str(metadata.get("source_id") or package.source_id)
        annotated.append(
            KnowledgeArtifact(
                signature=f"{source_id}::{artifact.signature}",
                text=artifact.text,
                structured_payload=artifact.structured_payload,
                metadata=metadata,
                textgrad_variable=artifact.textgrad_variable,
            )
        )
    return KnowledgePackage(
        source_id=package.source_id,
        artifacts=annotated,
        created_at=package.created_at,
        metadata=dict(package.metadata or {}),
    )


def _annotate_package_examples_sourceaware(package: KnowledgePackage) -> KnowledgePackage:
    annotated: list[KnowledgeArtifact] = []
    for artifact in package.artifacts:
        metadata = dict(artifact.metadata or {})
        metadata.setdefault("source_id", package.source_id)
        payload = artifact.structured_payload or {}
        payload_type = payload.get("type") if isinstance(payload, dict) else None
        source_id = str(metadata.get("source_id") or package.source_id)
        signature = artifact.signature
        if payload_type == "training_example":
            signature = f"{source_id}::{signature}"
        annotated.append(
            KnowledgeArtifact(
                signature=signature,
                text=artifact.text,
                structured_payload=artifact.structured_payload,
                metadata=metadata,
                textgrad_variable=artifact.textgrad_variable,
            )
        )
    return KnowledgePackage(
        source_id=package.source_id,
        artifacts=annotated,
        created_at=package.created_at,
        metadata=dict(package.metadata or {}),
    )


def _annotate_package_sourcecap(package: KnowledgePackage, *, copy_index_by_signature: dict[str, int], max_copies: int) -> KnowledgePackage:
    annotated: list[KnowledgeArtifact] = []
    for artifact in package.artifacts:
        metadata = dict(artifact.metadata or {})
        metadata.setdefault("source_id", package.source_id)
        base_signature = artifact.signature
        seen = copy_index_by_signature.get(base_signature, 0)
        if seen >= max_copies:
            continue
        copy_index_by_signature[base_signature] = seen + 1
        if seen == 0:
            signature = base_signature
        else:
            source_id = str(metadata.get("source_id") or package.source_id)
            signature = f"{source_id}::copy{seen + 1}::{base_signature}"
        annotated.append(
            KnowledgeArtifact(
                signature=signature,
                text=artifact.text,
                structured_payload=artifact.structured_payload,
                metadata=metadata,
                textgrad_variable=artifact.textgrad_variable,
            )
        )
    return KnowledgePackage(
        source_id=package.source_id,
        artifacts=annotated,
        created_at=package.created_at,
        metadata=dict(package.metadata or {}),
    )


def build_federated_runtime(rounds: int, client_count: int) -> SynapseRuntime:
    runtime = SynapseRuntime.build_local_runtime(REPO_ROOT, build_credentials(), client_count=client_count)
    for _ in range(max(1, rounds)):
        runtime.run_round()
    return runtime


def _centralized_direct_round(runtime: SynapseRuntime, *, sourceaware: bool = False, examples_sourceaware: bool = False, source_cap: int = 0) -> None:
    pooled_artifacts: list[KnowledgeArtifact] = []
    pooled_sources: list[str] = []
    copy_index_by_signature: dict[str, int] = {}
    if sourceaware:
        annotate_fn = _annotate_package_sourceaware
    elif examples_sourceaware:
        annotate_fn = _annotate_package_examples_sourceaware
    elif source_cap > 0:
        annotate_fn = None
    else:
        annotate_fn = _annotate_package
    for client_id in runtime.config.topology.client_ids:
        package = runtime.clients[client_id].prepare_for_edge()
        if not package.artifacts:
            continue
        annotated = _annotate_package_sourcecap(package, copy_index_by_signature=copy_index_by_signature, max_copies=source_cap) if source_cap > 0 else annotate_fn(package)
        pooled_sources.append(client_id)
        pooled_artifacts.extend(annotated.artifacts)

    if not pooled_artifacts:
        return

    pooled_package = KnowledgePackage(
        source_id=(
            "centralized-direct-sourceaware"
            if sourceaware
            else "centralized-direct-examples-sourceaware"
            if examples_sourceaware
            else "centralized-direct-sourcecap"
            if source_cap > 0
            else "centralized-direct"
        ),
        artifacts=pooled_artifacts,
        created_at=datetime.utcnow(),
        metadata={
            "sources": pooled_sources,
            "mode": (
                "direct_sourceaware"
                if sourceaware
                else "direct_examples_sourceaware"
                if examples_sourceaware
                else f"direct_sourcecap{source_cap}"
                if source_cap > 0
                else "direct"
            ),
        },
    )
    runtime.server.ingest_from_edge(pooled_package)


def _centralized_clustered_round(runtime: SynapseRuntime) -> None:
    for edge_id, client_ids in runtime.config.topology.edge_clusters.items():
        edge = runtime.edges[edge_id]
        packages: list[KnowledgePackage] = []
        for client_id in client_ids:
            package = runtime.clients[client_id].prepare_for_edge()
            if package.artifacts:
                packages.append(package)
        if not packages:
            continue
        merged = edge.merge_packages(packages)
        if merged:
            runtime.server.ingest_from_edge(merged)


def build_centralized_runtime(rounds: int, client_count: int, mode: str) -> SynapseRuntime:
    runtime = SynapseRuntime.build_local_runtime(REPO_ROOT, build_credentials(), client_count=client_count)
    if mode == "direct":
        round_fn = lambda rt: _centralized_direct_round(rt, sourceaware=False)
    elif mode == "direct_examples_sourceaware":
        round_fn = lambda rt: _centralized_direct_round(rt, sourceaware=False, examples_sourceaware=True)
    elif mode == "direct_sourceaware":
        round_fn = lambda rt: _centralized_direct_round(rt, sourceaware=True)
    elif mode == "direct_sourcecap2":
        round_fn = lambda rt: _centralized_direct_round(rt, source_cap=2)
    else:
        round_fn = _centralized_clustered_round
    for _ in range(max(1, rounds)):
        round_fn(runtime)
    return runtime


def summarize(results: list[dict[str, Any]]) -> tuple[float, float]:
    accuracies = [float(result["accuracy"]) for result in results]
    mean_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    sd_accuracy = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
    return mean_accuracy, sd_accuracy


def paired_stats(left: list[float], right: list[float]) -> dict[str, float]:
    diffs = [a - b for a, b in zip(left, right)]
    n = len(diffs)
    mean_diff = sum(diffs) / n
    sd_diff = statistics.stdev(diffs) if n > 1 else 0.0
    se_diff = sd_diff / math.sqrt(n) if n > 1 and sd_diff > 0 else 0.0
    if se_diff == 0.0:
        t_value = 0.0 if abs(mean_diff) < 1e-12 else math.copysign(math.inf, mean_diff)
    else:
        t_value = mean_diff / se_diff
    return {
        "mean_diff": mean_diff,
        "sd_diff": sd_diff,
        "se_diff": se_diff,
        "t_value": t_value,
    }


def write_seed_csv(path: Path, seeds: list[int], values: list[float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "acc"])
        for seed, value in zip(seeds, values):
            writer.writerow([seed, f"{value:.6f}"])


def render_markdown(
    *,
    sample_file: Path,
    sample_count: int,
    rounds: int,
    client_count: int,
    max_items: int,
    centralized_mode: str,
    federated_summary: dict[str, Any],
    centralized_summary: dict[str, Any],
    pair: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "### Fresh Table 27 Comparison",
            "",
            f"- sample_file: `{sample_file}`",
            f"- sample_count: `{sample_count}`",
            f"- rounds: `{rounds}`",
            f"- client_count: `{client_count}`",
            f"- max_items: `{max_items}`",
            f"- centralized_mode: `{centralized_mode}`",
            "",
            "| Arm | Mean acc. | SD | Seeds |",
            "| --- | ---: | ---: | --- |",
            (
                f"| federated | {federated_summary['mean_accuracy']:.3f} | {federated_summary['sd_accuracy']:.3f} | "
                + ", ".join(
                    f"{seed}={value:.3f}" for seed, value in federated_summary["per_seed_accuracy"].items()
                )
                + " |"
            ),
            (
                f"| centralized | {centralized_summary['mean_accuracy']:.3f} | {centralized_summary['sd_accuracy']:.3f} | "
                + ", ".join(
                    f"{seed}={value:.3f}" for seed, value in centralized_summary["per_seed_accuracy"].items()
                )
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
            "This is a current-codebase paired rerun. It does not by itself establish provenance for the submitted paper's Table 27 path.",
            "",
        ]
    )


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_seed_list(args.seeds)
    records = load_records(args.sample_file)

    with temporary_routing_alignment_profile():
        federated_runtime = build_federated_runtime(args.rounds, args.client_count)
        centralized_runtime = build_centralized_runtime(args.rounds, args.client_count, args.centralized_mode)

        federated_results = [
            evaluate_seed(
                runtime=federated_runtime,
                records=records,
                seed=seed,
                sample_count=args.sample_count,
                max_items=args.max_items,
            )
            for seed in seeds
        ]
        centralized_results = [
            evaluate_seed(
                runtime=centralized_runtime,
                records=records,
                seed=seed,
                sample_count=args.sample_count,
                max_items=args.max_items,
            )
            for seed in seeds
        ]

    for arm_name, arm_results in (("federated", federated_results), ("centralized", centralized_results)):
        arm_dir = args.output_dir / arm_name
        arm_dir.mkdir(parents=True, exist_ok=True)
        for result in arm_results:
            (arm_dir / f"routing_seed_{result['seed']}.json").write_text(
                json.dumps(result, indent=2),
                encoding="utf-8",
            )

    federated_mean, federated_sd = summarize(federated_results)
    centralized_mean, centralized_sd = summarize(centralized_results)

    federated_seed_acc = [float(result["accuracy"]) for result in federated_results]
    centralized_seed_acc = [float(result["accuracy"]) for result in centralized_results]
    pair = paired_stats(federated_seed_acc, centralized_seed_acc)

    federated_summary = {
        "mean_accuracy": federated_mean,
        "sd_accuracy": federated_sd,
        "per_seed_accuracy": {seed: value for seed, value in zip(seeds, federated_seed_acc)},
    }
    centralized_summary = {
        "mean_accuracy": centralized_mean,
        "sd_accuracy": centralized_sd,
        "per_seed_accuracy": {seed: value for seed, value in zip(seeds, centralized_seed_acc)},
    }

    write_seed_csv(args.output_dir / "federated_seed_values.csv", seeds, federated_seed_acc)
    write_seed_csv(args.output_dir / "centralized_seed_values.csv", seeds, centralized_seed_acc)

    summary = {
        "sample_file": str(args.sample_file),
        "sample_count": args.sample_count,
        "rounds": args.rounds,
        "client_count": args.client_count,
        "max_items": args.max_items,
        "centralized_mode": args.centralized_mode,
        "seeds": seeds,
        "federated": federated_summary,
        "centralized": centralized_summary,
        "paired": pair,
        "artifacts": {
            "federated_seed_csv": str(args.output_dir / "federated_seed_values.csv"),
            "centralized_seed_csv": str(args.output_dir / "centralized_seed_values.csv"),
        },
        "note": (
            "Fresh current-codebase paired comparison. Federated uses the existing runtime edge merge path; "
            "centralized=direct pools client-selected artifacts into one central compendium without edge cosine clustering. "
            "This is a current-runtime comparator, not a reconstruction of the submitted paper's historical Table 27 harness."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "summary.md").write_text(
        render_markdown(
            sample_file=args.sample_file,
            sample_count=args.sample_count,
            rounds=args.rounds,
            client_count=args.client_count,
            max_items=args.max_items,
            centralized_mode=args.centralized_mode,
            federated_summary=federated_summary,
            centralized_summary=centralized_summary,
            pair=pair,
        ),
        encoding="utf-8",
    )

    print(f"federated: mean={federated_mean:.3f}, sd={federated_sd:.3f}, seeds={federated_summary['per_seed_accuracy']}")
    print(
        f"centralized: mean={centralized_mean:.3f}, sd={centralized_sd:.3f}, seeds={centralized_summary['per_seed_accuracy']}"
    )
    print(
        f"paired: mean_diff={pair['mean_diff']:+.3f}, sd_diff={pair['sd_diff']:.3f}, "
        f"se_diff={pair['se_diff']:.3f}, t={pair['t_value']:+.3f}"
    )


if __name__ == "__main__":
    main()
