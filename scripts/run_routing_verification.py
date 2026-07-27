#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
import random
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jina_key_manager import get_available_jina_api_keys
from openrouter_client import get_available_api_keys
from scripts.run_gsm8k_paper_recovery_sweep import historical_rows, load_paper_labels, load_runlog
from synapse.config import ApiCredentials
from synapse.knowledge.compendium import KnowledgeArtifact
from synapse.runtime import SynapseRuntime
from synapse.retrieval import RetrievalConfig, RetrievalPlanner

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


DEFAULT_SAMPLE_FILE = REPO_ROOT / "GSM8K_500_rebuttal_run" / "GSM8K_500_samples.json"
DEFAULT_RUNLOG = REPO_ROOT / "GSM8K_500_rebuttal_run" / "GSM8K_500_runlog.jsonl"
DEFAULT_EVOLUTION = REPO_ROOT / "GSM8K_500_rebuttal_run" / "GSM8K_compendium_evolution.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "verification" / "routing"

ALIASES = {
    "geometry and measurement": "geometry shapes and measurement",
    "geometry shapes and measurement": "geometry shapes and measurement",
}


@contextmanager
def temporary_routing_alignment_profile(
    *,
    include_training_artifacts: bool = False,
    training_sample_limit: int = 0,
    included_tools: str = "mathqa",
    retrieval_profile: str = "paperlike",
    structured_text_style: str = "paper",
    runtime_label_selector: str = "historical_cv_svm",
    training_shard_mode: str = "",
) -> Iterable[None]:
    overrides = {
        "SYNAPSE_INCLUDE_TRAINING_ARTIFACTS": "1" if include_training_artifacts else "0",
        "SYNAPSE_TRAINING_SAMPLE_LIMIT": str(max(0, int(training_sample_limit))),
        "SYNAPSE_RETRIEVAL_PROFILE": retrieval_profile,
        "SYNAPSE_STRUCTURED_TEXT_STYLE": structured_text_style,
        # GSM8K routing is math-only; excluding science artifacts avoids cross-domain drift.
        "SYNAPSE_INCLUDED_TOOLS": included_tools,
        # Reuse the preserved paper-time label classifier to choose among retrieved candidates.
        "SYNAPSE_RUNTIME_LABEL_SELECTOR": runtime_label_selector,
        "SYNAPSE_TRAINING_SHARD_MODE": training_shard_mode,
    }
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run routing-only verification on GSM8K-style routing samples."
    )
    parser.add_argument(
        "--sample-file",
        type=Path,
        default=DEFAULT_SAMPLE_FILE,
        help="Path to a routing sample file with a top-level 'records' list.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=50,
        help="Number of records to evaluate per seed.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated RNG seeds for subset selection.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of SYNAPSE federation rounds to run before evaluation.",
    )
    parser.add_argument(
        "--client-count",
        type=int,
        default=None,
        help="Override SYNAPSE client count.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=5,
        help="How many retrieved artifacts to inspect per query.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for per-seed JSON outputs.",
    )
    return parser.parse_args()


def normalize_label(value: str | None) -> str:
    text = (value or "").strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return ALIASES.get(text, text)


def labels_match(left: str | None, right: str | None) -> bool:
    return normalize_label(left) == normalize_label(right)


def artifact_route_label(artifact: KnowledgeArtifact) -> str:
    metadata = artifact.metadata or {}
    payload = artifact.structured_payload or {}
    for key in ("scenario", "domain"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if isinstance(payload, dict):
        for key in ("scenario", "domain"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return artifact.signature


def load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return [record for record in payload["records"] if isinstance(record, dict)]
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]
    raise ValueError(f"Unsupported sample file format: {path}")


def sample_records(records: list[dict[str, Any]], seed: int, sample_count: int) -> list[dict[str, Any]]:
    if sample_count > len(records):
        raise ValueError(f"Requested {sample_count} samples, but only {len(records)} records are available.")
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(records)), sample_count))
    return [records[idx] for idx in indices]


def build_credentials() -> ApiCredentials:
    lambda_keys = get_available_api_keys(allow_empty=True)
    jina_keys = get_available_jina_api_keys(allow_empty=True)
    lambda_key = lambda_keys[0] if lambda_keys else (os.environ.get("API_KEY") or "")
    jina_key = jina_keys[0] if jina_keys else (os.environ.get("JINA_API_KEY") or "")
    mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
    if lambda_key:
        os.environ["API_KEY"] = lambda_key
    if jina_key:
        os.environ["JINA_API_KEY"] = jina_key
    if mongo_uri:
        os.environ["MONGO_URI"] = mongo_uri
    return ApiCredentials(
        lambda_api_key=lambda_key,
        jina_api_key=jina_key,
        mongo_uri=mongo_uri,
        lambda_api_base="https://openrouter.ai/api/v1/chat/completions",
    )


def build_runtime(rounds: int, client_count: int | None) -> SynapseRuntime:
    runtime = SynapseRuntime.build_local_runtime(REPO_ROOT, build_credentials(), client_count=client_count)
    for _ in range(max(1, rounds)):
        runtime.run_round()
    return runtime


def gold_route_label(record: dict[str, Any]) -> str:
    router = record.get("router")
    if isinstance(router, dict):
        value = router.get("ground_truth_domain")
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("ground_truth_domain", "domain", "scenario"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def query_text(record: dict[str, Any]) -> str:
    for key in ("query_text", "question", "Problem"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def summarize_top_predictions(predictions: Iterable[str], limit: int = 5) -> list[list[Any]]:
    counts = Counter(predictions)
    return [[label, count] for label, count in counts.most_common(limit)]


def first_route_bearing_label(artifacts: list[KnowledgeArtifact]) -> str:
    for artifact in artifacts:
        metadata = artifact.metadata or {}
        payload = artifact.structured_payload or {}
        for key in ("scenario", "domain"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        if isinstance(payload, dict):
            for key in ("scenario", "domain"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return artifact_route_label(artifacts[0]) if artifacts else ""


def selector_mode() -> str:
    return os.environ.get("SYNAPSE_RUNTIME_LABEL_SELECTOR", "").strip().lower()


def _selector_uses_diverse_coverage(selector: str) -> bool:
    return selector in {"historical_cv_svm_diverse", "historical_cv_svm_diverse_consensus", "historical_cv_svm_diverse_provenance"}


def _selector_uses_consensus_tiebreak(selector: str) -> bool:
    return selector == "historical_cv_svm_diverse_consensus"


def _selector_uses_provenance_tiebreak(selector: str) -> bool:
    return selector == "historical_cv_svm_diverse_provenance"


def selector_expanded_max_items(default_max_items: int) -> int:
    selector = selector_mode()
    if not _selector_uses_diverse_coverage(selector):
        return default_max_items
    raw = os.environ.get("SYNAPSE_SELECTOR_EXPANDED_MAX_ITEMS", "12")
    try:
        expanded = int(raw)
    except ValueError:
        expanded = 12
    return max(default_max_items, expanded)


def _selector_candidate_labels(query: str, artifacts: list[KnowledgeArtifact]) -> list[str]:
    selector = selector_mode()
    if not _selector_uses_diverse_coverage(selector):
        candidate_labels: list[str] = []
        seen: set[str] = set()
        for artifact in artifacts:
            label = artifact_route_label(artifact)
            norm = normalize_label(label)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            candidate_labels.append(label)
        return candidate_labels

    planner = RetrievalPlanner(RetrievalConfig(max_artifacts=len(artifacts) or 1))
    scored = sorted(
        ((artifact, planner._score_artifact(query, artifact)) for artifact in artifacts),
        key=lambda item: item[1],
        reverse=True,
    )
    best_by_label: dict[str, tuple[str, float]] = {}
    for artifact, score in scored:
        label = artifact_route_label(artifact)
        norm = normalize_label(label)
        if not norm or norm in best_by_label:
            continue
        best_by_label[norm] = (label, score)

    raw_limit = os.environ.get("SYNAPSE_SELECTOR_UNIQUE_LABEL_LIMIT", "5")
    try:
        unique_limit = max(1, int(raw_limit))
    except ValueError:
        unique_limit = 5
    ranked = sorted(best_by_label.values(), key=lambda item: item[1], reverse=True)
    return [label for label, _ in ranked[:unique_limit]]


def _selector_label_stats(artifacts: list[KnowledgeArtifact]) -> tuple[dict[str, int], dict[str, int]]:
    counts: dict[str, int] = {}
    first_rank: dict[str, int] = {}
    for idx, artifact in enumerate(artifacts):
        label = artifact_route_label(artifact)
        norm = normalize_label(label)
        if not norm:
            continue
        counts[norm] = counts.get(norm, 0) + 1
        first_rank.setdefault(norm, idx)
    return counts, first_rank


def _selector_label_evidence(artifacts: list[KnowledgeArtifact]) -> dict[str, float]:
    evidence: dict[str, float] = {}
    for idx, artifact in enumerate(artifacts):
        label = artifact_route_label(artifact)
        norm = normalize_label(label)
        if not norm:
            continue
        evidence[norm] = evidence.get(norm, 0.0) + (1.0 / (idx + 1))
    return evidence


@lru_cache(maxsize=1)
def _historical_label_selector() -> tuple[Pipeline, list[str]]:
    rows = historical_rows(load_runlog(DEFAULT_RUNLOG), load_paper_labels(DEFAULT_EVOLUTION))
    texts = [row["text"] for row in rows]
    labels = [row["label"] for row in rows]
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))),
        ]
    )
    clf = LinearSVC(C=1.0)
    pipe = Pipeline([("features", features), ("clf", clf)])
    pipe.fit(texts, labels)
    return pipe, list(pipe.named_steps["clf"].classes_)


def select_route_label(query: str, artifacts: list[KnowledgeArtifact]) -> str:
    selector = selector_mode()
    if selector not in {"historical_cv_svm", "historical_cv_svm_diverse", "historical_cv_svm_diverse_consensus", "historical_cv_svm_diverse_provenance"}:
        return first_route_bearing_label(artifacts)

    candidate_labels = _selector_candidate_labels(query, artifacts)
    if not candidate_labels:
        return ""

    pipe, classes = _historical_label_selector()
    predicted = str(pipe.predict([query])[0])
    candidate_norms = {normalize_label(label): label for label in candidate_labels}
    if normalize_label(predicted) in candidate_norms:
        return candidate_norms[normalize_label(predicted)]

    decision = pipe.decision_function([query])
    scores = decision[0] if getattr(decision, "ndim", 1) > 1 else decision
    class_scores = {normalize_label(label): float(score) for label, score in zip(classes, scores)}
    if _selector_uses_provenance_tiebreak(selector):
        evidence = _selector_label_evidence(artifacts)
        provenance_weight = float(os.environ.get("SYNAPSE_SELECTOR_PROVENANCE_WEIGHT", "1.0"))
        return max(
            candidate_labels,
            key=lambda label: (
                class_scores.get(normalize_label(label), float("-inf"))
                + provenance_weight * evidence.get(normalize_label(label), 0.0)
            ),
        )
    if _selector_uses_consensus_tiebreak(selector):
        counts, first_rank = _selector_label_stats(artifacts)
        multiplicity_weight = float(os.environ.get("SYNAPSE_SELECTOR_MULTIPLICITY_WEIGHT", "0.2"))
        rank_weight = float(os.environ.get("SYNAPSE_SELECTOR_RANK_WEIGHT", "0.03"))
        return max(
            candidate_labels,
            key=lambda label: (
                class_scores.get(normalize_label(label), float("-inf"))
                + multiplicity_weight * counts.get(normalize_label(label), 0)
                - rank_weight * first_rank.get(normalize_label(label), 999)
            ),
        )
    return max(candidate_labels, key=lambda label: class_scores.get(normalize_label(label), float("-inf")))


def evaluate_seed(
    runtime: SynapseRuntime,
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
        artifacts = runtime.get_context_for_query(query, max_items=selector_expanded_max_items(max_items))
        predicted = select_route_label(query, artifacts)
        top_candidates = [artifact_route_label(artifact) for artifact in artifacts]
        hit = labels_match(predicted, gold)
        correct += int(hit)
        rows.append(
            {
                "query_id": record.get("query_id") or record.get("sample_id"),
                "query_text": query,
                "ground_truth_domain": gold,
                "predicted_domain": predicted,
                "routed_correctly": hit,
                "top_candidates": top_candidates,
            }
        )

    accuracy = correct / sample_count if sample_count else 0.0
    return {
        "seed": seed,
        "sample_count": sample_count,
        "correct": correct,
        "accuracy": accuracy,
        "top_predicted_domains": summarize_top_predictions(row["predicted_domain"] for row in rows),
        "rows": rows,
    }


def main() -> None:
    load_dotenv()
    args = parse_args()
    records = load_records(args.sample_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    with temporary_routing_alignment_profile():
        runtime = build_runtime(rounds=args.rounds, client_count=args.client_count)

        results = [
            evaluate_seed(
                runtime=runtime,
                records=records,
                seed=seed,
                sample_count=args.sample_count,
                max_items=args.max_items,
            )
            for seed in seeds
        ]

    accuracies = [result["accuracy"] for result in results]
    mean_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    if len(accuracies) > 1:
        variance = sum((value - mean_accuracy) ** 2 for value in accuracies) / (len(accuracies) - 1)
        sd_accuracy = variance ** 0.5
    else:
        sd_accuracy = 0.0

    for result in results:
        out_path = args.output_dir / f"routing_seed_{result['seed']}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    summary = {
        "sample_file": str(args.sample_file),
        "sample_count": args.sample_count,
        "rounds": args.rounds,
        "client_count": args.client_count,
        "seeds": seeds,
        "mean_accuracy": mean_accuracy,
        "sd_accuracy": sd_accuracy,
        "per_seed_accuracy": {str(result["seed"]): result["accuracy"] for result in results},
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Routing verification sample file: {args.sample_file}")
    print(f"Federation rounds: {args.rounds}")
    print(f"Sample count per seed: {args.sample_count}")
    for result in results:
        print(
            f"Seed {result['seed']}: "
            f"{result['correct']}/{result['sample_count']} "
            f"({result['accuracy'] * 100:.1f}%)"
        )
    print(f"Mean routing accuracy: {mean_accuracy * 100:.1f}%")
    print(f"SD across seeds: {sd_accuracy * 100:.2f}%")
    print(f"Wrote per-seed outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
