#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_FILE = REPO_ROOT / "GSM8K_500_rebuttal_run" / "GSM8K_500_samples.json"
DEFAULT_COMPENDIUM = REPO_ROOT / "mathqa_tools_compendium.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "verification" / "routing_historical_local"

ALIASES = {
    "geometry and measurement": "geometry shapes and measurement",
    "geometry shapes and measurement": "geometry shapes and measurement",
    "general logic counting": "general logic and counting",
}

EXTRA_LABEL_TEXT = {
    "Number Theory": (
        "Solves discrete arithmetic and integer reasoning problems involving divisibility, "
        "factors, multiples, parity, sequences, and number patterns."
    ),
    "MathQA": (
        "General mathematical question answering across arithmetic, algebra, geometry, ratios, "
        "percentages, and multi-step word-problem solving."
    ),
    "SearchQA": (
        "Answers factoid questions that require retrieving a factual statement rather than solving "
        "a mathematical procedure."
    ),
    "Logic": "Handles formal logic, deduction, consistency checking, and symbolic reasoning problems.",
    "MMLUQA": (
        "Answers broad academic multiple-choice knowledge questions across science, humanities, "
        "social science, and reasoning."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a local-only reconstruction of the historical GSM8K routing candidate universe "
            "using TF-IDF over compendium scenario text."
        )
    )
    parser.add_argument("--sample-file", type=Path, default=DEFAULT_SAMPLE_FILE)
    parser.add_argument("--compendium", type=Path, default=DEFAULT_COMPENDIUM)
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def parse_seed_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def normalize_label(value: str | None) -> str:
    text = (value or "").strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return ALIASES.get(text, text)


def labels_match(left: str | None, right: str | None) -> bool:
    return normalize_label(left) == normalize_label(right)


def load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return [record for record in payload["records"] if isinstance(record, dict)]
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]
    raise ValueError(f"Unsupported sample file format: {path}")


def load_label_text(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    usage = payload.get("Textual_Compendium", {}).get("Usage_Scenarios", [])
    label_to_text = {
        row["scenario"]: row["context"]
        for row in usage
        if isinstance(row, dict) and isinstance(row.get("scenario"), str) and isinstance(row.get("context"), str)
    }
    if "Geometry: Shapes and Measurement" in label_to_text:
        label_to_text["Geometry and Measurement"] = label_to_text["Geometry: Shapes and Measurement"]
    label_to_text.update(EXTRA_LABEL_TEXT)
    return label_to_text


def historical_candidate_labels(records: list[dict[str, Any]], label_to_text: dict[str, str]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for record in records:
        router = record.get("router") or {}
        values: list[str] = []
        gold = router.get("ground_truth_domain")
        if isinstance(gold, str) and gold.strip():
            values.append(gold.strip())
        for candidate in router.get("top_candidates") or []:
            value = candidate.get("domain") if isinstance(candidate, dict) else None
            if isinstance(value, str) and value.strip():
                values.append(value.strip())
        for value in values:
            normalized = normalize_label(value)
            if normalized in seen:
                continue
            seen.add(normalized)
            label = "Geometry and Measurement" if normalized == "geometry shapes and measurement" else value
            if label in label_to_text:
                labels.append(label)
    return labels


def sample_records(records: list[dict[str, Any]], seed: int, sample_count: int) -> list[dict[str, Any]]:
    if sample_count > len(records):
        raise ValueError(f"Requested {sample_count} samples, but only {len(records)} records are available.")
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(records)), sample_count))
    return [records[idx] for idx in indices]


def query_text(record: dict[str, Any]) -> str:
    for key in ("query_text", "question", "Problem"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def gold_route_label(record: dict[str, Any]) -> str:
    router = record.get("router")
    if isinstance(router, dict):
        value = router.get("ground_truth_domain")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def build_matrix(labels: list[str], label_to_text: dict[str, str], records: list[dict[str, Any]]):
    texts = [label_to_text[label] for label in labels]
    queries = [query_text(record) for record in records]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(texts + queries)
    label_matrix = matrix[: len(texts)]
    query_matrix = matrix[len(texts) :]
    return cosine_similarity(query_matrix, label_matrix)


def evaluate_seed(
    *,
    records: list[dict[str, Any]],
    labels: list[str],
    scores: list[list[float]],
    seed: int,
    sample_count: int,
) -> dict[str, Any]:
    subset = sample_records(records, seed=seed, sample_count=sample_count)
    rows: list[dict[str, Any]] = []
    correct = 0
    record_index = {id(record): idx for idx, record in enumerate(records)}
    for record in subset:
        idx = record_index[id(record)]
        row_scores = scores[idx]
        best = max(range(len(labels)), key=row_scores.__getitem__)
        predicted = labels[best]
        gold = gold_route_label(record)
        hit = labels_match(predicted, gold)
        correct += int(hit)
        ranked = sorted(range(len(labels)), key=row_scores.__getitem__, reverse=True)[:5]
        rows.append(
            {
                "query_id": record.get("query_id") or record.get("sample_id"),
                "query_text": query_text(record),
                "ground_truth_domain": gold,
                "predicted_domain": predicted,
                "routed_correctly": hit,
                "top_candidates": [labels[i] for i in ranked],
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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.sample_file)
    label_to_text = load_label_text(args.compendium)
    labels = historical_candidate_labels(records, label_to_text)
    score_matrix = build_matrix(labels, label_to_text, records).tolist()
    seeds = parse_seed_list(args.seeds)

    results = [
        evaluate_seed(
            records=records,
            labels=labels,
            scores=score_matrix,
            seed=seed,
            sample_count=args.sample_count,
        )
        for seed in seeds
    ]
    for result in results:
        out_path = args.output_dir / f"routing_seed_{result['seed']}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    mean_accuracy, sd_accuracy = summarize(results)
    summary = {
        "sample_file": str(args.sample_file),
        "compendium": str(args.compendium),
        "candidate_labels": labels,
        "sample_count": args.sample_count,
        "seeds": seeds,
        "mean_accuracy": mean_accuracy,
        "sd_accuracy": sd_accuracy,
        "per_seed_accuracy": {str(result["seed"]): result["accuracy"] for result in results},
        "output_dir": str(args.output_dir),
        "note": (
            "Local-only reconstruction using the historical runlog label universe plus compendium "
            "scenario text. No external embedding or reranking services are used."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"candidate_labels={labels}")
    print(f"mean_accuracy={mean_accuracy:.3f}")
    print(f"sd_accuracy={sd_accuracy:.3f}")
    print(f"per_seed={summary['per_seed_accuracy']}")


if __name__ == "__main__":
    main()
