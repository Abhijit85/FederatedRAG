#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_FILE = REPO_ROOT / "GSM8K_500_rebuttal_run" / "GSM8K_500_samples.json"
DEFAULT_RUNLOG = REPO_ROOT / "GSM8K_500_rebuttal_run" / "GSM8K_500_runlog.jsonl"
DEFAULT_COMPENDIUM = REPO_ROOT / "mathqa_tools_compendium.json"
DEFAULT_EVOLUTION = REPO_ROOT / "GSM8K_500_rebuttal_run" / "GSM8K_compendium_evolution.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "verification" / "gsm8k_paper_mode_local"

ALIASES = {
    "geometry and measurement": "geometry shapes and measurement",
    "geometry shapes and measurement": "geometry shapes and measurement",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a local-only reconstruction of the submitted-paper GSM8K routing regime "
            "using the historical six-scenario universe plus exemplar-enriched label prototypes."
        )
    )
    parser.add_argument("--sample-file", type=Path, default=DEFAULT_SAMPLE_FILE)
    parser.add_argument("--runlog", type=Path, default=DEFAULT_RUNLOG)
    parser.add_argument("--compendium", type=Path, default=DEFAULT_COMPENDIUM)
    parser.add_argument("--evolution", type=Path, default=DEFAULT_EVOLUTION)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--examples-per-label", type=int, default=12)
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


def load_json_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return [record for record in payload["records"] if isinstance(record, dict)]
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]
    raise ValueError(f"Unsupported sample file format: {path}")


def load_runlog(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and obj.get("source_kind") == "gsm8k_derived":
                rows.append(obj)
    return rows


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
    for key in ("ground_truth_domain", "domain", "scenario"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def load_paper_labels(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("overall_unique_scenarios")
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"Expected overall_unique_scenarios in {path}")
    return [str(label) for label in labels]


def load_compendium_text(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    usage = payload.get("Textual_Compendium", {}).get("Usage_Scenarios", [])
    result: dict[str, str] = {}
    for row in usage:
        if not isinstance(row, dict):
            continue
        scenario = row.get("scenario")
        context = row.get("context")
        if isinstance(scenario, str) and isinstance(context, str) and scenario.strip() and context.strip():
            result[scenario.strip()] = context.strip()
    if "Geometry: Shapes and Measurement" in result and "Geometry and Measurement" not in result:
        result["Geometry and Measurement"] = result["Geometry: Shapes and Measurement"]
    return result


def sample_rows(rows: list[dict[str, Any]], seed: int, sample_count: int) -> list[dict[str, Any]]:
    if sample_count > len(rows):
        raise ValueError(f"Requested {sample_count} rows, but only {len(rows)} are available.")
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), sample_count))
    return [rows[idx] for idx in indices]


def build_label_prototypes(
    *,
    labels: list[str],
    compendium_text: dict[str, str],
    runlog_rows: list[dict[str, Any]],
    excluded_query_ids: set[str],
    examples_per_label: int,
) -> dict[str, str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in runlog_rows:
        qid = str(row.get("query_id") or "")
        if qid in excluded_query_ids:
            continue
        label = gold_route_label(row)
        if not label:
            continue
        matched = next((paper for paper in labels if labels_match(paper, label)), None)
        if not matched:
            continue
        text = query_text(row)
        if text:
            grouped[matched].append(text)

    prototypes: dict[str, str] = {}
    for label in labels:
        parts = [label]
        context = compendium_text.get(label)
        if context:
            parts.append(context)
        exemplars = grouped.get(label, [])[:examples_per_label]
        if exemplars:
            parts.append("Representative queries:")
            parts.extend(exemplars)
        prototypes[label] = "\n".join(parts)
    return prototypes


def classify_rows(
    *,
    rows: list[dict[str, Any]],
    prototypes: dict[str, str],
    labels: list[str],
) -> list[dict[str, Any]]:
    query_texts = [query_text(row) for row in rows]
    label_texts = [prototypes[label] for label in labels]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(label_texts + query_texts)
    label_matrix = matrix[: len(labels)]
    query_matrix = matrix[len(labels) :]
    scores = cosine_similarity(query_matrix, label_matrix)

    outputs: list[dict[str, Any]] = []
    for row, row_scores in zip(rows, scores.tolist()):
        ranked = sorted(range(len(labels)), key=row_scores.__getitem__, reverse=True)
        predicted = labels[ranked[0]]
        gold = gold_route_label(row)
        outputs.append(
            {
                "query_id": row.get("query_id") or row.get("sample_id"),
                "query_text": query_text(row),
                "ground_truth_domain": gold,
                "predicted_domain": predicted,
                "routed_correctly": labels_match(predicted, gold),
                "top_candidates": [labels[i] for i in ranked[:5]],
                "top_scores": [row_scores[i] for i in ranked[:5]],
            }
        )
    return outputs


def evaluate_seed(
    *,
    sample_records: list[dict[str, Any]],
    runlog_rows: list[dict[str, Any]],
    labels: list[str],
    compendium_text: dict[str, str],
    examples_per_label: int,
    seed: int,
    sample_count: int,
) -> dict[str, Any]:
    subset = sample_rows(sample_records, seed=seed, sample_count=sample_count)
    excluded = {str(row.get("query_id") or row.get("sample_id") or "") for row in subset}
    prototypes = build_label_prototypes(
        labels=labels,
        compendium_text=compendium_text,
        runlog_rows=runlog_rows,
        excluded_query_ids=excluded,
        examples_per_label=examples_per_label,
    )
    rows = classify_rows(rows=subset, prototypes=prototypes, labels=labels)
    correct = sum(int(row["routed_correctly"]) for row in rows)
    accuracy = correct / sample_count if sample_count else 0.0
    return {
        "seed": seed,
        "sample_count": sample_count,
        "correct": correct,
        "accuracy": accuracy,
        "paper_labels": labels,
        "prototype_preview": {label: prototypes[label][:400] for label in labels},
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

    sample_records = load_json_records(args.sample_file)
    runlog_rows = load_runlog(args.runlog)
    labels = load_paper_labels(args.evolution)
    compendium_text = load_compendium_text(args.compendium)
    seeds = parse_seed_list(args.seeds)

    results = [
        evaluate_seed(
            sample_records=sample_records,
            runlog_rows=runlog_rows,
            labels=labels,
            compendium_text=compendium_text,
            examples_per_label=args.examples_per_label,
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
        "runlog": str(args.runlog),
        "compendium": str(args.compendium),
        "evolution": str(args.evolution),
        "paper_labels": labels,
        "examples_per_label": args.examples_per_label,
        "sample_count": args.sample_count,
        "seeds": seeds,
        "mean_accuracy": mean_accuracy,
        "sd_accuracy": sd_accuracy,
        "per_seed_accuracy": {str(result["seed"]): result["accuracy"] for result in results},
        "output_dir": str(args.output_dir),
        "note": (
            "Local-only paper-mode reconstruction using the historical six-scenario GSM8K universe "
            "and exemplar-enriched label prototypes from the April 3, 2026 runlog."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"paper_labels={labels}")
    print(f"examples_per_label={args.examples_per_label}")
    print(f"mean_accuracy={mean_accuracy:.3f}")
    print(f"sd_accuracy={sd_accuracy:.3f}")
    print(f"per_seed={summary['per_seed_accuracy']}")


if __name__ == "__main__":
    main()
