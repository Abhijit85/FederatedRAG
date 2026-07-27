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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_FILE = REPO_ROOT / "GSM8K_500_rebuttal_run" / "GSM8K_500_samples.json"
DEFAULT_RUNLOG = REPO_ROOT / "GSM8K_500_rebuttal_run" / "GSM8K_500_runlog.jsonl"
DEFAULT_COMPENDIUM = REPO_ROOT / "mathqa_tools_compendium.json"
DEFAULT_EVOLUTION = REPO_ROOT / "GSM8K_500_rebuttal_run" / "GSM8K_compendium_evolution.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "verification" / "gsm8k_paper_recovery_sweep"

ALIASES = {
    "geometry and measurement": "geometry shapes and measurement",
    "geometry shapes and measurement": "geometry shapes and measurement",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark multiple local historical GSM8K reconstruction strategies against the "
            "paper-time six-scenario universe on shared 5-seed subsets."
        )
    )
    parser.add_argument("--sample-file", type=Path, default=DEFAULT_SAMPLE_FILE)
    parser.add_argument("--runlog", type=Path, default=DEFAULT_RUNLOG)
    parser.add_argument("--compendium", type=Path, default=DEFAULT_COMPENDIUM)
    parser.add_argument("--evolution", type=Path, default=DEFAULT_EVOLUTION)
    parser.add_argument("--sample-count", type=int, default=100)
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


def sample_indices(total: int, seed: int, sample_count: int) -> list[int]:
    if sample_count > total:
        raise ValueError(f"Requested {sample_count} rows, but only {total} are available.")
    rng = random.Random(seed)
    return sorted(rng.sample(range(total), sample_count))


def historical_rows(runlog_rows: list[dict[str, Any]], paper_labels: list[str]) -> list[dict[str, str]]:
    matched: list[dict[str, str]] = []
    for row in runlog_rows:
        label = gold_route_label(row)
        canonical = next((paper for paper in paper_labels if labels_match(paper, label)), None)
        if not canonical:
            continue
        matched.append(
            {
                "query_id": str(row.get("query_id") or row.get("sample_id") or ""),
                "text": query_text(row),
                "label": canonical,
            }
        )
    return matched


def label_prototypes(
    *,
    labels: list[str],
    compendium_text: dict[str, str],
    train_rows: list[dict[str, str]],
    examples_per_label: int,
) -> dict[str, str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in train_rows:
        grouped[row["label"]].append(row["text"])
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


def predict_prototype(
    *,
    train_rows: list[dict[str, str]],
    eval_rows: list[dict[str, str]],
    labels: list[str],
    compendium_text: dict[str, str],
    examples_per_label: int,
) -> list[str]:
    prototypes = label_prototypes(
        labels=labels,
        compendium_text=compendium_text,
        train_rows=train_rows,
        examples_per_label=examples_per_label,
    )
    label_texts = [prototypes[label] for label in labels]
    query_texts = [row["text"] for row in eval_rows]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(label_texts + query_texts)
    label_matrix = matrix[: len(labels)]
    query_matrix = matrix[len(labels) :]
    scores = cosine_similarity(query_matrix, label_matrix)
    return [labels[max(range(len(labels)), key=row.__getitem__)] for row in scores.tolist()]


def predict_nearest_neighbor(train_rows: list[dict[str, str]], eval_rows: list[dict[str, str]]) -> list[str]:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    train_matrix = vectorizer.fit_transform([row["text"] for row in train_rows])
    eval_matrix = vectorizer.transform([row["text"] for row in eval_rows])
    scores = cosine_similarity(eval_matrix, train_matrix)
    return [train_rows[max(range(len(train_rows)), key=row.__getitem__)]["label"] for row in scores.tolist()]


def predict_pipeline(
    train_rows: list[dict[str, str]],
    eval_rows: list[dict[str, str]],
    *,
    classifier: str,
) -> list[str]:
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))),
        ]
    )
    if classifier == "logreg":
        clf = LogisticRegression(max_iter=4000, C=3.0)
    elif classifier == "svm":
        clf = LinearSVC(C=1.0)
    elif classifier == "cnb":
        clf = ComplementNB(alpha=0.5)
    else:
        raise ValueError(f"Unsupported classifier: {classifier}")
    pipe = Pipeline([("features", features), ("clf", clf)])
    pipe.fit([row["text"] for row in train_rows], [row["label"] for row in train_rows])
    return list(pipe.predict([row["text"] for row in eval_rows]))


def cross_validated_predictions(rows: list[dict[str, str]], *, classifier: str) -> list[str]:
    labels = [row["label"] for row in rows]
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))),
        ]
    )
    if classifier == "logreg":
        clf = LogisticRegression(max_iter=4000, C=3.0)
    elif classifier == "svm":
        clf = LinearSVC(C=1.0)
    elif classifier == "cnb":
        clf = ComplementNB(alpha=0.5)
    else:
        raise ValueError(f"Unsupported classifier: {classifier}")
    pipe = Pipeline([("features", features), ("clf", clf)])
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    predictions = ["" for _ in rows]
    for train_idx, test_idx in splitter.split([row["text"] for row in rows], labels):
        train_rows = [rows[i] for i in train_idx]
        test_rows = [rows[i] for i in test_idx]
        preds = predict_pipeline(train_rows, test_rows, classifier=classifier)
        for idx, pred in zip(test_idx, preds):
            predictions[idx] = pred
    return predictions


def cross_validated_predictions_aux(rows: list[dict[str, str]]) -> list[str]:
    pseudo_labels: list[str] = []
    for row in rows:
        pseudo = row["label"]
        top_candidates = row.get("top_candidates") or []
        if row["label"] == "General Logic and Counting" and len(top_candidates) > 1 and top_candidates[1] == "Number Theory":
            pseudo = "Number Theory"
        elif row["label"] in {"Algebraic Word Problem Solver", "Geometry and Measurement"} and len(top_candidates) > 1 and top_candidates[1] == "MathQA":
            pseudo = "MathQA"
        pseudo_labels.append(pseudo)

    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))),
        ]
    )
    clf = LinearSVC(C=1.0)
    pipe = Pipeline([("features", features), ("clf", clf)])
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    predictions = ["" for _ in rows]
    for train_idx, test_idx in splitter.split([row["text"] for row in rows], pseudo_labels):
        train_rows = [rows[i] for i in train_idx]
        train_pseudo = [pseudo_labels[i] for i in train_idx]
        test_rows = [rows[i] for i in test_idx]
        pipe.fit([row["text"] for row in train_rows], train_pseudo)
        preds = list(pipe.predict([row["text"] for row in test_rows]))
        for idx, pred in zip(test_idx, preds):
            if pred == "Number Theory":
                pred = "General Logic and Counting"
            elif pred == "MathQA":
                pred = "Algebraic Word Problem Solver"
            predictions[idx] = pred
    return predictions


def evaluate_seed(predictions: list[str], rows: list[dict[str, str]], seed: int, sample_count: int) -> dict[str, Any]:
    indices = sample_indices(len(rows), seed=seed, sample_count=sample_count)
    subset = [rows[i] for i in indices]
    subset_preds = [predictions[i] for i in indices]
    correct = sum(int(labels_match(pred, row["label"])) for pred, row in zip(subset_preds, subset))
    accuracy = correct / sample_count if sample_count else 0.0
    return {
        "seed": seed,
        "sample_count": sample_count,
        "correct": correct,
        "accuracy": accuracy,
        "rows": [
            {
                "query_id": row["query_id"],
                "query_text": row["text"],
                "ground_truth_domain": row["label"],
                "predicted_domain": pred,
                "routed_correctly": labels_match(pred, row["label"]),
            }
            for row, pred in zip(subset, subset_preds)
        ],
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
    paper_labels = load_paper_labels(args.evolution)
    compendium_text = load_compendium_text(args.compendium)
    seeds = parse_seed_list(args.seeds)

    rows = historical_rows(runlog_rows, paper_labels)
    row_by_qid = {row["query_id"]: row for row in rows}
    eval_rows = [row_by_qid[str(record.get("query_id") or record.get("sample_id") or "")] for record in sample_records if str(record.get("query_id") or record.get("sample_id") or "") in row_by_qid]

    methods: dict[str, list[str]] = {}

    # Cross-validated classifiers on the historical query bank.
    for name in ("svm", "logreg", "cnb"):
        methods[f"cv_{name}"] = cross_validated_predictions(eval_rows, classifier=name)
    methods["cv_aux_collapse"] = cross_validated_predictions_aux(eval_rows)

    # Seed-specific train/eval reconstructions.
    for examples_per_label in (12, 30, 60):
        all_preds = [""] * len(eval_rows)
        for seed in seeds:
            idx = sample_indices(len(eval_rows), seed=seed, sample_count=args.sample_count)
            held_out = set(idx)
            train_rows = [row for i, row in enumerate(eval_rows) if i not in held_out]
            test_rows = [row for i, row in enumerate(eval_rows) if i in held_out]
            preds = predict_prototype(
                train_rows=train_rows,
                eval_rows=test_rows,
                labels=paper_labels,
                compendium_text=compendium_text,
                examples_per_label=examples_per_label,
            )
            for target_idx, pred in zip(idx, preds):
                all_preds[target_idx] = pred
        methods[f"prototype_{examples_per_label}"] = all_preds

    nn_preds = [""] * len(eval_rows)
    for seed in seeds:
        idx = sample_indices(len(eval_rows), seed=seed, sample_count=args.sample_count)
        held_out = set(idx)
        train_rows = [row for i, row in enumerate(eval_rows) if i not in held_out]
        test_rows = [row for i, row in enumerate(eval_rows) if i in held_out]
        preds = predict_nearest_neighbor(train_rows, test_rows)
        for target_idx, pred in zip(idx, preds):
            nn_preds[target_idx] = pred
    methods["query_bank_nn"] = nn_preds

    summary_rows: list[dict[str, Any]] = []
    for method_name, predictions in methods.items():
        results = [evaluate_seed(predictions, eval_rows, seed=seed, sample_count=args.sample_count) for seed in seeds]
        mean_accuracy, sd_accuracy = summarize(results)
        method_dir = args.output_dir / method_name
        method_dir.mkdir(parents=True, exist_ok=True)
        for result in results:
            (method_dir / f"routing_seed_{result['seed']}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        row = {
            "method": method_name,
            "paper_labels": paper_labels,
            "sample_count": args.sample_count,
            "seeds": seeds,
            "mean_accuracy": mean_accuracy,
            "sd_accuracy": sd_accuracy,
            "per_seed_accuracy": {str(result["seed"]): result["accuracy"] for result in results},
            "output_dir": str(method_dir),
        }
        (method_dir / "summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        summary_rows.append(row)
        print(f"{method_name}: mean={mean_accuracy:.3f}, sd={sd_accuracy:.3f}, seeds={row['per_seed_accuracy']}")

    combined = {
        "sample_file": str(args.sample_file),
        "runlog": str(args.runlog),
        "compendium": str(args.compendium),
        "evolution": str(args.evolution),
        "paper_labels": paper_labels,
        "sample_count": args.sample_count,
        "seeds": seeds,
        "methods": summary_rows,
        "note": (
            "Local-only recovery sweep over the historical six-scenario GSM8K universe. "
            "These methods reconstruct the paper-time artifact space from stored runlog and compendium assets."
        ),
    }
    (args.output_dir / "combined_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
