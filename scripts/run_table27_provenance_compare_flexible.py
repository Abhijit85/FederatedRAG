#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_gsm8k_paper_recovery_sweep import (  # noqa: E402
    DEFAULT_COMPENDIUM,
    DEFAULT_EVOLUTION,
    DEFAULT_RUNLOG,
    DEFAULT_SAMPLE_FILE,
    historical_rows,
    labels_match,
    load_json_records,
    load_paper_labels,
    load_runlog,
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "verification" / "table27_provenance_compare_flexible"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a provenance-faithful fresh Table 27 comparator over preserved paper-space GSM8K assets, "
            "with configurable classifier family for both federated and centralized arms."
        )
    )
    parser.add_argument("--sample-file", type=Path, default=DEFAULT_SAMPLE_FILE)
    parser.add_argument("--runlog", type=Path, default=DEFAULT_RUNLOG)
    parser.add_argument("--compendium", type=Path, default=DEFAULT_COMPENDIUM)
    parser.add_argument("--evolution", type=Path, default=DEFAULT_EVOLUTION)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--client-count", type=int, default=5)
    parser.add_argument("--shard-seed", type=int, default=0)
    parser.add_argument("--classifier", choices=("svm", "logreg", "cnb"), default="svm")
    parser.add_argument("--classifier-c", type=float, default=3.0)
    parser.add_argument("--nb-alpha", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def parse_seed_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def sample_indices(total: int, seed: int, sample_count: int) -> list[int]:
    if sample_count > total:
        raise ValueError(f"Requested {sample_count} rows, but only {total} rows are available.")
    rng = random.Random(seed)
    return sorted(rng.sample(range(total), sample_count))


def build_eval_rows(sample_file: Path, runlog: Path, evolution: Path) -> tuple[list[dict[str, str]], list[str]]:
    sample_records = load_json_records(sample_file)
    runlog_rows = load_runlog(runlog)
    paper_labels = load_paper_labels(evolution)
    rows = historical_rows(runlog_rows, paper_labels)
    row_by_qid = {row["query_id"]: row for row in rows}
    eval_rows = [
        row_by_qid[str(record.get("query_id") or record.get("sample_id") or "")]
        for record in sample_records
        if str(record.get("query_id") or record.get("sample_id") or "") in row_by_qid
    ]
    return eval_rows, paper_labels


def make_pipeline(classifier: str, c_value: float, nb_alpha: float) -> Pipeline:
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))),
        ]
    )
    if classifier == "svm":
        clf = LinearSVC(C=c_value)
    elif classifier == "logreg":
        clf = LogisticRegression(max_iter=4000, C=c_value)
    elif classifier == "cnb":
        clf = ComplementNB(alpha=nb_alpha)
    else:
        raise ValueError(f"Unsupported classifier: {classifier}")
    return Pipeline([("features", features), ("clf", clf)])


def stratified_client_shards(rows: list[dict[str, str]], client_count: int, shard_seed: int) -> list[list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)

    shards: list[list[dict[str, str]]] = [[] for _ in range(client_count)]
    for label, label_rows in grouped.items():
        rng = random.Random(f"{shard_seed}:{label}")
        permuted = list(label_rows)
        rng.shuffle(permuted)
        for idx, row in enumerate(permuted):
            shards[idx % client_count].append(row)
    return shards


def align_score_rows(classes: list[str], rows: list[list[float]], target_labels: list[str]) -> list[list[float]]:
    index = {label: idx for idx, label in enumerate(classes)}
    aligned: list[list[float]] = []
    for row in rows:
        out = [float("-inf")] * len(target_labels)
        for label_idx, label in enumerate(target_labels):
            src_idx = index.get(label)
            if src_idx is not None:
                out[label_idx] = float(row[src_idx])
        aligned.append(out)
    return aligned


def raw_score_rows(pipe: Pipeline, texts: list[str]) -> tuple[list[str], list[list[float]]]:
    clf = pipe.named_steps["clf"]
    classes = list(getattr(clf, "classes_", []))
    if hasattr(pipe, "predict_proba"):
        rows = pipe.predict_proba(texts).tolist()
        return classes, rows
    decision = pipe.decision_function(texts)
    if getattr(decision, "ndim", 1) == 1:
        # Binary margin -> expand to two columns in class order.
        rows = [[-float(value), float(value)] for value in decision.tolist()]
    else:
        rows = [[float(v) for v in row] for row in decision.tolist()]
    return classes, rows


def predict_centralized(
    *, train_rows: list[dict[str, str]], eval_rows: list[dict[str, str]], paper_labels: list[str], classifier: str, c_value: float, nb_alpha: float
) -> list[str]:
    pipe = make_pipeline(classifier, c_value, nb_alpha)
    pipe.fit([row["text"] for row in train_rows], [row["label"] for row in train_rows])
    classes, raw_rows = raw_score_rows(pipe, [row["text"] for row in eval_rows])
    aligned = align_score_rows(classes, raw_rows, paper_labels)
    return [paper_labels[max(range(len(paper_labels)), key=score_row.__getitem__)] for score_row in aligned]


def predict_federated(
    *, train_rows: list[dict[str, str]], eval_rows: list[dict[str, str]], paper_labels: list[str], client_count: int, shard_seed: int, classifier: str, c_value: float, nb_alpha: float
) -> list[str]:
    shards = stratified_client_shards(train_rows, client_count=client_count, shard_seed=shard_seed)
    score_sums = [[0.0] * len(paper_labels) for _ in eval_rows]
    total_weight = 0.0
    for shard in shards:
        if not shard:
            continue
        labels = {row["label"] for row in shard}
        if len(labels) < 2:
            continue
        pipe = make_pipeline(classifier, c_value, nb_alpha)
        pipe.fit([row["text"] for row in shard], [row["label"] for row in shard])
        classes, raw_rows = raw_score_rows(pipe, [row["text"] for row in eval_rows])
        aligned = align_score_rows(classes, raw_rows, paper_labels)
        weight = float(len(shard))
        total_weight += weight
        for i, row_scores in enumerate(aligned):
            finite_scores = [value for value in row_scores if value != float("-inf")]
            floor = (min(finite_scores) - 1.0) if finite_scores else -1.0
            for j, value in enumerate(row_scores):
                score_sums[i][j] += weight * (value if value != float("-inf") else floor)

    if total_weight <= 0:
        return [paper_labels[0]] * len(eval_rows)
    return [paper_labels[max(range(len(paper_labels)), key=score_row.__getitem__)] for score_row in score_sums]


def evaluate_seed_pair(
    *, rows: list[dict[str, str]], paper_labels: list[str], seed: int, sample_count: int, client_count: int, shard_seed: int, classifier: str, c_value: float, nb_alpha: float
) -> dict[str, Any]:
    held_out = set(sample_indices(len(rows), seed=seed, sample_count=sample_count))
    train_rows = [row for i, row in enumerate(rows) if i not in held_out]
    eval_rows = [rows[i] for i in sorted(held_out)]

    fed_preds = predict_federated(
        train_rows=train_rows,
        eval_rows=eval_rows,
        paper_labels=paper_labels,
        client_count=client_count,
        shard_seed=shard_seed,
        classifier=classifier,
        c_value=c_value,
        nb_alpha=nb_alpha,
    )
    cen_preds = predict_centralized(
        train_rows=train_rows,
        eval_rows=eval_rows,
        paper_labels=paper_labels,
        classifier=classifier,
        c_value=c_value,
        nb_alpha=nb_alpha,
    )

    fed_correct = sum(int(labels_match(pred, row["label"])) for pred, row in zip(fed_preds, eval_rows))
    cen_correct = sum(int(labels_match(pred, row["label"])) for pred, row in zip(cen_preds, eval_rows))

    paired_rows = []
    for row, fed_pred, cen_pred in zip(eval_rows, fed_preds, cen_preds):
        paired_rows.append(
            {
                "query_id": row["query_id"],
                "query_text": row["text"],
                "ground_truth_domain": row["label"],
                "federated_prediction": fed_pred,
                "centralized_prediction": cen_pred,
                "federated_correct": labels_match(fed_pred, row["label"]),
                "centralized_correct": labels_match(cen_pred, row["label"]),
            }
        )

    return {
        "seed": seed,
        "sample_count": sample_count,
        "client_count": client_count,
        "train_count": len(train_rows),
        "federated_accuracy": fed_correct / sample_count if sample_count else 0.0,
        "centralized_accuracy": cen_correct / sample_count if sample_count else 0.0,
        "paired_diff": (fed_correct - cen_correct) / sample_count if sample_count else 0.0,
        "rows": paired_rows,
    }


def mean_sd(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values) if values else 0.0
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, sd


def paired_stats(left: list[float], right: list[float]) -> dict[str, float]:
    diffs = [a - b for a, b in zip(left, right)]
    n = len(diffs)
    mean_diff = sum(diffs) / n if n else 0.0
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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seed_list(args.seeds)
    eval_rows, paper_labels = build_eval_rows(args.sample_file, args.runlog, args.evolution)

    per_seed = [
        evaluate_seed_pair(
            rows=eval_rows,
            paper_labels=paper_labels,
            seed=seed,
            sample_count=args.sample_count,
            client_count=args.client_count,
            shard_seed=args.shard_seed,
            classifier=args.classifier,
            c_value=args.classifier_c,
            nb_alpha=args.nb_alpha,
        )
        for seed in seeds
    ]

    for result in per_seed:
        (args.output_dir / f"paired_seed_{result['seed']}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    fed_values = [float(result["federated_accuracy"]) for result in per_seed]
    cen_values = [float(result["centralized_accuracy"]) for result in per_seed]
    fed_mean, fed_sd = mean_sd(fed_values)
    cen_mean, cen_sd = mean_sd(cen_values)
    pair = paired_stats(fed_values, cen_values)

    write_seed_csv(args.output_dir / "synapse_seed_values.csv", seeds, fed_values)
    write_seed_csv(args.output_dir / "centralized_seed_values.csv", seeds, cen_values)

    summary = {
        "sample_file": str(args.sample_file),
        "runlog": str(args.runlog),
        "compendium": str(args.compendium),
        "evolution": str(args.evolution),
        "paper_labels": paper_labels,
        "sample_count": args.sample_count,
        "seeds": seeds,
        "client_count": args.client_count,
        "shard_seed": args.shard_seed,
        "classifier": args.classifier,
        "classifier_c": args.classifier_c,
        "nb_alpha": args.nb_alpha,
        "federated": {
            "mean_accuracy": fed_mean,
            "sd_accuracy": fed_sd,
            "per_seed_accuracy": {str(seed): value for seed, value in zip(seeds, fed_values)},
        },
        "centralized": {
            "mean_accuracy": cen_mean,
            "sd_accuracy": cen_sd,
            "per_seed_accuracy": {str(seed): value for seed, value in zip(seeds, cen_values)},
        },
        "paired": pair,
        "artifacts": {
            "synapse_seed_csv": str(args.output_dir / 'synapse_seed_values.csv'),
            "centralized_seed_csv": str(args.output_dir / 'centralized_seed_values.csv'),
        },
        "note": (
            "Fresh provenance-faithful comparator over preserved April 3 paper-space GSM8K assets with configurable classifier family. "
            "Federated uses deterministic IID client shards with score aggregation; centralized uses one pooled classifier on the same held-out split."
        ),
    }
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(f"classifier={args.classifier}")
    print(f"federated_hist_iid5: mean={fed_mean:.3f}, sd={fed_sd:.3f}, seeds={summary['federated']['per_seed_accuracy']}")
    print(f"centralized_hist_pool: mean={cen_mean:.3f}, sd={cen_sd:.3f}, seeds={summary['centralized']['per_seed_accuracy']}")
    print(f"paired: mean_diff={pair['mean_diff']:+.3f}, sd_diff={pair['sd_diff']:.3f}, se_diff={pair['se_diff']:.3f}, t={pair['t_value']:+.3f}")


if __name__ == '__main__':
    main()
