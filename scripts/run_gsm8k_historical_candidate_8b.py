#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_gsm8k_router_cascade import (  # noqa: E402
    RouterSpec,
    ScenarioDoc,
    labels_match,
    local_router_decision,
)

DEFAULT_SAMPLE_FILE = REPO_ROOT / "GSM8K_500_rebuttal_run" / "GSM8K_500_samples.json"
DEFAULT_RUNLOG = REPO_ROOT / "GSM8K_500_rebuttal_run" / "GSM8K_500_runlog.jsonl"
DEFAULT_COMPENDIUM = REPO_ROOT / "mathqa_tools_compendium.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "verification" / "gsm8k_historical_candidate_8b"

AUX_CONTEXT = {
    "Number Theory": "Problems about integer arithmetic, divisibility, factors, multiples, parity, and counting patterns over whole numbers.",
    "MathQA": "General mathematical reasoning and multi-step arithmetic/algebra problems that do not fit a narrower tool-specific scenario.",
    "Logic": "Problems requiring discrete logical reasoning, inference, and elimination.",
    "SearchQA": "General question answering over retrieved supporting evidence.",
    "MMLUQA": "Broad academic question answering across mixed subjects.",
    "Geometry and Measurement": "Problems about shapes, distance, area, perimeter, volume, and measurement.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local 8B routing on preserved historical GSM8K candidate lists.")
    parser.add_argument("--sample-file", type=Path, default=DEFAULT_SAMPLE_FILE)
    parser.add_argument("--runlog", type=Path, default=DEFAULT_RUNLOG)
    parser.add_argument("--compendium", type=Path, default=DEFAULT_COMPENDIUM)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_seed_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def load_json_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return [row for row in payload["records"] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    raise ValueError(f"Unsupported sample file format: {path}")


def load_runlog_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict) or obj.get("source_kind") != "gsm8k_derived":
                continue
            qid = str(obj.get("query_id") or obj.get("sample_id") or "")
            if qid:
                rows[qid] = obj
    return rows


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
    if "Geometry: Shapes and Measurement" in result:
        result.setdefault("Geometry and Measurement", result["Geometry: Shapes and Measurement"])
    return result


def query_text(record: dict[str, Any]) -> str:
    for key in ("query_text", "question", "Problem"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def gold_route_label(record: dict[str, Any], runlog_row: dict[str, Any]) -> str:
    router = runlog_row.get("router") if isinstance(runlog_row, dict) else None
    if isinstance(router, dict):
        value = router.get("ground_truth_domain")
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("ground_truth_domain", "domain", "scenario"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def sample_records(records: list[dict[str, Any]], seed: int, sample_count: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(records)), sample_count))
    return [records[idx] for idx in indices]


def build_candidates(runlog_row: dict[str, Any], compendium_text: dict[str, str]) -> list[ScenarioDoc]:
    top_candidates = runlog_row.get("router", {}).get("top_candidates") or []
    docs: list[ScenarioDoc] = []
    for item in top_candidates:
        label = item.get("domain") if isinstance(item, dict) else item
        if not isinstance(label, str) or not label.strip():
            continue
        text = compendium_text.get(label) or AUX_CONTEXT.get(label) or f"Routing scenario label: {label}."
        docs.append(ScenarioDoc(label=label, text=text, embedding=[]))
    return docs


def infer_parse_ok(raw_response: str, candidates: list[str]) -> bool:
    for candidate in candidates:
        if labels_match(raw_response, candidate):
            return True
    raw = (raw_response or "").strip().lower()
    return any(candidate.lower() in raw or raw in candidate.lower() for candidate in candidates if raw)


def evaluate_seed(
    *,
    records: list[dict[str, Any]],
    runlog_rows: dict[str, dict[str, Any]],
    compendium_text: dict[str, str],
    seed: int,
    sample_count: int,
    spec: RouterSpec,
    max_tokens: int,
) -> dict[str, Any]:
    subset = sample_records(records, seed=seed, sample_count=sample_count)
    rows: list[dict[str, Any]] = []
    correct = 0
    total_latency = 0.0
    historical_top1_correct = 0
    gold_in_candidates = 0

    for record in subset:
        qid = str(record.get("query_id") or record.get("sample_id") or "")
        runlog_row = runlog_rows.get(qid)
        if runlog_row is None:
            continue
        query = query_text(record) or str(runlog_row.get("query_text") or "")
        gold = gold_route_label(record, runlog_row)
        candidates = build_candidates(runlog_row, compendium_text)
        candidate_labels = [candidate.label for candidate in candidates]
        if candidate_labels and labels_match(candidate_labels[0], gold):
            historical_top1_correct += 1
        gold_in_candidates += int(any(labels_match(gold, label) for label in candidate_labels))
        decision = local_router_decision(query, candidates, spec, max_tokens=max_tokens) if candidates else None
        predicted = decision.predicted_label if decision else ""
        hit = labels_match(predicted, gold)
        correct += int(hit)
        total_latency += decision.latency_seconds if decision else 0.0
        rows.append(
            {
                "query_id": qid,
                "query_text": query,
                "ground_truth_domain": gold,
                "predicted_domain": predicted,
                "routed_correctly": hit,
                "latency_seconds": decision.latency_seconds if decision else 0.0,
                "raw_response": decision.raw_response if decision else "",
                "parse_ok": infer_parse_ok(decision.raw_response if decision else "", candidate_labels),
                "option_scores_logprob": decision.option_scores_logprob if decision else None,
                "option_margin_logprob": decision.option_margin_logprob if decision else None,
                "answer_avg_logprob": decision.answer_avg_logprob if decision else None,
                "top_candidates": candidate_labels,
                "historical_top1": candidate_labels[0] if candidate_labels else None,
                "historical_top1_correct": bool(candidate_labels and labels_match(candidate_labels[0], gold)),
            }
        )
    n = len(rows)
    return {
        "seed": seed,
        "sample_count": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "mean_latency_seconds": total_latency / n if n else 0.0,
        "historical_top1_accuracy": historical_top1_correct / n if n else 0.0,
        "gold_in_candidates_rate": gold_in_candidates / n if n else 0.0,
        "parse_fail_rate": sum(1 for row in rows if not row["parse_ok"]) / n if n else 0.0,
        "rows": rows,
    }


def main() -> None:
    load_dotenv(REPO_ROOT / '.env')
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_json_records(args.sample_file)
    runlog_rows = load_runlog_rows(args.runlog)
    compendium_text = load_compendium_text(args.compendium)
    seeds = parse_seed_list(args.seeds)
    spec = RouterSpec(
        label="llama31_8b_historical_candidates",
        backend="local",
        model=args.model_path,
        local_model_path=args.model_path,
    )

    results = [
        evaluate_seed(
            records=records,
            runlog_rows=runlog_rows,
            compendium_text=compendium_text,
            seed=seed,
            sample_count=args.sample_count,
            spec=spec,
            max_tokens=args.max_tokens,
        )
        for seed in seeds
    ]
    out_dir = args.output_dir / spec.label
    out_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        (out_dir / f"routing_seed_{result['seed']}.json").write_text(json.dumps(result, indent=2), encoding='utf-8')

    accuracies = [float(result['accuracy']) for result in results]
    latencies = [float(result['mean_latency_seconds']) for result in results]
    hist_top1 = [float(result['historical_top1_accuracy']) for result in results]
    gold_cover = [float(result['gold_in_candidates_rate']) for result in results]
    parse_rates = [float(result['parse_fail_rate']) for result in results]
    summary = {
        'sample_file': str(args.sample_file),
        'runlog': str(args.runlog),
        'compendium': str(args.compendium),
        'sample_count': args.sample_count,
        'seeds': seeds,
        'max_tokens': args.max_tokens,
        'label': spec.label,
        'backend': spec.backend,
        'model': spec.model,
        'local_model_path': spec.local_model_path,
        'mean_accuracy': sum(accuracies) / len(accuracies),
        'sd_accuracy': statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
        'mean_latency_seconds': sum(latencies) / len(latencies),
        'sd_latency_seconds': statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        'mean_historical_top1_accuracy': sum(hist_top1) / len(hist_top1),
        'mean_gold_in_candidates_rate': sum(gold_cover) / len(gold_cover),
        'mean_parse_fail_rate': sum(parse_rates) / len(parse_rates),
        'per_seed_accuracy': {str(result['seed']): result['accuracy'] for result in results},
        'per_seed_historical_top1_accuracy': {str(result['seed']): result['historical_top1_accuracy'] for result in results},
        'per_seed_gold_in_candidates_rate': {str(result['seed']): result['gold_in_candidates_rate'] for result in results},
        'per_seed_parse_fail_rate': {str(result['seed']): result['parse_fail_rate'] for result in results},
        'output_dir': str(out_dir),
        'note': 'Local 8B reranking over preserved historical runlog candidate lists with compendium-backed contexts and label fallbacks for auxiliary historical labels.',
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (args.output_dir / 'combined_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
