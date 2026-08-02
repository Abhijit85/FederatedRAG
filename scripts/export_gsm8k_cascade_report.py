#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a completed GSM8K cascade run into three reporting layers: per-query JSONL, "
            "per-seed aggregates, and a summary grid suitable for rebuttal tables."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, sd


def safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return num / den


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_threshold(value: float | None) -> str:
    return "-" if value is None else f"{value:g}"


def canonicalize_label(text: str | None) -> str:
    if not text:
        return ""
    value = str(text).strip().lower()
    value = re.sub(r"^\s*[a-e0-9]+[\.:\)\-\s]+", "", value)
    value = value.replace("&", "and")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def infer_parse_ok(parsed_label: str | None, candidates: list[str]) -> bool | None:
    if not parsed_label:
        return False
    parsed_norm = canonicalize_label(parsed_label)
    if not parsed_norm:
        return False
    candidate_norms = [canonicalize_label(candidate) for candidate in candidates]
    if parsed_norm in candidate_norms:
        return True
    for candidate_norm in candidate_norms:
        if candidate_norm and (candidate_norm in parsed_norm or parsed_norm in candidate_norm):
            return True
    return False


def label_matches_top1(final_label: str | None, candidates: list[str]) -> bool:
    if not final_label or not candidates:
        return False
    return canonicalize_label(final_label) == canonicalize_label(candidates[0])


def collect_baseline_seed_rows(run_dir: Path, arm_label: str, arm_kind: str) -> list[dict[str, Any]]:
    arm_dir = run_dir / "baselines" / arm_label
    rows: list[dict[str, Any]] = []
    if not arm_dir.exists():
        return rows
    for path in sorted(arm_dir.glob("routing_seed_*.json")):
        payload = read_json(path)
        seed = int(payload["seed"])
        for row in payload.get("rows", []):
            candidates = row.get("top_candidates") or []
            final_label = row.get("predicted_domain")
            gold_label = row.get("ground_truth_domain")
            gold_position = None
            if gold_label in candidates:
                gold_position = candidates.index(gold_label) + 1
            rows.append(
                {
                    "seed": seed,
                    "arm": arm_kind,
                    "threshold": None,
                    "query_id": row.get("query_id"),
                    "candidates": candidates,
                    "gold_label": gold_label,
                    "gold_position": gold_position,
                    "tier1_raw_generation": row.get("raw_response"),
                    "tier1_parsed_label": final_label if arm_kind == "1b_only" else None,
                    "tier1_parse_ok": infer_parse_ok(final_label, candidates) if arm_kind == "1b_only" else None,
                    "tier1_seq_scores": row.get("option_scores_logprob"),
                    "tier1_margin": row.get("option_margin_logprob"),
                    "deferred": None,
                    "tier2_raw_generation": row.get("raw_response") if arm_kind == "8b_only" else None,
                    "tier2_parsed_label": final_label if arm_kind == "8b_only" else None,
                    "tier2_parse_ok": infer_parse_ok(final_label, candidates) if arm_kind == "8b_only" else None,
                    "final_label": final_label,
                    "correct": bool(row.get("routed_correctly")),
                    "latency_s": {
                        "tier1": row.get("latency_seconds") if arm_kind == "1b_only" else None,
                        "tier2": row.get("latency_seconds") if arm_kind == "8b_only" else None,
                    },
                }
            )
    return rows


def collect_cascade_seed_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in summary.get("results", []):
        threshold = result.get("threshold")
        for seed_result in result.get("seed_results", []):
            seed = int(seed_result["seed"])
            for row in seed_result.get("rows", []):
                candidates = row.get("top_candidates") or []
                gold_label = row.get("ground_truth_domain")
                gold_position = None
                if gold_label in candidates:
                    gold_position = candidates.index(gold_label) + 1
                rows.append(
                    {
                        "seed": seed,
                        "arm": "cascade",
                        "threshold": threshold,
                        "query_id": row.get("query_id"),
                        "candidates": candidates,
                        "gold_label": gold_label,
                        "gold_position": gold_position,
                        "tier1_raw_generation": row.get("small_raw_response"),
                        "tier1_parsed_label": row.get("small_predicted_domain"),
                        "tier1_parse_ok": row.get("small_parse_ok") if row.get("small_parse_ok") is not None else infer_parse_ok(row.get("small_predicted_domain"), candidates),
                        "tier1_seq_scores": row.get("small_option_scores_logprob"),
                        "tier1_margin": row.get("confidence_value"),
                        "deferred": bool(row.get("deferred_to_large")),
                        "tier2_raw_generation": row.get("large_raw_response"),
                        "tier2_parsed_label": row.get("large_predicted_domain"),
                        "tier2_parse_ok": infer_parse_ok(row.get("large_predicted_domain"), candidates) if row.get("deferred_to_large") else None,
                        "final_label": row.get("predicted_domain"),
                        "correct": bool(row.get("routed_correctly")),
                        "latency_s": {
                            "tier1": row.get("small_latency_seconds"),
                            "tier2": row.get("large_latency_seconds") if row.get("deferred_to_large") else None,
                        },
                    }
                )
    return rows


def write_jsonl(path: Path, records: list[dict[str, Any]], run_id: str, summary: dict[str, Any]) -> None:
    config = {
        "run_id": run_id,
        "sample_file": summary.get("sample_file"),
        "sample_count": summary.get("sample_count"),
        "seeds": summary.get("seeds"),
        "small_label": summary.get("small_label"),
        "large_label": summary.get("large_label"),
        "thresholds": summary.get("thresholds"),
        "confidence_signal": summary.get("confidence_signal"),
        "embed_model": summary.get("embed_model"),
        "k": summary.get("k"),
        "rounds": summary.get("rounds"),
        "client_count": summary.get("client_count"),
        "small_model_billions": summary.get("small_model_billions"),
        "large_model_billions": summary.get("large_model_billions"),
        "note": "Header config line for auditability. Subsequent lines are per-query records.",
    }
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"record_type": "config", **config}) + "\n")
        for record in records:
            fh.write(json.dumps({"record_type": "query", "run_id": run_id, **record}) + "\n")


def per_seed_aggregate(records: list[dict[str, Any]], arm: str, threshold: float | None) -> dict[str, Any]:
    n = len(records)
    correct = sum(int(r.get("correct")) for r in records)
    deferred_rows = [r for r in records if r.get("deferred") is True]
    kept_rows = [r for r in records if r.get("deferred") is False]
    if arm == "8b_only":
        parse_fail_count = sum(1 for r in records if r.get("tier2_parse_ok") is False)
    elif arm == "cascade":
        parse_fail_count = sum(
            1
            for r in records
            if (r.get("deferred") is True and r.get("tier2_parse_ok") is False)
            or (r.get("deferred") is False and r.get("tier1_parse_ok") is False)
        )
    else:
        parse_fail_count = sum(1 for r in records if r.get("tier1_parse_ok") is False)
    final_top1 = 0
    for r in records:
        if label_matches_top1(r.get("final_label"), r.get("candidates") or []):
            final_top1 += 1
    kept_latencies = [float(r["latency_s"]["tier1"]) for r in kept_rows if isinstance(r["latency_s"].get("tier1"), (int, float))]
    deferred_latencies = []
    for r in deferred_rows:
        tier1 = r["latency_s"].get("tier1")
        tier2 = r["latency_s"].get("tier2")
        total = 0.0
        ok = False
        if isinstance(tier1, (int, float)):
            total += float(tier1)
            ok = True
        if isinstance(tier2, (int, float)):
            total += float(tier2)
            ok = True
        if ok:
            deferred_latencies.append(total)
    return {
        "arm": arm,
        "threshold": threshold,
        "n_queries": n,
        "accuracy": safe_div(correct, n),
        "deferral_rate": safe_div(len(deferred_rows), n) if arm == "cascade" else None,
        "kept_acc": safe_div(sum(int(r.get("correct")) for r in kept_rows), len(kept_rows)) if arm == "cascade" else None,
        "deferred_acc": safe_div(sum(int(r.get("correct")) for r in deferred_rows), len(deferred_rows)) if arm == "cascade" else None,
        "parse_fail_rate": safe_div(parse_fail_count, n),
        "top1_pick_rate": safe_div(final_top1, n),
        "mean_latency_kept_s": sum(kept_latencies) / len(kept_latencies) if kept_latencies else None,
        "mean_latency_deferred_s": sum(deferred_latencies) / len(deferred_latencies) if deferred_latencies else None,
        "eight_b_invocation": 1.0 if arm == "8b_only" else (0.0 if arm == "1b_only" else safe_div(len(deferred_rows), n)),
        "flops_vs_8b": 1.0 if arm == "8b_only" else (0.125 if arm == "1b_only" else (0.125 * (1.0 - safe_div(len(deferred_rows), n)) + 1.0 * safe_div(len(deferred_rows), n))),
        "latency_query_s": (
            (sum(kept_latencies) + sum(deferred_latencies)) / (len(kept_latencies) + len(deferred_latencies))
            if (kept_latencies or deferred_latencies)
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else run_dir / "report_layers"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = read_json(run_dir / "summary.json")
    records: list[dict[str, Any]] = []
    records.extend(collect_baseline_seed_rows(run_dir, summary.get("small_label", "llama32_1b"), "1b_only"))
    records.extend(collect_baseline_seed_rows(run_dir, summary.get("large_label", "llama31_8b"), "8b_only"))
    records.extend(collect_cascade_seed_rows(summary))

    jsonl_path = output_dir / "per_query_log.jsonl"
    write_jsonl(jsonl_path, records, args.run_id, summary)

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for record in records:
        key = (record["arm"], normalize_threshold(record.get("threshold")), int(record["seed"]))
        grouped.setdefault(key, []).append(record)

    seed_rows = []
    for (arm, threshold_s, seed), group_records in sorted(grouped.items()):
        agg = per_seed_aggregate(group_records, arm=arm, threshold=(None if threshold_s == "-" else float(threshold_s)))
        agg["seed"] = seed
        seed_rows.append(agg)

    seed_csv = output_dir / "per_seed_aggregates.csv"
    seed_fields = [
        "arm", "threshold", "seed", "n_queries", "accuracy", "deferral_rate", "kept_acc", "deferred_acc",
        "parse_fail_rate", "top1_pick_rate", "mean_latency_kept_s", "mean_latency_deferred_s", "eight_b_invocation",
        "flops_vs_8b", "latency_query_s",
    ]
    with seed_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=seed_fields)
        writer.writeheader()
        for row in seed_rows:
            out = dict(row)
            out["threshold"] = normalize_threshold(row.get("threshold"))
            writer.writerow(out)

    summary_rows = []
    grouped_summary: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in seed_rows:
        key = (row["arm"], normalize_threshold(row.get("threshold")))
        grouped_summary.setdefault(key, []).append(row)
    for (arm, threshold_s), rows in sorted(grouped_summary.items()):
        def collect(name: str) -> list[float]:
            return [float(r[name]) for r in rows if isinstance(r.get(name), (int, float))]
        def mean_or_none(name: str) -> float | None:
            vals = collect(name)
            return sum(vals) / len(vals) if vals else None
        entry = {
            "arm": arm,
            "threshold": threshold_s,
            "deferral_mean": mean_or_none("deferral_rate"),
            "deferral_sd": mean_sd(collect("deferral_rate"))[1] if collect("deferral_rate") else None,
            "kept_acc_mean": mean_or_none("kept_acc"),
            "kept_acc_sd": mean_sd(collect("kept_acc"))[1] if collect("kept_acc") else None,
            "overall_acc_mean": mean_or_none("accuracy"),
            "overall_acc_sd": mean_sd(collect("accuracy"))[1] if collect("accuracy") else None,
            "eight_b_invocation_mean": mean_or_none("eight_b_invocation"),
            "flops_vs_8b_mean": mean_or_none("flops_vs_8b"),
            "latency_query_mean": mean_or_none("latency_query_s"),
        }
        summary_rows.append(entry)

    summary_csv = output_dir / "summary_grid.csv"
    summary_fields = [
        "arm", "threshold", "deferral_mean", "deferral_sd", "kept_acc_mean", "kept_acc_sd",
        "overall_acc_mean", "overall_acc_sd", "eight_b_invocation_mean", "flops_vs_8b_mean", "latency_query_mean",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    baselines = {row.get("label"): row for row in summary.get("baselines", [])}
    validation = {
        "8B endpoint": f"acc mean±SD across {len(summary.get('seeds', []))} seeds = {baselines.get(summary.get('large_label'), {}).get('mean_accuracy')} ± {baselines.get(summary.get('large_label'), {}).get('sd_accuracy')}",
        "1B endpoint": f"acc mean±SD = {baselines.get(summary.get('small_label'), {}).get('mean_accuracy')} ± {baselines.get(summary.get('small_label'), {}).get('sd_accuracy')}",
        "8B top1_pick_rate": None,
        "parse_fail_rate_1b_only": None,
        "parse_fail_rate_8b_only": None,
        "10 raw 8B generations eyeballed": "Not computed by exporter",
    }
    endpoint_rows_8b = [r for r in seed_rows if r["arm"] == "8b_only"]
    endpoint_rows_1b = [r for r in seed_rows if r["arm"] == "1b_only"]
    if endpoint_rows_8b:
        vals = [float(r["top1_pick_rate"]) for r in endpoint_rows_8b if isinstance(r.get("top1_pick_rate"), (int, float))]
        validation["8B top1_pick_rate"] = sum(vals) / len(vals) if vals else None
        vals = [float(r["parse_fail_rate"]) for r in endpoint_rows_8b if isinstance(r.get("parse_fail_rate"), (int, float))]
        validation["parse_fail_rate_8b_only"] = sum(vals) / len(vals) if vals else None
    if endpoint_rows_1b:
        vals = [float(r["parse_fail_rate"]) for r in endpoint_rows_1b if isinstance(r.get("parse_fail_rate"), (int, float))]
        validation["parse_fail_rate_1b_only"] = sum(vals) / len(vals) if vals else None
    (output_dir / "validation_block.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

    print(f"per_query_log={jsonl_path}")
    print(f"per_seed_aggregates={seed_csv}")
    print(f"summary_grid={summary_csv}")
    print(f"validation_block={output_dir / 'validation_block.json'}")


if __name__ == "__main__":
    main()
