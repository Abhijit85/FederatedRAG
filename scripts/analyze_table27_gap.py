#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{path} has no rows list")
    return [row for row in rows if isinstance(row, dict)]


def key(row: dict[str, Any]) -> str:
    return str(row.get("query_id") or "")


def normalize(label: str | None) -> str:
    return (label or "").strip().lower()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare per-query routing behavior between two Table 27 arms.")
    parser.add_argument("--runtime", type=Path, required=True, help="routing_seed_*.json from runtime arm")
    parser.add_argument("--reference", type=Path, required=True, help="routing_seed_*.json from reference arm")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    runtime_rows = {key(row): row for row in load_rows(args.runtime)}
    ref_rows = {key(row): row for row in load_rows(args.reference)}
    shared = sorted(set(runtime_rows) & set(ref_rows))
    if not shared:
        raise SystemExit("No shared query_ids between the two inputs.")

    total = len(shared)
    runtime_correct = 0
    ref_correct = 0
    oracle_top5 = 0
    ref_label_in_runtime_top5 = 0
    both_correct = 0
    runtime_only = 0
    ref_only = 0
    both_wrong = 0
    missed_gold_labels: Counter[str] = Counter()
    runtime_pred_labels: Counter[str] = Counter()

    sample_ref_only: list[str] = []

    for qid in shared:
        runtime = runtime_rows[qid]
        ref = ref_rows[qid]
        runtime_hit = bool(runtime.get("routed_correctly"))
        ref_hit = bool(ref.get("routed_correctly"))
        gold = str(runtime.get("ground_truth_domain") or ref.get("ground_truth_domain") or "")
        runtime_pred = str(runtime.get("predicted_domain") or "")
        ref_pred = str(ref.get("predicted_domain") or "")
        top5 = [str(item) for item in (runtime.get("top_candidates") or [])]

        runtime_correct += int(runtime_hit)
        ref_correct += int(ref_hit)
        runtime_pred_labels[runtime_pred] += 1

        if normalize(gold) in {normalize(item) for item in top5}:
            oracle_top5 += 1
        else:
            missed_gold_labels[gold] += 1

        if normalize(ref_pred) in {normalize(item) for item in top5}:
            ref_label_in_runtime_top5 += 1

        if runtime_hit and ref_hit:
            both_correct += 1
        elif runtime_hit and not ref_hit:
            runtime_only += 1
        elif not runtime_hit and ref_hit:
            ref_only += 1
            if len(sample_ref_only) < 15:
                sample_ref_only.append(
                    f"{qid} | gold={gold} | runtime={runtime_pred} | ref={ref_pred} | top5={top5}"
                )
        else:
            both_wrong += 1

    print("=== gap summary ===")
    print(f"shared queries: {total}")
    print(f"runtime accuracy:   {runtime_correct}/{total} = {runtime_correct/total:.3f}")
    print(f"reference accuracy: {ref_correct}/{total} = {ref_correct/total:.3f}")
    print()
    print("=== overlap ===")
    print(f"both correct: {both_correct}")
    print(f"runtime only: {runtime_only}")
    print(f"reference only: {ref_only}")
    print(f"both wrong: {both_wrong}")
    print()
    print("=== runtime top-k headroom ===")
    print(f"gold in runtime top5: {oracle_top5}/{total} = {oracle_top5/total:.3f}")
    print(
        f"reference predicted label in runtime top5: "
        f"{ref_label_in_runtime_top5}/{total} = {ref_label_in_runtime_top5/total:.3f}"
    )
    print()
    print("=== runtime label skew ===")
    for label, count in runtime_pred_labels.most_common(10):
        print(f"{label}: {count}")
    print()
    print("=== most-missed gold labels (runtime top5 miss) ===")
    for label, count in missed_gold_labels.most_common(10):
        print(f"{label}: {count}")
    print()
    print("=== sample reference-only wins ===")
    for line in sample_ref_only:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
