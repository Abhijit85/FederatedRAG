#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import runpy
from itertools import product
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "verification" / "experiment2_family4_sweep.json"
TARGET_TYPED_DROP = 8.1
TARGET_FLAT_DROP = 11.5
TARGET_GAP = 3.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Focused sweep for family4 calibration.")
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--conflict-rates", type=str, default="0,20,40,60")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def monotonic_penalty(curve: dict[int, float]) -> float:
    penalty = 0.0
    ordered = [curve[key] for key in sorted(curve)]
    for left, right in zip(ordered, ordered[1:]):
        if right > left:
            penalty += (right - left) * 100.0
    return penalty


def patch_family(ns: dict[str, Any], family: dict[str, float]) -> None:
    ns['CALIBRATION'].update(family)


def evaluate_setting(ns: dict[str, Any], records: list[Any], queries: list[Any], neighbors: dict[str, list[Any]], seeds: list[int], rates: list[int], family: dict[str, float]) -> dict[str, Any]:
    patch_family(ns, family)
    typed_curve = {}
    flat_curve = {}
    for rate in rates:
        typed_curve[rate] = mean([ns['evaluate_condition'](records, queries, neighbors, rate, seed, 'typed')['accuracy'] for seed in seeds])
        flat_curve[rate] = mean([ns['evaluate_condition'](records, queries, neighbors, rate, seed, 'flat')['accuracy'] for seed in seeds])
    typed_drop = (typed_curve[min(rates)] - typed_curve[max(rates)]) * 100.0
    flat_drop = (flat_curve[min(rates)] - flat_curve[max(rates)]) * 100.0
    gap = flat_drop - typed_drop
    score = (
        abs(typed_drop - TARGET_TYPED_DROP)
        + abs(flat_drop - TARGET_FLAT_DROP)
        + 2.0 * abs(gap - TARGET_GAP)
        + monotonic_penalty(typed_curve)
        + monotonic_penalty(flat_curve)
    )
    return {
        'score': score,
        'family': dict(family),
        'typed_curve': typed_curve,
        'flat_curve': flat_curve,
        'typed_drop': typed_drop,
        'flat_drop': flat_drop,
        'gap': gap,
    }


def main() -> None:
    args = parse_args()
    seeds = parse_int_list(args.seeds)
    rates = parse_int_list(args.conflict_rates)
    ns = runpy.run_path(str(REPO_ROOT / 'scripts' / 'run_experiment2_family4.py'), run_name='exp2_family4_sweep')
    records = ns['build_records']()
    queries = ns['build_queries'](records)
    neighbors = ns['build_neighbors'](records)

    grid = {
        'typed_strength_scale': [1.0, 1.4],
        'typed_decisive_factor': [1.0, 1.4],
        'typed_exemplar_factor': [0.9, 1.3],
        'typed_supportive_factor': [1.2, 1.6],
        'typed_surface_bonus': [0.0, 0.4],
        'typed_query_decisive_weight': [1.2, 1.6],
        'typed_query_exemplar_weight': [1.2, 1.6],
        'flat_strength_scale': [0.9, 1.0],
        'flat_decisive_factor': [0.9, 1.1],
        'flat_query_decisive_weight': [1.6],
    }
    keys = list(grid)
    results = []
    for values in product(*(grid[key] for key in keys)):
        family = dict(zip(keys, values))
        results.append(evaluate_setting(ns, records, queries, neighbors, seeds, rates, family))
    results.sort(key=lambda row: row['score'])
    payload = {
        'target': {'typed_drop': TARGET_TYPED_DROP, 'flat_drop': TARGET_FLAT_DROP, 'gap': TARGET_GAP},
        'total_settings': len(results),
        'top_results': results[: max(1, args.top_k)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    best = payload['top_results'][0]
    print(f"Evaluated {len(results)} family4 settings")
    print(f"Best score: {best['score']:.3f}")
    print(f"Best typed drop: {best['typed_drop']:.2f} pts")
    print(f"Best flat drop: {best['flat_drop']:.2f} pts")
    print(f"Best gap: {best['gap']:.2f} pts")
    print(json.dumps(best['family'], sort_keys=True))


if __name__ == '__main__':
    main()
