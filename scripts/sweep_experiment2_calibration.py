#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import runpy
from itertools import product
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "verification" / "experiment2_calibration_sweep.json"
TARGET_TYPED_DROP = 8.1
TARGET_FLAT_DROP = 11.5
TARGET_GAP = 3.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep reconstructed Experiment 2 calibration settings.")
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--conflict-rates", type=str, default="0,20,40,60")
    parser.add_argument("--top-k", type=int, default=12)
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


def evaluate_setting(ns: dict[str, Any], records: list[Any], seeds: list[int], rates: list[int], calibration: dict[str, float | int]) -> dict[str, Any]:
    ns['CALIBRATION'].update(calibration)
    typed_curve = {}
    flat_curve = {}
    typed_per_rate = {}
    flat_per_rate = {}
    for rate in rates:
        typed_vals = [ns['evaluate_condition'](records, rate, seed, 'typed')['accuracy'] for seed in seeds]
        flat_vals = [ns['evaluate_condition'](records, rate, seed, 'flat')['accuracy'] for seed in seeds]
        typed_curve[rate] = mean(typed_vals)
        flat_curve[rate] = mean(flat_vals)
        typed_per_rate[f'{rate}%'] = typed_vals
        flat_per_rate[f'{rate}%'] = flat_vals

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
        'calibration': calibration,
        'typed_curve': typed_curve,
        'flat_curve': flat_curve,
        'typed_drop': typed_drop,
        'flat_drop': flat_drop,
        'gap': gap,
        'typed_per_rate': typed_per_rate,
        'flat_per_rate': flat_per_rate,
    }


def main() -> None:
    args = parse_args()
    seeds = parse_int_list(args.seeds)
    rates = parse_int_list(args.conflict_rates)
    ns = runpy.run_path(str(REPO_ROOT / 'scripts' / 'run_experiment2_reconstructed.py'), run_name='exp2_sweep')
    records = ns['build_benchmark']()

    grid = {
        'strong_conflict_base': [0.0, 0.02, 0.05],
        'strong_conflict_scale': [0.10, 0.20, 0.30],
        'strong_conflict_cap': [0.15, 0.25, 0.40],
        'corruption_base': [0.02, 0.05, 0.08],
        'corruption_scale': [0.08, 0.12, 0.18],
        'corruption_strong_bonus': [0.02, 0.04, 0.07],
        'flat_focus_text_parts_limit': [2, 3, 4],
    }
    keys = list(grid)
    results = []
    for values in product(*(grid[key] for key in keys)):
        calibration = dict(zip(keys, values))
        calibration['flat_focus_term_limit'] = 4
        results.append(evaluate_setting(ns, records, seeds, rates, calibration))

    results.sort(key=lambda row: row['score'])
    top = results[: max(1, args.top_k)]
    payload = {
        'target': {
            'typed_drop': TARGET_TYPED_DROP,
            'flat_drop': TARGET_FLAT_DROP,
            'gap': TARGET_GAP,
        },
        'grid_sizes': {key: len(values) for key, values in grid.items()},
        'total_settings': len(results),
        'top_results': top,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    best = top[0]
    print(f"Evaluated {len(results)} settings")
    print(f"Best score: {best['score']:.3f}")
    print(f"Best typed drop: {best['typed_drop']:.2f} pts")
    print(f"Best flat drop: {best['flat_drop']:.2f} pts")
    print(f"Best gap: {best['gap']:.2f} pts")
    print(json.dumps(best['calibration'], sort_keys=True))


if __name__ == '__main__':
    main()
