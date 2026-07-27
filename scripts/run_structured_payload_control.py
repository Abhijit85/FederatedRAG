#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_routing_verification import (
    DEFAULT_SAMPLE_FILE,
    build_credentials,
    evaluate_seed,
    load_records,
)
from synapse.runtime import SynapseRuntime

DEFAULT_OUTPUT_DIR = REPO_ROOT / 'artifacts' / 'verification' / 'structured_payload_control'


def parse_seed_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(',') if part.strip()]


def slugify_condition(name: str) -> str:
    return name.strip().lower().replace('-', '_').replace(' ', '_')


@contextmanager
def temporary_structured_payload_mode(mode: str) -> Iterator[None]:
    key = 'SYNAPSE_STRUCTURED_PAYLOAD_MODE'
    previous = os.environ.get(key)
    os.environ[key] = mode
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run typed-vs-untyped structured-payload routing controls over shared seeds.'
    )
    parser.add_argument('--sample-file', type=Path, default=DEFAULT_SAMPLE_FILE)
    parser.add_argument('--sample-count', type=int, default=50)
    parser.add_argument('--seeds', type=str, default='1,2,3,4,5')
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--client-count', type=int, default=None)
    parser.add_argument('--max-items', type=int, default=5)
    parser.add_argument('--conditions', type=str, default='typed,untyped')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def build_runtime(rounds: int, client_count: int | None) -> SynapseRuntime:
    runtime = SynapseRuntime.build_local_runtime(REPO_ROOT, build_credentials(), client_count=client_count)
    for _ in range(max(1, rounds)):
        runtime.run_round()
    return runtime


def summarize_results(results: list[dict[str, Any]]) -> tuple[float, float]:
    accuracies = [float(result['accuracy']) for result in results]
    mean_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    sd_accuracy = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
    return mean_accuracy, sd_accuracy


def run_condition(
    condition: str,
    records: list[dict[str, Any]],
    seeds: list[int],
    sample_count: int,
    rounds: int,
    client_count: int | None,
    max_items: int,
    output_dir: Path,
) -> dict[str, Any]:
    condition_slug = slugify_condition(condition)
    condition_dir = output_dir / condition_slug
    condition_dir.mkdir(parents=True, exist_ok=True)

    with temporary_structured_payload_mode(condition):
        runtime = build_runtime(rounds=rounds, client_count=client_count)
        results = [
            evaluate_seed(
                runtime=runtime,
                records=records,
                seed=seed,
                sample_count=sample_count,
                max_items=max_items,
            )
            for seed in seeds
        ]

    mean_accuracy, sd_accuracy = summarize_results(results)
    for result in results:
        out_path = condition_dir / f"routing_seed_{result['seed']}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding='utf-8')

    summary = {
        'structured_payload_mode': condition,
        'sample_file': str(args.sample_file),
        'sample_count': sample_count,
        'rounds': rounds,
        'client_count': client_count,
        'seeds': seeds,
        'mean_accuracy': mean_accuracy,
        'sd_accuracy': sd_accuracy,
        'per_seed_accuracy': {str(result['seed']): result['accuracy'] for result in results},
        'output_dir': str(condition_dir),
    }
    (condition_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def build_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {row['structured_payload_mode']: row for row in rows}
    typed = indexed.get('typed')
    comparisons: list[dict[str, Any]] = []
    if not typed:
        return comparisons
    typed_seeds = typed.get('per_seed_accuracy', {})
    for name, row in indexed.items():
        if name == 'typed':
            continue
        deltas: dict[str, float] = {}
        for seed, typed_value in typed_seeds.items():
            other = row.get('per_seed_accuracy', {}).get(seed)
            if isinstance(other, (int, float)):
                deltas[seed] = float(typed_value) - float(other)
        comparisons.append({
            'baseline': 'typed',
            'variant': name,
            'typed_minus_variant_mean_delta': typed['mean_accuracy'] - row['mean_accuracy'],
            'typed_minus_variant_per_seed_delta': deltas,
        })
    return comparisons


def render_markdown(rows: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> str:
    parts = [
        '### Structured Payload Control',
        '',
        '| Condition | Seeds | Mean routing acc. | SD |',
        '| --- | --- | ---: | ---: |',
    ]
    for row in rows:
        seed_values = ', '.join(f"{int(seed)}={value:.3f}" for seed, value in row['per_seed_accuracy'].items())
        parts.append(
            f"| {row['structured_payload_mode']} | {seed_values} | {row['mean_accuracy']:.3f} | {row['sd_accuracy']:.3f} |"
        )
    if comparisons:
        parts.extend([
            '',
            '| Comparison | Mean delta | Per-seed deltas |',
            '| --- | ---: | --- |',
        ])
        for row in comparisons:
            delta_str = ', '.join(f"{seed}={value:+.3f}" for seed, value in row['typed_minus_variant_per_seed_delta'].items())
            parts.append(
                f"| typed - {row['variant']} | {row['typed_minus_variant_mean_delta']:+.3f} | {delta_str} |"
            )
    return '\n'.join(parts) + '\n'


if __name__ == '__main__':
    load_dotenv()
    args = parse_args()
    records = load_records(args.sample_file)
    seeds = parse_seed_list(args.seeds)
    conditions = [part.strip() for part in args.conditions.split(',') if part.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        run_condition(
            condition=condition,
            records=records,
            seeds=seeds,
            sample_count=args.sample_count,
            rounds=args.rounds,
            client_count=args.client_count,
            max_items=args.max_items,
            output_dir=args.output_dir,
        )
        for condition in conditions
    ]
    comparisons = build_comparison(rows)
    combined = {
        'sample_file': str(args.sample_file),
        'sample_count': args.sample_count,
        'rounds': args.rounds,
        'client_count': args.client_count,
        'seeds': seeds,
        'conditions': rows,
        'comparisons': comparisons,
    }
    (args.output_dir / 'combined_summary.json').write_text(json.dumps(combined, indent=2), encoding='utf-8')
    (args.output_dir / 'summary.md').write_text(render_markdown(rows, comparisons), encoding='utf-8')

    for row in rows:
        print(
            f"{row['structured_payload_mode']}: "
            f"mean={row['mean_accuracy']:.3f}, sd={row['sd_accuracy']:.3f}, "
            f"seeds={row['per_seed_accuracy']}"
        )
