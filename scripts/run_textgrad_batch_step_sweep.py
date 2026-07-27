#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / 'artifacts' / 'verification' / 'textgrad_table14_sweep'
DEFAULT_MIXED_QUERIES = REPO_ROOT / 'bbh_object_counting_eval_v3.json'
DEFAULT_TASK = 'BBH_object_counting'
DEFAULT_CLIENT_COUNT = 4
DEFAULT_ROUNDS = 1
DEFAULT_AGGREGATE = 'summarization'
DEFAULT_LOCAL_ENGINE = 'hf-local::/mnt/shared/shared_hf_home/hub/models--meta-llama--Llama-3.1-8B-Instruct/local_models/Llama-3.1-8B-Instruct'
DEFAULT_HF_HOME = '/mnt/shared/shared_hf_home'
DEFAULT_GPU = '7'


def parse_int_csv(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(',') if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run logged TextGrad batch×step cells for Table 14 / Section C.4.')
    parser.add_argument('--batches', type=str, default='3')
    parser.add_argument('--steps', type=str, default='1,3,5')
    parser.add_argument('--task', type=str, default=DEFAULT_TASK)
    parser.add_argument('--mixed-queries', type=Path, default=DEFAULT_MIXED_QUERIES)
    parser.add_argument('--client-count', type=int, default=DEFAULT_CLIENT_COUNT)
    parser.add_argument('--rounds', type=int, default=DEFAULT_ROUNDS)
    parser.add_argument('--aggregate-method', type=str, default=DEFAULT_AGGREGATE)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--evaluation-engine', type=str, default=DEFAULT_LOCAL_ENGINE)
    parser.add_argument('--hf-home', type=str, default=DEFAULT_HF_HOME)
    parser.add_argument('--cuda-visible-devices', type=str, default=DEFAULT_GPU)
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def current_commit() -> str | None:
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def load_textgrad_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line.startswith('{'):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and 'timestamp' in payload and 'payload' in payload:
            records.append(payload)
    return records


def select_new_central_record(before_count: int, after_records: list[dict[str, Any]], benchmark: str) -> dict[str, Any] | None:
    for record in after_records[before_count:]:
        if record.get('section') != 'central':
            continue
        payload = record.get('payload', {}) or {}
        if payload.get('mixed_queries') == benchmark:
            return record
    return None


def build_command(args: argparse.Namespace, batch_size: int, max_steps: int, snapshot_path: Path) -> list[str]:
    return [
        str(REPO_ROOT / '.venv' / 'bin' / 'python'),
        '-u',
        'scripts/run_fed_textgrad.py',
        '--task', args.task,
        '--client-count', str(args.client_count),
        '--rounds', str(args.rounds),
        '--aggregate-method', args.aggregate_method,
        '--evaluation-engine', args.evaluation_engine,
        '--batch-size', str(batch_size),
        '--max-steps', str(max_steps),
        '--mixed-queries', str(args.mixed_queries),
        '--output-snapshot', str(snapshot_path),
    ]


def run_cell(args: argparse.Namespace, batch_size: int, max_steps: int, benchmark: str, commit_sha: str | None) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    run_id = started.strftime('%Y%m%dT%H%M%SZ') + f'_b{batch_size}_s{max_steps}'
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = run_dir / 'snapshot.json'
    stdout_path = run_dir / 'stdout.log'
    metadata_path = run_dir / 'metadata.json'
    cmd = build_command(args, batch_size, max_steps, snapshot_path)

    metadata: dict[str, Any] = {
        'run_id': run_id,
        'started_at': started.isoformat(),
        'batch_size': batch_size,
        'max_steps': max_steps,
        'task': args.task,
        'mixed_queries': str(args.mixed_queries),
        'client_count': args.client_count,
        'rounds': args.rounds,
        'aggregate_method': args.aggregate_method,
        'evaluation_engine': args.evaluation_engine,
        'hf_home': args.hf_home,
        'cuda_visible_devices': args.cuda_visible_devices,
        'git_commit': commit_sha,
        'command': cmd,
        'stdout_path': str(stdout_path.relative_to(REPO_ROOT)),
        'snapshot_path': str(snapshot_path.relative_to(REPO_ROOT)),
        'status': 'dry_run' if args.dry_run else 'running',
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    if args.dry_run:
        return metadata

    log_path = REPO_ROOT / 'evaluation_on_textgrad_log.txt'
    before_records = load_textgrad_records(log_path)
    env = os.environ.copy()
    load_dotenv(REPO_ROOT / '.env', override=False)
    env.update({key: value for key, value in os.environ.items() if key in {'API_KEY', 'OPENROUTER_API_KEY', 'MODEL_NAME', 'JINA_API_KEY', 'MONGO_URI'}})
    env['HF_HOME'] = args.hf_home
    env['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    env['TEXTGRAD_HF_LOCAL_FILES_ONLY'] = '1'
    env.setdefault('TEXTGRAD_HF_DTYPE', 'bfloat16')

    with stdout_path.open('w', encoding='utf-8') as fh:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)

    ended = datetime.now(timezone.utc)
    after_records = load_textgrad_records(log_path)
    central_record = select_new_central_record(len(before_records), after_records, benchmark)

    metadata.update({
        'ended_at': ended.isoformat(),
        'returncode': proc.returncode,
        'status': 'ok' if proc.returncode == 0 else 'failed',
    })
    if central_record is not None:
        payload = central_record.get('payload', {}) or {}
        overall = payload.get('overall', {}) or {}
        metadata['central_log_record'] = {
            'timestamp': central_record.get('timestamp'),
            'accuracy': overall.get('accuracy'),
            'correct': overall.get('correct'),
            'total': overall.get('total'),
        }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


def render_summary(results: list[dict[str, Any]]) -> str:
    parts = [
        '### TextGrad Batch-Step Sweep Runs',
        '',
        '| Run ID | Batch | Steps | Status | Logged central acc. | Timestamp |',
        '| --- | ---: | ---: | --- | ---: | --- |',
    ]
    for result in results:
        record = result.get('central_log_record') or {}
        acc = record.get('accuracy')
        acc_text = 'n/a' if acc is None else f'{acc:.3f}'
        parts.append(
            f"| {result['run_id']} | {result['batch_size']} | {result['max_steps']} | {result['status']} | {acc_text} | {record.get('timestamp', 'n/a')} |"
        )
    return '\n'.join(parts) + '\n'


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    batches = parse_int_csv(args.batches)
    steps = parse_int_csv(args.steps)
    benchmark = str(args.mixed_queries)
    commit_sha = current_commit()

    results = []
    for batch_size in batches:
        for max_steps in steps:
            result = run_cell(args, batch_size, max_steps, benchmark, commit_sha)
            results.append(result)
            print(json.dumps({
                'run_id': result['run_id'],
                'batch_size': result['batch_size'],
                'max_steps': result['max_steps'],
                'status': result['status'],
                'central_accuracy': (result.get('central_log_record') or {}).get('accuracy'),
            }))
            if result['status'] == 'failed':
                break

    (args.output_root / 'summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    (args.output_root / 'summary.md').write_text(render_summary(results), encoding='utf-8')


if __name__ == '__main__':
    main()
