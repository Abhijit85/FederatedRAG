#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
TAU_ROOT = REPO_ROOT / 'external_datasets' / 'tau_bench' / 'tau-bench-main'
DEFAULT_LOG_DIR = REPO_ROOT / 'external_datasets' / 'tau_bench' / 'runs_synapse_retail'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the SYNAPSE retail bridge on tau-bench.')
    parser.add_argument('--model', type=str, default=os.environ.get('LOCAL_TAU_MODEL', 'Qwen/Qwen2.5-7B-Instruct'))
    parser.add_argument('--model-provider', type=str, default=os.environ.get('LOCAL_TAU_PROVIDER', 'local-hf'))
    parser.add_argument('--user-model', type=str, default=os.environ.get('LOCAL_TAU_USER_MODEL', 'Qwen/Qwen2.5-7B-Instruct'))
    parser.add_argument('--user-model-provider', type=str, default=os.environ.get('LOCAL_TAU_USER_PROVIDER', 'local-hf'))
    parser.add_argument('--task-split', type=str, default='test', choices=['train', 'dev', 'test'])
    parser.add_argument('--task-ids', type=int, nargs='*')
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--end-index', type=int, default=-1)
    parser.add_argument('--max-concurrency', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--log-dir', type=Path, default=DEFAULT_LOG_DIR)
    return parser.parse_args()


def main() -> None:
    load_dotenv(REPO_ROOT / '.env')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(REPO_ROOT) + os.pathsep + str(TAU_ROOT)

    args = parse_args()
    if args.model_provider != 'local-hf' or args.user_model_provider != 'local-hf':
        api_key = env.get('API_KEY') or env.get('OPENROUTER_API_KEY')
        if not api_key:
            raise RuntimeError('API_KEY / OPENROUTER_API_KEY missing')
        env['OPENROUTER_API_KEY'] = api_key
        env['OPENAI_API_KEY'] = api_key
        env['OPENAI_API_BASE'] = env.get('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
        env['OPENAI_BASE_URL'] = env['OPENAI_API_BASE']

    cmd = [
        str(REPO_ROOT / '.venv' / 'bin' / 'python'),
        str(TAU_ROOT / 'run.py'),
        '--agent-strategy', 'synapse-tool-calling',
        '--env', 'retail',
        '--model', args.model,
        '--model-provider', args.model_provider,
        '--user-model', args.user_model,
        '--user-model-provider', args.user_model_provider,
        '--user-strategy', 'llm',
        '--task-split', args.task_split,
        '--start-index', str(args.start_index),
        '--end-index', str(args.end_index),
        '--max-concurrency', str(args.max_concurrency),
        '--temperature', str(args.temperature),
        '--log-dir', str(args.log_dir),
    ]
    if args.task_ids:
        cmd.extend(['--task-ids', *[str(task_id) for task_id in args.task_ids]])

    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


if __name__ == '__main__':
    main()
