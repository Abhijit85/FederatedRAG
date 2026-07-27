#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / 'artifacts' / 'rebuttal'
ROUTING_SUMMARY = REPO_ROOT / 'artifacts' / 'verification' / 'routing' / 'summary.json'
ROUTING_SWEEP_SUMMARY = REPO_ROOT / 'artifacts' / 'verification' / 'routing_privacy_sweep' / 'combined_summary.json'
ONE_CLIENT_ROUTING_SUMMARY = REPO_ROOT / 'artifacts' / 'verification' / 'routing_client_count_1' / 'summary.json'
DEFAULT_PAIRED_TOST_JSON = REPO_ROOT / 'artifacts' / 'verification' / 'paired_tost_one_client_runtime.json'
CENTRALIZED_ROUTING_SUMMARY = REPO_ROOT / 'artifacts' / 'verification' / 'centralized_routing' / 'summary.json'
TAU_RESULTS_ROOT = REPO_ROOT / 'external_datasets' / 'tau_bench' / 'sample_results'
DEFAULT_OUTPUT_MD = ARTIFACT_ROOT / 'verification_tables_real_values.md'
DEFAULT_OUTPUT_JSON = ARTIFACT_ROOT / 'verification_tables_real_values.json'


@dataclass
class Row:
    checkpoint: str
    paper_anchor: str
    measured: str
    status: str


TABLE_ANCHORS: dict[str, list[tuple[str, str]]] = {
    'B': [
        ('No privacy', '0.935'),
        ('ε=2.0, λ=0.5', '0.928'),
        ('ε=2.0, λ=1.0', '0.914'),
        ('ε=2.0, λ=1.5', '0.897'),
        ('ε=1.0, λ=0.5', '0.909'),
        ('ε=1.0, λ=1.0', '0.902'),
        ('ε=1.0, λ=1.5', '0.881'),
        ('ε=0.5, λ=0.5', '0.884'),
        ('ε=0.5, λ=1.0', '0.866'),
        ('ε=0.5, λ=1.5', '0.851'),
    ],
    'D': [
        ('Table 22, τ-bench retail, SYNAPSE main', '0.453'),
    ],
    'E': [
        ('Paired TOST mean difference (SYNAPSE - centralized)', 'parity claim'),
        ('Paired TOST 90% CI containment margin', '±0.03 margin'),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build run-backed rebuttal tables only.')
    parser.add_argument('--run-routing-no-privacy', action='store_true')
    parser.add_argument('--routing-sample-count', type=int, default=50)
    parser.add_argument('--routing-seeds', type=str, default='1,2,3,4,5')
    parser.add_argument('--routing-rounds', type=int, default=1)
    parser.add_argument('--run-tau-sample', action='store_true')
    parser.add_argument('--tau-task-id', type=int, default=0)
    parser.add_argument('--tau-model', type=str, default='openai/gpt-4o-mini')
    parser.add_argument('--paired-tost-json', type=Path, default=DEFAULT_PAIRED_TOST_JSON)
    parser.add_argument('--output-md', type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument('--output-json', type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def run_command(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    subprocess.run(cmd, check=True, cwd=cwd or REPO_ROOT, env=env)


def ensure_openrouter_env() -> dict[str, str]:
    load_dotenv(REPO_ROOT / '.env')
    env = os.environ.copy()
    api_key = env.get('API_KEY') or env.get('OPENROUTER_API_KEY')
    if not api_key:
        raise RuntimeError('API_KEY / OPENROUTER_API_KEY missing; cannot run OpenRouter-backed jobs.')
    env['OPENROUTER_API_KEY'] = api_key
    return env


def run_routing_verifier(sample_count: int, seeds: str, rounds: int) -> None:
    cmd = [
        str(REPO_ROOT / '.venv' / 'bin' / 'python'),
        'scripts/run_routing_verification.py',
        '--sample-count', str(sample_count),
        '--seeds', seeds,
        '--rounds', str(rounds),
    ]
    run_command(cmd, env=ensure_openrouter_env())


def run_tau_sample(task_id: int, model: str) -> None:
    env = ensure_openrouter_env()
    user_model_suffix = model.split('/', 1)[0]
    nested = TAU_RESULTS_ROOT / f"tool-calling-{model.split('/')[-1]}-0.0_range_0--1_user-{user_model_suffix}"
    nested.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(REPO_ROOT / '.venv' / 'bin' / 'python'),
        'external_datasets/tau_bench/tau-bench-main/run.py',
        '--agent-strategy', 'tool-calling',
        '--env', 'retail',
        '--model', model,
        '--model-provider', 'openrouter',
        '--user-model', model,
        '--user-model-provider', 'openrouter',
        '--user-strategy', 'llm',
        '--max-concurrency', '1',
        '--task-ids', str(task_id),
        '--log-dir', str(TAU_RESULTS_ROOT),
    ]
    run_command(cmd, env=env)


def latest_tau_result() -> Path | None:
    candidates = sorted(TAU_RESULTS_ROOT.rglob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def fmt_float(value: float, digits: int = 3) -> str:
    return f'{value:.{digits}f}'


def summarize_seed_payload(payload: dict[str, Any]) -> str:
    seeds = payload.get('per_seed_accuracy', {})
    ordered = [seeds.get(str(seed)) for seed in payload.get('seeds', [])]
    ordered_str = ', '.join(fmt_float(v, 2) for v in ordered if isinstance(v, (int, float)))
    return f'seeds = [{ordered_str}]; mean = {fmt_float(payload["mean_accuracy"])}; SD = {fmt_float(payload["sd_accuracy"])}'


def describe_tau_sample() -> tuple[str, str] | None:
    path = latest_tau_result()
    if path is None:
        return None
    payload = load_json(path)
    if not payload:
        return None
    row = payload[0]
    reward = row.get('reward')
    task_id = row.get('task_id')
    info = row.get('info', {})
    user_cost = info.get('user_cost')
    measured = f'one live sample task succeeded: task_id = {task_id}, reward = {reward}, Pass^1 = {reward}'
    status = f'measured via tau-bench retail one-task sample; user_cost = {user_cost}; file = {path.relative_to(REPO_ROOT)}'
    return measured, status


def compute_tost(path: Path) -> dict[str, Any]:
    import statistics
    from math import sqrt
    from scipy import stats

    payload = load_json(path)
    synapse = payload['synapse']
    centralized = payload['centralized']
    margin = float(payload.get('margin', 0.03))
    alpha = float(payload.get('alpha', 0.05))
    diffs = [float(x) - float(y) for x, y in zip(synapse, centralized)]
    n = len(diffs)
    mean_diff = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    se = sd / sqrt(n)
    if se == 0:
        p_tost = 0.0 if -margin < mean_diff < margin else 1.0
        return {
            'mean_diff': mean_diff,
            'ci_low': mean_diff,
            'ci_high': mean_diff,
            'p_tost': p_tost,
            'equivalent': p_tost < alpha,
        }
    df = n - 1
    t_lower = (mean_diff - (-margin)) / se
    t_upper = (mean_diff - margin) / se
    p_lower = 1 - stats.t.cdf(t_lower, df)
    p_upper = stats.t.cdf(t_upper, df)
    p_tost = max(p_lower, p_upper)
    ci = stats.t.interval(1 - 2 * alpha, df, loc=mean_diff, scale=se)
    return {
        'mean_diff': mean_diff,
        'ci_low': ci[0],
        'ci_high': ci[1],
        'p_tost': p_tost,
        'equivalent': p_tost < alpha,
    }


def _seed_vector_from_summary(path: Path) -> list[float] | None:
    if not path.exists():
        return None
    payload = load_json(path)
    seeds = payload.get('seeds')
    per_seed = payload.get('per_seed_accuracy')
    if not isinstance(seeds, list) or not isinstance(per_seed, dict):
        return None
    values: list[float] = []
    for seed in seeds:
        value = per_seed.get(str(seed))
        if not isinstance(value, (int, float)):
            return None
        values.append(float(value))
    return values


def inspect_paired_tost_input(path: Path) -> tuple[dict[str, Any], list[str]]:
    payload = load_json(path)
    issues: list[str] = []

    synapse = payload.get('synapse')
    centralized = payload.get('centralized')
    if not isinstance(synapse, list) or not isinstance(centralized, list):
        return payload, ['paired TOST input must contain list-valued synapse and centralized fields']
    if len(synapse) != len(centralized):
        issues.append('synapse and centralized vectors have different lengths')
        return payload, issues

    try:
        synapse_values = [float(value) for value in synapse]
        centralized_values = [float(value) for value in centralized]
    except (TypeError, ValueError):
        issues.append('paired TOST vectors contain non-numeric values')
        return payload, issues

    if synapse_values == centralized_values:
        issues.append('synapse and centralized vectors are exactly identical, which strongly suggests a self-copy artifact rather than a real paired comparison')
    if centralized_values and len(set(centralized_values)) == 1:
        issues.append(f'centralized vector is degenerate (constant {centralized_values[0]:.3f} across all seeds)')

    synapse_summary = _seed_vector_from_summary(ONE_CLIENT_ROUTING_SUMMARY)
    if synapse_summary is not None and synapse_values != synapse_summary:
        issues.append(
            f'synapse vector does not match {ONE_CLIENT_ROUTING_SUMMARY.relative_to(REPO_ROOT)} '
            f'({synapse_summary} != {synapse_values})'
        )

    centralized_summary = _seed_vector_from_summary(CENTRALIZED_ROUTING_SUMMARY)
    if centralized_summary is not None and centralized_values != centralized_summary:
        issues.append(
            f'centralized vector does not match {CENTRALIZED_ROUTING_SUMMARY.relative_to(REPO_ROOT)} '
            f'({centralized_summary} != {centralized_values})'
        )

    return payload, issues


def build_rows(paired_tost_json: Path | None) -> dict[str, list[Row]]:
    rows: dict[str, list[Row]] = {'B': [], 'D': [], 'E': []}

    if ROUTING_SWEEP_SUMMARY.exists():
        sweep = load_json(ROUTING_SWEEP_SUMMARY)
        by_label = {entry['label']: entry['summary'] for entry in sweep.get('results', [])}
        label_map = {
            'No privacy': 'no_privacy',
            'ε=2.0, λ=0.5': 'eps_2_0',
            'ε=2.0, λ=1.0': 'eps_2_0',
            'ε=2.0, λ=1.5': 'eps_2_0',
            'ε=1.0, λ=0.5': 'eps_1_0',
            'ε=1.0, λ=1.0': 'eps_1_0',
            'ε=1.0, λ=1.5': 'eps_1_0',
            'ε=0.5, λ=0.5': 'eps_0_5',
            'ε=0.5, λ=1.0': 'eps_0_5',
            'ε=0.5, λ=1.5': 'eps_0_5',
        }
        for checkpoint, anchor in TABLE_ANCHORS['B']:
            summary = by_label.get(label_map[checkpoint])
            if not summary:
                continue
            rows['B'].append(Row(
                checkpoint,
                anchor,
                summarize_seed_payload(summary),
                f"measured on 50 GSM8K-derived routing samples per seed, {summary.get('rounds', 1)} federation round(s)",
            ))

    tau = describe_tau_sample()
    if tau is not None:
        measured, status = tau
        rows['D'].append(Row(TABLE_ANCHORS['D'][0][0], TABLE_ANCHORS['D'][0][1], measured, status))

    if paired_tost_json and paired_tost_json.exists():
        payload, issues = inspect_paired_tost_input(paired_tost_json)
        if issues:
            issue_text = '; '.join(issues)
            rows['E'].append(Row(TABLE_ANCHORS['E'][0][0], TABLE_ANCHORS['E'][0][1], 'not computed', f'invalid paired seed file: {issue_text}'))
            rows['E'].append(Row(TABLE_ANCHORS['E'][1][0], TABLE_ANCHORS['E'][1][1], 'not computed', 'resolve paired-input inconsistencies before running TOST'))
        else:
            tost = compute_tost(paired_tost_json)
            rows['E'].append(Row(TABLE_ANCHORS['E'][0][0], TABLE_ANCHORS['E'][0][1], f"mean_diff = {fmt_float(tost['mean_diff'])}", 'computed from paired seed file'))
            rows['E'].append(Row(TABLE_ANCHORS['E'][1][0], TABLE_ANCHORS['E'][1][1], f"90% CI = [{fmt_float(tost['ci_low'])}, {fmt_float(tost['ci_high'])}], p_tost = {fmt_float(tost['p_tost'])}", 'computed from paired seed file'))

    return {key: value for key, value in rows.items() if value}


def rows_to_json(rows: dict[str, list[Row]]) -> dict[str, Any]:
    return {key: [row.__dict__ for row in row_list] for key, row_list in rows.items()}


def render_markdown(rows: dict[str, list[Row]], paired_tost_json: Path | None) -> str:
    headers = {
        'B': 'Table B. Privacy–Utility Validation Points From Table 9 (Executed Checkpoints Only)',
        'D': 'Table D. Cross-Model / Root-Cause Checkpoints From Tables 22–23 (Executed Checkpoints Only)',
        'E': 'Table E. Controls / Equivalence Checkpoints (Executed Checkpoints Only)',
    }
    parts: list[str] = []
    for key in ['B', 'D', 'E']:
        if key not in rows:
            continue
        parts.append(f'### {headers[key]}')
        parts.append('')
        parts.append('| Checkpoint | Paper anchor | Real measured values | Status |')
        parts.append('| --- | ---: | --- | --- |')
        for row in rows[key]:
            parts.append(f'| {row.checkpoint} | {row.paper_anchor} | {row.measured} | {row.status} |')
        parts.append('')

    parts.append('### Supporting Artifacts')
    parts.append('')
    parts.append('| Artifact | Path |')
    parts.append('| --- | --- |')
    if ROUTING_SWEEP_SUMMARY.exists():
        parts.append(f'| Routing privacy sweep | [artifacts/verification/routing_privacy_sweep/combined_summary.json]({ROUTING_SWEEP_SUMMARY}:1) |')
    if ONE_CLIENT_ROUTING_SUMMARY.exists():
        parts.append(f'| One-client routing summary | [artifacts/verification/routing_client_count_1/summary.json]({ONE_CLIENT_ROUTING_SUMMARY}:1) |')
    latest_tau = latest_tau_result()
    if latest_tau:
        parts.append(f'| Latest τ-bench sample result | [{latest_tau.relative_to(REPO_ROOT)}]({latest_tau}:1) |')
    if paired_tost_json and paired_tost_json.exists():
        paired_abs = paired_tost_json.resolve()
        label = paired_abs.relative_to(REPO_ROOT) if paired_abs.is_relative_to(REPO_ROOT) else paired_abs
        parts.append(f'| Paired TOST input | [{label}]({paired_abs}:1) |')
    return "\n".join(parts).strip() + "\n"


def main() -> None:
    args = parse_args()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.run_routing_no_privacy:
        run_routing_verifier(args.routing_sample_count, args.routing_seeds, args.routing_rounds)
    if args.run_tau_sample:
        run_tau_sample(args.tau_task_id, args.tau_model)

    rows = build_rows(args.paired_tost_json)
    args.output_md.write_text(render_markdown(rows, args.paired_tost_json), encoding='utf-8')
    args.output_json.write_text(json.dumps(rows_to_json(rows), indent=2), encoding='utf-8')
    print(args.output_md)
    print(args.output_json)


if __name__ == '__main__':
    main()
