#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = REPO_ROOT / 'evaluation_on_textgrad_log.txt'
DEFAULT_TEXTGRAD_RUNNER = REPO_ROOT / 'scripts' / 'run_fed_textgrad.py'
DEFAULT_EXPECTED_RANGES = REPO_ROOT / 'artifacts' / 'rebuttal' / 'verification_tables_expected_ranges.md'
DEFAULT_OUTPUT_MD = REPO_ROOT / 'artifacts' / 'rebuttal' / 'table14_c4_audit.md'
DEFAULT_OUTPUT_JSON = REPO_ROOT / 'artifacts' / 'rebuttal' / 'table14_c4_audit.json'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Audit the Table 14 versus Section C.4 discrepancy from repo evidence.')
    parser.add_argument('--log-path', type=Path, default=DEFAULT_LOG)
    parser.add_argument('--runner-path', type=Path, default=DEFAULT_TEXTGRAD_RUNNER)
    parser.add_argument('--expected-ranges-path', type=Path, default=DEFAULT_EXPECTED_RANGES)
    parser.add_argument('--output-md', type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument('--output-json', type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def extract_textgrad_defaults(script_path: Path) -> dict[str, Any]:
    source = script_path.read_text(encoding='utf-8')
    tree = ast.parse(source)
    defaults: dict[str, Any] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != 'add_argument':
            continue
        if not node.args:
            continue
        first = node.args[0]
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            continue
        flag = first.value
        for keyword in node.keywords:
            if keyword.arg != 'default':
                continue
            try:
                defaults[flag] = ast.literal_eval(keyword.value)
            except (ValueError, SyntaxError):
                continue
    return {
        'aggregate_method': defaults.get('--aggregate-method'),
        'batch_size': defaults.get('--batch-size'),
        'max_steps': defaults.get('--max-steps'),
        'rounds': defaults.get('--rounds'),
    }


def load_textgrad_records(log_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not log_path.exists():
        return records
    for line in log_path.read_text(encoding='utf-8').splitlines():
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


def central_benchmark_rows(records: list[dict[str, Any]], benchmark: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.get('section') != 'central':
            continue
        payload = record.get('payload', {}) or {}
        if payload.get('mixed_queries') != benchmark:
            continue
        overall = payload.get('overall', {}) or {}
        rows.append({
            'timestamp': record.get('timestamp'),
            'benchmark': benchmark,
            'accuracy': overall.get('accuracy'),
            'correct': overall.get('correct'),
            'total': overall.get('total'),
        })
    return rows


def contains_exact_value(path: Path, needle: str) -> bool:
    if not path.exists():
        return False
    return needle in path.read_text(encoding='utf-8')


def render_markdown(payload: dict[str, Any]) -> str:
    defaults = payload['runner_defaults']
    rows = payload['bbh_v3_central_records']
    parts = [
        '### Table 14 vs. Section C.4 Audit',
        '',
        '| Question | Current repo evidence |',
        '| --- | --- |',
        f"| Current TextGrad runner defaults | aggregate=`{defaults.get('aggregate_method')}`, batch=`{defaults.get('batch_size')}`, steps=`{defaults.get('max_steps')}`, rounds=`{defaults.get('rounds')}` |",
        f"| Exact `0.901` preserved in TextGrad log | {payload['exact_0901_in_textgrad_log']} |",
        f"| Exact `0.901` present in rebuttal expected-ranges note | {payload['exact_0901_in_expected_ranges']} |",
        f"| Exact `0.92` preserved in TextGrad log | {payload['exact_092_in_textgrad_log']} |",
        '',
        '| Preserved central TextGrad evaluations on `bbh_object_counting_eval_v3.json` | Accuracy |',
        '| --- | ---: |',
    ]
    for row in rows:
        acc = row['accuracy']
        display = 'n/a' if acc is None else f"{acc:.3f} ({row['correct']}/{row['total']})"
        parts.append(f"| {row['timestamp']} | {display} |")
    parts.extend([
        '',
        'Interpretation:',
        f"The current checkout preserves a real `0.920` TextGrad result on `bbh_object_counting_eval_v3.json` but does not preserve a raw `0.901` TextGrad run artifact. The exact `0.901` value is present in the rebuttal expectations note, not in the committed TextGrad log. That means the repo supports the Table 14 headline as a real rounded measurement, but it does not preserve sufficient provenance to prove that Section C.4's `0.901` came from a committed batch=3 sweep artifact rather than a manual transcription or an external aggregation sheet.",
        '',
        f"A second limitation is that `evaluation_on_textgrad_log.txt` records benchmark outcomes but not the per-run CLI arguments, so even though the runner source defaults to batch=3 and max_steps=3, the log alone cannot prove which non-default flags were or were not used for a historical run.",
    ])
    return '\n'.join(parts) + '\n'


if __name__ == '__main__':
    args = parse_args()
    defaults = extract_textgrad_defaults(args.runner_path)
    records = load_textgrad_records(args.log_path)
    bbh_rows = central_benchmark_rows(records, 'bbh_object_counting_eval_v3.json')
    text_log = args.log_path.read_text(encoding='utf-8') if args.log_path.exists() else ''
    payload = {
        'runner_defaults': defaults,
        'bbh_v3_central_records': bbh_rows,
        'exact_0901_in_textgrad_log': '0.901' in text_log,
        'exact_092_in_textgrad_log': '0.92' in text_log,
        'exact_0901_in_expected_ranges': contains_exact_value(args.expected_ranges_path, '0.901'),
    }
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(payload), encoding='utf-8')
    args.output_json.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(args.output_md)
    print(args.output_json)
