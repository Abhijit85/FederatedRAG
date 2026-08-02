#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_gsm8k_router_cascade import RouterSpec, ScenarioDoc, local_router_decision, resolve_router_spec  # noqa: E402
from scripts.run_routing_verification import (  # noqa: E402
    artifact_route_label,
    build_credentials,
    build_runtime,
    gold_route_label,
    load_records,
    query_text,
    sample_records,
    select_route_label,
    selector_expanded_max_items,
    temporary_routing_alignment_profile,
)
from scripts.run_gsm8k_schema_control import temporary_structured_payload_mode  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run a 20-query diagnostic using schema_control retrieval with local LLM reranking.')
    parser.add_argument('--sample-file', type=Path, default=REPO_ROOT / 'GSM8K_500_rebuttal_run' / 'GSM8K_500_samples.json')
    parser.add_argument('--sample-count', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--client-count', type=int, default=5)
    parser.add_argument('--max-items', type=int, default=5)
    parser.add_argument('--max-tokens', type=int, default=8)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args()


def scenario_doc_from_artifact(artifact: Any) -> ScenarioDoc:
    label = artifact_route_label(artifact)
    payload = artifact.structured_payload or {}
    text = ''
    if isinstance(payload, dict):
        for key in ('scenario_context', 'context', 'scenario', 'domain'):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                text = value.strip()
                break
    if not text:
        text = str(getattr(artifact, 'text', '') or '').strip()
    if not text:
        text = f'Routing scenario label: {label}.'
    return ScenarioDoc(label=label, text=text, embedding=[])


def main() -> None:
    load_dotenv(REPO_ROOT / '.env')
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.sample_file)
    subset = sample_records(records, seed=args.seed, sample_count=args.sample_count)
    spec = resolve_router_spec(RouterSpec(label='llama31_8b_schema_diag', backend='local', model=args.model_path, local_model_path=args.model_path))

    rows = []
    correct = 0
    svm_correct = 0
    with temporary_structured_payload_mode('typed'), temporary_routing_alignment_profile():
        runtime = build_runtime(rounds=args.rounds, client_count=args.client_count)
        for record in subset:
            query = query_text(record)
            gold = gold_route_label(record)
            artifacts = runtime.get_context_for_query(query, max_items=selector_expanded_max_items(args.max_items))
            docs = [scenario_doc_from_artifact(artifact) for artifact in artifacts]
            docs = docs[:args.max_items]
            decision = local_router_decision(query, docs, spec, max_tokens=args.max_tokens) if docs else None
            predicted = decision.predicted_label if decision else ''
            hit = predicted == gold or (predicted == 'Geometry: Shapes and Measurement' and gold == 'Geometry and Measurement') or (predicted == 'Geometry and Measurement' and gold == 'Geometry: Shapes and Measurement')
            correct += int(hit)
            svm_pred = select_route_label(query, artifacts) if artifacts else ''
            svm_hit = svm_pred == gold or (svm_pred == 'Geometry: Shapes and Measurement' and gold == 'Geometry and Measurement') or (svm_pred == 'Geometry and Measurement' and gold == 'Geometry: Shapes and Measurement')
            svm_correct += int(svm_hit)
            rows.append({
                'query_id': record.get('query_id') or record.get('sample_id'),
                'query_text': query,
                'ground_truth_domain': gold,
                'svm_selector_prediction': svm_pred,
                'svm_selector_correct': svm_hit,
                'llm_prediction': predicted,
                'llm_correct': hit,
                'llm_raw_response': decision.raw_response if decision else '',
                'llm_option_scores_logprob': decision.option_scores_logprob if decision else None,
                'llm_option_margin_logprob': decision.option_margin_logprob if decision else None,
                'top_candidates': [doc.label for doc in docs],
                'top_candidate_text': [doc.text[:220] for doc in docs],
            })
    summary = {
        'sample_file': str(args.sample_file),
        'sample_count': args.sample_count,
        'seed': args.seed,
        'rounds': args.rounds,
        'client_count': args.client_count,
        'max_items': args.max_items,
        'max_tokens': args.max_tokens,
        'model_path': spec.local_model_path,
        'svm_selector_accuracy': svm_correct / len(rows) if rows else 0.0,
        'llm_accuracy': correct / len(rows) if rows else 0.0,
        'rows': rows,
        'note': 'Diagnostic: schema_control retrieval held fixed; only selector swapped from historical_cv_svm to local 8B exact-label reranker.',
    }
    (args.output_dir / 'diagnostic_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({
        'svm_selector_accuracy': summary['svm_selector_accuracy'],
        'llm_accuracy': summary['llm_accuracy'],
        'output': str(args.output_dir / 'diagnostic_summary.json'),
    }, indent=2))


if __name__ == '__main__':
    main()
