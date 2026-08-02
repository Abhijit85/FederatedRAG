#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compute a historical-object cascade sweep from saved endpoint rows.')
    parser.add_argument('--small-dir', type=Path, required=True)
    parser.add_argument('--large-dir', type=Path, required=True)
    parser.add_argument('--thresholds', type=str, default='0.5,1.0,1.5')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--small-model-billions', type=float, default=1.0)
    parser.add_argument('--large-model-billions', type=float, default=8.0)
    return parser.parse_args()


def parse_thresholds(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(',') if part.strip()]


def effective_cost_ratio(deferral_rate: float, small_model_billions: float, large_model_billions: float) -> float:
    if large_model_billions <= 0:
        return 1.0
    return (small_model_billions + deferral_rate * large_model_billions) / large_model_billions


def load_seed_rows(path: Path) -> dict[int, list[dict[str, Any]]]:
    rows = {}
    for file in sorted(path.glob('routing_seed_*.json')):
        payload = json.loads(file.read_text(encoding='utf-8'))
        rows[int(payload['seed'])] = payload['rows']
    return rows


def margin_from_scores(scores: dict[str, float] | None) -> float | None:
    if not scores:
        return None
    vals = sorted((float(v) for v in scores.values()), reverse=True)
    if len(vals) < 2:
        return None
    return vals[0] - vals[1]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = parse_thresholds(args.thresholds)
    small_rows = load_seed_rows(args.small_dir)
    large_rows = load_seed_rows(args.large_dir)
    seeds = sorted(set(small_rows) & set(large_rows))
    results = []
    for threshold in thresholds:
        seed_results = []
        for seed in seeds:
            small_by_qid = {row['query_id']: row for row in small_rows[seed]}
            large_by_qid = {row['query_id']: row for row in large_rows[seed]}
            qids = sorted(set(small_by_qid) & set(large_by_qid))
            rows = []
            correct = 0
            deferrals = 0
            kept_correct = 0
            deferred_correct = 0
            kept_total = 0
            deferred_total = 0
            total_latency = 0.0
            for qid in qids:
                s = small_by_qid[qid]
                l = large_by_qid[qid]
                margin = margin_from_scores(s.get('option_scores_logprob'))
                defer = not (isinstance(margin, (int, float)) and float(margin) >= threshold)
                pred = l['predicted_domain'] if defer else s['predicted_domain']
                hit = bool(l['routed_correctly']) if defer else bool(s['routed_correctly'])
                correct += int(hit)
                deferrals += int(defer)
                total_latency += float(s.get('latency_seconds') or 0.0)
                if defer:
                    total_latency += float(l.get('latency_seconds') or 0.0)
                    deferred_correct += int(hit)
                    deferred_total += 1
                else:
                    kept_correct += int(hit)
                    kept_total += 1
                rows.append({
                    'query_id': qid,
                    'ground_truth_domain': s['ground_truth_domain'],
                    'predicted_domain': pred,
                    'routed_correctly': hit,
                    'deferred_to_large': defer,
                    'small_predicted_domain': s['predicted_domain'],
                    'large_predicted_domain': l['predicted_domain'],
                    'top_candidates': s['top_candidates'],
                    'small_margin': margin,
                    'small_latency_seconds': s['latency_seconds'],
                    'large_latency_seconds': l['latency_seconds'] if defer else None,
                })
            n = len(qids)
            deferral_rate = deferrals / n if n else 0.0
            acc = correct / n if n else 0.0
            compute_ratio = effective_cost_ratio(deferral_rate, args.small_model_billions, args.large_model_billions)
            result = {
                'seed': seed,
                'threshold': threshold,
                'sample_count': n,
                'accuracy': acc,
                'correct': correct,
                'deferrals': deferrals,
                'deferral_rate': deferral_rate,
                'kept_acc': kept_correct / kept_total if kept_total else None,
                'deferred_acc': deferred_correct / deferred_total if deferred_total else None,
                'mean_latency_seconds': total_latency / n if n else 0.0,
                'effective_compute_ratio_vs_full_large': compute_ratio,
                'effective_compute_reduction_vs_full_large': 1.0 - compute_ratio,
                'rows': rows,
            }
            seed_results.append(result)
            threshold_dir = args.output_dir / f"threshold_{str(threshold).replace('.', 'p')}"
            threshold_dir.mkdir(parents=True, exist_ok=True)
            (threshold_dir / f'routing_seed_{seed}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        accuracies = [float(r['accuracy']) for r in seed_results]
        deferrals = [float(r['deferral_rate']) for r in seed_results]
        compute_ratios = [float(r['effective_compute_ratio_vs_full_large']) for r in seed_results]
        kept_accs = [float(r['kept_acc']) for r in seed_results if isinstance(r.get('kept_acc'), (int, float))]
        deferred_accs = [float(r['deferred_acc']) for r in seed_results if isinstance(r.get('deferred_acc'), (int, float))]
        mean_latency = [float(r['mean_latency_seconds']) for r in seed_results]
        summary = {
            'threshold': threshold,
            'mean_accuracy': sum(accuracies) / len(accuracies),
            'sd_accuracy': statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
            'mean_deferral_rate': sum(deferrals) / len(deferrals),
            'sd_deferral_rate': statistics.stdev(deferrals) if len(deferrals) > 1 else 0.0,
            'mean_kept_acc': sum(kept_accs) / len(kept_accs) if kept_accs else None,
            'mean_deferred_acc': sum(deferred_accs) / len(deferred_accs) if deferred_accs else None,
            'mean_latency_seconds': sum(mean_latency) / len(mean_latency),
            'mean_effective_compute_ratio_vs_full_large': sum(compute_ratios) / len(compute_ratios),
            'mean_effective_compute_reduction_vs_full_large': 1.0 - (sum(compute_ratios) / len(compute_ratios)),
            'per_seed_accuracy': {str(r['seed']): r['accuracy'] for r in seed_results},
            'per_seed_deferral_rate': {str(r['seed']): r['deferral_rate'] for r in seed_results},
            'seed_results': seed_results,
        }
        results.append(summary)
    combined = {
        'small_dir': str(args.small_dir),
        'large_dir': str(args.large_dir),
        'thresholds': thresholds,
        'seeds': seeds,
        'small_model_billions': args.small_model_billions,
        'large_model_billions': args.large_model_billions,
        'results': results,
        'note': 'Historical-object cascade computed directly from saved endpoint rows under the shared retrieval_rerank rank-prompt setup.',
    }
    (args.output_dir / 'summary.json').write_text(json.dumps(combined, indent=2), encoding='utf-8')
    print(json.dumps(combined, indent=2))


if __name__ == '__main__':
    main()
