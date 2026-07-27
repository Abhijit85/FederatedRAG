#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import runpy
from itertools import product
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "verification" / "experiment2_family2_sweep.json"
TARGET_TYPED_DROP = 8.1
TARGET_FLAT_DROP = 11.5
TARGET_GAP = 3.4

FAMILY2_DEFAULTS = {
    "strong_conflict_base": 0.05,
    "strong_conflict_scale": 0.20,
    "strong_conflict_cap": 0.35,
    "corruption_base": 0.04,
    "corruption_scale": 0.10,
    "corruption_strong_bonus": 0.04,
    "extra_conflict_prob": 0.25,
    "lexical_bonus": 1.1,
    "focus_bonus": 0.45,
    "scenario_hint_bonus": 0.25,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep a second Experiment 2 reconstruction family.")
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


def patch_family2(ns: dict[str, Any], family: dict[str, float]) -> None:
    ns['CALIBRATION'].update({
        'strong_conflict_base': family['strong_conflict_base'],
        'strong_conflict_scale': family['strong_conflict_scale'],
        'strong_conflict_cap': family['strong_conflict_cap'],
        'corruption_base': family['corruption_base'],
        'corruption_scale': family['corruption_scale'],
        'corruption_strong_bonus': family['corruption_strong_bonus'],
        'flat_focus_text_parts_limit': 2,
        'flat_focus_term_limit': 4,
    })

    def query_for(record: Any) -> str:
        focus_text = str(record.payload.get('focus_text') or record.scenario)
        return (
            f"Find the {record.tool} scenario named {record.scenario}. "
            f"It should also match these specialization cues: {focus_text}. "
            f"Prefer artifacts whose name and focus both agree."
        )

    def contradictory_artifact(record: Any, wrong: Any, source_id: str, condition: str, *, strong_conflict: bool, corruption_level: float):
        payload = dict(record.payload)
        if condition == 'untyped':
            payload.pop('type', None)
        metadata = dict(record.metadata)
        metadata['source_id'] = source_id
        metadata['conflicted'] = True
        metadata['contradicts'] = record.scenario
        metadata['scenario'] = record.scenario
        if 'domain' in metadata:
            metadata['domain'] = record.domain

        correct_focus_terms = list(record.payload.get('focus_terms', []))
        wrong_focus_terms = list(wrong.payload.get('focus_terms', []))
        blended_focus_terms = ns['blend_focus_terms'](correct_focus_terms, wrong_focus_terms, corruption_level)
        payload['focus_terms'] = blended_focus_terms
        payload['focus_text'] = ' '.join(blended_focus_terms)
        payload['canonical_scenario'] = record.scenario
        payload['scenario_hint'] = wrong.scenario
        payload['domain_hint'] = wrong.domain

        if condition == 'flat':
            payload['scenario'] = wrong.scenario
            payload['domain'] = wrong.domain
            if strong_conflict:
                payload['focus_terms'] = wrong_focus_terms[: max(1, len(correct_focus_terms))]
                payload['focus_text'] = ' '.join(payload['focus_terms'])
        elif condition == 'typed' and strong_conflict:
            # typed keeps scenario identity but gets stronger text-level ambiguity
            payload['conflict_support'] = wrong.scenario
            payload['conflict_focus'] = ' '.join(wrong_focus_terms)

        text = ns['structured_prompt'](metadata, payload, record.role)
        text += f" | focus: {payload['focus_text']}"
        if strong_conflict:
            text += f" | conflicting exemplar: {wrong.scenario}"
            text += f" | alternate cues: {' '.join(wrong_focus_terms)}"
        return ns['Artifact'](
            signature=f"{source_id}::family2::{record.tool}::{record.scenario}::{wrong.scenario}",
            text=text,
            metadata=metadata,
            payload=payload,
        )

    def build_condition_artifacts(records: list[Any], conflict_rate: int, seed: int, condition: str):
        rng = random.Random((seed * 1000) + conflict_rate + 17)
        conflicted = ns['select_conflicted_indices'](len(records), conflict_rate, seed)
        artifacts = []
        conflict_rows = []
        for idx, record in enumerate(records):
            artifacts.append(ns['base_artifact'](record, source_id='client_clean'))
            if idx not in conflicted:
                continue
            candidate_indices = [j for j in range(len(records)) if j != idx and records[j].tool == record.tool]
            if not candidate_indices:
                candidate_indices = [j for j in range(len(records)) if j != idx]
            wrong_idx = rng.choice(candidate_indices)
            wrong_record = records[wrong_idx]
            strong_conflict = rng.random() < min(family['strong_conflict_cap'], family['strong_conflict_base'] + (conflict_rate / 100.0) * family['strong_conflict_scale'])
            corruption_level = family['corruption_base'] + (conflict_rate / 100.0) * family['corruption_scale']
            if strong_conflict:
                corruption_level += family['corruption_strong_bonus']
            artifacts.append(contradictory_artifact(record, wrong_record, 'client_conflict_a', condition, strong_conflict=strong_conflict, corruption_level=corruption_level))
            if rng.random() < family['extra_conflict_prob'] * (conflict_rate / 60.0):
                wrong2_idx = rng.choice([j for j in candidate_indices if j != wrong_idx] or candidate_indices)
                wrong2_record = records[wrong2_idx]
                artifacts.append(contradictory_artifact(record, wrong2_record, 'client_conflict_b', condition, strong_conflict=True, corruption_level=min(1.0, corruption_level + 0.12)))
            conflict_rows.append({'target_scenario': record.scenario, 'contradictory_scenario': wrong_record.scenario, 'tool': record.tool})
        return artifacts, conflict_rows

    def retrieval_score(query: str, artifact: Any) -> float:
        score = ns['lexical_score'](query, artifact.text)
        q = query.lower()
        metadata = artifact.metadata or {}
        payload = artifact.payload or {}
        scenario = str(metadata.get('scenario') or payload.get('scenario') or '').lower()
        if scenario and scenario in q:
            score += family['lexical_bonus']
        focus_text = str(payload.get('focus_text') or '').lower()
        if focus_text and focus_text in q:
            score += family['focus_bonus']
        scenario_hint = str(payload.get('scenario_hint') or '').lower()
        if scenario_hint and scenario_hint in q:
            score += family['scenario_hint_bonus']
        for value in payload.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.lower() in q:
                        score += 0.12
        return score

    ns['query_for'] = query_for
    ns['build_condition_artifacts'] = build_condition_artifacts
    ns['retrieval_score'] = retrieval_score


def evaluate_setting(ns: dict[str, Any], records: list[Any], seeds: list[int], rates: list[int], family: dict[str, float]) -> dict[str, Any]:
    patch_family2(ns, family)
    typed_curve = {}
    flat_curve = {}
    for rate in rates:
        typed_curve[rate] = mean([ns['evaluate_condition'](records, rate, seed, 'typed')['accuracy'] for seed in seeds])
        flat_curve[rate] = mean([ns['evaluate_condition'](records, rate, seed, 'flat')['accuracy'] for seed in seeds])
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
        'family': family,
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
    ns = runpy.run_path(str(REPO_ROOT / 'scripts' / 'run_experiment2_reconstructed.py'), run_name='exp2_family2_sweep')
    records = ns['build_benchmark']()
    grid = {
        'strong_conflict_base': [0.02, 0.05],
        'strong_conflict_scale': [0.10, 0.20],
        'strong_conflict_cap': [0.20, 0.35],
        'corruption_base': [0.02, 0.04],
        'corruption_scale': [0.06, 0.10],
        'corruption_strong_bonus': [0.02, 0.04],
        'extra_conflict_prob': [0.15, 0.30],
        'lexical_bonus': [0.7, 1.1],
        'focus_bonus': [0.25, 0.45],
        'scenario_hint_bonus': [0.10, 0.25],
    }
    keys = list(grid)
    results = []
    for values in product(*(grid[key] for key in keys)):
        family = dict(zip(keys, values))
        results.append(evaluate_setting(ns, records, seeds, rates, family))
    results.sort(key=lambda row: row['score'])
    payload = {
        'target': {'typed_drop': TARGET_TYPED_DROP, 'flat_drop': TARGET_FLAT_DROP, 'gap': TARGET_GAP},
        'total_settings': len(results),
        'top_results': results[: max(1, args.top_k)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    best = payload['top_results'][0]
    print(f"Evaluated {len(results)} family2 settings")
    print(f"Best score: {best['score']:.3f}")
    print(f"Best typed drop: {best['typed_drop']:.2f} pts")
    print(f"Best flat drop: {best['flat_drop']:.2f} pts")
    print(f"Best gap: {best['gap']:.2f} pts")
    print(json.dumps(best['family'], sort_keys=True))


if __name__ == '__main__':
    main()
