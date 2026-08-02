#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pymongo import MongoClient

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jina_key_manager import get_named_jina_api_keys
from math_qa import JinaAIClient

DB_NAME = 'FredRag'
DEFAULT_RUNLOG = REPO_ROOT / 'GSM8K_500_rebuttal_run' / 'GSM8K_500_runlog.jsonl'
DEFAULT_COMPENDIUM = REPO_ROOT / 'mathqa_tools_compendium.json'

ALIASES = {
    'geometry and measurement': 'Geometry: Shapes and Measurement',
    'geometry shapes and measurement': 'Geometry: Shapes and Measurement',
}

PROFILE_OVERRIDES = {
    'General Logic and Counting': {
        'tool': 'General_Math_Tool',
        'category': 'general',
        'annex_terms': ['count', 'total', 'remaining', 'sum', 'difference', 'twice', 'each', 'left'],
        'merged_notes': 'Use for counting and everyday multi-step arithmetic that does not fit rate, finance, or algebra-heavy formulations.',
    },
    'Financial and Banking Calculator': {
        'tool': 'Financial_Calculator',
        'category': 'financial',
        'annex_terms': ['dollar', 'cost', 'price', 'profit', 'loss', 'discount', 'interest', 'money'],
        'merged_notes': 'Use for money, cost, price, revenue, change, discounts, and rate-of-return style arithmetic.',
    },
    'Algebraic Word Problem Solver': {
        'tool': 'Algebraic_Problem_Solver',
        'category': 'algebra',
        'annex_terms': ['equation', 'variable', 'unknown', 'integer', 'average', 'sum', 'difference', 'consecutive'],
        'merged_notes': 'Use when the word problem is best translated into equations over unknown quantities.',
    },
    'Work, Rate, and Time Analyzer': {
        'tool': 'Work_Time_Analyzer',
        'category': 'rate',
        'annex_terms': ['rate', 'time', 'speed', 'distance', 'hour', 'minute', 'together', 'per'],
        'merged_notes': 'Use for work-rate, speed-distance-time, and per-unit accumulation problems.',
    },
    'Geometry: Shapes and Measurement': {
        'tool': 'Algebraic_Problem_Solver',
        'category': 'geometry',
        'annex_terms': ['area', 'perimeter', 'volume', 'length', 'width', 'height', 'angle', 'radius'],
        'merged_notes': 'Use for measurement, geometry, perimeter, area, volume, and unit conversion across shapes.',
    },
    'Percentage and Proportion Solver': {
        'tool': 'General_Math_Tool',
        'category': 'percentage',
        'annex_terms': ['percent', 'percentage', 'ratio', 'proportion', 'fraction', 'share', 'part', 'out of'],
        'merged_notes': 'Use for percentages, proportions, ratios, and fraction-style relation problems.',
    },
    'SearchQA': {
        'tool': 'SearchQA',
        'category': 'search',
        'annex_terms': ['fact', 'lookup', 'search', 'question', 'answer'],
        'merged_notes': 'Use when the task is best framed as factual lookup rather than arithmetic reasoning.',
    },
    'Number Theory': {
        'tool': 'Number_Theory',
        'category': 'number_theory',
        'annex_terms': ['prime', 'factor', 'multiple', 'divisible', 'remainder', 'integer'],
        'merged_notes': 'Use for divisibility, factors, multiples, and other number-theoretic reasoning.',
    },
}

CANONICAL_BY_NORMALIZED = { ' '.join(key.lower().replace('&', ' and ').split()): key for key in PROFILE_OVERRIDES }


def normalize_label(text: str) -> str:
    return ' '.join((text or '').lower().replace('&', ' and ').split())


def canonical_label(label: str) -> str:
    norm = normalize_label(label)
    if norm in ALIASES:
        return ALIASES[norm]
    if norm in CANONICAL_BY_NORMALIZED:
        return CANONICAL_BY_NORMALIZED[norm]
    return label.strip()


def load_contexts(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text())
    usage = payload.get('Textual_Compendium', {}).get('Usage_Scenarios', [])
    out = {}
    for row in usage:
        if not isinstance(row, dict):
            continue
        scenario = str(row.get('scenario') or '').strip()
        context = str(row.get('context') or '').strip()
        if not scenario:
            continue
        out[scenario] = context
        out[normalize_label(scenario)] = context
        out[canonical_label(scenario)] = context
    return out


def scenario_profile(label: str, context: str) -> dict[str, Any]:
    canonical = canonical_label(label)
    override = PROFILE_OVERRIDES.get(canonical, {})
    return {
        'scenario': canonical,
        'tool': override.get('tool', canonical),
        'category': override.get('category', normalize_label(canonical).replace(' ', '_')),
        'scenario_context': context,
        'annex_terms': list(override.get('annex_terms', [])),
        'merged_notes': override.get('merged_notes', ''),
    }


def build_text(query: str, profile: dict[str, Any], top_gap: float | None) -> str:
    parts = [
        f'Problem: {query}',
        f"Tool Scenario: {profile['scenario']}",
        f"Context: {profile['scenario_context']}",
    ]
    if profile['merged_notes']:
        parts.append(f"Routing Notes: {profile['merged_notes']}")
    if profile['annex_terms']:
        parts.append(f"Keywords: {', '.join(profile['annex_terms'])}")
    if top_gap is not None:
        parts.append(f'Historical Retrieval Gap: {top_gap:.4f}')
    parts.append(f"Expected Tool: {profile['tool']}")
    return ' | '.join(parts)


def build_prototype_docs(contexts: dict[str, str]) -> list[dict[str, Any]]:
    docs = []
    seen: set[str] = set()
    for label, context in contexts.items():
        canonical = canonical_label(label)
        if canonical in seen:
            continue
        if canonical not in PROFILE_OVERRIDES:
            continue
        seen.add(canonical)
        profile = scenario_profile(canonical, context)
        text = (
            f"Prototype Routing Scenario: {profile['scenario']} | Context: {profile['scenario_context']} | "
            f"Routing Notes: {profile['merged_notes']} | Keywords: {', '.join(profile['annex_terms'])} | "
            f"Expected Tool: {profile['tool']}"
        )
        docs.append({
            '_id': f"prototype::{profile['scenario']}",
            'text': text,
            'metadata': {
                'query_id': '',
                'tool': profile['tool'],
                'scenario': profile['scenario'],
                'scenario_context': profile['scenario_context'],
                'merged_notes': profile['merged_notes'],
                'annex_terms': profile['annex_terms'],
                'category': profile['category'],
                'source_dataset': 'compendium_prototype',
                'original_problem': '',
                'route_confidence': None,
                'top_gap': None,
                'is_prototype': True,
            },
        })
    return docs


def iter_runlog_rows(path: Path):
    with path.open() as fh:
        for line in fh:
            obj = json.loads(line)
            if isinstance(obj, dict) and obj.get('source_kind') == 'gsm8k_derived':
                yield obj


def main() -> None:
    load_dotenv(REPO_ROOT / '.env')
    target = os.environ.get('QUERY_BANK_COLLECTION', 'routing_query_bank_v1')
    include_prototypes = os.environ.get('QUERY_BANK_INCLUDE_PROTOTYPES', '0') == '1'
    runlog = Path(os.environ.get('QUERY_BANK_RUNLOG', str(DEFAULT_RUNLOG)))
    compendium = Path(os.environ.get('QUERY_BANK_COMPENDIUM', str(DEFAULT_COMPENDIUM)))
    contexts = load_contexts(compendium)
    rows = list(iter_runlog_rows(runlog))
    jina = JinaAIClient(get_named_jina_api_keys())
    texts: list[str] = []
    docs: list[dict[str, Any]] = []
    counts = Counter()

    for row in rows:
        router = row.get('router') or {}
        label = str(router.get('ground_truth_domain') or '').strip()
        if not label:
            continue
        canonical = canonical_label(label)
        context = contexts.get(canonical) or contexts.get(label) or contexts.get(normalize_label(label)) or ''
        profile = scenario_profile(canonical, context)
        query = str(row.get('query_text') or '').strip()
        if not query:
            continue
        top_candidates = router.get('top_candidates') or []
        top_gap = None
        if len(top_candidates) >= 2:
            try:
                top_gap = float(top_candidates[0].get('cosine_score')) - float(top_candidates[1].get('cosine_score'))
            except Exception:
                top_gap = None
        text = build_text(query, profile, top_gap)
        texts.append(text)
        docs.append({
            '_id': str(row.get('sample_id') or row.get('query_id')),
            'text': text,
            'metadata': {
                'query_id': str(row.get('query_id') or ''),
                'tool': profile['tool'],
                'scenario': profile['scenario'],
                'scenario_context': profile['scenario_context'],
                'merged_notes': profile['merged_notes'],
                'annex_terms': profile['annex_terms'],
                'category': profile['category'],
                'source_dataset': row.get('source_dataset'),
                'original_problem': query,
                'route_confidence': (row.get('evaluation') or {}).get('route_confidence'),
                'top_gap': top_gap,
                'is_prototype': False,
            },
        })
        counts[profile['scenario']] += 1

    prototype_docs = build_prototype_docs(contexts) if include_prototypes else []
    docs.extend(prototype_docs)
    texts.extend(doc['text'] for doc in prototype_docs)

    embeddings = jina.get_embeddings(texts)
    for doc, emb in zip(docs, embeddings):
        doc['embedding'] = emb

    client = MongoClient(os.environ['MONGO_URI'])
    coll = client[DB_NAME][target]
    coll.delete_many({})
    coll.insert_many(docs)
    client.close()
    print({'target': target, 'count': len(docs), 'scenario_counts': dict(counts), 'prototype_count': len(prototype_docs), 'include_prototypes': include_prototypes})


if __name__ == '__main__':
    main()
