#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from pymongo import MongoClient

from math_qa import JinaAIClient, MongoRAGManager
from jina_key_manager import get_named_jina_api_keys
from mongo_utils import MongoVectorStore

DB_NAME = "FredRag"

GEOMETRY_TERMS = {
    "area", "perimeter", "volume", "surface", "circumference", "radius", "diameter", "triangle",
    "rectangle", "square", "circle", "cylinder", "sphere", "cube", "inch", "inches", "feet", "foot",
    "yard", "yards", "meter", "meters", "centimeter", "centimeters", "angle", "angles", "length", "width", "height"
}
PERCENT_TERMS = {
    "percent", "percentage", "%", "ratio", "proportion", "fractions", "fraction", "part", "share", "divided", "out of"
}
ALGEBRA_TERMS = {
    "equation", "solve", "variable", "number", "sum", "difference", "consecutive", "integer", "integers",
    "average", "mixture", "mixtures", "ages", "age", "coins", "digit", "digits"
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a richer math routing collection with scenario metadata.")
    parser.add_argument("--source", type=str, default="math_problems")
    parser.add_argument("--target", type=str, default="routing_math_scenarios_v1")
    parser.add_argument("--reembed", action="store_true", help="Regenerate embeddings for the rebuilt documents.")
    return parser.parse_args()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def contains_any(text: str, terms: set[str]) -> bool:
    blob = normalize(text)
    return any(term in blob for term in terms)


def assign_profile(doc: dict[str, Any]) -> dict[str, Any]:
    metadata = doc.get("metadata") or {}
    tool = str(metadata.get("tool") or "").strip()
    text = str(doc.get("text") or "")
    original_problem = str(metadata.get("original_problem") or text)
    blob = f"{text} {original_problem}"

    if tool == "Financial_Calculator":
        return {
            "tool": tool,
            "scenario": "Financial and Banking Calculator",
            "scenario_context": "Solves financial arithmetic such as gains, losses, discounts, interest, and transaction-value comparisons.",
            "annex_terms": ["profit", "loss", "discount", "interest", "banker", "price", "cost", "value"],
            "category": "gain",
        }
    if tool == "Work_Time_Analyzer":
        return {
            "tool": tool,
            "scenario": "Work, Rate, and Time Analyzer",
            "scenario_context": "Solves rate, speed, distance, time, and combined-work problems by composing per-unit rates and total durations.",
            "annex_terms": ["rate", "time", "speed", "distance", "hour", "minute", "work", "together"],
            "category": "physics",
        }
    if tool == "Algebraic_Problem_Solver":
        if contains_any(blob, GEOMETRY_TERMS):
            return {
                "tool": tool,
                "scenario": "Geometry: Shapes and Measurement",
                "scenario_context": "Handles measurements involving geometric shapes, perimeter, area, volume, and unit conversion across dimensions.",
                "annex_terms": ["length", "width", "height", "area", "volume", "circumference", "inch", "foot"],
                "category": "geometry",
            }
        return {
            "tool": tool,
            "scenario": "Algebraic Word Problem Solver",
            "scenario_context": "Solves algebraic word problems by translating relationships into equations over unknown quantities, averages, mixtures, and integer constraints.",
            "annex_terms": ["equation", "variable", "integer", "average", "mixture", "difference", "sum", "solve"],
            "category": "algebra",
        }
    if tool == "General_Math_Tool":
        if contains_any(blob, PERCENT_TERMS):
            return {
                "tool": tool,
                "scenario": "Percentage and Proportion Solver",
                "scenario_context": "Handles calculations involving percentages, proportions, ratios, and direct or inverse relationships.",
                "annex_terms": ["percent", "percentage", "ratio", "proportion", "fraction", "share", "part", "out of"],
                "category": "percentage",
            }
        return {
            "tool": tool,
            "scenario": "General Logic and Counting",
            "scenario_context": "Handles counting, simple arithmetic, and everyday multi-step word problems that do not require a domain-specific financial or rate model.",
            "annex_terms": ["count", "total", "remaining", "difference", "sum", "pieces", "steps", "students"],
            "category": "general",
        }
    if contains_any(blob, GEOMETRY_TERMS):
        return {
            "tool": "Algebraic_Problem_Solver",
            "scenario": "Geometry: Shapes and Measurement",
            "scenario_context": "Handles measurements involving geometric shapes, perimeter, area, volume, and unit conversion across dimensions.",
            "annex_terms": ["length", "width", "height", "area", "volume", "circumference", "inch", "foot"],
            "category": "geometry",
        }
    if contains_any(blob, PERCENT_TERMS):
        return {
            "tool": "General_Math_Tool",
            "scenario": "Percentage and Proportion Solver",
            "scenario_context": "Handles calculations involving percentages, proportions, ratios, and direct or inverse relationships.",
            "annex_terms": ["percent", "percentage", "ratio", "proportion", "fraction", "share", "part", "out of"],
            "category": "percentage",
        }
    if contains_any(blob, ALGEBRA_TERMS):
        return {
            "tool": "Algebraic_Problem_Solver",
            "scenario": "Algebraic Word Problem Solver",
            "scenario_context": "Solves algebraic word problems by translating relationships into equations over unknown quantities, averages, mixtures, and integer constraints.",
            "annex_terms": ["equation", "variable", "integer", "average", "mixture", "difference", "sum", "solve"],
            "category": "algebra",
        }
    return {
        "tool": "General_Math_Tool",
        "scenario": "General Logic and Counting",
        "scenario_context": "Handles counting, simple arithmetic, and everyday multi-step word problems that do not require a domain-specific financial or rate model.",
        "annex_terms": ["count", "total", "remaining", "difference", "sum", "pieces", "steps", "students"],
        "category": "general",
    }


def merged_notes(profile: dict[str, Any]) -> str:
    return profile["scenario_context"] + "; keywords: " + ", ".join(profile["annex_terms"][:6])


def render_text(problem: str, rationale: str, profile: dict[str, Any]) -> str:
    return (
        f"Problem: {problem} | Category: {profile['category']} | Tool Used: {profile['tool']} | "
        f"Tool Scenario: {profile['scenario']} | Context: {profile['scenario_context']} | "
        f"Keywords: {', '.join(profile['annex_terms'][:8])} | Rationale: {rationale}"
    )


def main() -> None:
    load_dotenv('.env')
    args = parse_args()
    client = MongoClient(os.environ['MONGO_URI'])
    db = client[DB_NAME]
    source = db[args.source]
    target = db[args.target]
    rows = list(source.find({}))
    if not rows:
        raise RuntimeError(f"source collection {args.source} is empty")

    rebuilt = []
    scenario_counts = Counter()
    tool_counts = Counter()
    texts = []
    for idx, doc in enumerate(rows):
        metadata = doc.get('metadata') or {}
        profile = assign_profile(doc)
        problem = str(metadata.get('original_problem') or doc.get('text') or '')
        rationale = str(metadata.get('rationale') or metadata.get('solution_steps') or '')
        text = render_text(problem, rationale, profile)
        rebuilt.append({
            '_id': str(doc.get('_id', idx)),
            'text': text,
            'metadata': {
                'tool': profile['tool'],
                'original_problem': problem,
                'scenario': profile['scenario'],
                'scenario_context': profile['scenario_context'],
                'merged_notes': merged_notes(profile),
                'annex_terms': profile['annex_terms'],
                'category': profile['category'],
                'source_collection': args.source,
            },
        })
        texts.append(text)
        scenario_counts[profile['scenario']] += 1
        tool_counts[profile['tool']] += 1

    if args.reembed:
        jina_client = JinaAIClient(get_named_jina_api_keys())
        embeddings = jina_client.get_embeddings(texts)
        for doc, emb in zip(rebuilt, embeddings):
            doc['embedding'] = emb
    else:
        for out_doc, in_doc in zip(rebuilt, rows):
            if 'embedding' in in_doc:
                out_doc['embedding'] = in_doc['embedding']

    target.delete_many({})
    target.insert_many(rebuilt)
    client.close()

    print({'target': args.target, 'count': len(rebuilt), 'scenario_counts': dict(scenario_counts), 'tool_counts': dict(tool_counts), 'reembed': args.reembed})


if __name__ == '__main__':
    main()
