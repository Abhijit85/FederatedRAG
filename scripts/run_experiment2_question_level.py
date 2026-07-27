#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "verification" / "experiment2_question_level"
DEFAULT_ANALYSIS_INPUT = REPO_ROOT / "artifacts" / "verification" / "experiment2_question_level_seeds.json"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "based", "be", "both", "by", "can", "common",
    "complex", "concerning", "context", "covers", "data", "determine", "do", "does", "for",
    "from", "general", "handles", "ideal", "identifies", "in", "including", "involving",
    "is", "it", "its", "like", "main", "of", "on", "or", "other", "outcomes", "problems",
    "properties", "provide", "question", "questions", "related", "select", "shown", "simple",
    "solves", "such", "system", "than", "that", "the", "their", "them", "these", "this",
    "those", "to", "tool", "true", "used", "using", "visual", "with",
}


@dataclass
class ScenarioRecord:
    scenario: str
    tool: str
    context: str
    domain: str
    cues: list[str]
    exemplar: str


@dataclass
class QueryCase:
    query_id: str
    target_scenario: str
    tool: str
    query: str


@dataclass
class Artifact:
    signature: str
    text: str
    metadata: dict[str, object]
    payload: dict[str, object]


@dataclass
class MergedArtifact:
    signature: str
    text: str
    metadata: dict[str, object]
    payload: dict[str, object]
    members: list[Artifact]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Question-level reconstruction of Experiment 2 using original compendium "
            "context text and paired 5-seed typed vs flat reruns."
        )
    )
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--conflict-rates", type=str, default="0,20,40,60")
    parser.add_argument("--conditions", type=str, default="typed,flat")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--analysis-input", type=Path, default=DEFAULT_ANALYSIS_INPUT)
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_str_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def normalize_text(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def unique_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in normalize_text(text):
        if token in STOPWORDS:
            continue
        if token not in tokens:
            tokens.append(token)
    return tokens


def guess_domain(scenario: str, tool: str) -> str:
    if ":" in scenario:
        return scenario.split(":", 1)[0].strip().lower()
    if tool == "mathqa":
        return "math"
    return "science"


def extract_exemplar(context: str) -> str:
    patterns = [
        r"such as ([^.]+)",
        r"like ([^.]+)",
        r"for questions involving ([^.]+)",
        r"for example, ([^.]+)",
    ]
    lower = context.lower()
    for pattern in patterns:
        match = re.search(pattern, lower)
        if match:
            text = match.group(1).strip(" .,'\"")
            if text:
                return text
    clauses = [part.strip(" .") for part in re.split(r"[.;]", context) if part.strip()]
    return clauses[-1] if clauses else context


def select_cues(scenario: str, context: str) -> list[str]:
    scenario_tokens = set(unique_tokens(scenario))
    context_tokens = [token for token in unique_tokens(context) if token not in scenario_tokens]
    cues = context_tokens[:6]
    if len(cues) < 4:
        for token in unique_tokens(scenario):
            if token not in cues:
                cues.append(token)
            if len(cues) >= 4:
                break
    return cues


def load_records(path: Path, tool: str) -> list[ScenarioRecord]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    scenarios = ((raw.get("Textual_Compendium") or {}).get("Usage_Scenarios")) or raw.get("Usage_Scenarios") or []
    records: list[ScenarioRecord] = []
    for item in scenarios:
        scenario = str(item.get("scenario") or "").strip()
        context = str(item.get("context") or "").strip()
        if not scenario or not context:
            continue
        records.append(
            ScenarioRecord(
                scenario=scenario,
                tool=tool,
                context=context,
                domain=guess_domain(scenario, tool),
                cues=select_cues(scenario, context),
                exemplar=extract_exemplar(context),
            )
        )
    return records


def build_records() -> list[ScenarioRecord]:
    return load_records(REPO_ROOT / "mathqa_tools_compendium.json", "mathqa") + load_records(
        REPO_ROOT / "scienceqa_tools_compendium.json", "scienceqa"
    )


def build_queries(records: list[ScenarioRecord]) -> list[QueryCase]:
    queries: list[QueryCase] = []
    for record in records:
        cue_text = ", ".join(record.cues[:3])
        queries.append(
            QueryCase(
                query_id=f"{record.tool}:{record.scenario}:named",
                target_scenario=record.scenario,
                tool=record.tool,
                query=(
                    f"Find the {record.tool} scenario named {record.scenario}. "
                    f"It should match these cues: {cue_text}."
                ),
            )
        )
        queries.append(
            QueryCase(
                query_id=f"{record.tool}:{record.scenario}:context",
                target_scenario=record.scenario,
                tool=record.tool,
                query=(
                    f"Which {record.tool} scenario handles tasks about {cue_text}? "
                    f"Prefer the scenario whose context best matches those cues."
                ),
            )
        )
        queries.append(
            QueryCase(
                query_id=f"{record.tool}:{record.scenario}:example",
                target_scenario=record.scenario,
                tool=record.tool,
                query=(
                    f"In {record.tool}, which scenario would answer a problem about {record.exemplar}? "
                    f"Match by specialization rather than by generic domain."
                ),
            )
        )
    return queries


def base_artifact(record: ScenarioRecord, source_id: str) -> Artifact:
    payload = {
        "scenario": record.scenario,
        "tool": record.tool,
        "domain": record.domain,
        "context": record.context,
        "cues": list(record.cues),
        "exemplar": record.exemplar,
    }
    metadata = {
        "scenario": record.scenario,
        "tool": record.tool,
        "domain": record.domain,
        "source_id": source_id,
    }
    text = (
        f"scenario: {record.scenario}\n"
        f"tool: {record.tool}\n"
        f"domain: {record.domain}\n"
        f"context: {record.context}\n"
        f"cues: {', '.join(record.cues)}\n"
        f"example: {record.exemplar}"
    )
    return Artifact(signature=f"{source_id}::{record.tool}::{record.scenario}", text=text, metadata=metadata, payload=payload)


def blend_lists(left: list[str], right: list[str], keep_left: int) -> list[str]:
    blended = list(left[:keep_left])
    for token in right:
        if token not in blended:
            blended.append(token)
    return blended


def contradictory_artifact(
    record: ScenarioRecord,
    wrong: ScenarioRecord,
    source_id: str,
    condition: str,
    rng: random.Random,
    conflict_rate: int,
) -> Artifact:
    keep_left = 2 if conflict_rate < 40 else 1
    payload = {
        "tool": record.tool,
        "domain": record.domain,
        "context": record.context,
        "scenario": record.scenario,
        "cues": list(record.cues),
        "exemplar": record.exemplar,
        "scenario_hint": wrong.scenario,
        "wrong_context": wrong.context,
        "wrong_cues": list(wrong.cues),
        "wrong_exemplar": wrong.exemplar,
    }
    metadata = {
        "scenario": record.scenario,
        "tool": record.tool,
        "domain": record.domain,
        "source_id": source_id,
        "conflicted": True,
        "contradicts": record.scenario,
    }

    if condition == "typed":
        payload["cues"] = blend_lists(record.cues, wrong.cues, 1 if conflict_rate >= 40 else keep_left)
        payload["context"] = f"{record.context} Alternate analysis: {wrong.context}"
        if rng.random() < 0.45 + (conflict_rate / 120.0):
            payload["exemplar"] = wrong.exemplar
    else:
        payload["scenario"] = record.scenario
        payload["domain"] = record.domain
        payload["cues"] = blend_lists(record.cues, wrong.cues, 2)
        payload["context"] = f"{record.context} Brief conflicting cue: {wrong.exemplar}."
        payload["exemplar"] = wrong.exemplar if rng.random() < 0.15 + (conflict_rate / 300.0) else record.exemplar

    text = (
        f"scenario: {payload['scenario']}\n"
        f"tool: {record.tool}\n"
        f"domain: {payload['domain']}\n"
        f"context: {payload['context']}\n"
        f"cues: {', '.join(payload['cues'])}\n"
        f"example: {payload['exemplar']}\n"
        f"alternate scenario: {wrong.scenario}\n"
        f"alternate cues: {', '.join(wrong.cues)}"
    )
    return Artifact(
        signature=f"{source_id}::conflict::{record.tool}::{record.scenario}::{wrong.scenario}",
        text=text,
        metadata=metadata,
        payload=payload,
    )


def tokenize_counter(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in normalize_text(text):
        counts[token] = counts.get(token, 0) + 1
    return counts


def cosine_similarity(left: Artifact, right: Artifact) -> float:
    lv = tokenize_counter(left.text)
    rv = tokenize_counter(right.text)
    overlap = set(lv) & set(rv)
    numerator = sum(lv[token] * rv[token] for token in overlap)
    left_norm = sum(value * value for value in lv.values()) ** 0.5
    right_norm = sum(value * value for value in rv.values()) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def cluster_artifacts(artifacts: list[Artifact], threshold: float = 0.84) -> list[list[Artifact]]:
    clusters: list[list[Artifact]] = []
    for artifact in artifacts:
        for cluster in clusters:
            if artifact.metadata.get("tool") != cluster[0].metadata.get("tool"):
                continue
            if cosine_similarity(artifact, cluster[0]) >= threshold:
                cluster.append(artifact)
                break
        else:
            clusters.append([artifact])
    return clusters


def flatten_cluster(cluster: list[Artifact]) -> tuple[str, dict[str, object], dict[str, object]]:
    metadata: dict[str, object] = {}
    payload: dict[str, object] = {}
    cues: list[str] = []
    exemplars: list[str] = []
    contexts: list[str] = []
    for artifact in cluster:
        metadata.update(artifact.metadata)
        payload.update(artifact.payload)
        for cue in artifact.payload.get("cues", []):
            if isinstance(cue, str) and cue not in cues:
                cues.append(cue)
        exemplar = artifact.payload.get("exemplar")
        if isinstance(exemplar, str) and exemplar not in exemplars:
            exemplars.append(exemplar)
        context = artifact.payload.get("context")
        if isinstance(context, str) and context not in contexts:
            contexts.append(context)
    payload["cues"] = cues[:4]
    if exemplars:
        payload["exemplar"] = exemplars[-1]
    if contexts:
        payload["context"] = " ".join(contexts[:2])
    text = (
        f"scenario: {payload.get('scenario', metadata.get('scenario', 'unknown'))}\n"
        f"tool: {payload.get('tool', metadata.get('tool', 'unknown'))}\n"
        f"domain: {payload.get('domain', metadata.get('domain', 'unknown'))}\n"
        f"context: {payload.get('context', '')}\n"
        f"cues: {', '.join(payload.get('cues', []))}\n"
        f"example: {payload.get('exemplar', '')}"
    )
    return text, metadata, payload


def merge_cluster(cluster: list[Artifact], condition: str) -> MergedArtifact:
    rep = max(cluster, key=lambda artifact: (len(artifact.text), len(artifact.payload)))
    if condition == "typed":
        return MergedArtifact(rep.signature, rep.text, dict(rep.metadata), dict(rep.payload), list(cluster))
    text, metadata, payload = flatten_cluster(cluster)
    return MergedArtifact(rep.signature, text, metadata, payload, list(cluster))


def retrieval_score(query: str, artifact: MergedArtifact) -> float:
    query_tokens = set(unique_tokens(query))
    score = 0.0
    tool = str(artifact.payload.get("tool") or artifact.metadata.get("tool") or "").lower()
    if tool:
        if tool in query.lower():
            score += 0.9
        else:
            score -= 0.6
    scenario = str(artifact.payload.get("scenario") or artifact.metadata.get("scenario") or "").lower()
    if scenario and scenario in query.lower():
        score += 1.4
    context = str(artifact.payload.get("context") or "").lower()
    for token in query_tokens:
        if token in context:
            score += 0.14
    for cue in artifact.payload.get("cues", []):
        if isinstance(cue, str) and cue.lower() in query.lower():
            score += 0.45
    exemplar = str(artifact.payload.get("exemplar") or "").lower()
    if exemplar and exemplar in query.lower():
        score += 0.9
    exemplar_tokens = [token for token in unique_tokens(exemplar) if token in query_tokens]
    score += 0.16 * len(exemplar_tokens)
    wrong_scenario = str(artifact.payload.get("scenario_hint") or "").lower()
    if wrong_scenario and wrong_scenario in query.lower():
        score += 0.25
    score += 0.03 * len(artifact.members)
    return score


def target_label(query_case: QueryCase) -> str:
    return query_case.target_scenario


def predicted_label(artifact: MergedArtifact) -> str:
    scenario = artifact.metadata.get("scenario")
    if isinstance(scenario, str) and scenario.strip():
        return scenario.strip()
    payload_scenario = artifact.payload.get("scenario")
    if isinstance(payload_scenario, str) and payload_scenario.strip():
        return payload_scenario.strip()
    return artifact.signature


def select_conflicted_indices(size: int, conflict_rate: int, seed: int) -> set[int]:
    count = round(size * conflict_rate / 100.0)
    count = max(0, min(size, count))
    if count == 0:
        return set()
    rng = random.Random(seed)
    return set(rng.sample(range(size), count))


def build_condition_artifacts(
    records: list[ScenarioRecord], conflict_rate: int, seed: int, condition: str
) -> tuple[list[Artifact], list[dict[str, Any]]]:
    rng = random.Random(seed * 1000 + conflict_rate)
    conflicted = select_conflicted_indices(len(records), conflict_rate, seed)
    artifacts: list[Artifact] = []
    conflicts: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        artifacts.append(base_artifact(record, "client_clean"))
        if idx not in conflicted:
            continue
        candidates = [candidate for j, candidate in enumerate(records) if j != idx and candidate.tool == record.tool]
        wrong = rng.choice(candidates)
        artifacts.append(contradictory_artifact(record, wrong, "client_conflict", condition, rng, conflict_rate))
        conflicts.append(
            {
                "target_scenario": record.scenario,
                "contradictory_scenario": wrong.scenario,
                "tool": record.tool,
            }
        )
    return artifacts, conflicts


def evaluate_condition(
    records: list[ScenarioRecord], queries: list[QueryCase], conflict_rate: int, seed: int, condition: str
) -> dict[str, Any]:
    artifacts, conflicts = build_condition_artifacts(records, conflict_rate, seed, condition)
    merged = [merge_cluster(cluster, condition) for cluster in cluster_artifacts(artifacts)]
    rows: list[dict[str, Any]] = []
    correct = 0
    for query_case in queries:
        ranked = sorted(merged, key=lambda artifact: retrieval_score(query_case.query, artifact), reverse=True)
        top = ranked[0]
        predicted = predicted_label(top)
        hit = predicted == target_label(query_case)
        correct += int(hit)
        rows.append(
            {
                "query_id": query_case.query_id,
                "query": query_case.query,
                "target_scenario": query_case.target_scenario,
                "predicted_scenario": predicted,
                "correct": hit,
                "top_signature": top.signature,
                "top_cluster_size": len(top.members),
                "top_score": retrieval_score(query_case.query, top),
            }
        )
    accuracy = correct / len(queries) if queries else 0.0
    return {
        "condition": condition,
        "seed": seed,
        "conflict_rate": conflict_rate,
        "sample_count": len(queries),
        "correct": correct,
        "accuracy": accuracy,
        "cluster_count": len(merged),
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
        "rows": rows,
    }


def summarize(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values) if values else 0.0
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, sd


def render_summary(condition: str, conflict_rate: int, results: list[dict[str, Any]]) -> str:
    values = [float(item["accuracy"]) for item in results]
    mean, sd = summarize(values)
    seed_str = ", ".join(f"{item['seed']}={item['accuracy']:.3f}" for item in results)
    return f"- {condition}, {conflict_rate}% conflict: {mean:.3f} ± {sd:.3f} ({seed_str})"


def main() -> None:
    args = parse_args()
    seeds = parse_int_list(args.seeds)
    conflict_rates = parse_int_list(args.conflict_rates)
    conditions = parse_str_list(args.conditions)
    records = build_records()
    queries = build_queries(records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis_input = {"typed": {}, "flat": {}}
    summary_lines = [
        "# Experiment 2 Question-Level Reconstruction",
        "",
        "- Records come directly from compendium scenario `context` fields.",
        "- Each scenario expands into three queries: named, context-only, and exemplar-style.",
        "- `typed` keeps a representative conflicting artifact; `flat` shallow-merges fields across the cluster.",
        "",
        "Per-condition results:",
    ]

    for condition in conditions:
        for rate in conflict_rates:
            per_seed: list[dict[str, Any]] = []
            for seed in seeds:
                result = evaluate_condition(records, queries, rate, seed, condition)
                per_seed.append(result)
                out_path = args.output_dir / condition / f"conflict_{rate:02d}" / f"seed_{seed}.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            summary_lines.append(render_summary(condition, rate, per_seed))
            if condition in analysis_input:
                analysis_input[condition][f"{rate}%"] = [float(item["accuracy"]) for item in per_seed]

    (args.output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    args.analysis_input.parent.mkdir(parents=True, exist_ok=True)
    args.analysis_input.write_text(json.dumps(analysis_input, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_dir / 'summary.md'}")
    print(f"Wrote {args.analysis_input}")


if __name__ == "__main__":
    main()
