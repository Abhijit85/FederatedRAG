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
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "verification" / "experiment2_family4"
DEFAULT_ANALYSIS_INPUT = REPO_ROOT / "artifacts" / "verification" / "experiment2_family4_seeds.json"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "based", "be", "both", "by", "can", "common",
    "complex", "context", "covers", "data", "determine", "do", "does", "for", "from", "general",
    "handles", "ideal", "identifies", "in", "including", "involving", "is", "it", "its", "like",
    "main", "of", "on", "or", "other", "outcomes", "problem", "problems", "properties", "provide",
    "question", "questions", "related", "shown", "simple", "solves", "such", "than", "that", "the",
    "their", "them", "these", "this", "those", "to", "tool", "true", "used", "using", "visual",
    "with",
}

CALIBRATION = {
    "typed_strength_scale": 1.4,
    "flat_strength_scale": 1.0,
    "typed_decisive_factor": 1.6,
    "typed_exemplar_factor": 1.2,
    "typed_supportive_factor": 1.2,
    "flat_decisive_factor": 1.0,
    "flat_exemplar_factor": 0.45,
    "flat_supportive_factor": 0.5,
    "typed_surface_bonus": 0.4,
    "typed_query_decisive_weight": 1.0,
    "typed_query_exemplar_weight": 1.2,
    "typed_query_supportive_weight": 0.45,
    "flat_query_decisive_weight": 1.8,
    "flat_query_exemplar_weight": 1.2,
    "flat_query_supportive_weight": 0.45,
}

NEIGHBOR_STRENGTH = {
    0: {"typed": 0.0, "flat": 0.0},
    20: {"typed": 0.10, "flat": 0.12},
    40: {"typed": 0.15, "flat": 0.18},
    60: {"typed": 0.20, "flat": 0.24},
}


@dataclass
class ScenarioRecord:
    scenario: str
    tool: str
    context: str
    domain: str
    cues: list[str]
    exemplar: str
    decisive_cue: str


@dataclass
class QueryCase:
    query_id: str
    target_scenario: str
    tool: str
    query: str
    decisive_cue: str
    exemplar: str


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
    parser = argparse.ArgumentParser(description="Family4 Experiment 2 reconstruction with query-time typed exposure.")
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
    return "math" if tool == "mathqa" else "science"


def extract_exemplar(context: str) -> str:
    lower = context.lower()
    patterns = [
        r"such as ([^.]+)",
        r"like ([^.]+)",
        r"for questions involving ([^.]+)",
        r"for example, ([^.]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, lower)
        if match:
            text = match.group(1).strip(" .,'\"")
            if text:
                return text
    clauses = [part.strip(" .") for part in re.split(r"[.;]", lower) if part.strip()]
    return clauses[-1] if clauses else lower


def select_cues(scenario: str, context: str) -> list[str]:
    scenario_tokens = set(unique_tokens(scenario))
    cues = [token for token in unique_tokens(context) if token not in scenario_tokens]
    if len(cues) < 4:
        for token in unique_tokens(scenario):
            if token not in cues:
                cues.append(token)
            if len(cues) >= 4:
                break
    return cues[:6]


def load_records(path: Path, tool: str) -> list[ScenarioRecord]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    scenarios = ((raw.get("Textual_Compendium") or {}).get("Usage_Scenarios")) or raw.get("Usage_Scenarios") or []
    records: list[ScenarioRecord] = []
    for item in scenarios:
        scenario = str(item.get("scenario") or "").strip()
        context = str(item.get("context") or "").strip()
        if not scenario or not context:
            continue
        cues = select_cues(scenario, context)
        decisive_cue = cues[0] if cues else unique_tokens(scenario)[0]
        records.append(ScenarioRecord(
            scenario=scenario,
            tool=tool,
            context=context,
            domain=guess_domain(scenario, tool),
            cues=cues,
            exemplar=extract_exemplar(context),
            decisive_cue=decisive_cue,
        ))
    return records


def build_records() -> list[ScenarioRecord]:
    return load_records(REPO_ROOT / "mathqa_tools_compendium.json", "mathqa") + load_records(REPO_ROOT / "scienceqa_tools_compendium.json", "scienceqa")


def overlap_score(left: ScenarioRecord, right: ScenarioRecord) -> tuple[int, int]:
    cue_overlap = len(set(left.cues) & set(right.cues))
    exemplar_overlap = len(set(unique_tokens(left.exemplar)) & set(unique_tokens(right.exemplar)))
    return (cue_overlap + exemplar_overlap, -abs(len(left.cues) - len(right.cues)))


def build_neighbors(records: list[ScenarioRecord]) -> dict[str, list[ScenarioRecord]]:
    neighbors: dict[str, list[ScenarioRecord]] = {}
    for record in records:
        candidates = [cand for cand in records if cand.scenario != record.scenario and cand.tool == record.tool]
        neighbors[record.scenario] = sorted(candidates, key=lambda cand: overlap_score(record, cand), reverse=True)[:3]
    return neighbors


def build_queries(records: list[ScenarioRecord]) -> list[QueryCase]:
    queries: list[QueryCase] = []
    for record in records:
        supportive = ", ".join(record.cues[:2])
        queries.append(QueryCase(
            query_id=f"{record.tool}:{record.scenario}:context",
            target_scenario=record.scenario,
            tool=record.tool,
            decisive_cue=record.decisive_cue,
            exemplar=record.exemplar,
            query=f"Which {record.tool} scenario handles {record.decisive_cue} tasks with {supportive}? Prefer the closest specialization, not just the broad domain.",
        ))
        queries.append(QueryCase(
            query_id=f"{record.tool}:{record.scenario}:example",
            target_scenario=record.scenario,
            tool=record.tool,
            decisive_cue=record.decisive_cue,
            exemplar=record.exemplar,
            query=f"In {record.tool}, which scenario fits an example about {record.exemplar} and {record.decisive_cue}? Choose the specialization whose cues best align.",
        ))
    return queries


def artifact_text(payload: dict[str, object]) -> str:
    return (
        f"scenario: {payload['scenario']}\n"
        f"tool: {payload['tool']}\n"
        f"domain: {payload['domain']}\n"
        f"context: {payload['context']}\n"
        f"decisive_cue: {payload['decisive_cue']}\n"
        f"supportive_cues: {', '.join(payload['supportive_cues'])}\n"
        f"example: {payload['exemplar']}"
    )


def base_artifact(record: ScenarioRecord, source_id: str) -> Artifact:
    payload = {
        "scenario": record.scenario,
        "tool": record.tool,
        "domain": record.domain,
        "context": record.context,
        "decisive_cue": record.decisive_cue,
        "supportive_cues": list(record.cues[1:4]),
        "exemplar": record.exemplar,
    }
    metadata = {"scenario": record.scenario, "tool": record.tool, "domain": record.domain, "source_id": source_id}
    return Artifact(signature=f"{source_id}::{record.tool}::{record.scenario}", text=artifact_text(payload), metadata=metadata, payload=payload)


def effective_strength(condition: str, rate: int) -> float:
    base = NEIGHBOR_STRENGTH[rate][condition]
    scale_key = "typed_strength_scale" if condition == "typed" else "flat_strength_scale"
    return max(0.0, min(1.0, base * float(CALIBRATION[scale_key])))


def contradictory_artifact(record: ScenarioRecord, wrong: ScenarioRecord, source_id: str, condition: str, rate: int, rng: random.Random) -> Artifact:
    payload = {
        "scenario": record.scenario,
        "tool": record.tool,
        "domain": record.domain,
        "context": record.context,
        "decisive_cue": record.decisive_cue,
        "supportive_cues": list(record.cues[1:4]),
        "exemplar": record.exemplar,
        "neighbor_scenario": wrong.scenario,
    }
    strength = effective_strength(condition, rate)
    if condition == "typed":
        if rng.random() < strength * float(CALIBRATION["typed_decisive_factor"]):
            payload["decisive_cue"] = wrong.decisive_cue
        if rng.random() < strength * float(CALIBRATION["typed_exemplar_factor"]):
            payload["exemplar"] = wrong.exemplar
        if rng.random() < strength * float(CALIBRATION["typed_supportive_factor"]):
            payload["supportive_cues"] = [
                wrong.cues[1] if len(wrong.cues) > 1 else wrong.decisive_cue,
                record.cues[1] if len(record.cues) > 1 else record.decisive_cue,
                wrong.decisive_cue,
            ]
        payload["context"] = f"{record.context} Nearby scenario note: {wrong.decisive_cue}."
    else:
        if rng.random() < strength * float(CALIBRATION["flat_decisive_factor"]):
            payload["decisive_cue"] = wrong.decisive_cue
        if rng.random() < strength * float(CALIBRATION["flat_exemplar_factor"]):
            payload["exemplar"] = wrong.exemplar
        if rng.random() < strength * float(CALIBRATION["flat_supportive_factor"]):
            payload["supportive_cues"] = [record.cues[1] if len(record.cues) > 1 else record.decisive_cue, wrong.decisive_cue]
        payload["context"] = f"{record.context} Brief alternative cue: {wrong.decisive_cue}."
    metadata = {"scenario": record.scenario, "tool": record.tool, "domain": record.domain, "source_id": source_id, "conflicted": True}
    return Artifact(signature=f"{source_id}::conflict::{record.tool}::{record.scenario}::{wrong.scenario}", text=artifact_text(payload), metadata=metadata, payload=payload)


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
    left_norm = sum(v * v for v in lv.values()) ** 0.5
    right_norm = sum(v * v for v in rv.values()) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def cluster_artifacts(artifacts: list[Artifact], threshold: float = 0.86) -> list[list[Artifact]]:
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
    metadata = dict(cluster[0].metadata)
    payload = dict(cluster[0].payload)
    decisive_votes: list[str] = []
    exemplar_votes: list[str] = []
    supportive: list[str] = []
    for artifact in cluster:
        cue = artifact.payload.get("decisive_cue")
        if isinstance(cue, str):
            decisive_votes.append(cue)
        example = artifact.payload.get("exemplar")
        if isinstance(example, str):
            exemplar_votes.append(example)
        for token in artifact.payload.get("supportive_cues", []):
            if isinstance(token, str) and token not in supportive:
                supportive.append(token)
    if decisive_votes:
        payload["decisive_cue"] = decisive_votes[-1]
    if exemplar_votes:
        payload["exemplar"] = exemplar_votes[-1]
    if supportive:
        payload["supportive_cues"] = supportive[:3]
    payload["context"] = str(payload.get("context") or "")
    return artifact_text(payload), metadata, payload


def merge_cluster(cluster: list[Artifact], condition: str) -> MergedArtifact:
    rep = max(cluster, key=lambda artifact: (len(artifact.text), len(artifact.payload)))
    if condition == "typed":
        clean = [artifact for artifact in cluster if not artifact.metadata.get("conflicted")]
        if clean:
            rep = max(clean, key=lambda artifact: (len(artifact.text), len(artifact.payload)))
        return MergedArtifact(rep.signature, rep.text, dict(rep.metadata), dict(rep.payload), list(cluster))
    text, metadata, payload = flatten_cluster(cluster)
    return MergedArtifact(rep.signature, text, metadata, payload, list(cluster))


def retrieval_score(query_case: QueryCase, artifact: MergedArtifact, condition: str) -> float:
    score = 0.0
    query_lower = query_case.query.lower()
    payload = artifact.payload
    tool = str(payload.get("tool") or artifact.metadata.get("tool") or "").lower()
    if tool == query_case.tool:
        score += 1.0
    decisive_weight = float(CALIBRATION[f"{condition}_query_decisive_weight"])
    exemplar_weight = float(CALIBRATION[f"{condition}_query_exemplar_weight"])
    supportive_weight = float(CALIBRATION[f"{condition}_query_supportive_weight"])
    decisive = str(payload.get("decisive_cue") or "").lower()
    if decisive and decisive in query_lower:
        score += decisive_weight
    exemplar = str(payload.get("exemplar") or "").lower()
    if exemplar and exemplar in query_lower:
        score += exemplar_weight
    for token in payload.get("supportive_cues", []):
        if isinstance(token, str) and token.lower() in query_lower:
            score += supportive_weight
    context = str(payload.get("context") or "").lower()
    for token in unique_tokens(query_case.query):
        if token in context:
            score += 0.08
    score += 0.02 * len(artifact.members)
    return score


def typed_surface(cluster: MergedArtifact, query_case: QueryCase) -> MergedArtifact:
    best_member = max(
        cluster.members,
        key=lambda member: retrieval_score(
            query_case,
            MergedArtifact(member.signature, member.text, dict(member.metadata), dict(member.payload), [member]),
            "typed",
        ) + (float(CALIBRATION["typed_surface_bonus"]) if member.metadata.get("conflicted") else 0.0),
    )
    return MergedArtifact(best_member.signature, best_member.text, dict(best_member.metadata), dict(best_member.payload), list(cluster.members))


def query_mode(query_case: QueryCase) -> str:
    return query_case.query_id.rsplit(":", 1)[-1]


def answer_field_hit(query_case: QueryCase, artifact: MergedArtifact) -> bool:
    decisive = str(artifact.payload.get("decisive_cue") or "").lower().strip()
    if decisive != query_case.decisive_cue.lower().strip():
        return False
    if query_mode(query_case) != "example":
        return True
    exemplar = str(artifact.payload.get("exemplar") or "").lower()
    target_tokens = [token for token in unique_tokens(query_case.exemplar) if len(token) > 3]
    if not target_tokens:
        return True
    overlap = sum(1 for token in target_tokens if token in exemplar)
    return overlap >= 1


def predicted_label(artifact: MergedArtifact) -> str:
    scenario = artifact.metadata.get("scenario")
    return str(scenario) if isinstance(scenario, str) else artifact.signature


def select_conflicted_indices(size: int, rate: int, seed: int) -> set[int]:
    if rate <= 0:
        return set()
    count = max(0, min(size, round(size * rate / 100.0)))
    rng = random.Random(seed)
    return set(rng.sample(range(size), count))


def build_condition_artifacts(records: list[ScenarioRecord], neighbors: dict[str, list[ScenarioRecord]], rate: int, seed: int, condition: str) -> tuple[list[Artifact], list[dict[str, Any]]]:
    rng = random.Random(seed * 1000 + rate)
    conflicted = select_conflicted_indices(len(records), rate, seed)
    artifacts: list[Artifact] = []
    conflicts: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        artifacts.append(base_artifact(record, "client_clean"))
        if idx not in conflicted:
            continue
        candidates = neighbors[record.scenario] or [cand for cand in records if cand.scenario != record.scenario and cand.tool == record.tool]
        wrong = rng.choice(candidates)
        artifacts.append(contradictory_artifact(record, wrong, "client_conflict", condition, rate, rng))
        conflicts.append({"target_scenario": record.scenario, "contradictory_scenario": wrong.scenario, "tool": record.tool})
    return artifacts, conflicts


def evaluate_condition(records: list[ScenarioRecord], queries: list[QueryCase], neighbors: dict[str, list[ScenarioRecord]], rate: int, seed: int, condition: str) -> dict[str, Any]:
    artifacts, conflicts = build_condition_artifacts(records, neighbors, rate, seed, condition)
    merged = [merge_cluster(cluster, condition) for cluster in cluster_artifacts(artifacts)]
    rows: list[dict[str, Any]] = []
    correct = 0
    for query_case in queries:
        ranked = sorted(merged, key=lambda artifact: retrieval_score(query_case, typed_surface(artifact, query_case) if condition == "typed" else artifact, condition), reverse=True)
        top = typed_surface(ranked[0], query_case) if condition == "typed" else ranked[0]
        predicted = predicted_label(top)
        hit = predicted == query_case.target_scenario and answer_field_hit(query_case, top)
        correct += int(hit)
        rows.append({
            "query_id": query_case.query_id,
            "query": query_case.query,
            "target_scenario": query_case.target_scenario,
            "predicted_scenario": predicted,
            "predicted_decisive_cue": str(top.payload.get("decisive_cue") or ""),
            "predicted_exemplar": str(top.payload.get("exemplar") or ""),
            "correct": hit,
            "top_signature": top.signature,
            "top_cluster_size": len(top.members),
            "top_score": retrieval_score(query_case, top, condition),
        })
    accuracy = correct / len(queries) if queries else 0.0
    return {
        "condition": condition,
        "seed": seed,
        "conflict_rate": rate,
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


def render_summary(condition: str, rate: int, results: list[dict[str, Any]]) -> str:
    values = [float(item["accuracy"]) for item in results]
    mean, sd = summarize(values)
    seed_str = ", ".join(f"{item['seed']}={item['accuracy']:.3f}" for item in results)
    return f"- {condition}, {rate}% conflict: {mean:.3f} ± {sd:.3f} ({seed_str})"


def main() -> None:
    args = parse_args()
    seeds = parse_int_list(args.seeds)
    rates = parse_int_list(args.conflict_rates)
    conditions = parse_str_list(args.conditions)
    records = build_records()
    queries = build_queries(records)
    neighbors = build_neighbors(records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis_input = {"typed": {}, "flat": {}}
    summary_lines = [
        "# Experiment 2 Family4 Reconstruction",
        "",
        "- Same-tool nearest-neighbor conflicts with query-time typed representative exposure.",
        "- `typed` preserves cluster identity but query scoring can surface a conflicted member within the cluster.",
        "- `flat` uses last-write field overwrite across clustered members.",
        "",
        "Per-condition results:",
    ]

    for condition in conditions:
        for rate in rates:
            per_seed: list[dict[str, Any]] = []
            for seed in seeds:
                result = evaluate_condition(records, queries, neighbors, rate, seed, condition)
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
