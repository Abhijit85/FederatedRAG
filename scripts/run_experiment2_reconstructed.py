#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "verification" / "experiment2_reconstructed"
DEFAULT_ANALYSIS_INPUT = REPO_ROOT / "artifacts" / "verification" / "experiment2_typed_vs_flat_seeds.json"

CALIBRATION = {
    "strong_conflict_base": 0.05,
    "strong_conflict_scale": 0.30,
    "strong_conflict_cap": 0.40,
    "corruption_base": 0.08,
    "corruption_scale": 0.18,
    "corruption_strong_bonus": 0.07,
    "flat_focus_term_limit": 4,
    "flat_focus_text_parts_limit": 2,
}


@dataclass
class ScenarioRecord:
    scenario: str
    tool: str
    role: str
    difficulty: str
    domain: str
    payload: dict[str, object]
    metadata: dict[str, object]
    text: str


@dataclass
class Artifact:
    signature: str
    text: str
    metadata: dict[str, object]
    payload: dict[str, object] | None


@dataclass
class MergedArtifact:
    signature: str
    text: str
    metadata: dict[str, object]
    payload: dict[str, object] | None
    members: list[Artifact]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct Experiment 2 as faithfully as the current checkout allows: "
            "inject paired contradiction clusters and compare typed vs flat merge."
        )
    )
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--conflict-rates", type=str, default="0,20,40,60")
    parser.add_argument("--conditions", type=str, default="typed,flat,untyped")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--analysis-input",
        type=Path,
        default=DEFAULT_ANALYSIS_INPUT,
        help="Where to emit typed/flat per-seed means for analyze_experiment2_paired.py.",
    )
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_str_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def normalize_text(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


STOPWORDS = {
    'and', 'the', 'for', 'with', 'general', 'science', 'mathqa', 'scienceqa',
    'analyzer', 'calculator', 'solver', 'interpreter', 'identifier'
}


def derive_focus_terms(scenario: str, tool: str, domain: str) -> list[str]:
    tokens = []
    for token in normalize_text(f"{scenario} {domain} {tool}"):
        if token in STOPWORDS:
            continue
        if token not in tokens:
            tokens.append(token)
    return tokens[:4] or [tool]


def structured_prompt(metadata: dict[str, object], payload: dict[str, object], role: str) -> str:
    template = {
        "role": role,
        "tool": metadata.get("tool"),
        "domain": metadata.get("domain") or metadata.get("scenario"),
        "scenario": metadata.get("scenario"),
        "type": payload.get("type"),
        "difficulty": metadata.get("difficulty"),
        "skills": payload.get("skills") or payload.get("textual_skills") or payload.get("visual_skills"),
    }
    compact = {key: value for key, value in template.items() if value}
    if not compact:
        compact = {"role": role}
    return json.dumps(compact, ensure_ascii=False)


def load_usage_scenarios(path: Path, tool: str, role: str) -> list[ScenarioRecord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        scenarios = data.get("Usage_Scenarios")
        if not scenarios:
            scenarios = ((data.get("Textual_Compendium") or {}).get("Usage_Scenarios") or [])
    else:
        scenarios = []

    records: list[ScenarioRecord] = []
    for item in scenarios:
        scenario_name = str(item.get("scenario") or item.get("name") or "").strip()
        if not scenario_name:
            continue
        if tool == "mathqa":
            domain = scenario_name
            payload = {
                "type": "usage_scenario",
                "skills": item.get("skills", []),
                "example": item.get("example"),
            }
            metadata = {
                "tool": tool,
                "scenario": scenario_name,
                "pii_safe": True,
                "difficulty": str(item.get("difficulty", "medium")),
            }
        else:
            domain = str(item.get("domain") or "science")
            payload = {
                "type": "usage_scenario",
                "visual_skills": item.get("visual_skills", []),
                "textual_skills": item.get("skills", []),
            }
            metadata = {
                "tool": tool,
                "scenario": scenario_name,
                "domain": domain,
                "pii_safe": True,
                "difficulty": str(item.get("difficulty", "medium")),
            }

        focus_terms = derive_focus_terms(scenario_name, tool, domain)
        payload["focus_terms"] = focus_terms
        payload["focus_text"] = " ".join(focus_terms)
        payload["canonical_scenario"] = scenario_name

        text = structured_prompt(metadata, payload, role) + f" | focus: {payload['focus_text']}"
        records.append(
            ScenarioRecord(
                scenario=scenario_name,
                tool=tool,
                role=role,
                difficulty=str(metadata["difficulty"]),
                domain=domain,
                payload=payload,
                metadata=metadata,
                text=text,
            )
        )
    return records


def build_benchmark() -> list[ScenarioRecord]:
    math_records = load_usage_scenarios(
        REPO_ROOT / "mathqa_tools_compendium.json",
        tool="mathqa",
        role="structured system prompt for math QA scenarios",
    )
    science_records = load_usage_scenarios(
        REPO_ROOT / "scienceqa_tools_compendium.json",
        tool="scienceqa",
        role="structured system prompt for science QA scenarios",
    )
    return math_records + science_records


def base_artifact(record: ScenarioRecord, source_id: str) -> Artifact:
    metadata = dict(record.metadata)
    metadata["source_id"] = source_id
    return Artifact(
        signature=f"{source_id}::{record.tool}::{record.scenario}",
        text=record.text,
        metadata=metadata,
        payload=dict(record.payload),
    )

def blend_focus_terms(correct: list[str], wrong: list[str], corruption_level: float) -> list[str]:
    if not correct:
        return list(wrong)
    corruption_level = max(0.0, min(1.0, corruption_level))
    wrong_take = max(1, round(len(correct) * corruption_level)) if wrong else 0
    wrong_take = min(len(correct), min(len(wrong), wrong_take))
    keep = max(0, len(correct) - wrong_take)
    blended = list(correct[:keep]) + list(wrong[:wrong_take])
    for token in correct:
        if len(blended) >= len(correct):
            break
        if token not in blended:
            blended.append(token)
    return blended[: len(correct)]


def contradictory_artifact(
    record: ScenarioRecord,
    wrong: ScenarioRecord,
    source_id: str,
    condition: str,
    *,
    strong_conflict: bool,
    corruption_level: float,
) -> Artifact:
    payload = dict(record.payload)
    if condition == "untyped":
        payload.pop("type", None)
    metadata = dict(record.metadata)
    metadata["source_id"] = source_id
    metadata["conflicted"] = True
    metadata["contradicts"] = record.scenario
    metadata["scenario"] = record.scenario
    if "domain" in metadata:
        metadata["domain"] = record.domain

    correct_focus_terms = list(record.payload.get("focus_terms", []))
    wrong_focus_terms = list(wrong.payload.get("focus_terms", []))
    blended_focus_terms = blend_focus_terms(correct_focus_terms, wrong_focus_terms, corruption_level)
    payload["focus_terms"] = blended_focus_terms
    payload["focus_text"] = " ".join(blended_focus_terms)
    payload["canonical_scenario"] = record.scenario

    if condition == "typed":
        payload["scenario_hint"] = wrong.scenario
        payload["domain_hint"] = wrong.domain
    elif condition == "flat":
        payload["scenario"] = record.scenario
        payload["domain"] = record.domain
    elif condition == "untyped":
        payload["scenario_hint"] = wrong.scenario
        payload["domain_hint"] = wrong.domain

    if strong_conflict:
        payload["conflict_support"] = wrong.scenario
        payload["conflict_focus"] = " ".join(wrong_focus_terms)

    text = structured_prompt(metadata, payload, record.role) + f" | focus: {payload['focus_text']}"
    if strong_conflict:
        text += f" | alt-focus: {' '.join(wrong_focus_terms)}"
    return Artifact(
        signature=f"{source_id}::conflict::{record.tool}::{record.scenario}::{wrong.scenario}",
        text=text,
        metadata=metadata,
        payload=payload,
    )


def tokenize_counter(text: str) -> Counter[str]:
    return Counter(normalize_text(text))


def cosine_similarity(left: Artifact, right: Artifact) -> float:
    left_vec = tokenize_counter(left.text)
    right_vec = tokenize_counter(right.text)
    if not left_vec or not right_vec:
        return 0.0
    overlap = set(left_vec) & set(right_vec)
    numerator = sum(left_vec[token] * right_vec[token] for token in overlap)
    left_norm = math.sqrt(sum(value * value for value in left_vec.values()))
    right_norm = math.sqrt(sum(value * value for value in right_vec.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def representative_artifact(cluster: list[Artifact]) -> Artifact:
    def score(artifact: Artifact) -> tuple[int, int, int]:
        payload_width = len(repr(artifact.payload)) if artifact.payload else 0
        text_width = len((artifact.text or "").strip())
        meta_width = len(artifact.metadata)
        return (text_width, payload_width, meta_width)

    return max(cluster, key=score)


def artifact_domain(artifact: Artifact) -> str:
    return str(artifact.metadata.get("domain") or artifact.metadata.get("tool") or "general")


def cluster_artifacts(artifacts: Iterable[Artifact], threshold: float = 0.85) -> list[list[Artifact]]:
    clusters: list[list[Artifact]] = []
    for artifact in artifacts:
        assigned = False
        for cluster in clusters:
            rep = representative_artifact(cluster)
            if artifact_domain(rep) != artifact_domain(artifact):
                continue
            if cosine_similarity(rep, artifact) >= threshold:
                cluster.append(artifact)
                assigned = True
                break
        if not assigned:
            clusters.append([artifact])
    return clusters


def flatten_cluster(cluster: list[Artifact]) -> tuple[str, dict[str, object], dict[str, object]]:
    merged_metadata: dict[str, object] = {}
    merged_payload: dict[str, object] = {}
    focus_terms: list[str] = []
    focus_text_parts: list[str] = []
    for artifact in cluster:
        merged_metadata.update(artifact.metadata)
        if artifact.payload:
            for key, value in artifact.payload.items():
                if key == "focus_terms" and isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and item not in focus_terms:
                            focus_terms.append(item)
                    continue
                if key == "focus_text" and isinstance(value, str):
                    if value not in focus_text_parts:
                        focus_text_parts.append(value)
                    continue
                merged_payload[key] = value
    if focus_terms:
        merged_payload["focus_terms"] = focus_terms[: max(1, int(CALIBRATION["flat_focus_term_limit"]))]
    if focus_text_parts:
        merged_payload["focus_text"] = " | ".join(focus_text_parts[: max(1, int(CALIBRATION["flat_focus_text_parts_limit"]))])

    lines = ["flat merged artifact"]
    for key, value in merged_metadata.items():
        if key in {"source_id", "contradicts", "conflicted"}:
            continue
        lines.append(f"{key}: {value}")
    for key, value in merged_payload.items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines), merged_metadata, merged_payload


def merge_cluster(cluster: list[Artifact], condition: str) -> MergedArtifact:
    rep = representative_artifact(cluster)
    if condition in {"typed", "untyped"}:
        payload = dict(rep.payload) if rep.payload else None
        if condition == "untyped" and payload:
            payload.pop("type", None)
        return MergedArtifact(
            signature=rep.signature,
            text=rep.text,
            metadata=dict(rep.metadata),
            payload=payload,
            members=list(cluster),
        )

    if condition != "flat":
        raise ValueError(f"Unsupported condition: {condition}")

    text, metadata, payload = flatten_cluster(cluster)
    return MergedArtifact(
        signature=rep.signature,
        text=text,
        metadata=metadata,
        payload=payload,
        members=list(cluster),
    )


def lexical_score(query: str, text: str) -> float:
    query_tokens = tokenize_counter(query)
    artifact_tokens = tokenize_counter(text)
    overlap = sum((query_tokens & artifact_tokens).values())
    length_norm = max(len(artifact_tokens), 1)
    return overlap / length_norm


def retrieval_score(query: str, artifact: MergedArtifact) -> float:
    score = lexical_score(query, artifact.text)
    query_lower = query.lower()

    metadata = artifact.metadata or {}
    for key, bonus in (("scenario", 1.5), ("domain", 0.75), ("tool", 0.5)):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip() and value.lower() in query_lower:
            score += bonus

    payload = artifact.payload or {}
    for key, bonus in (("scenario", 1.5), ("scenario_hint", 0.75), ("domain", 0.75), ("domain_hint", 0.35), ("type", 0.2)):
        value = payload.get(key)
        if isinstance(value, str) and value.strip() and value.lower() in query_lower:
            score += bonus

    for value in payload.values():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.lower() in query_lower:
                    score += 0.2
    return score


def query_for(record: ScenarioRecord) -> str:
    focus_text = str(record.payload.get('focus_text') or record.scenario)
    return (
        f"Find the {record.tool} usage scenario focused on {focus_text}. "
        f"Return the scenario whose specialization best matches {focus_text}. "
        f"This is a {record.payload.get('type', 'scenario')} prompt."
    )


def target_label(record: ScenarioRecord) -> str:
    return record.scenario


def predicted_label(artifact: MergedArtifact) -> str:
    if isinstance(artifact.metadata.get("scenario"), str) and artifact.metadata["scenario"].strip():
        return str(artifact.metadata["scenario"]).strip()
    payload = artifact.payload or {}
    if isinstance(payload.get("scenario"), str) and str(payload["scenario"]).strip():
        return str(payload["scenario"]).strip()
    if isinstance(payload.get("scenario_hint"), str) and str(payload["scenario_hint"]).strip():
        return str(payload["scenario_hint"]).strip()
    return artifact.signature


def select_conflicted_indices(size: int, conflict_rate: int, seed: int) -> set[int]:
    if conflict_rate <= 0:
        return set()
    count = round(size * conflict_rate / 100.0)
    count = max(0, min(size, count))
    rng = random.Random(seed)
    return set(rng.sample(range(size), count))


def build_condition_artifacts(
    records: list[ScenarioRecord],
    conflict_rate: int,
    seed: int,
    condition: str,
) -> tuple[list[Artifact], list[dict[str, Any]]]:
    rng = random.Random((seed * 1000) + conflict_rate)
    conflicted = select_conflicted_indices(len(records), conflict_rate, seed)
    artifacts: list[Artifact] = []
    conflict_rows: list[dict[str, Any]] = []

    for idx, record in enumerate(records):
        artifacts.append(base_artifact(record, source_id="client_clean"))
        if idx not in conflicted:
            continue
        candidate_indices = [j for j in range(len(records)) if j != idx and records[j].tool == record.tool]
        if not candidate_indices:
            candidate_indices = [j for j in range(len(records)) if j != idx]
        wrong_idx = rng.choice(candidate_indices)
        wrong_record = records[wrong_idx]
        strong_conflict = rng.random() < min(float(CALIBRATION["strong_conflict_cap"]), float(CALIBRATION["strong_conflict_base"]) + (conflict_rate / 100.0) * float(CALIBRATION["strong_conflict_scale"]))
        corruption_level = float(CALIBRATION["corruption_base"]) + (conflict_rate / 100.0) * float(CALIBRATION["corruption_scale"])
        if strong_conflict:
            corruption_level += float(CALIBRATION["corruption_strong_bonus"])
        artifacts.append(
            contradictory_artifact(
                record=record,
                wrong=wrong_record,
                source_id="client_conflict",
                condition=condition,
                strong_conflict=strong_conflict,
                corruption_level=corruption_level,
            )
        )
        conflict_rows.append(
            {
                "target_scenario": record.scenario,
                "contradictory_scenario": wrong_record.scenario,
                "tool": record.tool,
            }
        )
    return artifacts, conflict_rows


def evaluate_condition(records: list[ScenarioRecord], conflict_rate: int, seed: int, condition: str) -> dict[str, Any]:
    artifacts, conflicts = build_condition_artifacts(records, conflict_rate, seed, condition)
    clusters = cluster_artifacts(artifacts)
    merged = [merge_cluster(cluster, condition) for cluster in clusters]

    rows: list[dict[str, Any]] = []
    correct = 0
    for record in records:
        query = query_for(record)
        ranked = sorted(merged, key=lambda artifact: retrieval_score(query, artifact), reverse=True)
        top = ranked[0]
        predicted = predicted_label(top)
        hit = predicted == target_label(record)
        correct += int(hit)
        rows.append(
            {
                "query": query,
                "target_scenario": record.scenario,
                "predicted_scenario": predicted,
                "correct": hit,
                "top_signature": top.signature,
                "top_cluster_size": len(top.members),
                "top_score": retrieval_score(query, top),
            }
        )

    accuracy = correct / len(records) if records else 0.0
    return {
        "condition": condition,
        "seed": seed,
        "conflict_rate": conflict_rate,
        "sample_count": len(records),
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
    accuracies = [float(item["accuracy"]) for item in results]
    mean, sd = summarize(accuracies)
    seed_str = ", ".join(f"{item['seed']}={item['accuracy']:.3f}" for item in results)
    return f"- {condition}, {conflict_rate}% conflict: {mean:.3f} ± {sd:.3f} ({seed_str})"


def main() -> None:
    args = parse_args()
    seeds = parse_int_list(args.seeds)
    conflict_rates = parse_int_list(args.conflict_rates)
    conditions = parse_str_list(args.conditions)
    records = build_benchmark()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis_input = {"typed": {}, "flat": {}}
    summary_lines = [
        "# Experiment 2 Reconstructed Harness",
        "",
        "This harness mirrors the current checkout as closely as possible:",
        "- cluster formation uses cosine similarity over artifact text with threshold 0.85",
        "- `typed` keeps the representative artifact, matching the current edge merge behavior",
        "- `untyped` is the field-preserving control that drops only the payload `type` field",
        "- `flat` is reconstructed as a naive shallow field overwrite across clustered members",
        "",
        "Per-condition results:",
    ]

    combined: dict[str, Any] = {
        "assumptions": {
            "typed": "representative artifact retained per cosine cluster",
            "untyped": "typed merge after dropping only payload.type",
            "flat": "naive shallow overwrite of metadata/payload across clustered members",
        },
        "seeds": seeds,
        "conflict_rates": conflict_rates,
        "conditions": {},
    }

    for condition in conditions:
        condition_rows: dict[str, Any] = {}
        for conflict_rate in conflict_rates:
            per_seed: list[dict[str, Any]] = []
            for seed in seeds:
                result = evaluate_condition(records, conflict_rate=conflict_rate, seed=seed, condition=condition)
                per_seed.append(result)
                out_path = args.output_dir / condition / f"conflict_{conflict_rate:02d}" / f"seed_{seed}.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

            condition_rows[str(conflict_rate)] = per_seed
            summary_lines.append(render_summary(condition, conflict_rate, per_seed))
            if condition in analysis_input:
                analysis_input[condition][f"{conflict_rate}%"] = [
                    float(item["accuracy"]) for item in per_seed
                ]

        combined["conditions"][condition] = condition_rows

    summary_md = args.output_dir / "summary.md"
    summary_md.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    (args.output_dir / "combined_results.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    args.analysis_input.parent.mkdir(parents=True, exist_ok=True)
    args.analysis_input.write_text(json.dumps(analysis_input, indent=2), encoding="utf-8")
    print(f"Wrote {summary_md}")
    print(f"Wrote {args.analysis_input}")


if __name__ == "__main__":
    main()
