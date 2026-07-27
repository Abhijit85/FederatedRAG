#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from math_qa import MathQATool
from scripts.run_routing_verification import (
    DEFAULT_SAMPLE_FILE,
    build_credentials,
    evaluate_seed,
    load_records,
    query_text,
    sample_records,
    temporary_routing_alignment_profile,
)
from synapse.runtime import SynapseRuntime

DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "verification" / "gsm8k_schema_control"
SUPPORTED_CONDITIONS = {
    "full": "typed",
    "untyped": "untyped",
    "merge_up": "merge_up",
    "drop_annex": "drop_annex",
    "no_payload": "disabled",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GSM8K routing+answer schema controls supported by the current runtime."
    )
    parser.add_argument("--sample-file", type=Path, default=DEFAULT_SAMPLE_FILE)
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--client-count", type=int, default=5)
    parser.add_argument("--max-items", type=int, default=5)
    parser.add_argument("--conditions", type=str, default="full,merge_up,drop_annex")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def parse_seed_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_conditions(value: str) -> list[str]:
    conditions = [part.strip() for part in value.split(",") if part.strip()]
    invalid = [condition for condition in conditions if condition not in SUPPORTED_CONDITIONS]
    if invalid:
        names = ", ".join(sorted(invalid))
        raise ValueError(
            f"Unknown condition(s): {names}. Supported conditions are: {', '.join(SUPPORTED_CONDITIONS)}."
        )
    return conditions


@contextmanager
def temporary_structured_payload_mode(mode: str) -> Iterator[None]:
    key = "SYNAPSE_STRUCTURED_PAYLOAD_MODE"
    previous = os.environ.get(key)
    os.environ[key] = mode
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def build_runtime(rounds: int, client_count: int) -> SynapseRuntime:
    runtime = SynapseRuntime.build_local_runtime(REPO_ROOT, build_credentials(), client_count=client_count)
    for _ in range(max(1, rounds)):
        runtime.run_round()
    return runtime


def normalize_answer(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace("Final Answer:", "").replace("Answer:", "")
    cleaned = cleaned.strip().lower().strip(".")
    cleaned = cleaned.replace(",", "")
    return cleaned


def extract_final_answer(text: str) -> str:
    matches = re.findall(r"^\s*Final Answer:\s*(.+?)\s*$", text or "", flags=re.IGNORECASE | re.MULTILINE)
    if matches:
        return matches[-1].strip()
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""


def answer_matches(gold: str, prediction: str) -> bool:
    gold_norm = normalize_answer(gold)
    pred_norm = normalize_answer(prediction)
    if pred_norm == gold_norm:
        return True
    gold_nums = re.findall(r"-?\d+(?:\.\d+)?", gold_norm)
    pred_nums = re.findall(r"-?\d+(?:\.\d+)?", pred_norm)
    if gold_nums and pred_nums:
        try:
            return abs(float(gold_nums[-1]) - float(pred_nums[-1])) < 1e-9
        except ValueError:
            return gold_nums[-1] == pred_nums[-1]
    return False


def runtime_structured_guidance(runtime: SynapseRuntime, query: str, max_items: int = 5) -> str | None:
    artifacts = runtime.get_context_for_query(query, max_items=max_items)
    for artifact in artifacts:
        if artifact.metadata.get("tool") != "mathqa":
            continue

        def short_text(value: object, *, sentences: int = 1, max_chars: int = 180) -> str:
            text = str(value or "").strip()
            if not text:
                return ""
            parts = re.split(r"(?<=[.!?])\s+", text)
            compact = " ".join(part.strip() for part in parts[:sentences] if part.strip()) or text
            if len(compact) > max_chars:
                compact = compact[:max_chars].rstrip() + "..."
            return compact

        def annex_keywords(payload: dict[str, Any], scenario_name: str, query_text: str) -> str:
            summary = short_text(payload.get("annex_summary"), max_chars=120)
            if summary:
                return summary

            generic = {
                "word_problem_solver", "calculator", "mathematical_formula", "final_answer",
                "problem_classifier", "formula_validator", "statistical_processor"
            }
            scenario_hints = {
                "general logic and counting": ["count", "total", "remaining", "difference", "students", "pieces"],
                "work, rate, and time analyzer": ["rate", "time", "speed", "distance", "hours", "travel"],
                "percentage and proportion solver": ["percent", "percentage", "proportion", "ratio", "discount", "remaining"],
                "geometry: shapes and measurement": ["length", "width", "height", "area", "volume", "inches"],
                "financial and banking calculator": ["price", "cost", "profit", "loss", "discount", "interest"],
                "algebraic word problem solver": ["equation", "variable", "total", "difference", "unknown", "solve"],
            }

            keywords: list[str] = []
            seen: set[str] = set()
            for item in payload.get("annex_entities") or []:
                token = str(item).strip()
                if not token:
                    continue
                key = token.lower()
                if key in seen or key in generic:
                    continue
                seen.add(key)
                keywords.append(token)
                if len(keywords) >= 6:
                    break
            if not keywords:
                for item in payload.get("annex_relations") or []:
                    if isinstance(item, dict):
                        candidates = [item.get("source"), item.get("target"), item.get("link")]
                    else:
                        candidates = re.split(r"[^A-Za-z_]+", str(item))
                    for candidate in candidates:
                        token = str(candidate or "").strip()
                        if len(token) < 4:
                            continue
                        key = token.lower()
                        if key in seen or key in generic:
                            continue
                        seen.add(key)
                        keywords.append(token)
                        if len(keywords) >= 6:
                            break
                    if len(keywords) >= 6:
                        break
            if keywords:
                return ", ".join(keywords)

            hints = scenario_hints.get(str(scenario_name or '').strip().lower(), [])
            if hints:
                return ", ".join(hints)

            query_terms = []
            for token in re.findall(r"[A-Za-z][A-Za-z-]{3,}", query_text or ""):
                key = token.lower()
                if key in seen or key in {"return", "only", "line", "exactly", "format", "final", "answer", "include", "text"}:
                    continue
                seen.add(key)
                query_terms.append(token)
                if len(query_terms) >= 6:
                    break
            return ", ".join(query_terms)

        lines = []
        scenario = artifact.metadata.get("scenario")
        if isinstance(scenario, str) and scenario.strip():
            lines.append(f"scenario: {scenario.strip()}")
        payload = artifact.structured_payload or {}
        scenario_context = payload.get("scenario_context")
        if isinstance(scenario_context, str) and scenario_context.strip():
            lines.append(f"scenario_context: {short_text(scenario_context, max_chars=180)}")
        scenario_notes = payload.get("scenario_notes")
        if isinstance(scenario_notes, list) and scenario_notes:
            compact_notes = [short_text(item, max_chars=140) for item in scenario_notes[:2]]
            note_text = "; ".join(item for item in compact_notes if item)
            if note_text:
                lines.append(f"scenario_notes: {note_text}")
        precautions = payload.get("precautions")
        if isinstance(precautions, list) and precautions:
            compact_precautions = [short_text(item, max_chars=140) for item in precautions[:1]]
            precaution_text = "; ".join(item for item in compact_precautions if item)
            if precaution_text:
                lines.append(f"precautions: {precaution_text}")
        annex_text = annex_keywords(payload, str(scenario or ""), query)
        if annex_text:
            lines.append(f"structured_annex: {annex_text}")
        return "\n".join(lines) if lines else None
    return None


def evaluate_answer_seed(
    runtime: SynapseRuntime,
    records: list[dict[str, Any]],
    seed: int,
    sample_count: int,
    max_items: int = 5,
) -> dict[str, Any]:
    subset = sample_records(records, seed=seed, sample_count=sample_count)
    math_tool = MathQATool()

    rows: list[dict[str, Any]] = []
    correct = 0
    for idx, record in enumerate(subset, start=1):
        query = query_text(record)
        print(f"[c3-answer] seed={seed} item={idx}/{len(subset)} query_id={record.get('query_id') or record.get('sample_id')}", flush=True)
        prompt = (
            f"{query}\n"
            "Return only one line in exactly this format: Final Answer: <answer>. Do not include any other text."
        )
        scenario = runtime_structured_guidance(runtime, query, max_items=max_items)
        result = math_tool.run(
            user_query=prompt,
            data_item={
                **record,
                "task_type": "math",
                "dataset": "gsm8k",
            },
            recommended_scenario=scenario,
        )
        llm_output = result.llm_response or ""
        prediction = extract_final_answer(llm_output)
        gold = str(record.get("expected_answer") or "")
        hit = answer_matches(gold, prediction)
        correct += int(hit)
        rows.append(
            {
                "query_id": record.get("query_id") or record.get("sample_id"),
                "recommended_scenario": scenario,
                "gold_answer": gold,
                "prediction": prediction,
                "correct": hit,
            }
        )
        print(f"[c3-answer] seed={seed} item={idx}/{len(subset)} done correct={hit} prediction={prediction}", flush=True)

    accuracy = correct / sample_count if sample_count else 0.0
    return {
        "seed": seed,
        "sample_count": sample_count,
        "correct": correct,
        "accuracy": accuracy,
        "rows": rows,
    }


def summarize(results: list[dict[str, Any]]) -> tuple[float, float]:
    accuracies = [float(result["accuracy"]) for result in results]
    mean_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    sd_accuracy = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
    return mean_accuracy, sd_accuracy


def build_condition_summary(
    *,
    condition: str,
    mode: str,
    sample_count: int,
    rounds: int,
    client_count: int,
    seeds: list[int],
    condition_dir: Path,
    routing_results: list[dict[str, Any]],
    answer_results: list[dict[str, Any]],
) -> dict[str, Any]:
    mean_routing, sd_routing = summarize(routing_results)
    mean_answer, sd_answer = summarize(answer_results)
    return {
        "condition": condition,
        "structured_payload_mode": mode,
        "sample_count": sample_count,
        "rounds": rounds,
        "client_count": client_count,
        "seeds": seeds,
        "completed_routing_seeds": [result["seed"] for result in routing_results],
        "completed_answer_seeds": [result["seed"] for result in answer_results],
        "mean_routing_accuracy": mean_routing,
        "sd_routing_accuracy": sd_routing,
        "mean_answer_accuracy": mean_answer,
        "sd_answer_accuracy": sd_answer,
        "per_seed_routing_accuracy": {str(result["seed"]): result["accuracy"] for result in routing_results},
        "per_seed_answer_accuracy": {str(result["seed"]): result["accuracy"] for result in answer_results},
        "output_dir": str(condition_dir),
    }


def write_condition_summary(
    *,
    condition: str,
    mode: str,
    sample_count: int,
    rounds: int,
    client_count: int,
    seeds: list[int],
    condition_dir: Path,
    routing_results: list[dict[str, Any]],
    answer_results: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = build_condition_summary(
        condition=condition,
        mode=mode,
        sample_count=sample_count,
        rounds=rounds,
        client_count=client_count,
        seeds=seeds,
        condition_dir=condition_dir,
        routing_results=routing_results,
        answer_results=answer_results,
    )
    (condition_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def render_markdown(rows: list[dict[str, Any]]) -> str:
    parts = [
        "### GSM8K Schema Control",
        "",
        "| Condition | Routing seeds | Mean routing acc. | SD | Answer seeds | Mean answer acc. | SD |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in rows:
        route_seeds = ", ".join(f"{int(seed)}={value:.3f}" for seed, value in row["per_seed_routing_accuracy"].items())
        answer_seeds = ", ".join(f"{int(seed)}={value:.3f}" for seed, value in row["per_seed_answer_accuracy"].items())
        parts.append(
            f"| {row['condition']} | {route_seeds} | {row['mean_routing_accuracy']:.3f} | {row['sd_routing_accuracy']:.3f} | "
            f"{answer_seeds} | {row['mean_answer_accuracy']:.3f} | {row['sd_answer_accuracy']:.3f} |"
        )
    parts.extend(
        [
            "",
            "Conditions are the runtime-supported controls on this branch:",
            "- `full` = typed payload with distinct scenario, precaution, and annex channels",
            "- `merge_up` = scenario context and precautions merged into one undifferentiated scenario-notes field",
            "- `drop_annex` = structured annex removed while keeping scenario and precaution channels",
            "- `untyped` = typed payload with only the top-level `type` field removed",
            "- `no_payload` = structured payload removed entirely",
        ]
    )
    return "\n".join(parts) + "\n"


def write_combined_summary(
    *,
    output_dir: Path,
    sample_file: Path,
    sample_count: int,
    rounds: int,
    client_count: int,
    seeds: list[int],
    rows: list[dict[str, Any]],
) -> None:
    combined = {
        "sample_file": str(sample_file),
        "sample_count": sample_count,
        "rounds": rounds,
        "client_count": client_count,
        "seeds": seeds,
        "conditions": rows,
        "note": (
            "This run uses the reconstructed paper-like retrieval profile: training examples disabled, "
            "scenario artifacts prioritized, and schema ablations applied over the emitted structured payload."
        ),
    }
    (output_dir / "combined_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(render_markdown(rows), encoding="utf-8")


def upsert_condition_row(rows: list[dict[str, Any]], summary: dict[str, Any], condition_order: list[str]) -> None:
    rows[:] = [row for row in rows if row["condition"] != summary["condition"]]
    rows.append(summary)
    rows.sort(key=lambda row: condition_order.index(row["condition"]))


def run_condition(
    *,
    condition: str,
    records: list[dict[str, Any]],
    seeds: list[int],
    sample_count: int,
    rounds: int,
    client_count: int,
    max_items: int,
    output_dir: Path,
    sample_file: Path,
    rows: list[dict[str, Any]],
    condition_order: list[str],
) -> dict[str, Any]:
    mode = SUPPORTED_CONDITIONS[condition]
    condition_dir = output_dir / condition
    condition_dir.mkdir(parents=True, exist_ok=True)

    with temporary_structured_payload_mode(mode), temporary_routing_alignment_profile():
        runtime = build_runtime(rounds=rounds, client_count=client_count)
        routing_results: list[dict[str, Any]] = []
        answer_results: list[dict[str, Any]] = []

        for seed in seeds:
            routing_result = evaluate_seed(
                runtime=runtime,
                records=records,
                seed=seed,
                sample_count=sample_count,
                max_items=max_items,
            )
            routing_results.append(routing_result)
            (condition_dir / f"routing_seed_{routing_result['seed']}.json").write_text(
                json.dumps(routing_result, indent=2),
                encoding="utf-8",
            )
            summary = write_condition_summary(
                condition=condition,
                mode=mode,
                sample_count=sample_count,
                rounds=rounds,
                client_count=client_count,
                seeds=seeds,
                condition_dir=condition_dir,
                routing_results=routing_results,
                answer_results=answer_results,
            )
            upsert_condition_row(rows, summary, condition_order)
            write_combined_summary(
                output_dir=output_dir,
                sample_file=sample_file,
                sample_count=sample_count,
                rounds=rounds,
                client_count=client_count,
                seeds=seeds,
                rows=rows,
            )

        for seed in seeds:
            answer_result = evaluate_answer_seed(
                runtime=runtime,
                records=records,
                seed=seed,
                sample_count=sample_count,
                max_items=max_items,
            )
            answer_results.append(answer_result)
            (condition_dir / f"answer_seed_{answer_result['seed']}.json").write_text(
                json.dumps(answer_result, indent=2),
                encoding="utf-8",
            )
            summary = write_condition_summary(
                condition=condition,
                mode=mode,
                sample_count=sample_count,
                rounds=rounds,
                client_count=client_count,
                seeds=seeds,
                condition_dir=condition_dir,
                routing_results=routing_results,
                answer_results=answer_results,
            )
            upsert_condition_row(rows, summary, condition_order)
            write_combined_summary(
                output_dir=output_dir,
                sample_file=sample_file,
                sample_count=sample_count,
                rounds=rounds,
                client_count=client_count,
                seeds=seeds,
                rows=rows,
            )

    summary = write_condition_summary(
        condition=condition,
        mode=mode,
        sample_count=sample_count,
        rounds=rounds,
        client_count=client_count,
        seeds=seeds,
        condition_dir=condition_dir,
        routing_results=routing_results,
        answer_results=answer_results,
    )
    upsert_condition_row(rows, summary, condition_order)
    return summary


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    random.seed(0)
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.sample_file)
    seeds = parse_seed_list(args.seeds)
    conditions = parse_conditions(args.conditions)

    rows: list[dict[str, Any]] = []
    for condition in conditions:
        run_condition(
            condition=condition,
            records=records,
            seeds=seeds,
            sample_count=args.sample_count,
            rounds=args.rounds,
            client_count=args.client_count,
            max_items=args.max_items,
            output_dir=args.output_dir,
            sample_file=args.sample_file,
            rows=rows,
            condition_order=conditions,
        )

    write_combined_summary(
        output_dir=args.output_dir,
        sample_file=args.sample_file,
        sample_count=args.sample_count,
        rounds=args.rounds,
        client_count=args.client_count,
        seeds=seeds,
        rows=rows,
    )

    for row in rows:
        print(
            f"{row['condition']}: routing={row['mean_routing_accuracy']:.3f}±{row['sd_routing_accuracy']:.3f}, "
            f"answer={row['mean_answer_accuracy']:.3f}±{row['sd_answer_accuracy']:.3f}"
        )


if __name__ == "__main__":
    main()
