#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from math_qa import JinaAIClient, MongoRAGManager  # noqa: E402
from scripts.run_gsm8k_router_cascade import (  # noqa: E402
    RouterSpec,
    ScenarioDoc,
    labels_match,
    local_router_decision,
    resolve_router_spec,
)
from scripts.run_routing_verification import build_credentials, gold_route_label, load_records, query_text  # noqa: E402


DEFAULT_SAMPLE_FILE = REPO_ROOT / "GSM8K_500_rebuttal_run" / "GSM8K_500_samples.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "verification" / "gsm8k_jina_mongo_8b_router"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run GSM8K routing through the memory-path pipeline: Jina query embedding -> "
            "Mongo vector search -> candidate scenario list -> local 8B rerank."
        )
    )
    parser.add_argument("--sample-file", type=Path, default=DEFAULT_SAMPLE_FILE)
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--seeds", type=str, default="42,123,456,789,1024")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--search-k", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument(
        "--decision-mode",
        type=str,
        default="parse_or_score",
        choices=("parse_or_score", "score_only", "fused_score", "retrieval_top1", "retrieval_margin_gate"),
    )
    parser.add_argument(
        "--retrieval-margin-threshold",
        type=float,
        default=0.0,
        help="If decision-mode=retrieval_margin_gate, accept retrieval top-1 when top1-top2 score gap exceeds this threshold.",
    )
    parser.add_argument(
        "--retrieval-alpha",
        type=float,
        default=0.0,
        help="Weight applied to retrieval rank prior when decision-mode=fused_score.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/mnt/shared/shared_hf_home/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
    )
    parser.add_argument("--mongo-collection", type=str, default=os.environ.get("MATHQA_COLLECTION", "math_problems"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_seed_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def sample_records(records: list[dict[str, Any]], seed: int, sample_count: int) -> list[dict[str, Any]]:
    if sample_count > len(records):
        raise ValueError(f"Requested {sample_count} rows, but only {len(records)} are available.")
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(records)), sample_count))
    return [records[idx] for idx in indices]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return float("-inf")
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
    right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return float("-inf")
    return dot / (left_norm * right_norm)


def mongo_doc_to_scenario(doc: dict[str, Any]) -> ScenarioDoc | None:
    metadata = doc.get("metadata") or {}
    raw_text = str(doc.get("text") or "").strip()
    scenario = metadata.get("scenario")
    if not isinstance(scenario, str) or not scenario.strip():
        marker = "Tool Scenario:"
        if marker in raw_text:
            after = raw_text.split(marker, 1)[1].strip()
            scenario = after.split(". Context:", 1)[0].strip()
    if not isinstance(scenario, str) or not scenario.strip():
        return None
    context = metadata.get("scenario_context")
    if (not isinstance(context, str) or not context.strip()) and ". Context:" in raw_text:
        context = raw_text.split(". Context:", 1)[1].strip()
    merged_notes = metadata.get("merged_notes")
    annex_terms = metadata.get("annex_terms") or []
    parts: list[str] = []
    if isinstance(context, str) and context.strip():
        parts.append(context.strip())
    if isinstance(merged_notes, str) and merged_notes.strip():
        parts.append(merged_notes.strip())
    if isinstance(annex_terms, list) and annex_terms:
        annex = ", ".join(str(term).strip() for term in annex_terms[:8] if str(term).strip())
        if annex:
            parts.append(f"Annex terms: {annex}")
    text = " ".join(parts).strip() or raw_text or f"Routing scenario label: {scenario.strip()}."
    return ScenarioDoc(label=scenario.strip(), text=text, embedding=[])


def retrieve_candidates(
    manager: MongoRAGManager, query: str, *, top_k: int, search_k: int, exclude_query_id: str | None = None
) -> tuple[list[ScenarioDoc], list[dict[str, Any]], list[float]]:
    query_embedding = manager.jina_client.get_embeddings([query])[0]
    try:
        raw = manager.vector_store.search(query_embedding, num_results=max(search_k, top_k))
        if not raw:
            raise RuntimeError("empty vector search result")
    except Exception:
        raw = manager.vector_store.search_manual(query_embedding, num_results=max(search_k, top_k))
    filtered = manager._filter_math_documents(raw, max(search_k, top_k))

    docs: list[ScenarioDoc] = []
    raw_rows: list[dict[str, Any]] = []
    doc_scores: list[float] = []
    seen: set[str] = set()
    next_rank = 1
    for doc in filtered:
        metadata = doc.get("metadata") or {}
        if exclude_query_id and str(metadata.get("query_id") or "") == exclude_query_id:
            continue
        rank = next_rank
        next_rank += 1
        scenario = mongo_doc_to_scenario(doc)
        if scenario is None:
            continue
        norm = scenario.label.strip().lower()
        if norm in seen:
            continue
        seen.add(norm)
        score = doc.get("score")
        if score is None:
            score = cosine_similarity(query_embedding, doc.get("embedding") or [])
        docs.append(scenario)
        doc_scores.append(float(score))
        raw_rows.append(
            {
                "rank": rank,
                "scenario": scenario.label,
                "retrieval_prior": 1.0 / rank,
                "retrieval_score": float(score),
                "tool": (doc.get("metadata") or {}).get("tool"),
                "scenario_context": (doc.get("metadata") or {}).get("scenario_context"),
                "text_preview": str(doc.get("text") or "")[:220],
            }
        )
        if len(docs) >= top_k:
            break
    return docs, raw_rows, doc_scores


def choose_prediction(
    *,
    decision: Any | None,
    candidates: list[ScenarioDoc],
    retrieved_rows: list[dict[str, Any]],
    decision_mode: str,
    retrieval_alpha: float,
    retrieval_margin_threshold: float,
) -> tuple[str, dict[str, float] | None, float | None, bool]:
    retrieval_prior = {
        str(row.get("scenario")): float(row.get("retrieval_prior") or 0.0)
        for row in retrieved_rows
        if row.get("scenario")
    }
    top1_score = float(retrieved_rows[0].get("retrieval_score")) if retrieved_rows else None
    top2_score = float(retrieved_rows[1].get("retrieval_score")) if len(retrieved_rows) > 1 else None
    retrieval_margin = (top1_score - top2_score) if top1_score is not None and top2_score is not None else None
    if decision_mode == "retrieval_top1":
        return candidates[0].label, None, retrieval_margin, False
    if decision_mode == "retrieval_margin_gate" and retrieval_margin is not None and retrieval_margin >= retrieval_margin_threshold:
        return candidates[0].label, None, retrieval_margin, False
    if decision is None:
        return candidates[0].label, None, retrieval_margin, False
    option_scores = decision.option_scores_logprob or {}
    if decision_mode == "parse_or_score":
        return decision.predicted_label, None, retrieval_margin, True
    fused_scores: dict[str, float] = {}
    for candidate in candidates:
        label = candidate.label
        base = float(option_scores.get(label, float("-inf")))
        if decision_mode == "fused_score" or decision_mode == "retrieval_margin_gate":
            base += retrieval_alpha * retrieval_prior.get(label, 0.0)
        fused_scores[label] = base
    predicted = max(fused_scores.items(), key=lambda item: item[1])[0] if fused_scores else decision.predicted_label
    return predicted, fused_scores, retrieval_margin, True


def evaluate_seed(
    *,
    records: list[dict[str, Any]],
    seed: int,
    sample_count: int,
    manager: MongoRAGManager,
    spec: RouterSpec,
    top_k: int,
    search_k: int,
    max_tokens: int,
    decision_mode: str,
    retrieval_alpha: float,
    retrieval_margin_threshold: float,
) -> dict[str, Any]:
    subset = sample_records(records, seed=seed, sample_count=sample_count)
    rows: list[dict[str, Any]] = []
    correct = 0
    total_latency = 0.0
    empty_candidates = 0
    deferred_count = 0

    for record in subset:
        query = query_text(record)
        gold = gold_route_label(record)
        query_id = str(record.get("query_id") or record.get("sample_id") or "")
        candidates, retrieved_rows, _doc_scores = retrieve_candidates(
            manager,
            query,
            top_k=top_k,
            search_k=search_k,
            exclude_query_id=query_id or None,
        )
        if not candidates:
            empty_candidates += 1
            rows.append(
                {
                    "query_id": record.get("query_id") or record.get("sample_id"),
                    "query_text": query,
                    "ground_truth_domain": gold,
                    "predicted_domain": "",
                    "routed_correctly": False,
                    "top_candidates": [],
                    "retrieved_rows": [],
                    "empty_candidate_set": True,
                }
            )
            continue
        top1_score = float(retrieved_rows[0].get("retrieval_score")) if retrieved_rows else None
        top2_score = float(retrieved_rows[1].get("retrieval_score")) if len(retrieved_rows) > 1 else None
        retrieval_margin = (top1_score - top2_score) if top1_score is not None and top2_score is not None else None
        should_defer = decision_mode not in {"retrieval_top1"} and not (
            decision_mode == "retrieval_margin_gate"
            and retrieval_margin is not None
            and retrieval_margin >= retrieval_margin_threshold
        )
        decision = local_router_decision(query, candidates, spec, max_tokens=max_tokens) if should_defer else None
        predicted_label, fused_scores, retrieval_margin, used_llm = choose_prediction(
            decision=decision,
            candidates=candidates,
            retrieved_rows=retrieved_rows,
            decision_mode=decision_mode,
            retrieval_alpha=retrieval_alpha,
            retrieval_margin_threshold=retrieval_margin_threshold,
        )
        hit = labels_match(predicted_label, gold)
        correct += int(hit)
        if used_llm and decision is not None:
            deferred_count += 1
            total_latency += float(decision.latency_seconds)
        rows.append(
            {
                "query_id": record.get("query_id") or record.get("sample_id"),
                "query_text": query,
                "ground_truth_domain": gold,
                "predicted_domain": predicted_label,
                "routed_correctly": hit,
                "decision_mode": decision_mode,
                "deferred_to_llm": bool(used_llm and decision is not None),
                "raw_response": decision.raw_response if decision is not None else "",
                "parse_or_score_prediction": decision.predicted_label if decision is not None else "",
                "option_scores_logprob": decision.option_scores_logprob if decision is not None else None,
                "fused_option_scores": fused_scores,
                "retrieval_margin": retrieval_margin,
                "retrieval_margin_threshold": retrieval_margin_threshold,
                "option_margin_logprob": decision.option_margin_logprob if decision is not None else None,
                "answer_avg_logprob": decision.answer_avg_logprob if decision is not None else None,
                "latency_seconds": decision.latency_seconds if decision is not None else 0.0,
                "top_candidates": [candidate.label for candidate in candidates],
                "retrieved_rows": retrieved_rows,
                "empty_candidate_set": False,
            }
        )

    accuracy = correct / sample_count if sample_count else 0.0
    mean_latency = total_latency / max(1, deferred_count) if deferred_count else 0.0
    return {
        "seed": seed,
        "sample_count": sample_count,
        "correct": correct,
        "accuracy": accuracy,
        "empty_candidate_sets": empty_candidates,
        "deferred_count": deferred_count,
        "deferral_rate": (deferred_count / sample_count) if sample_count else 0.0,
        "mean_latency_seconds": mean_latency,
        "rows": rows,
    }


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("MONGO_URI"):
        raise RuntimeError("MONGO_URI must be set for the Jina->Mongo->8B router pipeline.")

    _ = build_credentials()
    records = load_records(args.sample_file)
    seeds = parse_seed_list(args.seeds)
    jina_client = JinaAIClient(os.environ.get("JINA_API_KEY") and [os.environ["JINA_API_KEY"]] or [])
    manager = MongoRAGManager(jina_client=jina_client, collection_name=args.mongo_collection)
    spec = resolve_router_spec(
        RouterSpec(
            label="llama31_8b_jina_mongo_router",
            backend="local",
            model=args.model_path,
            local_model_path=args.model_path,
        )
    )

    per_seed = [
        evaluate_seed(
            records=records,
            seed=seed,
            sample_count=args.sample_count,
            manager=manager,
            spec=spec,
            top_k=args.top_k,
            search_k=args.search_k,
            max_tokens=args.max_tokens,
            decision_mode=args.decision_mode,
            retrieval_alpha=args.retrieval_alpha,
            retrieval_margin_threshold=args.retrieval_margin_threshold,
        )
        for seed in seeds
    ]

    for result in per_seed:
        out_path = args.output_dir / f"routing_seed_{result['seed']}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    accuracies = [float(item["accuracy"]) for item in per_seed]
    latencies = [float(item["mean_latency_seconds"]) for item in per_seed]
    summary = {
        "sample_file": str(args.sample_file),
        "sample_count": args.sample_count,
        "seeds": seeds,
        "top_k": args.top_k,
        "search_k": args.search_k,
        "decision_mode": args.decision_mode,
        "retrieval_alpha": args.retrieval_alpha,
        "retrieval_margin_threshold": args.retrieval_margin_threshold,
        "model_path": spec.local_model_path,
        "mongo_collection": args.mongo_collection,
        "mean_accuracy": statistics.mean(accuracies) if accuracies else 0.0,
        "sd_accuracy": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
        "mean_latency_seconds": statistics.mean(latencies) if latencies else 0.0,
        "sd_latency_seconds": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "per_seed_accuracy": {str(item["seed"]): item["accuracy"] for item in per_seed},
        "per_seed_empty_candidate_sets": {str(item["seed"]): item["empty_candidate_sets"] for item in per_seed},
        "per_seed_deferral_rate": {str(item["seed"]): item["deferral_rate"] for item in per_seed},
        "mean_deferral_rate": statistics.mean(float(item["deferral_rate"]) for item in per_seed) if per_seed else 0.0,
        "output_dir": str(args.output_dir),
        "note": "Pipeline follows memory-path design: Jina query embedding -> Mongo vector search -> unique scenario candidates -> local 8B rerank.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
