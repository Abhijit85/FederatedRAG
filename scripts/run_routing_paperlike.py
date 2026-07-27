#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from math_qa import JinaAIClient
from openrouter_client import chat_completion, get_available_api_keys
from scripts.run_routing_verification import DEFAULT_SAMPLE_FILE, build_credentials, gold_route_label, load_records
from synapse.runtime import SynapseRuntime

DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "verification" / "routing_paperlike"

ALIASES = {
    "geometry and measurement": "geometry shapes and measurement",
    "geometry shapes and measurement": "geometry shapes and measurement",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a paper-like routing verifier: Jina embeddings over scenario artifacts + LLM reranker."
    )
    parser.add_argument("--sample-file", type=Path, default=DEFAULT_SAMPLE_FILE)
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--client-count", type=int, default=5)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--embed-model", type=str, default="jina-embeddings-v2-base-en")
    parser.add_argument("--reranker-model", type=str, default="meta-llama/llama-3.1-8b-instruct")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def parse_seed_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def normalize_label(value: str | None) -> str:
    text = (value or "").strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return ALIASES.get(text, text)


def labels_match(left: str | None, right: str | None) -> bool:
    return normalize_label(left) == normalize_label(right)


def query_text(record: dict[str, Any]) -> str:
    for key in ("query_text", "question", "Problem"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def sample_records(records: list[dict[str, Any]], seed: int, sample_count: int) -> list[dict[str, Any]]:
    if sample_count > len(records):
        raise ValueError(f"Requested {sample_count} samples, but only {len(records)} records are available.")
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(records)), sample_count))
    return [records[idx] for idx in indices]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


@dataclass
class ScenarioDoc:
    label: str
    text: str
    embedding: list[float]


def collect_scenario_docs(rounds: int, client_count: int, jina_client: JinaAIClient) -> list[ScenarioDoc]:
    runtime = SynapseRuntime.build_local_runtime(REPO_ROOT, build_credentials(), client_count=client_count)
    for _ in range(max(1, rounds)):
        runtime.run_round()

    seen: dict[str, str] = {}
    for artifact in runtime.server.compendium.build_snapshot().artifacts:
        metadata = artifact.metadata or {}
        label = metadata.get("scenario") or metadata.get("domain")
        if not isinstance(label, str) or not label.strip():
            continue
        seen.setdefault(label.strip(), artifact.text)

    labels = list(seen)
    previous_model = os.environ.get("JINA_EMBED_MODEL")
    os.environ["JINA_EMBED_MODEL"] = args.embed_model
    try:
        embeddings = jina_client.get_embeddings([seen[label] for label in labels])
    finally:
        if previous_model is None:
            os.environ.pop("JINA_EMBED_MODEL", None)
        else:
            os.environ["JINA_EMBED_MODEL"] = previous_model
    return [ScenarioDoc(label=label, text=seen[label], embedding=embedding) for label, embedding in zip(labels, embeddings)]


def rerank_with_llm(query: str, candidates: list[ScenarioDoc], model: str) -> str:
    options = []
    for index, candidate in enumerate(candidates, start=1):
        options.append(f"{index}. {candidate.label}\nContext: {candidate.text}")
    prompt = (
        "You are a routing reranker for math word problems.\n"
        "Choose the single best scenario label for the query from the candidate list.\n"
        "Return only the exact scenario label, with no explanation.\n\n"
        f"Query:\n{query}\n\n"
        f"Candidates:\n{chr(10).join(options)}\n"
    )
    response = chat_completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=64,
        temperature=0,
    )
    content = response.choices[0].message.content or ""
    cleaned = content.strip().splitlines()[0].strip()
    for candidate in candidates:
        if labels_match(cleaned, candidate.label):
            return candidate.label
    for candidate in candidates:
        if normalize_label(candidate.label) in normalize_label(cleaned):
            return candidate.label
    return cleaned


def evaluate_seed(
    *,
    records: list[dict[str, Any]],
    scenario_docs: list[ScenarioDoc],
    jina_client: JinaAIClient,
    seed: int,
    sample_count: int,
    k: int,
    reranker_model: str,
    embed_model: str,
) -> dict[str, Any]:
    subset = sample_records(records, seed=seed, sample_count=sample_count)
    rows: list[dict[str, Any]] = []
    correct = 0

    previous_model = os.environ.get("JINA_EMBED_MODEL")
    os.environ["JINA_EMBED_MODEL"] = embed_model
    try:
        query_embeddings = jina_client.get_embeddings([query_text(record) for record in subset])
    finally:
        if previous_model is None:
            os.environ.pop("JINA_EMBED_MODEL", None)
        else:
            os.environ["JINA_EMBED_MODEL"] = previous_model

    for record, query_embedding in zip(subset, query_embeddings):
        query = query_text(record)
        gold = gold_route_label(record)
        ranked = sorted(
            scenario_docs,
            key=lambda doc: cosine_similarity(query_embedding, doc.embedding),
            reverse=True,
        )[:k]
        predicted = rerank_with_llm(query, ranked, reranker_model) if ranked else ""
        hit = labels_match(predicted, gold)
        correct += int(hit)
        rows.append(
            {
                "query_id": record.get("query_id") or record.get("sample_id"),
                "query_text": query,
                "ground_truth_domain": gold,
                "predicted_domain": predicted,
                "routed_correctly": hit,
                "top_candidates": [candidate.label for candidate in ranked],
            }
        )

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


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    global args
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not get_available_api_keys(allow_empty=True):
        raise RuntimeError("OpenRouter API_KEY is required for LLM reranking.")

    records = load_records(args.sample_file)
    seeds = parse_seed_list(args.seeds)
    jina_client = JinaAIClient(api_keys=os.environ.get("JINA_API_KEY") and [os.environ["JINA_API_KEY"]] or [])
    scenario_docs = collect_scenario_docs(args.rounds, args.client_count, jina_client)

    results = [
        evaluate_seed(
            records=records,
            scenario_docs=scenario_docs,
            jina_client=jina_client,
            seed=seed,
            sample_count=args.sample_count,
            k=args.k,
            reranker_model=args.reranker_model,
            embed_model=args.embed_model,
        )
        for seed in seeds
    ]

    for result in results:
        out_path = args.output_dir / f"routing_seed_{result['seed']}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    mean_accuracy, sd_accuracy = summarize(results)
    summary = {
        "sample_file": str(args.sample_file),
        "sample_count": args.sample_count,
        "rounds": args.rounds,
        "client_count": args.client_count,
        "k": args.k,
        "embed_model": args.embed_model,
        "reranker_model": args.reranker_model,
        "scenario_count": len(scenario_docs),
        "seeds": seeds,
        "mean_accuracy": mean_accuracy,
        "sd_accuracy": sd_accuracy,
        "per_seed_accuracy": {str(result["seed"]): result["accuracy"] for result in results},
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"scenario_count={len(scenario_docs)}")
    print(f"reranker_model={args.reranker_model}")
    print(f"mean_accuracy={mean_accuracy:.3f}")
    print(f"sd_accuracy={sd_accuracy:.3f}")
    print(f"per_seed={summary['per_seed_accuracy']}")


if __name__ == "__main__":
    main()
