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
import time
from dataclasses import dataclass
from functools import lru_cache
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

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - optional local inference dependency
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "verification" / "gsm8k_small_router_sweep"

ALIASES = {
    "geometry and measurement": "geometry shapes and measurement",
    "geometry shapes and measurement": "geometry shapes and measurement",
}


@dataclass
class ScenarioDoc:
    label: str
    text: str
    embedding: list[float]


@dataclass
class RouterSpec:
    label: str
    model: str
    backend: str
    local_model_path: str | None = None
    device_map: str = "auto"


@dataclass
class LocalChatBackend:
    tokenizer: Any
    model: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the reviewer-requested small-LLM router sweep on the GSM8K paperlike routing metric. "
            "This harness keeps retrieval fixed and varies only the router backend."
        )
    )
    parser.add_argument("--sample-file", type=Path, default=DEFAULT_SAMPLE_FILE)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--seeds", type=str, default="42,123,456,789,1024")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--client-count", type=int, default=5)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--embed-model", type=str, default="jina-embeddings-v2-base-en")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--gate-label", type=str, default="llama31_8b")
    parser.add_argument("--gate-min", type=float, default=0.85)
    parser.add_argument("--gate-max", type=float, default=0.98)
    parser.add_argument("--arm", action="append", default=[], help="Repeatable arm spec: label|backend|model_or_path")
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


def collect_scenario_docs(rounds: int, client_count: int, jina_client: JinaAIClient, embed_model: str) -> list[ScenarioDoc]:
    runtime = SynapseRuntime.build_local_runtime(REPO_ROOT, build_credentials(), client_count=client_count)
    for _ in range(max(1, rounds)):
        runtime.run_round()

    seen: dict[str, str] = {}
    for artifact in runtime.server.compendium.build_snapshot().artifacts:
        metadata = artifact.metadata or {}
        tool = metadata.get("tool")
        label = metadata.get("scenario") or metadata.get("domain")
        if tool != "mathqa":
            continue
        if not isinstance(label, str) or not label.strip():
            continue
        seen.setdefault(label.strip(), artifact.text)

    labels = list(seen)
    previous_model = os.environ.get("JINA_EMBED_MODEL")
    os.environ["JINA_EMBED_MODEL"] = embed_model
    try:
        embeddings = jina_client.get_embeddings([seen[label] for label in labels])
    finally:
        if previous_model is None:
            os.environ.pop("JINA_EMBED_MODEL", None)
        else:
            os.environ["JINA_EMBED_MODEL"] = previous_model
    return [ScenarioDoc(label=label, text=seen[label], embedding=embedding) for label, embedding in zip(labels, embeddings)]


def parse_arm_specs(values: list[str]) -> list[RouterSpec]:
    if not values:
        return [
            RouterSpec(
                label="qwen25_0p5b",
                backend="local",
                model="Qwen/Qwen2.5-0.5B-Instruct",
                local_model_path="/mnt/shared/shared_hf_home/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
            ),
            RouterSpec(
                label="llama32_1b",
                backend="openrouter",
                model="meta-llama/llama-3.2-1b-instruct",
            ),
            RouterSpec(
                label="llama32_3b",
                backend="local",
                model="meta-llama/Llama-3.2-3B-Instruct",
                local_model_path="/mnt/shared/shared_hf_home/hub/models--meta-llama--Llama-3.1-8B-Instruct/local_models/Llama-3.2-3B-Instruct",
            ),
            RouterSpec(
                label="llama31_8b",
                backend="local",
                model="meta-llama/Llama-3.1-8B-Instruct",
                local_model_path="/mnt/shared/shared_hf_home/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
            ),
        ]

    specs: list[RouterSpec] = []
    for value in values:
        parts = value.split("|")
        if len(parts) != 3:
            raise ValueError(f"Expected --arm label|backend|model_or_path, got: {value}")
        label, backend, model_or_path = (part.strip() for part in parts)
        if backend not in {"local", "openrouter"}:
            raise ValueError(f"Unsupported backend '{backend}' in arm: {value}")
        if backend == "local":
            specs.append(
                RouterSpec(
                    label=label,
                    backend=backend,
                    model=model_or_path,
                    local_model_path=model_or_path,
                )
            )
        else:
            specs.append(RouterSpec(label=label, backend=backend, model=model_or_path))
    return specs


@lru_cache(maxsize=8)
def _load_local_backend(model_path: str, device_map: str) -> LocalChatBackend:
    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError("transformers/torch are required for local router inference.")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map=device_map,
        local_files_only=True,
    )
    model.eval()
    return LocalChatBackend(tokenizer=tokenizer, model=model)


def _local_chat_completion(spec: RouterSpec, prompt: str, max_tokens: int) -> str:
    if not spec.local_model_path:
        raise ValueError(f"Local router arm {spec.label} is missing a local_model_path.")
    backend = _load_local_backend(spec.local_model_path, spec.device_map)
    messages = [
        {"role": "system", "content": "You are a precise routing reranker for math word problems."},
        {"role": "user", "content": prompt},
    ]
    rendered = backend.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = backend.tokenizer(rendered, return_tensors="pt")
    encoded = {key: value.to(backend.model.device) for key, value in encoded.items()}
    with torch.no_grad():
        output_ids = backend.model.generate(
            **encoded,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=backend.tokenizer.pad_token_id,
            eos_token_id=backend.tokenizer.eos_token_id,
        )
    prompt_len = encoded["input_ids"].shape[1]
    return backend.tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()


def rerank_with_router(query: str, candidates: list[ScenarioDoc], spec: RouterSpec, max_tokens: int) -> tuple[str, float]:
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
    started = time.perf_counter()
    if spec.backend == "local":
        content = _local_chat_completion(spec, prompt, max_tokens=max_tokens)
    else:
        response = chat_completion(
            model=spec.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
        )
        content = response.choices[0].message.content or ""
    latency = time.perf_counter() - started
    cleaned = content.strip().splitlines()[0].strip()
    for candidate in candidates:
        if labels_match(cleaned, candidate.label):
            return candidate.label, latency
    for candidate in candidates:
        if normalize_label(candidate.label) in normalize_label(cleaned):
            return candidate.label, latency
    return cleaned, latency


def evaluate_seed(
    *,
    records: list[dict[str, Any]],
    scenario_docs: list[ScenarioDoc],
    jina_client: JinaAIClient,
    seed: int,
    sample_count: int,
    k: int,
    embed_model: str,
    spec: RouterSpec,
    max_tokens: int,
) -> dict[str, Any]:
    subset = sample_records(records, seed=seed, sample_count=sample_count)
    rows: list[dict[str, Any]] = []
    correct = 0
    total_latency = 0.0

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
        predicted, latency = rerank_with_router(query, ranked, spec, max_tokens=max_tokens) if ranked else ("", 0.0)
        hit = labels_match(predicted, gold)
        total_latency += latency
        correct += int(hit)
        rows.append(
            {
                "query_id": record.get("query_id") or record.get("sample_id"),
                "query_text": query,
                "ground_truth_domain": gold,
                "predicted_domain": predicted,
                "routed_correctly": hit,
                "latency_seconds": latency,
                "top_candidates": [candidate.label for candidate in ranked],
            }
        )

    accuracy = correct / sample_count if sample_count else 0.0
    mean_latency = total_latency / sample_count if sample_count else 0.0
    return {
        "seed": seed,
        "sample_count": sample_count,
        "correct": correct,
        "accuracy": accuracy,
        "mean_latency_seconds": mean_latency,
        "rows": rows,
    }


def summarize(results: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    accuracies = [float(result["accuracy"]) for result in results]
    latencies = [float(result["mean_latency_seconds"]) for result in results]
    mean_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    sd_accuracy = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    sd_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    return mean_accuracy, sd_accuracy, mean_latency, sd_latency


def run_arm(
    *,
    spec: RouterSpec,
    records: list[dict[str, Any]],
    scenario_docs: list[ScenarioDoc],
    jina_client: JinaAIClient,
    seeds: list[int],
    sample_count: int,
    k: int,
    embed_model: str,
    max_tokens: int,
    output_dir: Path,
) -> dict[str, Any]:
    arm_dir = output_dir / spec.label
    arm_dir.mkdir(parents=True, exist_ok=True)
    results = [
        evaluate_seed(
            records=records,
            scenario_docs=scenario_docs,
            jina_client=jina_client,
            seed=seed,
            sample_count=sample_count,
            k=k,
            embed_model=embed_model,
            spec=spec,
            max_tokens=max_tokens,
        )
        for seed in seeds
    ]
    for result in results:
        (arm_dir / f"routing_seed_{result['seed']}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    mean_accuracy, sd_accuracy, mean_latency, sd_latency = summarize(results)
    summary = {
        "label": spec.label,
        "backend": spec.backend,
        "model": spec.model,
        "local_model_path": spec.local_model_path,
        "sample_count": sample_count,
        "k": k,
        "seeds": seeds,
        "mean_accuracy": mean_accuracy,
        "sd_accuracy": sd_accuracy,
        "mean_latency_seconds": mean_latency,
        "sd_latency_seconds": sd_latency,
        "per_seed_accuracy": {str(result["seed"]): result["accuracy"] for result in results},
        "per_seed_latency_seconds": {str(result["seed"]): result["mean_latency_seconds"] for result in results},
        "output_dir": str(arm_dir),
    }
    (arm_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def render_markdown(rows: list[dict[str, Any]], gate_label: str, gate_min: float, gate_max: float) -> str:
    parts = [
        "### GSM8K Small LLM Router Sweep",
        "",
        f"- 8B gate label: `{gate_label}` must land in [`{gate_min:.3f}`, `{gate_max:.3f}`] to count as the expected routing scale.",
        "",
        "| Router | Backend | Mean routing acc. | SD | Mean latency (s/query) | SD latency |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        parts.append(
            f"| {row['label']} | {row['backend']} | {row['mean_accuracy']:.3f} | {row['sd_accuracy']:.3f} | "
            f"{row['mean_latency_seconds']:.3f} | {row['sd_latency_seconds']:.3f} |"
        )
    return "\n".join(parts) + "\n"


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    arm_specs = parse_arm_specs(args.arm)
    if any(spec.backend == "openrouter" for spec in arm_specs) and not get_available_api_keys(allow_empty=True):
        raise RuntimeError("At least one OpenRouter API_KEY is required for openrouter-backed router arms.")

    records = load_records(args.sample_file)
    seeds = parse_seed_list(args.seeds)
    jina_client = JinaAIClient(api_keys=os.environ.get("JINA_API_KEY") and [os.environ["JINA_API_KEY"]] or [])
    scenario_docs = collect_scenario_docs(
        rounds=args.rounds,
        client_count=args.client_count,
        jina_client=jina_client,
        embed_model=args.embed_model,
    )

    rows = [
        run_arm(
            spec=spec,
            records=records,
            scenario_docs=scenario_docs,
            jina_client=jina_client,
            seeds=seeds,
            sample_count=args.sample_count,
            k=args.k,
            embed_model=args.embed_model,
            max_tokens=args.max_tokens,
            output_dir=args.output_dir,
        )
        for spec in arm_specs
    ]

    gate_row = next((row for row in rows if row["label"] == args.gate_label), None)
    gate = {
        "label": args.gate_label,
        "present": gate_row is not None,
        "expected_min": args.gate_min,
        "expected_max": args.gate_max,
        "mean_accuracy": gate_row["mean_accuracy"] if gate_row else None,
        "passed": bool(gate_row and args.gate_min <= float(gate_row["mean_accuracy"]) <= args.gate_max),
    }

    combined = {
        "sample_file": str(args.sample_file),
        "sample_count": args.sample_count,
        "rounds": args.rounds,
        "client_count": args.client_count,
        "k": args.k,
        "embed_model": args.embed_model,
        "seeds": seeds,
        "scenario_count": len(scenario_docs),
        "gate": gate,
        "arms": rows,
        "note": (
            "This is an explicit LLM-router sweep on the paperlike GSM8K routing metric. "
            "If the 8B gate fails, the run should not be presented as a reproduction of the submitted ~0.92 anchor."
        ),
    }
    (args.output_dir / "combined_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    (args.output_dir / "summary.md").write_text(
        render_markdown(rows, args.gate_label, args.gate_min, args.gate_max),
        encoding="utf-8",
    )

    for row in rows:
        print(
            f"{row['label']}: mean={row['mean_accuracy']:.3f}, sd={row['sd_accuracy']:.3f}, "
            f"latency={row['mean_latency_seconds']:.3f}s, backend={row['backend']}, model={row['model']}"
        )
    print(
        f"gate[{gate['label']}]: present={gate['present']} mean={gate['mean_accuracy']} "
        f"range=[{args.gate_min:.3f}, {args.gate_max:.3f}] passed={gate['passed']}"
    )


if __name__ == "__main__":
    main()
