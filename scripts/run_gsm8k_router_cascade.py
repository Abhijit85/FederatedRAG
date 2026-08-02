#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from math_qa import JinaAIClient
from openrouter_client import chat_completion, get_available_api_keys
from scripts.run_gsm8k_small_router_sweep import (
    DEFAULT_OUTPUT_DIR as DEFAULT_BASE_OUTPUT_DIR,
    RouterSpec,
    ScenarioDoc,
    collect_scenario_docs,
    cosine_similarity,
    evaluate_seed as evaluate_router_seed,
    load_records,
    gold_route_label,
    parse_arm_specs,
    parse_seed_list,
    query_text,
    run_arm,
)

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - optional local inference dependency
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "verification" / "gsm8k_router_cascade"


@dataclass
class LocalBackend:
    tokenizer: Any
    model: Any


@dataclass
class RouterDecision:
    predicted_label: str
    latency_seconds: float
    answer_avg_logprob: float | None
    option_margin_logprob: float | None
    option_scores_logprob: dict[str, float] | None
    raw_response: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a two-tier GSM8K routing cascade over the existing top-k retrieval pipeline. "
            "The small router handles confident queries; low-confidence cases defer to the larger router."
        )
    )
    parser.add_argument("--sample-file", type=Path, default=REPO_ROOT / "GSM8K_500_rebuttal_run" / "GSM8K_500_samples.json")
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--client-count", type=int, default=5)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--embed-model", type=str, default="jina-embeddings-v2-base-en")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument(
        "--confidence-signal",
        type=str,
        default="option_margin_logprob",
        choices=("option_margin_logprob", "answer_avg_logprob"),
    )
    parser.add_argument("--thresholds", type=str, default="0.5,1.0,1.5")
    parser.add_argument("--small-label", type=str, default="llama32_1b")
    parser.add_argument("--large-label", type=str, default="llama31_8b")
    parser.add_argument("--small-model-billions", type=float, default=1.0)
    parser.add_argument("--large-model-billions", type=float, default=8.0)
    parser.add_argument("--run-baselines", action="store_true")
    parser.add_argument("--arm", action="append", default=[], help="Repeatable arm spec: label|backend|model_or_path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def parse_threshold_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def resolve_router_spec(spec: RouterSpec) -> RouterSpec:
    if spec.backend != "local" or not spec.local_model_path:
        return spec
    if "models--meta-llama--Llama-3.1-8B-Instruct/snapshots/" in spec.local_model_path:
        return RouterSpec(
            label=spec.label,
            backend=spec.backend,
            model=spec.model,
            local_model_path=(
                "/mnt/shared/shared_hf_home/hub/models--meta-llama--Llama-3.1-8B-Instruct/local_models/"
                "Llama-3.1-8B-Instruct"
            ),
            device_map=spec.device_map,
        )
    return spec


def labels_match(left: str | None, right: str | None) -> bool:
    def normalize(value: str | None) -> str:
        text = (value or "").strip().lower()
        text = text.replace("&", " and ")
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    aliases = {
        "geometry and measurement": "geometry shapes and measurement",
        "geometry shapes and measurement": "geometry shapes and measurement",
    }
    return aliases.get(normalize(left), normalize(left)) == aliases.get(normalize(right), normalize(right))


def build_label_prompt(query: str, candidates: list[ScenarioDoc]) -> str:
    options = []
    for index, candidate in enumerate(candidates, start=1):
        options.append(f"{index}. {candidate.label}\nContext: {candidate.text}")
    return (
        "You are a routing reranker for math word problems.\n"
        "Choose the single best scenario label for the query from the candidate list.\n"
        "Return only the exact scenario label, with no explanation.\n\n"
        f"Query:\n{query}\n\n"
        f"Candidates:\n{chr(10).join(options)}\n\n"
        "Answer:"
    )


_local_backend_cache: dict[tuple[str, str], LocalBackend] = {}


def load_local_backend(model_path: str, device_map: str) -> LocalBackend:
    key = (model_path, device_map)
    cached = _local_backend_cache.get(key)
    if cached is not None:
        return cached
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
    backend = LocalBackend(tokenizer=tokenizer, model=model)
    _local_backend_cache[key] = backend
    return backend


def _render_local_chat_prompt(backend: LocalBackend, prompt: str) -> Any:
    messages = [
        {"role": "system", "content": "You are a precise routing reranker for math word problems."},
        {"role": "user", "content": prompt},
    ]
    rendered = backend.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return backend.tokenizer(rendered, return_tensors="pt")["input_ids"].to(backend.model.device)


def score_local_candidates(spec: RouterSpec, prompt: str, candidate_labels: list[str]) -> dict[str, float]:
    if not spec.local_model_path:
        raise ValueError(f"Local router arm {spec.label} is missing a local_model_path.")
    backend = load_local_backend(spec.local_model_path, spec.device_map)
    prompt_ids = _render_local_chat_prompt(backend, prompt)
    scores: dict[str, float] = {}
    with torch.no_grad():
        for label in candidate_labels:
            continuation = backend.tokenizer(label, add_special_tokens=False, return_tensors="pt")["input_ids"].to(
                backend.model.device
            )
            full_ids = torch.cat([prompt_ids, continuation], dim=1)
            logits = backend.model(full_ids).logits[:, :-1, :]
            labels = full_ids[:, 1:]
            log_probs = torch.log_softmax(logits, dim=-1)
            gathered = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            continuation_start = prompt_ids.shape[1] - 1
            continuation_logprobs = gathered[:, continuation_start:]
            scores[label] = float(continuation_logprobs.sum().detach().cpu())
    return scores


def generate_local_label(spec: RouterSpec, prompt: str, max_tokens: int) -> str:
    if not spec.local_model_path:
        raise ValueError(f"Local router arm {spec.label} is missing a local_model_path.")
    backend = load_local_backend(spec.local_model_path, spec.device_map)
    prompt_ids = _render_local_chat_prompt(backend, prompt)
    attention_mask = torch.ones_like(prompt_ids)
    with torch.no_grad():
        output_ids = backend.model.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=backend.tokenizer.pad_token_id,
            eos_token_id=backend.tokenizer.eos_token_id,
        )
    prompt_len = prompt_ids.shape[1]
    return backend.tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()


def local_router_decision(query: str, candidates: list[ScenarioDoc], spec: RouterSpec, max_tokens: int) -> RouterDecision:
    prompt = build_label_prompt(query, candidates)
    started = time.perf_counter()
    raw_response = generate_local_label(spec, prompt, max_tokens=max_tokens)
    candidate_labels = [candidate.label for candidate in candidates]
    option_scores = score_local_candidates(spec, prompt, candidate_labels)
    latency = time.perf_counter() - started
    ranked = sorted(option_scores.items(), key=lambda item: item[1], reverse=True)
    best_label, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else None
    predicted_label = raw_response
    for candidate in candidates:
        if labels_match(raw_response, candidate.label):
            predicted_label = candidate.label
            break
    if predicted_label == raw_response:
        for candidate in candidates:
            if candidate.label.lower() in raw_response.lower() or raw_response.lower() in candidate.label.lower():
                predicted_label = candidate.label
                break
    if predicted_label == raw_response:
        predicted_label = best_label
    return RouterDecision(
        predicted_label=predicted_label,
        latency_seconds=latency,
        answer_avg_logprob=best_score / max(1, len(best_label.split())),
        option_margin_logprob=(best_score - second_score) if second_score is not None else None,
        option_scores_logprob=option_scores,
        raw_response=raw_response,
    )


def extract_openrouter_logprobs(response: Any) -> tuple[float | None, dict[str, float] | None]:
    choice = response.choices[0]
    logprobs = getattr(choice, "logprobs", None)
    content = getattr(logprobs, "content", None) if logprobs is not None else None
    if not content:
        return None, None

    token_logprobs: list[float] = []
    top_map: dict[str, float] = {}
    for idx, token_info in enumerate(content):
        logprob = getattr(token_info, "logprob", None)
        if isinstance(logprob, (int, float)):
            token_logprobs.append(float(logprob))
        if idx == 0:
            top_logprobs = getattr(token_info, "top_logprobs", None) or []
            for alt in top_logprobs:
                token = normalize_option_token(getattr(alt, "token", ""))
                alt_logprob = getattr(alt, "logprob", None)
                if token and isinstance(alt_logprob, (int, float)):
                    prev = top_map.get(token)
                    if prev is None or float(alt_logprob) > prev:
                        top_map[token] = float(alt_logprob)
            token = normalize_option_token(getattr(token_info, "token", ""))
            if token and isinstance(logprob, (int, float)):
                prev = top_map.get(token)
                if prev is None or float(logprob) > prev:
                    top_map[token] = float(logprob)
    avg = sum(token_logprobs) / len(token_logprobs) if token_logprobs else None
    return avg, top_map or None


def openrouter_router_decision(query: str, candidates: list[ScenarioDoc], spec: RouterSpec, max_tokens: int) -> RouterDecision:
    prompt = build_label_prompt(query, candidates)
    started = time.perf_counter()
    response = chat_completion(
        model=spec.model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,
        logprobs=True,
        top_logprobs=20,
    )
    latency = time.perf_counter() - started
    content = response.choices[0].message.content or ""
    raw_response = content.strip()
    option = extract_option_from_text(raw_response)
    avg_logprob, top_map = extract_openrouter_logprobs(response)
    if option and option in OPTION_KEYS[: len(candidates)]:
        predicted_label = candidates[OPTION_KEYS.index(option)].label
    else:
        predicted_label = raw_response
        for candidate in candidates:
            if labels_match(raw_response, candidate.label):
                predicted_label = candidate.label
                break
    margin = None
    if top_map:
        scores = [value for key, value in top_map.items() if key in OPTION_KEYS[: len(candidates)]]
        if len(scores) >= 2:
            top_two = sorted(scores, reverse=True)[:2]
            margin = top_two[0] - top_two[1]
    return RouterDecision(
        predicted_label=predicted_label,
        latency_seconds=latency,
        answer_avg_logprob=avg_logprob,
        option_margin_logprob=margin,
        option_scores_logprob=top_map,
        raw_response=raw_response,
    )


def router_decision(query: str, candidates: list[ScenarioDoc], spec: RouterSpec, max_tokens: int) -> RouterDecision:
    if spec.backend == "local":
        return local_router_decision(query, candidates, spec, max_tokens=max_tokens)
    return openrouter_router_decision(query, candidates, spec, max_tokens=max_tokens)


def effective_cost_ratio(deferral_rate: float, small_model_billions: float, large_model_billions: float) -> float:
    if large_model_billions <= 0:
        return 1.0
    return (small_model_billions + deferral_rate * large_model_billions) / large_model_billions


def prepare_cached_decisions(
    *,
    records: list[dict[str, Any]],
    scenario_docs: list[ScenarioDoc],
    jina_client: JinaAIClient,
    seeds: list[int],
    sample_count: int,
    k: int,
    embed_model: str,
    small_spec: RouterSpec,
    large_spec: RouterSpec,
    max_tokens: int,
) -> dict[int, list[dict[str, Any]]]:
    previous_model = os.environ.get("JINA_EMBED_MODEL")
    os.environ["JINA_EMBED_MODEL"] = embed_model
    try:
        subsets = {seed: __import__('random').Random(seed).sample(range(len(records)), sample_count) for seed in seeds}
        cached: dict[int, list[dict[str, Any]]] = {}
        for seed, indices in subsets.items():
            indices = sorted(indices)
            subset = [records[idx] for idx in indices]
            embeddings = jina_client.get_embeddings([query_text(record) for record in subset])
            rows: list[dict[str, Any]] = []
            for record, query_embedding in zip(subset, embeddings):
                query = query_text(record)
                gold = gold_route_label(record)
                ranked = sorted(
                    scenario_docs,
                    key=lambda doc: cosine_similarity(query_embedding, doc.embedding),
                    reverse=True,
                )[:k]
                small = router_decision(query, ranked, small_spec, max_tokens=max_tokens) if ranked else RouterDecision("", 0.0, None, None, None, "")
                large = router_decision(query, ranked, large_spec, max_tokens=max_tokens) if ranked else RouterDecision("", 0.0, None, None, None, "")
                rows.append(
                    {
                        "query_id": record.get("query_id") or record.get("sample_id"),
                        "query_text": query,
                        "ground_truth_domain": gold,
                        "top_candidates": [candidate.label for candidate in ranked],
                        "small": {
                            "predicted_domain": small.predicted_label,
                            "latency_seconds": small.latency_seconds,
                            "answer_avg_logprob": small.answer_avg_logprob,
                            "option_margin_logprob": small.option_margin_logprob,
                            "option_scores_logprob": small.option_scores_logprob,
                            "raw_response": small.raw_response,
                        },
                        "large": {
                            "predicted_domain": large.predicted_label,
                            "latency_seconds": large.latency_seconds,
                            "raw_response": large.raw_response,
                        },
                    }
                )
            cached[seed] = rows
    finally:
        if previous_model is None:
            os.environ.pop("JINA_EMBED_MODEL", None)
        else:
            os.environ["JINA_EMBED_MODEL"] = previous_model
    return cached


def summarize_cascade_threshold(
    *,
    cached_rows: dict[int, list[dict[str, Any]]],
    threshold: float,
    confidence_signal: str,
    sample_count: int,
    small_model_billions: float,
    large_model_billions: float,
) -> dict[str, Any]:
    seed_results: list[dict[str, Any]] = []
    for seed, rows_in in cached_rows.items():
        rows_out: list[dict[str, Any]] = []
        correct = 0
        deferred = 0
        total_latency = 0.0
        for row in rows_in:
            confidence_value = row["small"].get(confidence_signal)
            accept_small = isinstance(confidence_value, (int, float)) and float(confidence_value) >= threshold
            deferred += int(not accept_small)
            predicted = row["small"]["predicted_domain"] if accept_small else row["large"]["predicted_domain"]
            total_latency += float(row["small"]["latency_seconds"])
            if not accept_small:
                total_latency += float(row["large"]["latency_seconds"])
            hit = labels_match(predicted, row["ground_truth_domain"])
            correct += int(hit)
            rows_out.append(
                {
                    "query_id": row["query_id"],
                    "query_text": row["query_text"],
                    "ground_truth_domain": row["ground_truth_domain"],
                    "predicted_domain": predicted,
                    "routed_correctly": hit,
                    "deferred_to_large": not accept_small,
                    "confidence_signal": confidence_signal,
                    "confidence_value": confidence_value,
                    "small_predicted_domain": row["small"]["predicted_domain"],
                    "large_predicted_domain": row["large"]["predicted_domain"],
                    "top_candidates": row["top_candidates"],
                }
            )
        deferral_rate = deferred / sample_count if sample_count else 0.0
        accuracy = correct / sample_count if sample_count else 0.0
        compute_ratio = effective_cost_ratio(deferral_rate, small_model_billions, large_model_billions)
        seed_results.append(
            {
                "seed": seed,
                "threshold": threshold,
                "sample_count": sample_count,
                "accuracy": accuracy,
                "correct": correct,
                "deferrals": deferred,
                "deferral_rate": deferral_rate,
                "mean_latency_seconds": total_latency / sample_count if sample_count else 0.0,
                "effective_compute_ratio_vs_full_large": compute_ratio,
                "effective_compute_reduction_vs_full_large": 1.0 - compute_ratio,
                "rows": rows_out,
            }
        )

    mean_accuracy = sum(float(item["accuracy"]) for item in seed_results) / len(seed_results) if seed_results else 0.0
    mean_deferral = sum(float(item["deferral_rate"]) for item in seed_results) / len(seed_results) if seed_results else 0.0
    mean_compute_ratio = (
        sum(float(item["effective_compute_ratio_vs_full_large"]) for item in seed_results) / len(seed_results)
        if seed_results
        else 1.0
    )
    return {
        "threshold": threshold,
        "confidence_signal": confidence_signal,
        "mean_accuracy": mean_accuracy,
        "sd_accuracy": statistics.stdev(float(item["accuracy"]) for item in seed_results) if len(seed_results) > 1 else 0.0,
        "mean_deferral_rate": mean_deferral,
        "sd_deferral_rate": statistics.stdev(float(item["deferral_rate"]) for item in seed_results) if len(seed_results) > 1 else 0.0,
        "mean_effective_compute_ratio_vs_full_large": mean_compute_ratio,
        "mean_effective_compute_reduction_vs_full_large": 1.0 - mean_compute_ratio,
        "per_seed_accuracy": {str(item["seed"]): item["accuracy"] for item in seed_results},
        "per_seed_deferral_rate": {str(item["seed"]): item["deferral_rate"] for item in seed_results},
        "per_seed_effective_compute_ratio_vs_full_large": {
            str(item["seed"]): item["effective_compute_ratio_vs_full_large"] for item in seed_results
        },
        "seed_results": seed_results,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    baselines = summary.get("baselines") or []
    lines = [
        "### GSM8K Two-Tier Router Cascade",
        "",
        f"- Small router: `{summary['small_label']}`",
        f"- Large router: `{summary['large_label']}`",
        f"- Confidence signal: `{summary['confidence_signal']}`",
        f"- Sample count: {summary['sample_count']}",
        f"- Seeds: {', '.join(str(seed) for seed in summary['seeds'])}",
        "",
    ]
    if baselines:
        lines.extend(
            [
                "| Baseline router | Mean routing acc. | SD |",
                "| --- | ---: | ---: |",
            ]
        )
        for row in baselines:
            lines.append(f"| {row['label']} | {row['mean_accuracy']:.3f} | {row['sd_accuracy']:.3f} |")
        lines.append("")
    lines.extend(
        [
            "| Threshold | Mean routing acc. | SD | Mean deferral rate | Mean compute reduction vs full large |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["results"]:
        lines.append(
            f"| {row['threshold']:.3f} | {row['mean_accuracy']:.3f} | {row['sd_accuracy']:.3f} | "
            f"{row['mean_deferral_rate']:.3f} | {row['mean_effective_compute_reduction_vs_full_large']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    arm_specs = [resolve_router_spec(spec) for spec in parse_arm_specs(args.arm)]
    by_label = {spec.label: spec for spec in arm_specs}
    small_spec = by_label.get(args.small_label)
    large_spec = by_label.get(args.large_label)
    if small_spec is None or large_spec is None:
        raise ValueError(
            f"Both --small-label={args.small_label!r} and --large-label={args.large_label!r} must be present in the default or explicit arm list."
        )
    if any(spec.backend == "openrouter" for spec in (small_spec, large_spec)) and not get_available_api_keys(allow_empty=True):
        raise RuntimeError("At least one OpenRouter API_KEY is required for openrouter-backed router arms.")

    records = load_records(args.sample_file)
    seeds = parse_seed_list(args.seeds)
    thresholds = parse_threshold_list(args.thresholds)
    jina_client = JinaAIClient(api_keys=os.environ.get("JINA_API_KEY") and [os.environ["JINA_API_KEY"]] or [])
    scenario_docs = collect_scenario_docs(
        rounds=args.rounds,
        client_count=args.client_count,
        jina_client=jina_client,
        embed_model=args.embed_model,
    )

    baselines = []
    if args.run_baselines:
        baseline_dir = args.output_dir / "baselines"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        for spec in (small_spec, large_spec):
            baselines.append(
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
                    output_dir=baseline_dir,
                )
            )

    cached_rows = prepare_cached_decisions(
        records=records,
        scenario_docs=scenario_docs,
        jina_client=jina_client,
        seeds=seeds,
        sample_count=args.sample_count,
        k=args.k,
        embed_model=args.embed_model,
        small_spec=small_spec,
        large_spec=large_spec,
        max_tokens=args.max_tokens,
    )

    results = []
    for threshold in thresholds:
        threshold_summary = summarize_cascade_threshold(
            cached_rows=cached_rows,
            threshold=threshold,
            confidence_signal=args.confidence_signal,
            sample_count=args.sample_count,
            small_model_billions=args.small_model_billions,
            large_model_billions=args.large_model_billions,
        )
        results.append(threshold_summary)
        threshold_slug = str(threshold).replace("-", "neg_").replace(".", "p")
        out_dir = args.output_dir / f"threshold_{threshold_slug}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for seed_result in threshold_summary["seed_results"]:
            (out_dir / f"routing_seed_{seed_result['seed']}.json").write_text(json.dumps(seed_result, indent=2), encoding="utf-8")

    summary = {
        "sample_file": str(args.sample_file),
        "sample_count": args.sample_count,
        "rounds": args.rounds,
        "client_count": args.client_count,
        "k": args.k,
        "embed_model": args.embed_model,
        "seeds": seeds,
        "thresholds": thresholds,
        "confidence_signal": args.confidence_signal,
        "small_label": small_spec.label,
        "large_label": large_spec.label,
        "small_model_billions": args.small_model_billions,
        "large_model_billions": args.large_model_billions,
        "baselines": baselines,
        "results": results,
        "base_router_sweep_reference": str(DEFAULT_BASE_OUTPUT_DIR),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")

    for row in results:
        print(
            f"threshold={row['threshold']:.3f} acc={row['mean_accuracy']:.3f} sd={row['sd_accuracy']:.3f} "
            f"deferral={row['mean_deferral_rate']:.3f} compute_reduction={row['mean_effective_compute_reduction_vs_full_large']:.3f}"
        )


if __name__ == "__main__":
    main()
