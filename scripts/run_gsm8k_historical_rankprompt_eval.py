#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_gsm8k_router_cascade import RouterSpec, ScenarioDoc, labels_match, load_local_backend  # noqa: E402

DEFAULT_SAMPLE_FILE = REPO_ROOT / "GSM8K_500_rebuttal_run" / "GSM8K_500_samples.json"
DEFAULT_RUNLOG = REPO_ROOT / "GSM8K_500_rebuttal_run" / "GSM8K_500_runlog.jsonl"

PROMPT_PREFIXES = {
    "plain_index": (
        "You are ranking candidate routing scenarios for a math word problem.\n"
        "Choose the single best candidate index for solving the query.\n"
        "Return only the index number of the best candidate."
    ),
    "retrieval_rerank": (
        "These are retrieved routing labels. Rerank them and choose the best final route.\n"
        "Output only the best candidate index."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a single local model on preserved historical candidate lists with a rank prompt.")
    parser.add_argument("--sample-file", type=Path, default=DEFAULT_SAMPLE_FILE)
    parser.add_argument("--runlog", type=Path, default=DEFAULT_RUNLOG)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--variant", choices=tuple(PROMPT_PREFIXES.keys()), default="retrieval_rerank")
    parser.add_argument("--decision-mode", choices=("parse_or_score", "score_only"), default="score_only")
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_seed_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def load_json_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return [row for row in payload["records"] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    raise ValueError(f"Unsupported sample file format: {path}")


def load_runlog_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict) or obj.get("source_kind") != "gsm8k_derived":
                continue
            qid = str(obj.get("query_id") or obj.get("sample_id") or "")
            if qid:
                rows[qid] = obj
    return rows


def query_text(record: dict[str, Any]) -> str:
    for key in ("query_text", "question", "Problem"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def gold_route_label(record: dict[str, Any], runlog_row: dict[str, Any]) -> str:
    router = runlog_row.get("router") if isinstance(runlog_row, dict) else None
    if isinstance(router, dict):
        value = router.get("ground_truth_domain")
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("ground_truth_domain", "domain", "scenario"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def sample_records(records: list[dict[str, Any]], seed: int, sample_count: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(records)), sample_count))
    return [records[idx] for idx in indices]


def build_candidates(runlog_row: dict[str, Any]) -> list[ScenarioDoc]:
    top_candidates = runlog_row.get("router", {}).get("top_candidates") or []
    docs: list[ScenarioDoc] = []
    for item in top_candidates:
        label = item.get("domain") if isinstance(item, dict) else item
        if isinstance(label, str) and label.strip():
            docs.append(ScenarioDoc(label=label, text=f"Routing scenario label: {label}.", embedding=[]))
    return docs


def build_prompt(query: str, candidates: list[ScenarioDoc], variant: str) -> str:
    options = [f"{i}. {c.label}" for i, c in enumerate(candidates, start=1)]
    return (
        f"{PROMPT_PREFIXES[variant]}\n\n"
        f"Query:\n{query}\n\n"
        f"Candidates:\n{chr(10).join(options)}\n\n"
        "Best candidate index:"
    )


def render_prompt_ids(spec: RouterSpec, prompt: str):
    backend = load_local_backend(spec.local_model_path, spec.device_map)
    messages = [
        {"role": "system", "content": "You are a precise routing classifier for math word problems."},
        {"role": "user", "content": prompt},
    ]
    rendered = backend.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = backend.tokenizer(rendered, return_tensors="pt")
    return backend, encoded["input_ids"].to(backend.model.device)


def score_choice_tokens(spec: RouterSpec, prompt: str, n_choices: int) -> dict[str, float]:
    import torch
    backend, prompt_ids = render_prompt_ids(spec, prompt)
    scores: dict[str, float] = {}
    with torch.no_grad():
        for idx in range(1, n_choices + 1):
            token_text = str(idx)
            continuation = backend.tokenizer(token_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(backend.model.device)
            full_ids = torch.cat([prompt_ids, continuation], dim=1)
            logits = backend.model(full_ids).logits[:, :-1, :]
            labels = full_ids[:, 1:]
            log_probs = torch.log_softmax(logits, dim=-1)
            gathered = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            continuation_start = prompt_ids.shape[1] - 1
            continuation_logprobs = gathered[:, continuation_start:]
            scores[token_text] = float(continuation_logprobs.sum().detach().cpu())
    return scores


def generate_choice(spec: RouterSpec, prompt: str, max_tokens: int) -> str:
    import torch
    backend, prompt_ids = render_prompt_ids(spec, prompt)
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


def parse_choice(raw_response: str, n_choices: int) -> int | None:
    match = re.search(r"\b([1-9])\b", raw_response or "")
    if not match:
        return None
    idx = int(match.group(1))
    return idx if 1 <= idx <= n_choices else None


def evaluate_seed(*, records: list[dict[str, Any]], runlog_rows: dict[str, dict[str, Any]], seed: int, sample_count: int, spec: RouterSpec, max_tokens: int, variant: str, decision_mode: str) -> dict[str, Any]:
    subset = sample_records(records, seed=seed, sample_count=sample_count)
    rows: list[dict[str, Any]] = []
    correct = 0
    total_latency = 0.0
    historical_top1_correct = 0
    gold_in_candidates = 0
    parsed_count = 0
    for record in subset:
        qid = str(record.get("query_id") or record.get("sample_id") or "")
        runlog_row = runlog_rows.get(qid)
        if runlog_row is None:
            continue
        query = query_text(record) or str(runlog_row.get("query_text") or "")
        gold = gold_route_label(record, runlog_row)
        candidates = build_candidates(runlog_row)
        candidate_labels = [candidate.label for candidate in candidates]
        if candidate_labels and labels_match(candidate_labels[0], gold):
            historical_top1_correct += 1
        gold_in_candidates += int(any(labels_match(gold, label) for label in candidate_labels))
        prompt = build_prompt(query, candidates, variant)
        started = time.perf_counter()
        raw = generate_choice(spec, prompt, max_tokens=max_tokens)
        scores = score_choice_tokens(spec, prompt, len(candidates))
        latency = time.perf_counter() - started
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        parsed_choice = parse_choice(raw, len(candidates))
        if decision_mode == "score_only":
            chosen = int(ranked[0][0])
        else:
            chosen = parsed_choice or int(ranked[0][0])
        predicted = candidates[chosen - 1].label
        hit = labels_match(predicted, gold)
        correct += int(hit)
        total_latency += latency
        parsed_count += int(parsed_choice is not None)
        rows.append(
            {
                "query_id": qid,
                "query_text": query,
                "ground_truth_domain": gold,
                "predicted_domain": predicted,
                "routed_correctly": hit,
                "raw_response": raw,
                "parsed_choice": parsed_choice,
                "latency_seconds": latency,
                "option_scores_logprob": scores,
                "top_candidates": candidate_labels,
                "historical_top1_correct": bool(candidate_labels and labels_match(candidate_labels[0], gold)),
            }
        )
    n = len(rows)
    return {
        "seed": seed,
        "sample_count": n,
        "accuracy": correct / n if n else 0.0,
        "correct": correct,
        "mean_latency_seconds": total_latency / n if n else 0.0,
        "historical_top1_accuracy": historical_top1_correct / n if n else 0.0,
        "gold_in_candidates_rate": gold_in_candidates / n if n else 0.0,
        "parse_success_rate": parsed_count / n if n else 0.0,
        "rows": rows,
    }


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_json_records(args.sample_file)
    runlog_rows = load_runlog_rows(args.runlog)
    seeds = parse_seed_list(args.seeds)
    spec = RouterSpec(label=args.label, backend="local", model=args.model_path, local_model_path=args.model_path)
    results = [
        evaluate_seed(
            records=records,
            runlog_rows=runlog_rows,
            seed=seed,
            sample_count=args.sample_count,
            spec=spec,
            max_tokens=args.max_tokens,
            variant=args.variant,
            decision_mode=args.decision_mode,
        )
        for seed in seeds
    ]
    out_dir = args.output_dir / spec.label
    out_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        (out_dir / f"routing_seed_{result['seed']}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    accuracies = [float(result["accuracy"]) for result in results]
    latencies = [float(result["mean_latency_seconds"]) for result in results]
    hist_top1 = [float(result["historical_top1_accuracy"]) for result in results]
    gold_cover = [float(result["gold_in_candidates_rate"]) for result in results]
    parse_rates = [float(result["parse_success_rate"]) for result in results]
    summary = {
        "sample_file": str(args.sample_file),
        "runlog": str(args.runlog),
        "sample_count": args.sample_count,
        "seeds": seeds,
        "max_tokens": args.max_tokens,
        "variant": args.variant,
        "decision_mode": args.decision_mode,
        "label": spec.label,
        "backend": spec.backend,
        "model": spec.model,
        "local_model_path": spec.local_model_path,
        "mean_accuracy": sum(accuracies) / len(accuracies),
        "sd_accuracy": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
        "mean_latency_seconds": sum(latencies) / len(latencies),
        "sd_latency_seconds": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "mean_historical_top1_accuracy": sum(hist_top1) / len(hist_top1),
        "mean_gold_in_candidates_rate": sum(gold_cover) / len(gold_cover),
        "mean_parse_success_rate": sum(parse_rates) / len(parse_rates),
        "per_seed_accuracy": {str(result["seed"]): result["accuracy"] for result in results},
        "output_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "combined_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
