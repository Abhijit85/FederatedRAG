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

PROMPT_VARIANTS = {
    "plain_index": (
        "You are ranking candidate routing scenarios for a math word problem.\n"
        "Choose the single best candidate index for solving the query.\n"
        "Return only the index number of the best candidate."
    ),
    "strict_index": (
        "Select the best routing candidate for the math problem.\n"
        "Answer with exactly one digit: 1, 2, or 3.\n"
        "Do not output words or explanation."
    ),
    "best_match": (
        "Pick the candidate whose label best matches the math problem type.\n"
        "Return only the candidate number."
    ),
    "retrieval_rerank": (
        "These are retrieved routing labels. Rerank them and choose the best final route.\n"
        "Output only the best candidate index."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep historical label-only rank prompts for the local 8B router.")
    parser.add_argument("--sample-file", type=Path, default=DEFAULT_SAMPLE_FILE)
    parser.add_argument("--runlog", type=Path, default=DEFAULT_RUNLOG)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--max-tokens", type=int, default=4)
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


def build_rank_prompt(query: str, candidates: list[ScenarioDoc], variant: str) -> str:
    options = [f"{i}. {c.label}" for i, c in enumerate(candidates, start=1)]
    return (
        f"{PROMPT_VARIANTS[variant]}\n\n"
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
    backend, prompt_ids = render_prompt_ids(spec, prompt)
    import torch
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
    backend, prompt_ids = render_prompt_ids(spec, prompt)
    import torch
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


def evaluate_variant(records: list[dict[str, Any]], runlog_rows: dict[str, dict[str, Any]], seed: int, sample_count: int, spec: RouterSpec, max_tokens: int, variant: str, decision_mode: str) -> dict[str, Any]:
    subset = sample_records(records, seed=seed, sample_count=sample_count)
    rows=[]
    correct=0
    total_latency=0.0
    parsed=0
    for record in subset:
        qid = str(record.get("query_id") or record.get("sample_id") or "")
        runlog_row = runlog_rows.get(qid)
        if runlog_row is None:
            continue
        query = query_text(record) or str(runlog_row.get("query_text") or "")
        gold = gold_route_label(record, runlog_row)
        candidates = build_candidates(runlog_row)
        prompt = build_rank_prompt(query, candidates, variant)
        started=time.perf_counter()
        raw=generate_choice(spec, prompt, max_tokens=max_tokens)
        scores=score_choice_tokens(spec, prompt, len(candidates))
        latency=time.perf_counter()-started
        ranked=sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        parsed_choice=parse_choice(raw, len(candidates))
        if decision_mode == 'score_only':
            chosen = int(ranked[0][0])
        else:
            chosen = parsed_choice or int(ranked[0][0])
        pred = candidates[chosen-1].label
        hit = labels_match(pred, gold)
        parsed += int(parsed_choice is not None)
        correct += int(hit)
        total_latency += latency
        rows.append({'query_id': qid, 'gold': gold, 'predicted': pred, 'raw': raw, 'parsed_choice': parsed_choice, 'top_candidates': [c.label for c in candidates], 'correct': hit})
    n=len(rows)
    return {
        'seed': seed,
        'variant': variant,
        'decision_mode': decision_mode,
        'accuracy': correct/n if n else 0.0,
        'parse_success_rate': parsed/n if n else 0.0,
        'mean_latency_seconds': total_latency/n if n else 0.0,
        'rows': rows,
    }


def main() -> None:
    load_dotenv(REPO_ROOT / '.env')
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_json_records(args.sample_file)
    runlog_rows = load_runlog_rows(args.runlog)
    seeds = parse_seed_list(args.seeds)
    spec = RouterSpec(label='llama31_8b_historical_rankprompt_sweep', backend='local', model=args.model_path, local_model_path=args.model_path)
    all_results=[]
    for variant in PROMPT_VARIANTS:
        for decision_mode in ('parse_or_score','score_only'):
            seed_results=[evaluate_variant(records, runlog_rows, seed, args.sample_count, spec, args.max_tokens, variant, decision_mode) for seed in seeds]
            accs=[r['accuracy'] for r in seed_results]
            parses=[r['parse_success_rate'] for r in seed_results]
            lats=[r['mean_latency_seconds'] for r in seed_results]
            summary={
                'variant': variant,
                'decision_mode': decision_mode,
                'seeds': seeds,
                'mean_accuracy': sum(accs)/len(accs),
                'sd_accuracy': statistics.stdev(accs) if len(accs)>1 else 0.0,
                'mean_parse_success_rate': sum(parses)/len(parses),
                'mean_latency_seconds': sum(lats)/len(lats),
                'per_seed_accuracy': {str(r['seed']): r['accuracy'] for r in seed_results},
            }
            out_dir=args.output_dir / f'{variant}__{decision_mode}'
            out_dir.mkdir(parents=True, exist_ok=True)
            for r in seed_results:
                (out_dir / f"routing_seed_{r['seed']}.json").write_text(json.dumps(r, indent=2), encoding='utf-8')
            (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
            all_results.append(summary)
            print(json.dumps(summary))
    combined={'sample_file': str(args.sample_file), 'runlog': str(args.runlog), 'sample_count': args.sample_count, 'seeds': seeds, 'max_tokens': args.max_tokens, 'results': all_results}
    (args.output_dir / 'combined_summary.json').write_text(json.dumps(combined, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
