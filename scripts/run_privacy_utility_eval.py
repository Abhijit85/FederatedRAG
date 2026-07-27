#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jina_key_manager import get_available_jina_api_keys
from math_qa import MathQATool
from openrouter_client import get_available_api_keys
from synapse.agent import SynapseAgent
from synapse.config import ApiCredentials
from synapse.runtime import SynapseRuntime

try:
    import numpy as np
except Exception:  # pragma: no cover - optional
    np = None

try:
    import torch
except Exception:  # pragma: no cover - optional
    torch = None


DEFAULT_DATASET = REPO_ROOT / "train_new.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "verification" / "privacy_utility_eval"


@dataclass
class AttackMetrics:
    linkage_accuracy: float
    auroc: float
    pair_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a privacy-utility evaluation on the current SYNAPSE runtime."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--client-count", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument(
        "--lambda-scale",
        type=float,
        default=None,
        help="Proxy text-privacy strength. Maps to adaptive probability/distortion multipliers.",
    )
    parser.add_argument("--label", type=str, default="custom")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _build_credentials() -> ApiCredentials:
    load_dotenv(REPO_ROOT / ".env")
    lambda_keys = get_available_api_keys(allow_empty=True)
    jina_keys = get_available_jina_api_keys(allow_empty=True)
    lambda_key = lambda_keys[0] if lambda_keys else (os.environ.get("API_KEY") or "")
    jina_key = jina_keys[0] if jina_keys else (os.environ.get("JINA_API_KEY") or "")
    mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
    if lambda_key:
        os.environ["API_KEY"] = lambda_key
    if jina_key:
        os.environ["JINA_API_KEY"] = jina_key
    os.environ["MONGO_URI"] = mongo_uri
    return ApiCredentials(
        lambda_api_key=lambda_key,
        jina_api_key=jina_key,
        mongo_uri=mongo_uri,
        lambda_api_base="https://openrouter.ai/api/v1/chat/completions",
    )


def _configure_privacy_env(epsilon: float | None, lambda_scale: float | None) -> dict[str, str | None]:
    previous = {
        key: os.environ.get(key)
        for key in (
            "SYNAPSE_ENABLE_DP",
            "SYNAPSE_DP_EPSILON",
            "SYNAPSE_ADAPTIVE_TEXT_NOISE",
            "SYNAPSE_ADAPTIVE_PROBABILITY_MULT",
            "SYNAPSE_ADAPTIVE_DISTORT_MULT",
            "PROMPT_ATTACK",
            "MATHQA_FORCE_OPTION_ONLY",
        )
    }
    os.environ["PROMPT_ATTACK"] = "0"
    os.environ["MATHQA_FORCE_OPTION_ONLY"] = "1"
    if epsilon is None and lambda_scale is None:
        os.environ["SYNAPSE_ENABLE_DP"] = "0"
        os.environ["SYNAPSE_ADAPTIVE_TEXT_NOISE"] = "0"
        os.environ.pop("SYNAPSE_DP_EPSILON", None)
        os.environ.pop("SYNAPSE_ADAPTIVE_PROBABILITY_MULT", None)
        os.environ.pop("SYNAPSE_ADAPTIVE_DISTORT_MULT", None)
        return previous

    os.environ["SYNAPSE_ENABLE_DP"] = "1"
    os.environ["SYNAPSE_DP_EPSILON"] = str(epsilon if epsilon is not None else 1.0)
    if lambda_scale is None:
        os.environ["SYNAPSE_ADAPTIVE_TEXT_NOISE"] = "0"
        os.environ.pop("SYNAPSE_ADAPTIVE_PROBABILITY_MULT", None)
        os.environ.pop("SYNAPSE_ADAPTIVE_DISTORT_MULT", None)
    else:
        os.environ["SYNAPSE_ADAPTIVE_TEXT_NOISE"] = "1"
        os.environ["SYNAPSE_ADAPTIVE_PROBABILITY_MULT"] = str(lambda_scale)
        os.environ["SYNAPSE_ADAPTIVE_DISTORT_MULT"] = str(lambda_scale)
    return previous


def _restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _load_records(path: Path, sample_count: int, seed: int) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a list dataset at {path}")
    if sample_count > len(records):
        raise ValueError(f"Requested {sample_count} rows, but only {len(records)} are available.")
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(records)), sample_count))
    return [records[idx] for idx in indices]


def _normalize_answer(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace("Final Answer:", "").replace("Answer:", "")
    return cleaned.strip().lower().strip(".")


def _parse_options(options: Any) -> dict[str, str]:
    if isinstance(options, list):
        option_text = ", ".join(str(item) for item in options)
    else:
        option_text = str(options or "")
    matches = re.findall(r"([a-e])\s*\)\s*(.*?)(?=\s*,\s*[a-e]\s*\)|$)", option_text, flags=re.IGNORECASE)
    return {letter.lower(): text.strip().lower().strip(".") for letter, text in matches}


def _extract_prediction(result: Any, options: Any) -> str:
    parsed_output = getattr(result, "parsed_output", None) or {}
    final_answer = parsed_output.get("final_answer")
    if isinstance(final_answer, str) and final_answer.strip():
        return _normalize_answer(final_answer)

    llm_output = getattr(result, "llm_response", "") or ""
    lower = llm_output.lower()
    letter_match = re.search(r"(?:final answer|answer|option)\s*[:\-]?\s*([a-e])\b", lower)
    if letter_match:
        return letter_match.group(1)

    option_map = _parse_options(options)
    for letter, option_text in option_map.items():
        if option_text and option_text in lower:
            return letter
    return _normalize_answer(llm_output)


def _answers_match(gold: str, prediction: str) -> bool:
    return _normalize_answer(gold) == _normalize_answer(prediction)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _similarity(left: str, right: str) -> float:
    left_tokens = _tokenize(left)
    right_tokens = _tokenize(right)
    token_score = 0.0
    if left_tokens or right_tokens:
        token_score = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
    sequence_score = SequenceMatcher(a=left, b=right).ratio()
    return 0.5 * token_score + 0.5 * sequence_score


def _binary_auroc(positive_scores: list[float], negative_scores: list[float]) -> float | None:
    if not positive_scores or not negative_scores:
        return None
    wins = 0.0
    total = 0
    for pos in positive_scores:
        for neg in negative_scores:
            total += 1
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total if total else None


def _compute_attack_metrics(attack_pairs: list[dict[str, str]]) -> AttackMetrics:
    if len(attack_pairs) < 2:
        return AttackMetrics(linkage_accuracy=1.0 if attack_pairs else 0.0, auroc=None, pair_count=len(attack_pairs))

    positives: list[float] = []
    negatives: list[float] = []
    correct = 0

    raw_by_signature = {pair["signature"]: pair["raw_text"] for pair in attack_pairs}
    raw_items = list(raw_by_signature.items())

    for pair in attack_pairs:
        observed = pair["observed_text"]
        signature = pair["signature"]
        candidate_scores = []
        for raw_sig, raw_text in raw_items:
            score = _similarity(raw_text, observed)
            candidate_scores.append((raw_sig, score))
            if raw_sig == signature:
                positives.append(score)
            else:
                negatives.append(score)
        best_sig, _ = max(candidate_scores, key=lambda item: item[1])
        correct += int(best_sig == signature)

    auroc = _binary_auroc(positives, negatives)
    return AttackMetrics(
        linkage_accuracy=correct / len(attack_pairs),
        auroc=auroc,
        pair_count=len(attack_pairs),
    )


def _evaluate_seed(
    *,
    dataset_path: Path,
    sample_count: int,
    seed: int,
    rounds: int,
    client_count: int | None,
) -> dict[str, Any]:
    _seed_everything(seed)
    records = _load_records(dataset_path, sample_count=sample_count, seed=seed)
    runtime = SynapseRuntime.build_local_runtime(REPO_ROOT, _build_credentials(), client_count=client_count)
    for _ in range(max(1, rounds)):
        runtime.run_round()

    math_tool = MathQATool()
    agent = SynapseAgent(runtime=runtime, tool_registry={"mathqa": math_tool})

    rows: list[dict[str, Any]] = []
    correct = 0
    for idx, item in enumerate(records, start=1):
        question = item.get("question") or item.get("Problem") or ""
        options = item.get("options")
        option_text = ", ".join(str(opt) for opt in options) if isinstance(options, list) else str(options or "")
        query = question
        if option_text:
            query += f"\nOptions: {option_text}"
        query += "\nReturn only one line in exactly this format: Final Answer: <option letter>. Do not include any other text."
        result = agent.run(
            query=query,
            data_item={
                **item,
                "task_type": "math",
                "dataset": "mathqa",
            },
        )
        prediction = _extract_prediction(result, options)
        gold = str(item.get("correct") or item.get("answer") or item.get("Answer") or "")
        hit = _answers_match(gold, prediction)
        correct += int(hit)
        rows.append(
            {
                "index": idx,
                "problem": question,
                "gold": gold,
                "prediction": prediction,
                "correct": hit,
            }
        )

    client_metrics: dict[str, dict[str, Any]] = {}
    client_linkage = []
    client_aurocs = []
    for client_id, client in runtime.clients.items():
        attack_pairs = client.get_attack_artifacts()
        metrics = _compute_attack_metrics(attack_pairs)
        client_metrics[client_id] = {
            **asdict(metrics),
            "sample_pairs": attack_pairs[:3],
        }
        client_linkage.append(metrics.linkage_accuracy)
        if metrics.auroc is not None:
            client_aurocs.append(metrics.auroc)

    pct_clients_lt_point1 = (
        sum(value < 0.10 for value in client_linkage) / len(client_linkage)
        if client_linkage
        else None
    )
    mean_auroc = sum(client_aurocs) / len(client_aurocs) if client_aurocs else None

    return {
        "seed": seed,
        "sample_count": sample_count,
        "correct": correct,
        "accuracy": correct / sample_count if sample_count else 0.0,
        "mean_auroc": mean_auroc,
        "pct_clients_lt_point1": pct_clients_lt_point1,
        "client_metrics": client_metrics,
        "rows": rows,
    }


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous_env = _configure_privacy_env(args.epsilon, args.lambda_scale)
    try:
        seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
        results = [
            _evaluate_seed(
                dataset_path=args.dataset,
                sample_count=args.sample_count,
                seed=seed,
                rounds=args.rounds,
                client_count=args.client_count,
            )
            for seed in seeds
        ]
    finally:
        _restore_env(previous_env)

    accuracies = [result["accuracy"] for result in results]
    mean_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    if len(accuracies) > 1:
        variance = sum((value - mean_accuracy) ** 2 for value in accuracies) / (len(accuracies) - 1)
        sd_accuracy = variance ** 0.5
    else:
        sd_accuracy = 0.0

    aurocs = [result["mean_auroc"] for result in results if result["mean_auroc"] is not None]
    client_pct = [
        result["pct_clients_lt_point1"] for result in results if result["pct_clients_lt_point1"] is not None
    ]

    for result in results:
        out_path = args.output_dir / f"{args.label}_seed_{result['seed']}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    summary = {
        "label": args.label,
        "dataset": str(args.dataset),
        "sample_count": args.sample_count,
        "rounds": args.rounds,
        "client_count": args.client_count,
        "epsilon": args.epsilon,
        "lambda_scale": args.lambda_scale,
        "seeds": seeds,
        "mean_accuracy": mean_accuracy,
        "sd_accuracy": sd_accuracy,
        "mean_auroc": sum(aurocs) / len(aurocs) if aurocs else None,
        "mean_pct_clients_lt_point1": sum(client_pct) / len(client_pct) if client_pct else None,
        "per_seed_accuracy": {str(result["seed"]): result["accuracy"] for result in results},
        "per_seed_auroc": {str(result["seed"]): result["mean_auroc"] for result in results},
        "per_seed_pct_clients_lt_point1": {
            str(result["seed"]): result["pct_clients_lt_point1"] for result in results
        },
        "output_dir": str(args.output_dir),
    }
    summary_path = args.output_dir / f"{args.label}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Label: {args.label}")
    print(f"Dataset: {args.dataset}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Lambda scale: {args.lambda_scale}")
    print(f"Mean accuracy: {mean_accuracy:.4f}")
    print(f"SD accuracy: {sd_accuracy:.4f}")
    print(f"Mean AUROC: {summary['mean_auroc']}")
    print(f"Mean % clients < 0.10: {summary['mean_pct_clients_lt_point1']}")
    print(f"Wrote summary to: {summary_path}")


if __name__ == "__main__":
    main()
