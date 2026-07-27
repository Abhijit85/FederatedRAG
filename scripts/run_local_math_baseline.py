#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jina_key_manager import get_available_jina_api_keys
from openrouter_client import get_available_api_keys
from math_qa import MathQATool
from synapse.config import ApiCredentials
from synapse.runtime import SynapseRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a clean local no-privacy math baseline.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "train_new.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--client-count", type=int, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "verification" / "local_math_baseline_train_new.json")
    return parser.parse_args()


def _normalize_answer(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace("Final Answer:", "").replace("Answer:", "")
    return cleaned.strip().lower().strip(".")


def _parse_options(options: str | None) -> dict[str, str]:
    if not options:
        return {}
    matches = re.findall(r"([a-e])\s*\)\s*(.*?)(?=\s*,\s*[a-e]\s*\)|$)", options, flags=re.IGNORECASE)
    return {letter.lower(): text.strip().lower().strip(".") for letter, text in matches}


def _extract_final_answer(text: str, options: str | None = None) -> str:
    marker = "Final Answer:"
    idx = text.rfind(marker)
    if idx != -1:
        return text[idx + len(marker):].strip()

    lower = text.lower()
    letter_match = re.search(r"(?:final answer|answer|option)\s*[:\-]?\s*([a-e])\b", lower)
    if letter_match:
        return letter_match.group(1)

    option_map = _parse_options(options)
    for letter, option_text in option_map.items():
        if option_text and option_text in lower:
            return letter

    return text.strip()


def _answers_match(gold: str, prediction: str) -> bool:
    gold_norm = _normalize_answer(gold)
    pred_norm = _normalize_answer(prediction)
    if pred_norm == gold_norm:
        return True
    if pred_norm in {"a", "b", "c", "d", "e"} and gold_norm in {"a", "b", "c", "d", "e"}:
        return pred_norm == gold_norm
    return False


def _build_credentials() -> ApiCredentials:
    load_dotenv(ROOT / ".env")
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


def _load_dataset(path: Path, limit: Optional[int]) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list dataset at {path}")
    if limit is not None:
        return data[:limit]
    return data


def main() -> None:
    args = parse_args()
    os.environ["SYNAPSE_ENABLE_DP"] = "0"
    os.environ["PROMPT_ATTACK"] = "0"
    os.environ["MATHQA_FORCE_OPTION_ONLY"] = "1"

    records = _load_dataset(args.dataset, args.limit)
    credentials = _build_credentials()
    runtime = SynapseRuntime.build_local_runtime(ROOT, credentials, client_count=args.client_count)
    for _ in range(max(1, args.rounds)):
        runtime.run_round()

    math_tool = MathQATool()

    rows: list[dict[str, Any]] = []
    correct = 0
    for idx, item in enumerate(records, start=1):
        question = item.get("question") or item.get("Problem") or ""
        options = item.get("options")
        query = f"{question}\nOptions: {options}" if options else question
        query += "\nReturn only one line in exactly this format: Final Answer: <option letter>. Do not include any other text."
        result = math_tool.run(user_query=query, data_item=item)
        llm_output = result.llm_response or ""
        parsed_final = None
        if getattr(result, "parsed_output", None):
            parsed_final = result.parsed_output.get("final_answer")
        pred = _normalize_answer(parsed_final or _extract_final_answer(llm_output, options))
        gold = str(item.get("correct") or item.get("answer") or "")
        hit = _answers_match(gold, pred)
        correct += int(hit)
        rows.append(
            {
                "index": idx,
                "problem": question,
                "gold": gold,
                "prediction": pred,
                "correct": hit,
                "raw_output": llm_output,
            }
        )
        print(f"[{idx}/{len(records)}] hit={int(hit)} gold={gold} pred={pred}")

    total = len(rows)
    accuracy = correct / total if total else 0.0
    payload = {
        "dataset": str(args.dataset),
        "limit": args.limit,
        "rounds": args.rounds,
        "client_count": args.client_count,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "misses": [row for row in rows if not row["correct"]],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"accuracy": accuracy, "correct": correct, "total": total, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
