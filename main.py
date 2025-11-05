import argparse
import json
import logging
import os
import sys
import statistics
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from a .env file at the start
load_dotenv()
model_name_env = os.environ.get("MODEL_NAME", "").strip()
os.environ.setdefault("VLM_MODEL", model_name_env or "gpt-4o-mini")

from math_qa import MathQATool
from science_qa import ScienceQATool
from synapse.agent import SynapseAgent
from synapse.config import ApiCredentials
from synapse.runtime import SynapseRuntime
from openrouter_client import get_available_api_keys
from jina_key_manager import get_available_jina_api_keys

# --- 1. LOGGING SETUP ---
class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message != '\n':
            self.level(message.strip())

    def flush(self):
        pass


log_formatter = logging.Formatter('%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("evaluation_log.txt", mode='w')
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

sys.stdout = LoggerWriter(logger.info)
sys.stderr = LoggerWriter(logger.error)
# --- END LOGGING SETUP ---


import collections


def evaluate_mixed_queries(
    agent: SynapseAgent,
    test_file: str = "mixed_queries.json",
    dataset_label: Optional[str] = None,
):
    """Evaluate the SYNAPSE agent on the mixed benchmark file and compute accuracy."""
    label = dataset_label or Path(test_file).name
    print(f"\n--- 3. EVALUATING SYNAPSE AGENT ON DATASET: {label} ---")
    try:
        with open(test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        print(f"[✓] Successfully loaded '{test_file}'.")
    except Exception as e:
        print(f"❌ Error loading test file: {e}")
        return None

    per_dataset = collections.defaultdict(lambda: {"correct": 0, "total": 0})
    bbh_metrics = {"correct": 0, "total": 0}

    def _record(dataset_name: str, hit: int) -> None:
        bucket = per_dataset[dataset_name]
        bucket["correct"] += hit
        bucket["total"] += 1
        if _is_bbh_dataset(dataset_name):
            bbh_metrics["correct"] += hit
            bbh_metrics["total"] += 1

    metrics = {
        "math": {"correct": 0, "total": 0},
        "science": {"correct": 0, "total": 0},
    }

    def _extract_final_answer(text: str) -> str:
        marker = "Final Answer:"
        idx = text.rfind(marker)
        if idx == -1:
            return ""
        answer = text[idx + len(marker):].strip()
        # When answers are like "Final Answer: copepod" keep everything after the marker.
        return answer

    for item in test_data:
        is_science_query = (
            item.get("type") == "science"
            or item.get("domain") == "science"
            or bool(item.get("image"))
        )
        if is_science_query:
            query_text = item["question"]
            data_payload = item
            metrics["science"]["total"] += 1
            print(f"\n--- Processing ScienceQA Query: '{query_text[:80]}...' ---")
        else:
            question_text = item.get("question") or item.get("Problem") or ""
            options = item.get("options")
            if options:
                query_text = f"{question_text}\nOptions: {options}"
            else:
                query_text = question_text
            data_payload = item
            metrics["math"]["total"] += 1
            print(f"\n--- Processing MathQA Query: '{query_text[:80]}...' ---")

        dataset_name = item.get("dataset") or dataset_label or Path(test_file).name
        try:
            result = agent.run(query=query_text, data_item=data_payload)
        except Exception as exc:
            print(f"[✗] Agent execution failed: {exc}")
            continue

        if result and getattr(result, "llm_response", None):
            llm_output = result.llm_response
            print("\n" + "=" * 25 + " AGENT REASONING & FINAL OUTPUT " + "=" * 25)
            print(llm_output)
            print("=" * 80)
            print("[✓] Query processed successfully.")

            final_answer = _normalize_answer(_extract_final_answer(llm_output))
            if is_science_query:
                gold_idx = item.get("answer")
                gold_text = item.get("choices")
                correct = False
                if isinstance(gold_idx, int) and gold_text and 0 <= gold_idx < len(gold_text):
                    gold_answer = gold_text[gold_idx]
                    correct = gold_answer.strip().lower() in final_answer.lower()
                elif isinstance(gold_idx, str):
                    correct = gold_idx.strip().lower() in final_answer.lower()
                else:
                    # fall back to matching the first choice containing the final answer
                    correct = any(choice.strip().lower() in final_answer.lower() for choice in gold_text or [])
                hit = int(bool(correct))
                metrics["science"]["correct"] += hit
                _record(dataset_name, hit)
            else:
                gold = item.get("correct")
                if not gold:
                    gold = item.get("answer")
                if isinstance(gold, str):
                    hit = int(_answers_match(gold, final_answer, dataset_name))
                    metrics["math"]["correct"] += hit
                    _record(dataset_name, hit)
                elif isinstance(gold, (list, tuple)):
                    if any(_answers_match(g, final_answer, dataset_name) for g in gold):
                        metrics["math"]["correct"] += 1
                        _record(dataset_name, 1)
        else:
            print("[✗] Agent failed to produce a final result for this query.")

    total_correct = metrics["math"]["correct"] + metrics["science"]["correct"]
    total_questions = metrics["math"]["total"] + metrics["science"]["total"]

    def _accuracy(correct: int, total: int) -> Optional[float]:
        if total == 0:
            return None
        return correct / total

    def _format_accuracy(correct: int, total: int) -> str:
        if total == 0:
            return "n/a"
        return f"{correct}/{total} ({correct / total * 100:.1f}%)"

    print("\n--- METRICS ---")
    print(f"Math Accuracy: {_format_accuracy(metrics['math']['correct'], metrics['math']['total'])}")
    print(f"Science Accuracy: {_format_accuracy(metrics['science']['correct'], metrics['science']['total'])}")
    print(f"Overall Accuracy: {_format_accuracy(total_correct, total_questions)}")
    if per_dataset:
        print("Dataset accuracies:")
        for name, bucket in per_dataset.items():
            print(f"  · {name}: {_format_accuracy(bucket['correct'], bucket['total'])}")
    if bbh_metrics["total"]:
        print(f"BBH Accuracy: {_format_accuracy(bbh_metrics['correct'], bbh_metrics['total'])}")

    print("\n--- AGENT EVALUATION COMPLETE ---")
    print("Full output has been saved to evaluation_log.txt")

    return {
        "label": label,
        "math": {
            "correct": metrics["math"]["correct"],
            "total": metrics["math"]["total"],
            "accuracy": _accuracy(metrics["math"]["correct"], metrics["math"]["total"]),
        },
        "datasets": {
            name: {
                "correct": bucket["correct"],
                "total": bucket["total"],
                "accuracy": (bucket["correct"] / bucket["total"]) if bucket["total"] else None,
            }
            for name, bucket in per_dataset.items()
        },
        "bbh": {
            "correct": bbh_metrics["correct"],
            "total": bbh_metrics["total"],
            "accuracy": _accuracy(bbh_metrics["correct"], bbh_metrics["total"]),
        },
        "science": {
            "correct": metrics["science"]["correct"],
            "total": metrics["science"]["total"],
            "accuracy": _accuracy(metrics["science"]["correct"], metrics["science"]["total"]),
        },
        "overall": {
            "correct": total_correct,
            "total": total_questions,
            "accuracy": _accuracy(total_correct, total_questions),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a SYNAPSE federation round and evaluate the agent.")
    parser.add_argument(
        "--test-file",
        type=Path,
        default=Path("mixed_queries.json"),
        help="Primary evaluation dataset for the global run.",
    )
    parser.add_argument(
        "--client-data-dir",
        type=Path,
        help="Directory containing per-client evaluation datasets (JSON files). Each file is evaluated individually.",
    )
    parser.add_argument(
        "--client-count",
        type=int,
        help="Number of synthetic clients to spawn locally. Overrides SYNAPSE_CLIENT_COUNT if provided.",
    )
    parser.add_argument(
        "--skip-global-eval",
        action="store_true",
        help="Skip evaluation of the aggregated (central) SYNAPSE agent after the federation round.",
    )
    parser.add_argument(
        "--evaluate-clients",
        action="store_true",
        help="Evaluate per-client datasets (requires --client-data-dir).",
    )
    return parser.parse_args()


def main():
    """
    Run a full SYNAPSE federation round and evaluate the resulting agent.
    """
    args = parse_args()

    available_lambda_keys = get_available_api_keys(allow_empty=True)
    lambda_key = available_lambda_keys[0] if available_lambda_keys else None
    if not lambda_key:
        lambda_key = os.environ.get("LAMBDA_API_KEY") or os.environ.get("LAMDA_API_KEY", "")
        if lambda_key:
            print("⚠️ Detected deprecated env var for Lambda API key. Please rename it to API_KEY or API_KEY_<n>.")
    jina_keys = get_available_jina_api_keys(allow_empty=True)
    jina_key = jina_keys[0] if jina_keys else ""
    mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
    if not lambda_key or not jina_key:
        print("⚠️ Warning: API keys are missing; downstream tool calls may fail.")

    # Ensure downstream libraries receive the keys even if they import later.
    if lambda_key:
        os.environ["API_KEY"] = lambda_key
    if jina_key:
        os.environ["JINA_API_KEY"] = jina_key
    if mongo_uri:
        os.environ["MONGO_URI"] = mongo_uri

    credentials = ApiCredentials(
        lambda_api_key=lambda_key,
        jina_api_key=jina_key,
        mongo_uri=mongo_uri,
        lambda_api_base="https://openrouter.ai/api/v1/chat/completions",
    )

    runtime = SynapseRuntime.build_local_runtime(
        Path.cwd(),
        credentials,
        client_count=args.client_count,
    )
    print("\n--- 1. SYNAPSE FEDERATION ROUND ---")
    runtime.run_round()
    summary = runtime.summarize_round()
    print(f"SYNAPSE Round Summary: {summary}")

    tool_registry = {
        "mathqa": MathQATool(),
        "scienceqa": ScienceQATool(),
    }
    agent = SynapseAgent(runtime=runtime, tool_registry=tool_registry)

    runtime.export_snapshot(Path("synapse_global_snapshot.json"))
    print("✅ Exported SYNAPSE snapshot to 'synapse_global_snapshot.json'.")

    if args.skip_global_eval:
        print("ℹ️ Skipping global evaluation as requested.")
    else:
        evaluate_mixed_queries(agent, test_file=str(args.test_file), dataset_label=args.test_file.name)

    client_metrics = []
    if args.evaluate_clients:
        if not args.client_data_dir:
            print("⚠️ --evaluate-clients was set but no --client-data-dir provided; skipping client evaluations.")
        else:
            dataset_dir = args.client_data_dir
            if not dataset_dir.exists():
                print(f"⚠️ Client data directory '{dataset_dir}' does not exist; skipping per-client evaluations.")
            else:
                dataset_paths = sorted(p for p in dataset_dir.glob("*.json") if p.is_file())
                if not dataset_paths:
                    print(f"⚠️ No JSON datasets found in '{dataset_dir}'.")
                for data_path in dataset_paths:
                    metrics = evaluate_mixed_queries(agent, test_file=str(data_path), dataset_label=data_path.stem)
                    if metrics:
                        client_metrics.append(metrics)

    if client_metrics:
        def _format_percent(value: Optional[float]) -> str:
            return f"{value * 100:.1f}%" if value is not None else "n/a"

        print("\n--- FEDERATED CLIENT BENCHMARK SUMMARY ---")
        overall_values = [m["overall"]["accuracy"] for m in client_metrics if m["overall"]["accuracy"] is not None]
        math_values = [m["math"]["accuracy"] for m in client_metrics if m["math"]["accuracy"] is not None]
        science_values = [m["science"]["accuracy"] for m in client_metrics if m["science"]["accuracy"] is not None]

        if overall_values:
            macro = sum(overall_values) / len(overall_values)
            spread = max(overall_values) - min(overall_values)
            stdev = statistics.pstdev(overall_values) if len(overall_values) > 1 else 0.0
            print(f"Macro overall accuracy: {_format_percent(macro)}")
            print(f"Overall accuracy spread: {_format_percent(spread)} (max - min)")
            print(f"Overall accuracy σ: {_format_percent(stdev)}")
        if math_values:
            print(f"Macro math accuracy: {_format_percent(sum(math_values) / len(math_values))}")
        if science_values:
            print(f"Macro science accuracy: {_format_percent(sum(science_values) / len(science_values))}")

        for metrics in client_metrics:
            dataset_details = metrics.get("datasets", {})
            dataset_summary = ", ".join(
                f"{name}={_format_percent(bucket['accuracy'])}" for name, bucket in dataset_details.items()
            )
            if not dataset_summary:
                dataset_summary = "datasets=n/a"
            bbh_acc = metrics.get("bbh", {}).get("accuracy")
            print(
                f"  · {metrics['label']}: "
                f"overall={_format_percent(metrics['overall']['accuracy'])}, "
                f"math={_format_percent(metrics['math']['accuracy'])}, "
                f"science={_format_percent(metrics['science']['accuracy'])}, "
                f"bbh={_format_percent(bbh_acc)}, "
                f"{dataset_summary}"
            )
def _normalize_answer(text: str) -> str:
    """
    Normalize answer strings by removing common prefixes and punctuation.
    """
    cleaned = text.strip()
    cleaned = cleaned.replace("Final Answer:", "").replace("Answer:", "")
    cleaned = cleaned.strip().lower().strip(".")
    return cleaned


def _is_bbh_dataset(name: str | None) -> bool:
    return isinstance(name, str) and name.upper().startswith("BBH")


def _extract_numeric_value(text: str) -> Optional[int]:
    match = re.findall(r"-?\d+", text)
    if not match:
        return None
    try:
        return int(match[-1])
    except ValueError:
        return None


def _answers_match(gold: str, prediction: str, dataset_name: str) -> bool:
    if _is_bbh_dataset(dataset_name):
        gold_val = _extract_numeric_value(gold)
        pred_val = _extract_numeric_value(prediction)
        return gold_val is not None and pred_val == gold_val
    return _normalize_answer(gold) == prediction


if __name__ == "__main__":
    main()
