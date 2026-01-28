#!/usr/bin/env python3
"""Run a SYNAPSE federation round with TextGrad optimisation."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from torch.utils.data import Dataset as TorchDataset, RandomSampler, Subset, random_split

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def load_project_env() -> None:
    """
    Ensure the repository root is on sys.path and .env variables are loaded before imports.
    """
    load_dotenv()
    from openrouter_client import get_available_api_keys  # noqa: WPS433
    return get_available_api_keys


get_available_api_keys = load_project_env()

def sync_tool_model_env() -> None:
    """
    Ensure downstream tools have a default model before heavy imports that
    read EVAL_MODEL/VLM_MODEL.
    """
    test_engine = os.environ.get("TEXTGRAD_TEST_ENGINE") or os.environ.get("MODEL_NAME")
    if test_engine:
        os.environ.setdefault("EVAL_MODEL", test_engine)
        os.environ.setdefault("VLM_MODEL", test_engine)


sync_tool_model_env()

from math_qa import MathQATool
from science_qa import ScienceQATool
from synapse.agent import SynapseAgent
from synapse.clients.textgrad_trainer import TextGradPromptTrainer
from synapse.config import ApiCredentials
from synapse.runtime import SynapseRuntime
from synapse.textgrad_support import TextGradSettings
from third_party.textgrad import BlackboxLLM, TextualGradientDescent
from third_party.textgrad.tasks import DataLoader, load_task


class LocalPromptDataset(TorchDataset):
    """
    Lightweight dataset wrapping custom JSON prompt/answer pairs for TextGrad.
    """

    def __init__(self, samples: List[tuple[str, str]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.samples[idx]


def _resolve_runtime_defaults() -> tuple[bool, str, str, int]:
    """
    Decide whether we should default to online models or offline mocks.
    Online defaults are enabled as soon as MODEL_NAME is populated. If
    SYNAPSE_CLIENT_COUNT is missing, fall back to 4 clients.
    """
    env_model = os.environ.get("MODEL_NAME", "").strip()
    env_client = os.environ.get("SYNAPSE_CLIENT_COUNT", "").strip()
    online_ready = bool(env_model)

    if online_ready:
        eval_default = env_model
        test_default = env_model
        try:
            client_default = int(env_client) if env_client else 4
        except ValueError:
            client_default = 4
    else:
        eval_default = "offline-mock"
        test_default = "offline-mock"
        client_default = 2

    return online_ready, eval_default, test_default, client_default



def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _format_accuracy_value(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{value * 100:.1f}%"
    except TypeError:
        return "n/a"


def _format_metric_bucket(name: str, bucket: Dict[str, Any]) -> str:
    correct = bucket.get("correct")
    total = bucket.get("total")
    accuracy = bucket.get("accuracy")
    return (
        f"  · {name}: {correct}/{total} ({_format_accuracy_value(accuracy)})"
        if correct is not None and total is not None
        else f"  · {name}: {bucket}"
    )


def _format_textgrad_log_entry(record: Dict[str, Any]) -> str:
    timestamp = record.get("timestamp", "")
    section = record.get("section", "")
    payload: Dict[str, Any] = record.get("payload", {}) or {}

    lines = ["=" * 72, f"[{timestamp}] TextGrad evaluation ({section})"]

    mixed_queries = payload.get("mixed_queries")
    if mixed_queries:
        lines.append(f"Benchmark: {mixed_queries}")

    label = payload.get("label")
    if label:
        lines.append(f"Dataset label: {label}")

    dataset_path = payload.get("dataset_path")
    if dataset_path:
        lines.append(f"Dataset path: {dataset_path}")

    domains = payload.get("domains")
    if isinstance(domains, dict) and domains:
        lines.append("Domain metrics:")
        for name in sorted(domains.keys()):
            bucket = domains[name] or {}
            lines.append(_format_metric_bucket(name, bucket))

    datasets = payload.get("datasets")
    if isinstance(datasets, dict) and datasets:
        lines.append("Dataset breakdown:")
        for name in sorted(datasets.keys()):
            bucket = datasets[name] or {}
            lines.append(_format_metric_bucket(name, bucket))

    overall = payload.get("overall")
    if isinstance(overall, dict) and overall:
        lines.append("Overall:")
        lines.append(_format_metric_bucket("aggregate", overall))

    central = payload.get("central")
    if isinstance(central, dict):
        label = central.get("label")
        if label:
            lines.append(f"Central dataset: {label}")
        central_overall = central.get("overall")
        if isinstance(central_overall, dict):
            lines.append("Central overall:")
            lines.append(_format_metric_bucket("overall", central_overall))
        central_domains = central.get("domains")
        if isinstance(central_domains, dict) and central_domains:
            lines.append("Central domains:")
            for name, bucket in central_domains.items():
                lines.append(_format_metric_bucket(name, bucket))
        central_datasets = central.get("datasets")
        if isinstance(central_datasets, dict) and central_datasets:
            lines.append("Central datasets:")
            for name, bucket in central_datasets.items():
                lines.append(_format_metric_bucket(name, bucket))

    client_summary = payload.get("client_summary")
    if isinstance(client_summary, dict):
        lines.append("Client summary:")
        macro_overall = client_summary.get("macro_overall")
        if macro_overall is not None:
            lines.append(f"  Macro overall accuracy: {_format_accuracy_value(macro_overall)}")
        spread = client_summary.get("overall_spread")
        if spread is not None:
            lines.append(f"  Overall spread: {_format_accuracy_value(spread)}")
        stdev = client_summary.get("overall_stdev")
        if stdev is not None:
            lines.append(f"  Overall σ: {_format_accuracy_value(stdev)}")
        macro_math = client_summary.get("macro_math")
        if macro_math is not None:
            lines.append(f"  Macro math accuracy: {_format_accuracy_value(macro_math)}")
        macro_science = client_summary.get("macro_science")
        if macro_science is not None:
            lines.append(f"  Macro science accuracy: {_format_accuracy_value(macro_science)}")

        details = client_summary.get("details") or []
        if details:
            lines.append("  Per-client accuracies:")
            for detail in details:
                lines.append(
                    "    · "
                    + f"{detail.get('label', 'client')}: overall={_format_accuracy_value(detail.get('overall'))}, "
                    + f"math={_format_accuracy_value(detail.get('math'))}, "
                    + f"science={_format_accuracy_value(detail.get('science'))}"
                )
                dataset_map = detail.get("datasets") or {}
                if dataset_map:
                    dataset_str = ", ".join(
                        f"{name}={_format_accuracy_value(value)}" for name, value in dataset_map.items()
                    )
                    lines.append(f"       datasets: {dataset_str}")

    extra_keys = {"mixed_queries", "label", "dataset_path", "domains", "datasets", "overall", "central", "client_summary"}
    other_items = {k: v for k, v in payload.items() if k not in extra_keys}
    if other_items:
        lines.append("Additional details:")
        for key, value in other_items.items():
            lines.append(f"  · {key}: {value}")

    lines.append("")
    return "\n".join(lines)


def _append_textgrad_log(section: str, payload: Dict[str, Any]) -> None:
    """
    Append a human-readable and JSON record describing TextGrad evaluation output.
    """
    log_path = Path("evaluation_on_textgrad_log.txt")
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "section": section,
        "payload": payload,
    }
    try:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(_format_textgrad_log_entry(record))
            fh.write(json.dumps(record))
            fh.write("\n\n")
    except OSError as exc:
        print(f"⚠️ Failed to write TextGrad log: {exc}")


def _format_question(entry: Dict[str, Any]) -> Optional[str]:
    question = entry.get("question") or entry.get("Problem") or entry.get("prompt")
    if not question:
        return None
    parts = [str(question).strip()]

    options = entry.get("options")
    if isinstance(options, list) and options:
        parts.append("Options: " + "; ".join(map(str, options)))
    elif isinstance(options, str) and options.strip():
        parts.append("Options: " + options.strip())

    choices = entry.get("choices")
    if isinstance(choices, list) and choices:
        parts.append("Choices: " + "; ".join(map(str, choices)))

    lecture = entry.get("lecture")
    if isinstance(lecture, str) and lecture.strip():
        parts.append("Lecture: " + lecture.strip())

    hint = entry.get("hint")
    if isinstance(hint, str) and hint.strip():
        parts.append("Hint: " + hint.strip())

    return "\n".join(parts).strip()


def _format_answer(entry: Dict[str, Any]) -> Optional[str]:
    answer = entry.get("answer")
    if answer is None:
        answer = entry.get("Answer") or entry.get("correct")
    choices = entry.get("choices")
    if isinstance(answer, int) and isinstance(choices, list) and 0 <= answer < len(choices):
        return str(choices[answer])
    if isinstance(answer, (int, float)):
        return str(answer)
    if isinstance(answer, str) and answer.strip():
        return answer.strip()
    return None


def _load_json_prompt_dataset(path: Path) -> Optional[LocalPromptDataset]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[!] Failed to load client training data '{path}': {exc}")
        return None
    if not isinstance(payload, list):
        print(f"[!] Expected a list of samples in '{path}'. Skipping.")
        return None
    samples: List[tuple[str, str]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        question = _format_question(entry)
        answer = _format_answer(entry)
        if question and answer:
            samples.append((question, answer))
    if not samples:
        print(f"[!] No valid samples found in '{path}'.")
        return None
    return LocalPromptDataset(samples)


def _load_client_train_sets(directory: Optional[Path], client_ids: List[str]) -> Dict[str, LocalPromptDataset]:
    if directory is None:
        return {}
    if not directory.exists():
        print(f"⚠️ Client training directory '{directory}' does not exist; using default task splits.")
        return {}
    mapping: Dict[str, LocalPromptDataset] = {}
    for idx, client_id in enumerate(client_ids, start=1):
        candidates = sorted(directory.glob(f"client_{idx}_*.json"))
        if not candidates:
            candidates = sorted(directory.glob(f"{client_id}*.json"))
        if not candidates:
            continue
        dataset = _load_json_prompt_dataset(candidates[0])
        if dataset:
            mapping[client_id] = dataset
            print(f"[Train] Loaded custom dataset for {client_id} from '{candidates[0]}'.")
    return mapping


def parse_args() -> argparse.Namespace:
    online_ready, eval_default, test_default, client_default = _resolve_runtime_defaults()
    sample_default = _env_flag("TEXTGRAD_SAMPLE_WITH_REPLACEMENT", default=False)

    parser = argparse.ArgumentParser(description="Run TextGrad-enabled SYNAPSE federation.")
    parser.add_argument("--task", type=str, default="BBH_object_counting", help="TextGrad task to optimise.")
    parser.add_argument("--client-count", type=int, default=client_default, help="Number of SYNAPSE clients.")
    parser.add_argument("--rounds", type=int, default=1, help="Federation rounds to execute.")
    parser.add_argument("--aggregate-method", type=str, default="summarization", choices=["concat", "summarization", "sum_uid"], help="Aggregation method for client prompts.")
    parser.add_argument("--evaluation-engine", type=str, default=eval_default, help="LLM used to compute textual gradients (overrides TEXTGRAD_EVAL_ENGINE).")
    parser.add_argument("--test-engine", type=str, default=test_default, help="LLM used for client-side testing.")
    parser.add_argument("--batch-size", type=int, default=3, help="Training batch size per client.")
    parser.add_argument("--max-steps", type=int, default=3, help="Maximum optimisation steps per client.")
    parser.add_argument("--disable-proximal", action="store_true", help="Disable proximal rejection of harmful updates.")
    parser.add_argument("--mixed-queries", type=Path, default=Path("mixed_queries.json"), help="Benchmark file for post-round evaluation.")
    parser.add_argument("--output-snapshot", type=Path, default=Path("synapse_textgrad_snapshot.json"), help="Path to export the final compendium snapshot.")
    parser.add_argument(
        "--client-data-dir",
        type=Path,
        help="Optional directory of per-client evaluation JSON files (e.g., BBH slices) to score after training.",
    )
    parser.add_argument(
        "--client-train-dir",
        type=Path,
        help="Optional directory of per-client JSON training files (client_{k}_*.json) for heterogeneous TextGrad.",
    )
    parser.add_argument(
        "--evaluate-clients",
        action="store_true",
        help="Evaluate the aggregated agent on each dataset found in --client-data-dir.",
    )
    parser.add_argument(
        "--sample-with-replacement",
        action="store_true",
        default=sample_default,
        help="Draw TextGrad batches with replacement (samples may repeat before epoch ends).",
    )

    args = parser.parse_args()
    args.test_engine = args.evaluation_engine  # enforce single-model usage
    setattr(args, "online_ready", online_ready)

    if not online_ready:
        if args.evaluation_engine != "offline-mock" or args.test_engine != "offline-mock":
            print("⚠️ MODEL_NAME not populated; forcing offline mock engines.")
            args.evaluation_engine = "offline-mock"
            args.test_engine = "offline-mock"
    return args


def prepare_credentials() -> ApiCredentials:
    available_lambda_keys = get_available_api_keys(allow_empty=True)
    lambda_key = available_lambda_keys[0] if available_lambda_keys else os.environ.get("API_KEY", "")
    jina_key = os.environ.get("JINA_API_KEY", "")
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
    )


def ensure_textgrad_env(args: argparse.Namespace) -> None:
    os.environ["SYNAPSE_TEXTGRAD_ENABLED"] = "1"
    os.environ["SYNAPSE_TEXTGRAD_EVAL_ENGINE"] = args.evaluation_engine
    os.environ["SYNAPSE_TEXTGRAD_TEST_ENGINE"] = args.test_engine
    os.environ["SYNAPSE_TEXTGRAD_AGGREGATE"] = args.aggregate_method
    os.environ["SYNAPSE_TEXTGRAD_BATCH_SIZE"] = str(args.batch_size)
    os.environ["SYNAPSE_TEXTGRAD_MAX_STEPS"] = str(args.max_steps)
    os.environ["SYNAPSE_TEXTGRAD_PROXIMAL"] = "0" if args.disable_proximal else "1"
    os.environ["SYNAPSE_CLIENT_COUNT"] = str(args.client_count)
    os.environ["EVAL_MODEL"] = args.test_engine
    os.environ["VLM_MODEL"] = args.test_engine
    os.environ["TEXTGRAD_SAMPLE_WITH_REPLACEMENT"] = "1" if args.sample_with_replacement else "0"


def make_client_splits(dataset, client_count: int) -> List:
    base_length = len(dataset) // client_count
    lengths = [base_length] * client_count
    for i in range(len(dataset) - base_length * client_count):
        lengths[i] += 1
    return list(random_split(dataset, lengths))


def evaluate_agent(agent: SynapseAgent, mixed_queries: Path, dataset_label: str | None = None) -> Dict[str, Any]:
    if not mixed_queries.exists():
        raise FileNotFoundError(f"Benchmark file '{mixed_queries}' not found.")
    with mixed_queries.open("r", encoding="utf-8") as fh:
        benchmark = json.load(fh)

    metrics = {
        "math": {"correct": 0, "total": 0},
        "science": {"correct": 0, "total": 0},
    }
    dataset_metrics: Dict[str, Dict[str, int]] = {}

    for item in benchmark:
        question = item.get("question") or item.get("Problem") or ""
        domain = item.get("domain") or item.get("dataset") or "math"
        dataset_name = item.get("dataset") or dataset_label or mixed_queries.name
        dataset_bucket = dataset_metrics.setdefault(dataset_name, {"correct": 0, "total": 0})
        try:
            result = agent.run(question, data_item=item)
            prediction = result.llm_response or ""
        except Exception as exc:
            print(f"[!] Agent failed to answer '{question[:50]}...': {exc}")
            continue
        gold_answer = item.get("answer") or item.get("Answer") or item.get("correct")
        if not gold_answer:
            continue
        normalized_prediction = _normalize_answer_text(_extract_final_answer(prediction or ""))
        metrics.setdefault(domain, {"correct": 0, "total": 0})
        metrics[domain]["total"] += 1
        hit = int(_answers_match_textgrad(gold_answer, normalized_prediction, dataset_name))
        metrics[domain]["correct"] += hit
        dataset_bucket["total"] += 1
        dataset_bucket["correct"] += hit

    def _to_accuracy(bucket: Dict[str, int]) -> Dict[str, float | int | None]:
        total = bucket["total"]
        correct = bucket["correct"]
        accuracy = (correct / total) if total else None
        return {"correct": correct, "total": total, "accuracy": accuracy}

    dataset_breakdown = {name: _to_accuracy(bucket) for name, bucket in dataset_metrics.items()}
    if dataset_breakdown:
        print("[Eval] Dataset accuracy breakdown:")
        for name, bucket in dataset_breakdown.items():
            acc = bucket["accuracy"]
            if acc is None:
                print(f"  · {name}: n/a (0 samples)")
            else:
                print(f"  · {name}: {bucket['correct']}/{bucket['total']} ({acc * 100:.1f}%)")

    overall_bucket = {
        "correct": sum(bucket["correct"] for bucket in metrics.values()),
        "total": sum(bucket["total"] for bucket in metrics.values()),
    }
    overall_bucket["accuracy"] = (
        overall_bucket["correct"] / overall_bucket["total"] if overall_bucket["total"] else None
    )

    return {
        "label": dataset_label or mixed_queries.name,
        "domains": {domain: _to_accuracy(bucket) for domain, bucket in metrics.items()},
        "datasets": dataset_breakdown,
        "overall": overall_bucket,
    }


def _normalize_answer_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace("Final Answer:", "").replace("Answer:", "")
    cleaned = cleaned.strip().lower().strip(".")
    return cleaned


def _extract_final_answer(response: str) -> str:
    marker = "Final Answer:"
    idx = response.rfind(marker)
    if idx == -1:
        return response
    return response[idx + len(marker):].strip()


def _is_bbh_dataset(name: Optional[str]) -> bool:
    return isinstance(name, str) and name.upper().startswith("BBH")


def _extract_numeric_value(text: str) -> Optional[int]:
    match = re.findall(r"-?\d+", text)
    if not match:
        return None
    try:
        return int(match[-1])
    except ValueError:
        return None


def _answers_match_textgrad(gold: str, prediction: str, dataset_name: str) -> bool:
    if _is_bbh_dataset(dataset_name):
        gold_val = _extract_numeric_value(gold)
        pred_val = _extract_numeric_value(prediction)
        return gold_val is not None and pred_val == gold_val
    return _normalize_answer_text(gold) == prediction


def train_clients(
    runtime: SynapseRuntime,
    settings: TextGradSettings,
    train_splits,
    eval_fn,
    validation_dataset=None,
    sample_with_replacement: bool = False,
    client_specific_datasets: Optional[Dict[str, TorchDataset]] = None,
) -> None:
    trainer = TextGradPromptTrainer(settings)
    settings.ensure_engines()
    evaluation_engine = settings.evaluation_engine
    test_engine = settings.test_engine or settings.evaluation_engine

    for idx, (client_id, client) in enumerate(runtime.clients.items()):
        artifacts = client.collect_local_artifacts()
        train_subset = None
        if client_specific_datasets:
            train_subset = client_specific_datasets.get(client_id)
        if train_subset is None:
            train_subset = train_splits[min(idx, len(train_splits) - 1)]
        total_questions = len(train_subset)
        if total_questions == 0:
            continue
        if sample_with_replacement:
            sampler = RandomSampler(train_subset, replacement=True)
            dataloader = DataLoader(train_subset, batch_size=settings.batch_size, sampler=sampler)
        else:
            dataloader = DataLoader(train_subset, batch_size=settings.batch_size, shuffle=True)

        validation_samples = None
        if validation_dataset:
            sample_count = min(len(validation_dataset), settings.batch_size)
            validation_samples = [validation_dataset[i] for i in range(sample_count)]

        for artifact in artifacts:
            if artifact.textgrad_variable is None:
                continue
            system_prompt = artifact.textgrad_variable
            model = BlackboxLLM(test_engine, system_prompt)
            optimizer = TextualGradientDescent(engine=test_engine, parameters=[system_prompt])

            def _log_progress(processed: int, total: int, *, client_id=client_id, signature=artifact.signature) -> None:
                remaining = max(total - processed, 0)
                short_sig = signature[:8]
                print(
                    f"[TextGrad][{client_id}:{short_sig}] Processed {processed}/{total} questions "
                    f"({remaining} remaining)."
                )

            results = trainer.train_batches(
                dataloader,
                model,
                optimizer,
                eval_fn,
                system_prompt,
                validation_samples=validation_samples,
                total_questions=total_questions,
                progress_callback=_log_progress,
            )
            if not results:
                continue
            last_result = results[-1]
            artifact.metadata["textgrad_last_loss"] = last_result.batch_loss
            artifact.metadata["textgrad_updated_loss"] = last_result.updated_loss
            artifact.metadata["textgrad_update_accepted"] = last_result.accepted
            artifact.metadata["textgrad_complexity"] = last_result.complexity


def main() -> None:
    load_dotenv()
    args = parse_args()
    ensure_textgrad_env(args)

    credentials = prepare_credentials()
    runtime = SynapseRuntime.build_local_runtime(Path.cwd(), credentials, client_count=args.client_count)
    client_ids = list(runtime.clients.keys())
    client_train_sets = _load_client_train_sets(args.client_train_dir, client_ids)

    settings = runtime.textgrad_settings
    settings.enabled = True
    settings.evaluation_engine_name = args.evaluation_engine
    settings.test_engine_name = args.test_engine
    settings.aggregate_method = args.aggregate_method
    settings.batch_size = args.batch_size
    settings.max_steps = args.max_steps
    settings.proximal_update = not args.disable_proximal
    settings.ensure_engines()

    train_set, val_set, _, eval_fn = load_task(args.task, evaluation_api=settings.evaluation_engine)
    train_limit = min(len(train_set), 20)
    train_subset = Subset(train_set, list(range(train_limit)))
    train_splits = make_client_splits(train_subset, args.client_count)

    for round_idx in range(args.rounds):
        print(f"[Round {round_idx + 1}] Starting client training …")
        train_clients(
            runtime,
            settings,
            train_splits,
            eval_fn,
            validation_dataset=val_set,
            sample_with_replacement=args.sample_with_replacement,
            client_specific_datasets=client_train_sets,
        )
        print(f"[Round {round_idx + 1}] Finished client training.")
        runtime.run_round()
        runtime.server.distribute_snapshot()
        print(f"[Round {round_idx + 1}] Snapshot distributed.")

    if args.output_snapshot:
        runtime.export_snapshot(args.output_snapshot)
        print(f"✅ Exported TextGrad snapshot to '{args.output_snapshot}'.")

    print("[Eval] Initialising MathQATool and ScienceQATool …")
    tool_registry = {
        "mathqa": MathQATool(),
        "scienceqa": ScienceQATool(),
    }
    print("[Eval] Tool registry ready. Beginning mixed-query evaluation …")
    agent = SynapseAgent(runtime=runtime, tool_registry=tool_registry)
    def _format_accuracy(value: float | None) -> str:
        return f"{value * 100:.1f}%" if value is not None else "n/a"

    central_metrics: Dict[str, Any] | None = None

    try:
        metrics = evaluate_agent(agent, args.mixed_queries)
        central_metrics = metrics
        domain_summary = {domain: bucket["accuracy"] for domain, bucket in metrics["domains"].items()}
        print("Federated TextGrad domain accuracies:", {k: _format_accuracy(v) for k, v in domain_summary.items()})
        if metrics["datasets"]:
            dataset_summary = {
                name: _format_accuracy(bucket["accuracy"]) for name, bucket in metrics["datasets"].items()
            }
            print("Federated TextGrad dataset accuracies:", dataset_summary)
        if metrics["overall"]["accuracy"] is not None:
            print(f"Federated TextGrad overall accuracy: {_format_accuracy(metrics['overall']['accuracy'])}")
        _append_textgrad_log(
            "central",
            {
                "mixed_queries": str(args.mixed_queries),
                "domains": metrics["domains"],
                "datasets": metrics["datasets"],
                "overall": metrics["overall"],
            },
        )
    except Exception as exc:
        print(f"⚠️ Skipping mixed-query evaluation: {exc}")

    client_metrics = []
    client_summary_payload: Optional[Dict[str, Any]] = None
    if args.evaluate_clients:
        if not args.client_data_dir:
            print("⚠️ --evaluate-clients was set but --client-data-dir is missing; skipping per-client evaluations.")
        elif not args.client_data_dir.exists():
            print(f"⚠️ Client data directory '{args.client_data_dir}' does not exist; skipping per-client evaluations.")
        else:
            dataset_paths = sorted(
                p
                for p in args.client_data_dir.glob("*.json")
                if p.is_file() and "summary" not in p.stem.lower()
            )
            if not dataset_paths:
                print(f"⚠️ No JSON datasets found in '{args.client_data_dir}'.")
            for data_path in dataset_paths:
                try:
                    result = evaluate_agent(agent, data_path, dataset_label=data_path.stem)
                    client_metrics.append(result)
                    _append_textgrad_log(
                        "client",
                        {
                            "dataset_path": str(data_path),
                            "label": result["label"],
                            "domains": result["domains"],
                            "datasets": result["datasets"],
                            "overall": result["overall"],
                        },
                    )
                except Exception as exc:
                    print(f"[!] Failed to evaluate {data_path}: {exc}")

    if client_metrics:
        print("\n--- TextGrad Client Benchmark Summary ---")

        overall_values = [entry["overall"]["accuracy"] for entry in client_metrics if entry["overall"]["accuracy"] is not None]
        math_values = [entry["domains"].get("math", {}).get("accuracy") for entry in client_metrics]
        math_values = [value for value in math_values if value is not None]
        science_values = [entry["domains"].get("science", {}).get("accuracy") for entry in client_metrics]
        science_values = [value for value in science_values if value is not None]

        macro = spread = stdev = None
        if overall_values:
            macro = sum(overall_values) / len(overall_values)
            spread = max(overall_values) - min(overall_values)
            stdev = statistics.pstdev(overall_values) if len(overall_values) > 1 else 0.0
            print(f"Macro overall accuracy: {_format_accuracy(macro)}")
            print(f"Overall accuracy spread: {_format_accuracy(spread)} (max - min)")
            print(f"Overall accuracy σ: {_format_accuracy(stdev)}")
        if math_values:
            print(f"Macro math accuracy: {_format_accuracy(sum(math_values) / len(math_values))}")
        if science_values:
            print(f"Macro science accuracy: {_format_accuracy(sum(science_values) / len(science_values))}")

        client_details = []
        for entry in client_metrics:
            overall = _format_accuracy(entry["overall"]["accuracy"])
            dataset_snippets = ", ".join(
                f"{name}={_format_accuracy(bucket['accuracy'])}" for name, bucket in entry["datasets"].items()
            ) or "no dataset metrics"
            print(f"  · {entry['label']}: overall={overall}; datasets: {dataset_snippets}")

            domains = entry.get("domains", {})
            client_details.append(
                {
                    "label": entry["label"],
                    "overall": entry["overall"].get("accuracy"),
                    "math": domains.get("math", {}).get("accuracy"),
                    "science": domains.get("science", {}).get("accuracy"),
                    "datasets": {name: bucket.get("accuracy") for name, bucket in entry["datasets"].items()},
                }
            )

        client_summary_payload = {
            "macro_overall": macro,
            "overall_spread": spread,
            "overall_stdev": stdev,
            "macro_math": (sum(math_values) / len(math_values)) if math_values else None,
            "macro_science": (sum(science_values) / len(science_values)) if science_values else None,
            "details": client_details,
        }

    summary_payload: Dict[str, Any] = {}
    if central_metrics:
        summary_payload["central"] = {
            "label": central_metrics.get("label"),
            "domains": central_metrics.get("domains"),
            "datasets": central_metrics.get("datasets"),
            "overall": central_metrics.get("overall"),
        }
    if client_summary_payload:
        summary_payload["client_summary"] = client_summary_payload
    if summary_payload:
        _append_textgrad_log("run_summary", summary_payload)


if __name__ == "__main__":
    main()
