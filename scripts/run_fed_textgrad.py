#!/usr/bin/env python3
"""Run a SYNAPSE federation round with TextGrad optimisation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Dict, List

from dotenv import load_dotenv
from torch.utils.data import random_split

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
    test_engine = os.environ.get("TEXTGRAD_TEST_ENGINE")
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


def _resolve_runtime_defaults() -> tuple[bool, str, str, int]:
    """
    Decide whether we should default to online models or offline mocks.
    Online defaults are only enabled when both TEXTGRAD_EVAL_ENGINE and
    SYNAPSE_CLIENT_COUNT are populated in the environment.
    """
    env_eval = os.environ.get("TEXTGRAD_EVAL_ENGINE", "").strip()
    env_client = os.environ.get("SYNAPSE_CLIENT_COUNT", "").strip()
    env_test = os.environ.get("TEXTGRAD_TEST_ENGINE", "").strip()
    online_ready = bool(env_eval and env_client)

    if online_ready:
        eval_default = env_eval
        test_default = env_test if env_test else "gpt-3.5-turbo-0125"
        try:
            client_default = int(env_client)
        except ValueError:
            client_default = 4
    else:
        eval_default = "offline-mock"
        test_default = "offline-mock"
        client_default = 2

    return online_ready, eval_default, test_default, client_default


def parse_args() -> argparse.Namespace:
    online_ready, eval_default, test_default, client_default = _resolve_runtime_defaults()

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

    args = parser.parse_args()
    setattr(args, "online_ready", online_ready)

    if not online_ready:
        if args.evaluation_engine != "offline-mock" or args.test_engine != "offline-mock":
            print("⚠️ TEXTGRAD_EVAL_ENGINE/SYNAPSE_CLIENT_COUNT not populated; forcing offline mock engines.")
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


def make_client_splits(dataset, client_count: int) -> List:
    base_length = len(dataset) // client_count
    lengths = [base_length] * client_count
    for i in range(len(dataset) - base_length * client_count):
        lengths[i] += 1
    return list(random_split(dataset, lengths))


def evaluate_agent(agent: SynapseAgent, mixed_queries: Path) -> Dict[str, float]:
    if not mixed_queries.exists():
        raise FileNotFoundError(f"Benchmark file '{mixed_queries}' not found.")
    with mixed_queries.open("r", encoding="utf-8") as fh:
        benchmark = json.load(fh)

    metrics = {
        "math": {"correct": 0, "total": 0},
        "science": {"correct": 0, "total": 0},
    }

    for item in benchmark:
        question = item.get("question") or item.get("Problem") or ""
        domain = item.get("domain") or item.get("dataset") or "math"
        try:
            result = agent.run(question, data_item=item)
            prediction = result.llm_response or ""
        except Exception as exc:
            print(f"[!] Agent failed to answer '{question[:50]}...': {exc}")
            continue
        final_answer = item.get("answer") or item.get("Answer") or ""
        if not final_answer:
            continue
        metrics.setdefault(domain, {"correct": 0, "total": 0})
        metrics[domain]["total"] += 1
        metrics[domain]["correct"] += int(str(final_answer).strip().lower() in prediction.lower())

    return {
        domain: (bucket["correct"] / bucket["total"]) if bucket["total"] else 0.0
        for domain, bucket in metrics.items()
    }


def train_clients(
    runtime: SynapseRuntime,
    settings: TextGradSettings,
    train_splits,
    eval_fn,
) -> None:
    trainer = TextGradPromptTrainer(settings)
    settings.ensure_engines()
    evaluation_engine = settings.evaluation_engine
    test_engine = settings.test_engine or settings.evaluation_engine

    for idx, (client_id, client) in enumerate(runtime.clients.items()):
        artifacts = client.collect_local_artifacts()
        train_subset = train_splits[min(idx, len(train_splits) - 1)]
        if len(train_subset) == 0:
            continue
        dataloader = DataLoader(train_subset, batch_size=settings.batch_size, shuffle=True)

        for artifact in artifacts:
            if artifact.textgrad_variable is None:
                continue
            system_prompt = artifact.textgrad_variable
            model = BlackboxLLM(test_engine, system_prompt)
            optimizer = TextualGradientDescent(engine=evaluation_engine, parameters=[system_prompt])

            results = trainer.train_batches(dataloader, model, optimizer, eval_fn, system_prompt)
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

    settings = runtime.textgrad_settings
    settings.enabled = True
    settings.evaluation_engine_name = args.evaluation_engine
    settings.test_engine_name = args.test_engine
    settings.aggregate_method = args.aggregate_method
    settings.batch_size = args.batch_size
    settings.max_steps = args.max_steps
    settings.proximal_update = not args.disable_proximal
    settings.ensure_engines()

    train_set, _, _, eval_fn = load_task(args.task, evaluation_api=settings.evaluation_engine)
    train_splits = make_client_splits(train_set, args.client_count)

    for round_idx in range(args.rounds):
        train_clients(runtime, settings, train_splits, eval_fn)
        runtime.run_round()
        runtime.server.distribute_snapshot()
        print(f"[Round {round_idx + 1}] Completed federation and snapshot export.")

    if args.output_snapshot:
        runtime.export_snapshot(args.output_snapshot)
        print(f"✅ Exported TextGrad snapshot to '{args.output_snapshot}'.")

    tool_registry = {
        "mathqa": MathQATool(),
        "scienceqa": ScienceQATool(),
    }
    agent = SynapseAgent(runtime=runtime, tool_registry=tool_registry)
    try:
        metrics = evaluate_agent(agent, args.mixed_queries)
        print("Federated TextGrad evaluation metrics:", metrics)
    except Exception as exc:
        print(f"⚠️ Skipping mixed-query evaluation: {exc}")


if __name__ == "__main__":
    main()
