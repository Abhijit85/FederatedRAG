import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from math_qa import MathQATool
from science_qa import ScienceQATool
from synapse.agent import SynapseAgent
from synapse.config import ApiCredentials
from synapse.runtime import SynapseRuntime

# Load environment variables from a .env file at the start
load_dotenv()

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


def evaluate_mixed_queries(agent: SynapseAgent, test_file: str = "mixed_queries.json"):
    """
    Evaluate the SYNAPSE agent on the mixed benchmark file.
    """
    print("\n--- 3. EVALUATING SYNAPSE AGENT ON MIXED DATASET ---")
    try:
        with open(test_file, "r", encoding='utf-8') as f:
            test_data = json.load(f)
        print(f"[✓] Successfully loaded '{test_file}'.")
    except Exception as e:
        print(f"❌ Error loading test file: {e}")
        return

    for item in test_data:
        is_science_query = 'question' in item
        if is_science_query:
            query_text = item['question']
            data_payload = item
            print(f"\n--- Processing ScienceQA Query: '{query_text[:80]}...' ---")
        else:
            query_text = f"{item.get('Problem', '')}\nOptions: {item.get('options', '')}"
            data_payload = None
            print(f"\n--- Processing MathQA Query: '{query_text[:80]}...' ---")

        try:
            result = agent.run(query=query_text, data_item=data_payload)
        except Exception as exc:
            print(f"[✗] Agent execution failed: {exc}")
            continue

        if result and getattr(result, "llm_response", None):
            print("\n" + "="*25 + " AGENT REASONING & FINAL OUTPUT " + "="*25)
            print(result.llm_response)
            print("="*80)
            print("[✓] Query processed successfully.")
        else:
            print("[✗] Agent failed to produce a final result for this query.")

    print("\n--- AGENT EVALUATION COMPLETE ---")
    print("Full output has been saved to evaluation_log.txt")


def main():
    """
    Run a full SYNAPSE federation round and evaluate the resulting agent.
    """
    lambda_key = os.environ.get("LAMDA_API_KEY", "")
    jina_key = os.environ.get("JINA_API_KEY", "")
    mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017")

    if not lambda_key or not jina_key:
        print("⚠️ Warning: API keys are missing; downstream tool calls may fail.")

    credentials = ApiCredentials(
        lambda_api_key=lambda_key,
        jina_api_key=jina_key,
        mongo_uri=mongo_uri,
    )

    runtime = SynapseRuntime.build_local_runtime(Path.cwd(), credentials)
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

    evaluate_mixed_queries(agent, test_file="mixed_queries.json")


if __name__ == "__main__":
    main()
