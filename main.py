import json
import os
import re
import logging
import sys
from dotenv import load_dotenv

# Import the new manager and all necessary classes
from CompendiumManager import CompendiumManager
from CompendiumAwareAgent import CompendiumAwareAgent
from math_qa import MathQATool
from science_qa import ScienceQATool
from CompendiumBuilder import CompendiumEntry

# Load environment variables from a .env file at the start
load_dotenv()

# --- 1. LOGGING SETUP ---
# This class will redirect all print() statements to the logger
class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message != '\n': # avoid logging empty lines
            self.level(message.strip())

    def flush(self):
        pass # Required for file-like object

# Configure the logger to write to a file and the console
log_formatter = logging.Formatter('%(message)s') # Keep the log clean
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler("evaluation_log.txt", mode='w') # 'w' to overwrite file each run
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# Redirect stdout and stderr to the logger
sys.stdout = LoggerWriter(logger.info)
sys.stderr = LoggerWriter(logger.error)
# --- END LOGGING SETUP ---

def evaluate_mixed_queries(agent, test_file="mixed_queries.json"):
    """
    Loads a mixed dataset and runs a full evaluation of the agent's
    multi-tool coordination and problem-solving capabilities.
    """
    print("\n--- 3. EVALUATING AGENT ON MIXED DATASET ---")
    try:
        with open(test_file, "r", encoding='utf-8') as f:
            test_data = json.load(f)
        print(f"[✓] Successfully loaded '{test_file}'.")
    except Exception as e:
        print(f"❌ Error loading test file: {e}")
        return

    # (Optional) Add scoring logic here if needed, similar to previous versions.
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

        # --- Execute the agent's new, advanced routing logic ---
        result = agent.route_query(query=query_text, data_item=data_payload)
        
        if result and result.llm_response:
            print("\n" + "="*25 + " AGENT REASONING & FINAL OUTPUT " + "="*25)
            print(result.llm_response)
            print("="*80)
            print("[✓] Query processed successfully.")
        else:
            print("[✗] Agent failed to produce a final result for this query.")
            # Here, you could trigger the self-improvement mechanism
            # print("[!] Triggering self-improvement...")
            # agent.compendium_manager.self_improve_compendium(query_text)
            
    print("\n--- AGENT EVALUATION COMPLETE ---")
    print("Full output has been saved to evaluation_log.txt")


def main():
    """
    Main function to orchestrate the entire pipeline:
    1. Manage Compendiums (Merge, Deduplicate).
    2. Initialize the Agent with the final knowledge base.
    3. Run the Evaluation.
    """
    if not os.environ.get("LAMDA_API_KEY") or not os.environ.get("JINA_API_KEY"):
        print("❌ Error: API keys must be set in your .env file.")
        return
    
    # --- PHASE 1: COMPENDIUM MANAGEMENT ---
    print("\n--- 1. MANAGING KNOWLEDGE COMPENDIUMS ---")
    compendium_manager = CompendiumManager()
    
    # Merge existing compendiums into a single master file.
    # The LLM will handle deduplication and merging of scenarios.
    source_files = ["mathqa_tools_compendium.json", "scienceqa_tools_compendium.json"]
    final_compendium_path = "final_compendium.json"
    final_compendium = compendium_manager.merge_compendiums(source_files, final_compendium_path)

    if not final_compendium:
        print("❌ Critical error: Could not create the final compendium. Aborting.")
        return
        
    # --- PHASE 2: AGENT INITIALIZATION & EVALUATION ---
    print("\n--- 2. INITIALIZING COMPENDIUM-AWARE AGENT ---")
    tool_map = {"mathqa": MathQATool(), "scienceqa": ScienceQATool()}
    
    # The agent is now initialized with the single, master compendium path.
    # It will load this file and build its own vector store internally.
    agent = CompendiumAwareAgent(tools=tool_map, final_compendium_path=final_compendium_path)

    # The rest of the evaluation proceeds using the new agent and its advanced capabilities.
    evaluate_mixed_queries(agent, test_file="mixed_queries.json")

if __name__ == "__main__":
    main()
