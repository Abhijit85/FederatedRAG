import json
import os
import re
import logging
import sys
from dotenv import load_dotenv

from math_qa import MathQATool
from science_qa import ScienceQATool
from CompendiumBuilder import CompendiumEntry
from CompendiumAwareAgent import CompendiumAwareAgent

load_dotenv()

# --- LOGGING SETUP ---
class LoggerWriter:
    def __init__(self, level):
        self.level = level
    def write(self, message):
        if message != '\n':
            self.level(message.strip())
    def flush(self):
        pass

log_formatter = logging.Formatter('%(message)s') # Keep the log clean
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

def preprocess_compendium_dict(comp_dict):
    tc = comp_dict.get("Textual_Compendium", {})
    scenarios = tc.get("Usage_Scenarios", [])
    tc["Usage_Scenarios"] = [s["scenario"] + ": " + s["context"] if isinstance(s, dict) else str(s) for s in scenarios]
    precautions = tc.get("Precautions", [])
    tc["Precautions"] = [p["precaution"] + ": " + p["details"] if isinstance(p, dict) else str(p) for p in precautions]
    mtc = tc.get("Multi_tool_Coordination", {})
    examples = mtc.get("Examples", [])
    mtc["Examples"] = [e["example"] if isinstance(e, dict) else str(e) for e in examples]
    tc["Multi_tool_Coordination"] = mtc
    if "Solving_Protocol" not in tc:
        tc["Solving_Protocol"] = []
    comp_dict["Textual_Compendium"] = tc
    return comp_dict

def evaluate_mixed_queries(agent, test_file="mixed_queries.json"):
    print("\n--- 4. EVALUATING AGENT ON MIXED DATASET ---")
    try:
        with open(test_file, "r", encoding='utf-8') as f:
            test_data = json.load(f)
        print(f"[✓] Successfully loaded '{test_file}'.")
    except Exception as e:
        print(f"❌ Error loading test file: {e}")
        return

    total_queries = 0
    correct_tool_selections = 0
    correct_final_answers = 0

    for item in test_data:
        total_queries += 1
        is_science_query = 'question' in item
        
        if is_science_query:
            query_text = item['question']
            data_payload = item
            print(f"\n--- Processing ScienceQA Query: '{query_text[:80]}...' ---")
        else:
            query_text = f"{item.get('Problem', '')}\nOptions: {item.get('options', '')}"
            data_payload = None
            print(f"\n--- Processing MathQA Query: '{query_text[:80]}...' ---")

        # Get the agent's routing decision
        retrieved_docs = agent.vector_search_filter.search(query_text, top_k=1)
        if not retrieved_docs:
            print("[!] Agent could not select a tool.")
            continue
        
        chosen_tool_name = retrieved_docs[0]['tool_name']
        ground_truth_tool = item.get("correct_tool")
        
        print(f"  Tool Selected: '{chosen_tool_name}'")

        # Get the final answer from the agent
        result = agent.route_query(query=query_text, data_item=data_payload)
        
        # --- NEW DISPLAY AND SCORING LOGIC ---
        if result:
            # Print the full reasoning block
            print("\n" + "="*25 + " AGENT REASONING " + "="*25)
            print(result.llm_response)
            print("="*70 + "\n")

            if result.parsed_output:
                if is_science_query:
                    predicted = result.parsed_output.get('answer_index', -1)
                    ground_truth = item.get('answer')
                    print("Ground Truth: {ground_truth}")
                    
                else: # MathQA
                    predicted = result.parsed_output.get('final_answer')
                    ground_truth = item.get('correct')
                    print(f" Ground Truth: '{ground_truth}'")
                    
            else:
                print("  [!] Could not parse final answer from reasoning.")
        else:
            print("  [!] Agent failed to produce a result.")

    # --- FINAL REPORT ---
    tool_accuracy = (correct_tool_selections / total_queries * 100) if total_queries > 0 else 0
    answer_accuracy = (correct_final_answers / total_queries * 100) if total_queries > 0 else 0

    print("\n" + "="*50)
    print("--- AGENT EVALUATION COMPLETE ---")
    print("="*50)
    print(f"Total Queries Processed: {total_queries}")
    print("-" * 25)
    print("="*50)
    print("Full output has been saved to evaluation_log.txt")

def main():
    if not os.environ.get("LAMDA_API_KEY") or not os.environ.get("JINA_API_KEY"):
        print("❌ Error: API keys must be set in your .env file.")
        return
    
    print("--- 1. LOADING KNOWLEDGE COMPENDIUMS ---")
    tool_map = {"mathqa": MathQATool(), "scienceqa": ScienceQATool()}
    compendium_map = {}

    for name in ["mathqa", "scienceqa"]:
        filename = f"{name}_tools_compendium.json"
        if os.path.exists(filename):
            with open(filename, "r") as f:
                try:
                    comp_dict = preprocess_compendium_dict(json.load(f))
                    compendium_map[name] = CompendiumEntry(**comp_dict)
                    print(f"[✓] Loaded '{filename}'.")
                except Exception as e:
                    print(f"[!] Error validating {filename}: {e}")
        else:
            print(f"[!] '{filename}' not found.")

    if not compendium_map:
        print("❌ Error: No compendiums were loaded. Cannot initialize agent.")
        return

    print("\n--- 2. INITIALIZING COMPENDIUM-AWARE AGENT ---")
    agent = CompendiumAwareAgent(tools=tool_map, compendium_map=compendium_map)
    print("[✓] Agent initialized successfully.")

    evaluate_mixed_queries(agent, test_file="mixed_queries.json")

if __name__ == "__main__":
    main()