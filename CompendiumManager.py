# CompendiumManager.py
import json
import os
import requests
from typing import List, Dict
from CompendiumBuilder import CompendiumBuilder, CompendiumEntry
from vector_search import VectorSearchFilter

# --- CONFIG ---
API_KEY = os.environ.get("LAMDA_API_KEY")
MODEL = "llama3.1-8b-instruct"
LAMBDA_API = "https://api.lambda.ai/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

class CompendiumManager:
    def __init__(self):
        self.compendium_builder = CompendiumBuilder()
        self.vector_search = VectorSearchFilter()

    def merge_compendiums(self, file_paths: List[str], output_path: str = "final_compendium.json"):
        """
        Merges multiple compendium files into a single master compendium using an LLM to
        deduplicate and combine overlapping usage scenarios.
        """
        all_scenarios = []
        base_structure = None

        for path in file_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    comp_data = json.load(f)
                    if not base_structure:
                        base_structure = comp_data # Use the first file as a template
                    
                    scenarios = comp_data.get("Textual_Compendium", {}).get("Usage_Scenarios", [])
                    # Use a set to keep track of unique scenarios to avoid duplicates from the start
                    all_scenarios.extend(scenarios)

        if not base_structure:
            print("❌ No compendiums found to merge.")
            return None

        # Create a list of unique scenarios based on the 'scenario' key
        unique_scenarios = list({s['scenario']: s for s in all_scenarios}.values())

        # FIX: Refined prompt to be more explicit about the expected JSON output
        prompt = f"""
        You are an AI knowledge base architect. Your task is to merge and deduplicate the following list of tool usage scenarios into a single, cohesive list.
        - Combine scenarios that are semantically similar.
        - Preserve unique and distinct scenarios.
        - Your response MUST be a valid JSON object with a single key "merged_scenarios", containing the final array of scenario objects.

        **Scenarios to Merge:**
        {json.dumps(unique_scenarios, indent=2)}

        Respond ONLY with the valid JSON object.
        """
        
        # FIX: Enforce a JSON object response from the API for better reliability.
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
            "response_format": {"type": "json_object"} 
        }

        try:
            res = requests.post(LAMBDA_API, json=payload, headers=HEADERS)
            res.raise_for_status()
            
            response_data = json.loads(res.json()['choices'][0]['message']['content'])
            merged_scenarios_list = response_data["merged_scenarios"]

            base_structure["Textual_Compendium"]["Usage_Scenarios"] = merged_scenarios_list

            with open(output_path, 'w') as f:
                json.dump(base_structure, f, indent=4)
            print(f"✅ Successfully merged {len(file_paths)} compendiums into '{output_path}'.")
            return base_structure
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            print(f"❌ Failed to merge compendiums: {e}")
            return None

    def self_improve_compendium(self, failed_query: str, compendium_path: str = "final_compendium.json"):
        """
        When no tool is adequate, this function uses an LLM to add new knowledge
        (scenarios, precautions, etc.) to the master compendium.
        """
        print(f"\n--- 🧠 Triggering Compendium Self-Improvement for query: '{failed_query[:50]}...' ---")
        
        with open(compendium_path, 'r') as f:
            compendium_data = json.load(f)

        prompt = f"""
        You are an AI system architect with full control to edit a tool compendium.
        The system failed to find an adequate tool for the following user query, indicating a knowledge gap.
        
        **Failed User Query:**
        "{failed_query}"

        **Existing Compendium (JSON):**
        {json.dumps(compendium_data, indent=2)}

        Your task is to improve the compendium. You can:
        1.  Add a new, specific `Usage_Scenario` to an existing tool.
        2.  Refine an existing `Usage_Scenario` to be more descriptive.
        3.  Add a new `Precaution` to prevent future misclassifications.
        
        Respond with ONLY the updated, complete JSON of the compendium.
        """
        
        # Enforce JSON object response for reliability
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8192,
            "response_format": {"type": "json_object"}
        }

        try:
            res = requests.post(LAMBDA_API, json=payload, headers=HEADERS)
            res.raise_for_status()
            updated_compendium_str = res.json()['choices'][0]['message']['content']
            
            # Validate and save the new compendium
            updated_compendium = json.loads(updated_compendium_str)
            with open(compendium_path, 'w') as f:
                json.dump(updated_compendium, f, indent=4)
            print(f"✅ Compendium successfully updated by LLM.")
            return updated_compendium
        except Exception as e:
            print(f"❌ LLM-driven update failed: {e}")
            return None
