import json
import os
from typing import List, Dict
from CompendiumBuilder import CompendiumBuilder
from pymongo import MongoClient
from openrouter_client import chat_completion, get_openrouter_client

# --- CONFIG ---
MODEL = os.environ.get("RERANK_MODEL", "llama3.1-8b-instruct")
get_openrouter_client()

# --- DATABASE CONFIG ---
MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME = "FredRag"
COMPENDIUM_COLLECTION = "master_compendium"

class CompendiumManager:
    def __init__(self):
        self.compendium_builder = CompendiumBuilder()

    def merge_compendiums(self, file_paths: List[str]) -> Dict:
        """
        Merges tool compendiums using an LLM and then saves the result
        to a dedicated collection in MongoDB.
        """
        all_scenarios = []
        base_structure = None

        for path in file_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    comp_data = json.load(f)
                    if not base_structure:
                        base_structure = comp_data
                    all_scenarios.extend(comp_data.get("Textual_Compendium", {}).get("Usage_Scenarios", []))

        if not base_structure:
            print("❌ No compendiums found to merge.")
            return None

        unique_scenarios = list({s['scenario']: s for s in all_scenarios}.values())
        prompt = f"""
        You are an AI knowledge base architect. Merge and deduplicate the following tool usage scenarios into a single, cohesive list.
        Your response MUST be a valid JSON object with a single key "merged_scenarios".

        Scenarios: {json.dumps(unique_scenarios, indent=2)}
        """
        
        try:
            completion = chat_completion(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            response_json = completion.choices[0].message.content
            merged_scenarios_data = json.loads(response_json)
            merged_scenarios = merged_scenarios_data.get("merged_scenarios", [])

            base_structure["Textual_Compendium"]["Usage_Scenarios"] = merged_scenarios
            print("✅ Compendiums successfully merged by LLM.")

            # --- Save the merged result to MongoDB ---
            try:
                print(f"Saving master compendium to MongoDB collection: '{COMPENDIUM_COLLECTION}'...")
                client = MongoClient(MONGO_URI)
                db = client[DB_NAME]
                collection = db[COMPENDIUM_COLLECTION]
                
                # Use replace_one with upsert=True to ensure there is only ever one master document.
                collection.replace_one({"_id": "master_compendium_document"}, base_structure, upsert=True)
                print("✅ Master compendium successfully saved to MongoDB.")
                client.close()
            except Exception as e:
                print(f"❌ Failed to save master compendium to MongoDB: {e}")

            return base_structure

        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"❌ LLM-driven merge failed: {e}")
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
        try:
            completion = chat_completion(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8192,
                response_format={"type": "json_object"},
            )
            updated_compendium_str = completion.choices[0].message.content
            
            # Validate and save the new compendium
            updated_compendium = json.loads(updated_compendium_str)
            with open(compendium_path, 'w') as f:
                json.dump(updated_compendium, f, indent=4)
            print(f"✅ Compendium successfully updated by LLM.")
            return updated_compendium
        except Exception as e:
            print(f"❌ LLM-driven update failed: {e}")
            return None
