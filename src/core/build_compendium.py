# build_compendium.py

import os
from dotenv import load_dotenv
from CompendiumManager import CompendiumManager
from populate_vector_store import populate_vectors
from pymongo import MongoClient
from openrouter_client import get_available_api_keys
from jina_key_manager import get_available_jina_api_keys

# --- CONFIGURATION ---
load_dotenv()

COMPENDIUM_FILES = [
    "mathqa_tools_compendium.json",
    "scienceqa_tools_compendium.json",
    "mmlu_tools_compendium.json",
    "truthfulqa_tools_compendium.json"
]
DB_NAME = "FredRag"

def main():
    """
    Runs the full database setup process.
    1. Merges local compendium files into 'master_compendium'.
    2. Populates the 'vectors' collection for searching.
    3. Ensures the 'logs' collection exists for the application logger.
    """
    print("--- Starting Full Database Setup Process ---")

    # Check for necessary environment variables
    mongo_uri = os.environ.get("MONGO_URI")
    available_keys = get_available_api_keys(allow_empty=True)
    lambda_api_key = available_keys[0] if available_keys else None
    if not lambda_api_key:
        lambda_api_key = os.environ.get("LAMBDA_API_KEY") or os.environ.get("LAMDA_API_KEY")
        if lambda_api_key:
            print("⚠️ Detected deprecated env var for Lambda API key. Please rename it to API_KEY or API_KEY_<n>.")
    jina_keys = get_available_jina_api_keys(allow_empty=True)
    jina_api_key = jina_keys[0] if jina_keys else None

    if not all([mongo_uri, lambda_api_key, jina_api_key]):
        print("❌ Error: MONGO_URI, an API_KEY, and at least one JINA_API_KEY must be set in your .env file.")
        return

    # Keep backwards compatibility for libraries expecting the primary env var.
    if lambda_api_key:
        os.environ["API_KEY"] = lambda_api_key
    if jina_api_key:
        os.environ["JINA_API_KEY"] = jina_api_key

    # --- Step 1: Build and Save the Master Compendium ---
    print("\n--- Building Master Compendium ---")
    manager = CompendiumManager()
    print(f"Merging the following files: {COMPENDIUM_FILES}")
    merged_compendium_data = manager.merge_compendiums(file_paths=COMPENDIUM_FILES)

    # --- Step 2: Populate the Vector Store from In-Memory Data ---
    if merged_compendium_data:
        print("\n--- Populating Vector Store ---")
        populate_vectors(merged_compendium_data)
    else:
        print("❌ Halting process: Master compendium could not be built.")
        
    # --- Step 3: Ensure 'logs' collection exists ---
    print("\n--- Ensuring 'logs' collection exists ---")
    try:
        client = MongoClient(mongo_uri)
        db = client[DB_NAME]
        if "logs" not in db.list_collection_names():
            db.create_collection("logs")
            print("✅ 'logs' collection successfully created.")
        else:
            print("✅ 'logs' collection already exists.")
        client.close()
    except Exception as e:
        print(f"❌ Failed to create 'logs' collection: {e}")

    print("\n--- Full Database Setup Complete ---")
    print("Your MongoDB database is now ready for the agent.")

if __name__ == "__main__":
    main()
