import os
from dotenv import load_dotenv
from CompendiumManager import CompendiumManager
from populate_vector_store import populate_vectors

# --- CONFIGURATION ---
load_dotenv()

# List the paths to your individual tool compendium JSON files.
COMPENDIUM_FILES = [
    "mathqa_tools_compendium.json",
    "scienceqa_tools_compendium.json" 
]

def main():
    """
    Runs the full database setup process:
    1. Merges local compendium files and saves the master compendium to MongoDB.
    2. Immediately uses the in-memory master compendium to populate the vector store.
    """
    print("--- Starting Full Database Setup Process ---")
    
    # Check for necessary environment variables
    mongo_uri = os.environ.get("MONGO_URI")
    lambda_api_key = os.environ.get("LAMDA_API_KEY")
    jina_api_key = os.environ.get("JINA_API_KEY")

    if not all([mongo_uri, lambda_api_key, jina_api_key]):
        print("❌ Error: MONGO_URI, LAMDA_API_KEY, and JINA_API_KEY must be set in your .env file.")
        return

    # --- Step 1: Build and Save the Master Compendium ---
    print("\n--- Building Master Compendium ---")
    manager = CompendiumManager()
    print(f"Merging the following files: {COMPENDIUM_FILES}")
    
    # Capture the returned compendium data in a variable
    merged_compendium = manager.merge_compendiums(file_paths=COMPENDIUM_FILES, compendium_id="master_compendium")
    
    # --- Step 2: Populate the Vector Store ---
    if merged_compendium:
        print("\n--- Populating Vector Store ---")
        # Pass the in-memory data directly to the populate function
        populate_vectors(merged_compendium)
    else:
        print("❌ Halting process: Master compendium could not be built.")
    
    print("\n--- Full Database Setup Process Complete ---")
    print("Your MongoDB database is now ready for the agent.")

if __name__ == "__main__":
    main()
