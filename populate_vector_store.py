import os
import json
import requests
from dotenv import load_dotenv
from mongo_utils import MongoVectorStore

# --- CONFIGURATION ---
load_dotenv()
JINA_API_KEY = os.environ.get("JINA_API_KEY")
MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME = "FredRag"
VECTOR_COLLECTION_NAME = "vectors"
JINA_EMBED_API_URL = "https://api.jina.ai/v1/embeddings"

class JinaAIClient:
    """A client to interact with Jina AI APIs for embeddings."""
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("JINA_API_KEY not found in environment.")
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def get_embeddings(self, texts):
        """Generates embeddings for a list of texts using Jina AI."""
        try:
            response = requests.post(
                JINA_EMBED_API_URL, headers=self.headers,
                json={"model": "jina-embeddings-v2-base-en", "input": texts}
            )
            response.raise_for_status()
            return [item['embedding'] for item in response.json()['data']]
        except requests.exceptions.RequestException as e:
            print(f"Error getting embeddings from Jina AI: {e}")
            return []

def populate_vectors(compendium_data: dict):
    """
    Generates embeddings for each scenario from the provided compendium data
    and stores them in the MongoDB vector store.
    """
    print("--- Starting Vector Store Population ---")

    if not compendium_data:
        print("[!] Error: No compendium data provided to populate vectors.")
        return

    # 1. Initialize clients
    jina_client = JinaAIClient(JINA_API_KEY)
    vector_store = MongoVectorStore(MONGO_URI, DB_NAME, VECTOR_COLLECTION_NAME)
    
    # 2. Extract scenario texts from the in-memory data
    scenarios = compendium_data.get("Textual_Compendium", {}).get("Usage_Scenarios", [])
    if not scenarios:
        print("[!] No usage scenarios found in the compendium. Nothing to populate.")
        return

    texts_to_embed = [f"Tool Scenario: {s['scenario']}. Context: {s['context']}" for s in scenarios]
    print(f"Found {len(texts_to_embed)} scenarios to embed.")

    # 3. Generate embeddings
    print("Generating embeddings with Jina AI...")
    vectors = jina_client.get_embeddings(texts_to_embed)

    if not vectors or len(vectors) != len(texts_to_embed):
        print("[!] Failed to generate embeddings. Aborting population.")
        return

    # 4. Add vectors to MongoDB
    print("Adding vectors to MongoDB...")
    try:
        vector_store.collection.delete_many({})
        print("Cleared existing vectors from the collection.")
        vector_store.add_vectors(vectors, texts_to_embed)
        print(f"[✓] Successfully added {len(vectors)} vectors to the '{VECTOR_COLLECTION_NAME}' collection.")
    except Exception as e:
        print(f"[!] An error occurred while adding vectors to MongoDB: {e}")

    print("--- Vector Store Population Complete ---")

if __name__ == "__main__":
    print("This script is intended to be called from build_compendium.py")
    pass
