import os
import json
import requests
from dotenv import load_dotenv
from mongo_utils import MongoVectorStore
import time

# --- CONFIGURATION ---
load_dotenv()
JINA_API_KEY = os.environ.get("JINA_API_KEY")
MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME = "FredRag"
VECTOR_COLLECTION_NAME = "vectors"
JINA_EMBED_API_URL = "https://api.jina.ai/v1/embeddings"

class JinaAIClient:
    """A client to interact with Jina AI APIs for embeddings."""
    # Add this validation to your JinaAIClient initialization
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("JINA_API_KEY not found in environment.")
        if not api_key.startswith('jina_'):
            print("⚠️ Warning: API key format doesn't look like a valid Jina AI key")
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def get_embeddings(self, texts):
        """
        Generates embeddings for a list of texts using Jina AI.
        The correct payload structure expects a simple list of strings.
        """
        payload = {
            "model": "jina-embeddings-v2-base-en",
            "input": texts  # Just the list of strings, not objects
        }

        retries = 3
        backoff_factor = 0.5

        for i in range(retries):
            try:
                response = requests.post(JINA_EMBED_API_URL, headers=self.headers, json=payload)
                response.raise_for_status()
                return [item['embedding'] for item in response.json()['data']]
            except requests.exceptions.RequestException as e:
                print(f"Error getting embeddings from Jina AI: {e}")
                if i < retries - 1:
                    sleep_time = backoff_factor * (2 ** i)
                    print(f"Retrying in {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                else:
                    print("All retries failed.")
                    return []


def populate_vectors(compendium_data: dict):
    """
    Populates the MongoDB 'vectors' collection with embeddings of tool scenarios.
    """
    print("--- Starting Vector Store Population ---")

    vector_store = MongoVectorStore(MONGO_URI, DB_NAME, VECTOR_COLLECTION_NAME)
    
    try:
        jina_client = JinaAIClient(JINA_API_KEY)
        
        scenarios = compendium_data.get("Textual_Compendium", {}).get("Usage_Scenarios", [])
        if not scenarios:
            print("[!] No usage scenarios found. Nothing to populate.")
            return

        texts_to_embed = []
        for s in scenarios:
            if 'scenario' in s:
                texts_to_embed.append(f"Tool Scenario: {s['scenario']}. Context: {s.get('context', '')}")
            else:
                print(f"⚠️ Warning: Skipping a scenario because it is missing the 'scenario' key. Data: {s}")

        if not texts_to_embed:
            print("[!] No valid scenarios found to embed. Aborting population.")
            return

        print(f"Found {len(texts_to_embed)} scenarios to embed.")

        print("Generating embeddings with Jina AI...")
        vectors = jina_client.get_embeddings(texts_to_embed)

        if not vectors:
            print("[!] Failed to generate embeddings. Aborting population.")
            return
            
        if len(vectors) != len(texts_to_embed):
            print(f"[!] Mismatch: requested {len(texts_to_embed)} embeddings, got {len(vectors)}")
            # Continue with what we have, but log the issue
            vectors = vectors[:len(texts_to_embed)]  # Truncate if needed

        print("Adding vectors to MongoDB...")
        vector_store.collection.delete_many({})
        print("Cleared existing vectors from the collection.")
        vector_store.add_vectors(vectors, texts_to_embed)
        print(f"[✓] Successfully added {len(vectors)} vectors to the '{VECTOR_COLLECTION_NAME}' collection.")
    
    except Exception as e:
        print(f"[!] An error occurred during vector population: {e}")
    
    finally:
        vector_store.close()
        print("MongoDB connection for population has been closed.")

    print("--- Vector Store Population Complete ---")