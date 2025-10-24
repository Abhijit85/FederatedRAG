import json
import os
import time
from typing import List, Sequence

import requests
from dotenv import load_dotenv

from jina_key_manager import JinaAPIKeyRotator, get_named_jina_api_keys
from mongo_utils import MongoVectorStore

# --- CONFIGURATION ---
load_dotenv()
MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME = "FredRag"
VECTOR_COLLECTION_NAME = "vectors"
JINA_EMBED_API_URL = "https://api.jina.ai/v1/embeddings"

class JinaAIClient:
    """A client to interact with Jina AI APIs for embeddings."""
    def __init__(self, api_keys: Sequence[str] | Sequence[tuple[str, str]] | None = None):
        self._rotator = JinaAPIKeyRotator(api_keys)

    @staticmethod
    def _build_headers(api_key: str) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def _post_embeddings(self, texts: List[str]) -> requests.Response:
        """
        Generates embeddings for a list of texts using Jina AI.
        The correct payload structure expects a simple list of strings.
        """
        payload = {
            "model": "jina-embeddings-v2-base-en",
            "input": texts  # Just the list of strings, not objects
        }
        return self._rotator.execute(
            lambda api_key: requests.post(
                JINA_EMBED_API_URL,
                headers=self._build_headers(api_key),
                json=payload,
            )
        )

    def get_embeddings(self, texts: List[str]):
        response = self._post_embeddings(texts)
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]

    def get_embeddings_with_retry(self, texts: List[str], *, retries: int = 3, backoff_factor: float = 0.5):
        for i in range(retries):
            try:
                response = self._post_embeddings(texts)
                response.raise_for_status()
                return [item["embedding"] for item in response.json()["data"]]
            except (requests.exceptions.RequestException, RuntimeError) as e:
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
        jina_client = JinaAIClient(get_named_jina_api_keys())
        
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
        vectors = jina_client.get_embeddings_with_retry(texts_to_embed)

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
