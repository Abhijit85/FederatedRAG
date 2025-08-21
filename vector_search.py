# vector_search.py

import requests
import os
import numpy as np
from CompendiumBuilder import CompendiumEntry # Import the compendium structure

class VectorSearchFilter:
    """
    A client to interact with Jina AI for embedding and to perform
    in-memory vector searches against a pre-populated store of tool scenarios.
    """
    def __init__(self, api_key=None):
        # The JINA_API_KEY from your .env file will be used here
        self.api_key = api_key or os.environ.get("JINA_API_KEY")
        if not self.api_key:
            raise ValueError("JINA_API_KEY environment variable not set.")
        
        self.embed_url = "https://api.jina.ai/v1/embeddings"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # --- NEW: In-memory store for scenario embeddings ---
        self.scenario_store = []

    def get_embedding(self, text: str) -> list:
        """Generates an embedding for a single text using Jina AI."""
        try:
            response = requests.post(
                self.embed_url,
                headers=self.headers,
                json={"model": "jina-embeddings-v2-base-en", "input": [text]}
            )
            response.raise_for_status()
            return response.json()['data'][0]['embedding']
        except Exception as e:
            print(f"❌ Jina embedding failed: {e}")
            return []

    # --- NEW: The missing method ---
    def add_scenarios_from_compendium(self, compendium: CompendiumEntry, tool_name: str):
        """
        Extracts scenarios from a compendium, generates embeddings,
        and adds them to the internal scenario_store.
        """
        scenarios = compendium.Textual_Compendium.Usage_Scenarios
        print(f"  -> Indexing {len(scenarios)} scenarios for tool: '{tool_name}'...")
        for scenario_text in scenarios:
            if not scenario_text or not isinstance(scenario_text, str):
                continue
            
            embedding = self.get_embedding(scenario_text)
            if embedding:
                self.scenario_store.append({
                    "scenario": scenario_text,
                    "embedding": np.array(embedding), # Store as numpy array for calculations
                    "tool_name": tool_name # Tag with the parent tool name
                })

    @staticmethod
    def cosine_similarity(vec_a, vec_b):
        """Helper to compute cosine similarity between two numpy vectors."""
        return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

    # --- NEW: The primary search method ---
    def search(self, query: str, top_k: int = 3) -> list:
        """
        Finds the top-k most similar scenarios from the pre-populated store.
        """
        if not self.scenario_store:
            print("⚠️ Warning: Vector store is empty. Search cannot be performed.")
            return []
            
        query_embedding = np.array(self.get_embedding(query))
        if query_embedding.size == 0:
            return []

        # Calculate similarity against every scenario in the store
        similarities = []
        for entry in self.scenario_store:
            sim = self.cosine_similarity(query_embedding, entry["embedding"])
            similarities.append((entry, sim))
            
        # Sort by similarity and return the top-k results
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return the data, not the similarity score
        return [entry for entry, sim in similarities[:top_k]]