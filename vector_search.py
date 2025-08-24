# vector_search.py

import os
import chromadb
from chromadb.utils import embedding_functions
import numpy as np
from CompendiumBuilder import CompendiumEntry

class VectorSearchFilter:
    """
    A client that uses a persistent ChromaDB collection to store and search
    tool scenarios based on vector embeddings.
    """
    def __init__(self):
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        # Initialize a persistent ChromaDB client.
        # This saves the database to a 'chroma_db' folder, ensuring data persists.
        client = chromadb.PersistentClient(path="chroma_db")
        
        # Get or create the collection for compendium scenarios.
        self.collection = client.get_or_create_collection(
            name="compendium_scenarios",
            embedding_function=self.embedding_function
        )

    def add_scenarios_from_compendium(self, compendium: CompendiumEntry, tool_name: str):
        """Populates the ChromaDB collection with scenarios from a compendium."""
        scenarios = compendium.Textual_Compendium.Usage_Scenarios
        if not scenarios:
            return

        print(f"-> Indexing {len(scenarios)} scenarios for tool: '{tool_name}'...")
        
        documents = []
        metadatas = []
        ids = []
        
        for i, scenario in enumerate(scenarios):
            context = f": {scenario.context}" if scenario.context else ""
            scenario_text = f"{scenario.scenario}{context}"
            
            documents.append(scenario_text)
            metadatas.append({
                "scenario": scenario_text,
                "tool_name": tool_name
            })
            ids.append(f"scenario_{tool_name}_{i}")

        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

    def search(self, query: str, top_k: int = 3) -> list:
        """
        Finds the top-k most similar scenarios from the ChromaDB collection.
        """
        if self.collection.count() == 0:
            print("⚠️ Warning: ChromaDB vector store is empty. Search cannot be performed.")
            return []
            
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        if not results or not results.get('metadatas') or not results['metadatas'][0]:
            return []

        return results['metadatas'][0]
