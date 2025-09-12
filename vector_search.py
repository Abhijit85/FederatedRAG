# vector_search.py

import os
from mongo_utils import MongoVectorStore
from populate_vector_store import JinaAIClient # Re-use the Jina client for embeddings

class VectorSearchFilter:
    """
    A client that uses a MongoDB collection to search for tool scenarios
    based on vector embeddings.
    """
    def __init__(self):
        # Load environment variables
        JINA_API_KEY = os.environ.get("JINA_API_KEY")
        MONGO_URI = os.environ.get("MONGO_URI")
        DB_NAME = "FredRag"
        VECTOR_COLLECTION_NAME = "vectors"

        if not all([JINA_API_KEY, MONGO_URI]):
            raise ValueError("JINA_API_KEY and MONGO_URI must be set in your .env file.")
            
        self.jina_client = JinaAIClient(JINA_API_KEY)
        # Connect to the 'vectors' collection via MongoVectorStore
        self.vector_store = MongoVectorStore(MONGO_URI, DB_NAME, VECTOR_COLLECTION_NAME)

    def search(self, query: str, top_k: int = 3) -> list:
        """
        Finds the top-k most similar tool scenarios from the MongoDB collection
        using Atlas Vector Search.
        """
        if self.vector_store.collection.count_documents({}) == 0:
            print("⚠️ Warning: MongoDB 'vectors' collection is empty. Search cannot be performed.")
            return []
            
        # 1. Get the embedding for the user's query
        query_vector = self.jina_client.get_embeddings([query])
        
        if not query_vector:
            print("❌ Failed to generate an embedding for the query.")
            return []
        
        # 2. Perform the vector search in MongoDB
        results = self.vector_store.search(query_vector[0], num_results=top_k)
        
        # 3. Extract just the 'text' field from the results to match the agent's expectation
        return [result.get('text', '') for result in results]