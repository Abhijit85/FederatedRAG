import datetime
import os

import numpy as np
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

try:
    import certifi
except ImportError:  # pragma: no cover - certifi is optional but recommended for TLS
    certifi = None


class MongoVectorStore:
    def __init__(self, db_uri, db_name, collection_name):
        """
        Initializes the MongoDB vector store.
        """
        tls_kwargs = {}
        normalized_uri = (db_uri or "").lower()
        uses_tls = normalized_uri.startswith("mongodb+srv://") or "tls=true" in normalized_uri or "ssl=true" in normalized_uri
        if uses_tls and certifi:
            tls_kwargs["tlsCAFile"] = certifi.where()
        elif uses_tls and not certifi:
            print("⚠️ TLS connection requested but 'certifi' package is missing. Install it to avoid certificate errors.")

        self.client = MongoClient(db_uri, **tls_kwargs)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.search_index_name = os.environ.get("MONGO_VECTOR_INDEX_NAME", "vector_index")
        self.embedding_path = os.environ.get("MONGO_VECTOR_PATH", "embedding")
        self.similarity = os.environ.get("MONGO_VECTOR_SIMILARITY", "cosine")

    def _infer_embedding_dimensions(self):
        doc = self.collection.find_one({self.embedding_path: {"$exists": True}}, {self.embedding_path: 1})
        if not doc:
            return None
        embedding = doc.get(self.embedding_path) or []
        return len(embedding) if isinstance(embedding, list) else None

    def has_search_index(self, index_name=None):
        name = index_name or self.search_index_name
        try:
            indexes = list(self.collection.list_search_indexes(name))
        except Exception:
            return False
        return any(index.get("name") == name for index in indexes)

    def ensure_vector_search_index(self, dimensions=None, index_name=None):
        """
        Create the Atlas Vector Search index if it does not already exist.
        Safe to call repeatedly.
        """
        name = index_name or self.search_index_name
        if self.has_search_index(name):
            return {"created": False, "index_name": name}

        dims = dimensions or self._infer_embedding_dimensions()
        if not dims:
            raise RuntimeError(
                "Unable to infer embedding dimensions for Atlas Vector Search. "
                "Populate the collection first or pass dimensions explicitly."
            )

        definition = {
            "fields": [
                {
                    "type": "vector",
                    "path": self.embedding_path,
                    "numDimensions": int(dims),
                    "similarity": self.similarity,
                }
            ]
        }
        model = SearchIndexModel(definition=definition, name=name, type="vectorSearch")
        self.collection.create_search_index(model=model)
        return {
            "created": True,
            "index_name": name,
            "dimensions": int(dims),
            "similarity": self.similarity,
        }

    def add_vectors(self, vectors, texts):
        """
        Adds vectors and their corresponding texts to the collection.
        The vector field is named 'embedding' to match the search index.
        """
        documents = []
        for i, text in enumerate(texts):
            document = {
                "text": text,
                self.embedding_path: vectors[i],
            }
            documents.append(document)
        self.collection.insert_many(documents)

    def search(self, query_vector, num_results=5):
        """
        Performs a vector search in the collection using Atlas Vector Search.
        """
        results = self.collection.aggregate([
            {
                "$vectorSearch": {
                    "index": self.search_index_name,
                    "path": self.embedding_path,
                    "queryVector": query_vector,
                    "numCandidates": 100,
                    "limit": num_results,
                }
            }
        ])
        return list(results)

    def search_manual(self, query_vector, num_results=5):
        """
        Performs a manual vector search for testing without a vector index.
        NOTE: This is much slower and not for production use.
        """
        print("[!] Performing manual vector search for testing purposes.")
        query_vec = np.array(query_vector)
        all_docs = list(self.collection.find({}, {self.embedding_path: 1, "text": 1, "metadata": 1}))

        similarities = []
        for doc in all_docs:
            doc_vec = np.array(doc.get(self.embedding_path, []))
            if doc_vec.size == 0:
                continue

            similarity = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            similarities.append((similarity, doc))

        similarities.sort(key=lambda x: x[0], reverse=True)

        top_results = [doc for score, doc in similarities[:num_results]]
        return top_results

    def close(self):
        """Closes the MongoDB client connection to free up resources."""
        self.client.close()


class MongoLogger:
    def __init__(self, db_uri, db_name, collection_name="logs"):
        """Initializes the MongoDB logger."""
        tls_kwargs = {}
        normalized_uri = (db_uri or "").lower()
        uses_tls = normalized_uri.startswith("mongodb+srv://") or "tls=true" in normalized_uri or "ssl=true" in normalized_uri
        if uses_tls and certifi:
            tls_kwargs["tlsCAFile"] = certifi.where()
        elif uses_tls and not certifi:
            print("⚠️ TLS connection requested but 'certifi' package is missing. Install it to avoid certificate errors.")

        self.client = MongoClient(db_uri, **tls_kwargs)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def log_entry(self, user_id, log_data):
        """Logs an entry event with structured data."""
        log_entry = {
            "timestamp_entry": datetime.datetime.utcnow(),
            "user_id": user_id,
            "log_data": log_data,
            "timestamp_exit": None,
            "llm_response": None,
        }
        return self.collection.insert_one(log_entry).inserted_id

    def log_exit(self, log_id, llm_response: str):
        """Logs an exit event by updating the log entry with the final LLM response."""
        update_fields = {
            "timestamp_exit": datetime.datetime.utcnow(),
            "llm_response": llm_response,
        }
        self.collection.update_one({"_id": log_id}, {"$set": update_fields})
