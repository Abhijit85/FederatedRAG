import pymongo
from pymongo import MongoClient
import datetime
import numpy as np

class MongoVectorStore:
    def __init__(self, db_uri, db_name, collection_name):
        """
        Initializes the MongoDB vector store.
        """
        self.client = MongoClient(db_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def add_vectors(self, vectors, texts):
        """
        Adds vectors and their corresponding texts to the collection.
        The vector field is named 'embedding' to match the search index.
        """
        documents = []
        for i, text in enumerate(texts):
            document = {
                "text": text,
                "embedding": vectors[i]
            }
            documents.append(document)
        self.collection.insert_many(documents)

    def search(self, query_vector, num_results=5):
        """
        Performs a vector search in the collection using Atlas Vector Search.
        """
        results = self.collection.aggregate([
            {
                '$vectorSearch': {
                    'index': 'vector_index',
                    'path': 'embedding',
                    'queryVector': query_vector,
                    'numCandidates': 100,
                    'limit': num_results
                }
            }
        ])
        return list(results)
        
    def search_manual(self, query_vector, num_results=5):
        """
        Performs a manual vector search for testing without an Atlas index.
        This is much slower and should only be used for local testing.
        """
        print("[!] Performing manual vector search for testing purposes.")
        
        # --- THIS IS THE FIX ---
        # Ensure it fetches the 'embedding' field from the database.
        all_docs = list(self.collection.find({}, {"text": 1, "embedding": 1}))
        
        if not all_docs:
            return []

        query_vec = np.array(query_vector)
        
        similarities = []
        for doc in all_docs:
            # Ensure it accesses the 'embedding' field from the document.
            doc_vec = np.array(doc.get('embedding', []))
            if doc_vec.size == 0:
                continue # Skip documents that might be missing an embedding
            
            similarity = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            similarities.append((similarity, doc))
            
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        top_results = [doc for score, doc in similarities[:num_results]]
        return top_results


class MongoLogger:
    def __init__(self, db_uri, db_name, collection_name="logs"):
        """
        Initializes the MongoDB logger.
        """
        self.client = MongoClient(db_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def log_entry(self, user_id, log_data):
        """
        Logs an entry event.
        """
        log_entry = {
            "timestamp_entry": datetime.datetime.utcnow(),
            "user_id": user_id,
            "log_data": log_data,
            "timestamp_exit": None
        }
        return self.collection.insert_one(log_entry).inserted_id

    def log_exit(self, log_id):
        """
        Logs an exit event by updating the log entry.
        """
        self.collection.update_one(
            {"_id": log_id},
            {"$set": {"timestamp_exit": datetime.datetime.utcnow()}}
        )
