import os
import json
import re
import requests
import pandas as pd
from dotenv import load_dotenv
from agenttools import BaseTool, ToolUsageExample
from mongo_utils import MongoVectorStore
from typing import Dict, Optional
import base64
from PIL import Image
from io import BytesIO
from openrouter_client import chat_completion

# --- 1. CONFIGURATION ---
load_dotenv()
JINA_API_KEY = os.environ.get("JINA_API_KEY")
if not JINA_API_KEY:
    raise ValueError("JINA_API_KEY environment variable not set.")

MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME = "FredRag"

# Using a different embedding model for multimodal data
JINA_MULTIMODAL_EMBED_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_RERANK_API_URL = "https://api.jina.ai/v1/rerank"

VLM_MODEL = os.environ.get("VLM_MODEL")
if not VLM_MODEL:
    raise ValueError("VLM_MODEL environment variable not set.")


# --- 2. RAG SYSTEM COMPONENTS ---

class JinaAIClient:
    """A client to interact with Jina AI APIs for embeddings and reranking."""
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def get_multimodal_embeddings(self, texts, images):
        """Generates multimodal embeddings for text-image pairs."""
        # NOTE: This is a conceptual implementation. The actual Jina API might differ.
        # For this example, we will embed text only as a placeholder for a true multimodal model.
        response = requests.post(
            JINA_MULTIMODAL_EMBED_API_URL, headers=self.headers,
            json={"model": "jina-embeddings-v2-base-en", "input": texts}
        )
        response.raise_for_status()
        return [item['embedding'] for item in response.json()['data']]

    def rerank_documents(self, query, documents):
        response = requests.post(
            JINA_RERANK_API_URL, headers=self.headers,
            json={"model": "jina-reranker-v2-base-multilingual", "query": query, "documents": documents, "top_n": len(documents)}
        )
        response.raise_for_status()
        return response.json()['results']

class VLMClient:
    """A client for the Vision Language Model."""
    def __init__(self, model):
        self.model = model

    def generate_response(self, prompt, image_data_uri):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_uri}}
        ]}]
        
        response = chat_completion(
            model=self.model,
            messages=messages,
            max_tokens=2048,
        )
        return response.choices[0].message.content

class MongoRAGManager:
    """A manager for the ScienceQA RAG system using MongoDB."""
    def __init__(self, jina_client, collection_name="science_problems"):
        self.vector_store = MongoVectorStore(MONGO_URI, DB_NAME, collection_name)
        self.jina_client = jina_client
        print(f"✅ MongoDB RAG collection '{collection_name}' is ready.")

    def count(self):
        return self.vector_store.collection.count_documents({})

    def add_documents(self, documents_df):
        print("Embedding documents for Science RAG...")
        # Placeholder for multimodal embeddings
        embeddings = self.jina_client.get_multimodal_embeddings(
            documents_df["text_for_embedding"].tolist(),
            documents_df["image_b64"].tolist()
        )
        
        documents_to_insert = []
        for i, row in documents_df.iterrows():
            doc = {
                "_id": row["id"],
                "text": row["text_for_embedding"],
                "embedding": embeddings[i],
                "metadata": {
                    "original_question": row["original_question"],
                    "image_b64": row["image_b64"]
                }
            }
            documents_to_insert.append(doc)

        self.vector_store.collection.insert_many(documents_to_insert)
        print(f"✅ Added {len(documents_df)} documents to MongoDB Science RAG collection.")

    def query(self, user_query, image_b64, n_results=3):
        # Placeholder for multimodal query embedding
        query_embedding = self.jina_client.get_multimodal_embeddings([user_query], [image_b64])[0]
        return self.vector_store.search(query_embedding)

class RAGSystem:
    """Orchestrates the Science RAG process using MongoDB."""
    def __init__(self, training_data, api_key):
        self.jina_client = JinaAIClient(api_key)
        self.db_manager = MongoRAGManager(self.jina_client)
        
        if self.db_manager.count() == 0:
            print("Science RAG collection is empty. Populating with new data...")
            processed_docs_df = self._load_and_preprocess_data(training_data)
            self.db_manager.add_documents(processed_docs_df)
        else:
            print("ScienceQA RAG system is already populated.")

    def _load_and_preprocess_data(self, training_data):
        processed_docs = []
        for i, item in enumerate(training_data):
            # Ensure the item has an image
            if not item.get("image"):
                continue
            
            text_for_embedding = (
                f"Question: {item['question']} | Choices: {item['choices']} | "
                f"Topic: {item['topic']} | Lecture: {item.get('lecture', '')}"
            )
            processed_docs.append({
                "id": str(i),
                "text_for_embedding": text_for_embedding,
                "original_question": item['question'],
                "image_b64": item['image'] 
            })
        return pd.DataFrame(processed_docs)

    def answer_question(self, user_query, image_b64, choices, scenario):
        print(f"\n🔎 Querying Science RAG system for: '{user_query[:50]}...'")
        retrieved_docs = self.db_manager.query(user_query, image_b64, n_results=3)
        
        if not retrieved_docs:
            context_str = "No relevant documents found."
        else:
            docs_to_rerank = [doc['text'] for doc in retrieved_docs]
            print(f"\n🔄 Reranking {len(docs_to_rerank)} documents for relevance...")
            reranked_results = self.jina_client.rerank_documents(user_query, docs_to_rerank)
            print("✅ Reranking complete.")
            context_chunks = []
            for i, doc in enumerate(reranked_results):
                doc_payload = doc.get("document")
                if isinstance(doc_payload, dict):
                    doc_text = doc_payload.get("text", "")
                else:
                    doc_text = doc_payload or ""
                context_chunks.append(
                    f"Example {i+1} (Relevance: {doc.get('relevance_score', 0):.2f}):\n{doc_text}"
                )
            context_str = "\n---\n".join(context_chunks)

        prompt = f"""
        You are an expert scientific analyst.Your task is to answer the user's question based on the provided image.
        Analyze the provided image, text, and knowledge base context to answer the multiple-choice question. You have been assigned the specific analytical lens of {scenario} for this problem; use it to guide your reasoning. 
        Provide a step-by-step rationale and conclude with the final answer in the format 'Final Answer: [The correct choice text]'.

        **Context from Knowledge Base:**
        {context_str}

        **User's Question:**
        {user_query}

        **Choices:**
        {json.dumps(choices)}

        **Your Response:**
        """
        print("\n🤖 Generating final answer with VLM...")
        vlm_client = VLMClient(VLM_MODEL)
        return vlm_client.generate_response(prompt, image_b64)

# --- 3. THE FINAL SCIENCEQA TOOL FOR THE AGENT ---

class ScienceQATool(BaseTool):
    """
    A tool for solving science questions by leveraging a MongoDB-based RAG system.
    """
    def __init__(self):
        super().__init__("scienceqa")
        self.description = "A tool for solving science questions using a Retrieval-Augmented Generation system with multimodal capabilities."
        
        try:
            with open('scienceqa_challenge_test.json', 'r') as f:
                training_data = json.load(f)
            print("✅ Successfully loaded 'scienceqa_challenge_test.json' for ScienceQATool.")
            self.rag_system = RAGSystem(training_data, JINA_API_KEY)
        except FileNotFoundError:
            print("❌ CRITICAL ERROR: 'scienceqa_challenge_test.json' not found. ScienceQATool will not work.")
            self.rag_system = None
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Could not initialize RAG system for ScienceQATool: {e}")
            self.rag_system = None

    def run(self, user_query: str, data_item: Optional[Dict] = None ,recommended_scenario: str = None) -> ToolUsageExample:
        if not self.rag_system:
            return self._create_error_response(user_query, "RAG system not initialized.")
        if not data_item or not data_item.get("image"):
            return self._create_error_response(user_query, "No image provided for ScienceQATool.")

        try:
            full_response_text = self.rag_system.answer_question(
                user_query,
                data_item["image"],
                data_item["choices"],
                recommended_scenario
            )
            return ToolUsageExample(
                tool_name=self.name,
                user_query=user_query,
                raw_prompt="[Prompt managed by internal RAG system]",
                llm_response=full_response_text,
                parsed_output={"response": full_response_text}
            )
        except Exception as e:
            return self._create_error_response(user_query, f"An error occurred in the Science RAG system: {e}")

    def _create_error_response(self, user_query, error_message):
        print(f"❌ {error_message}")
        return ToolUsageExample(
            tool_name=self.name,
            user_query=user_query,
            raw_prompt="[Error occurred]",
            llm_response=error_message,
            parsed_output={"error": error_message}
        )
