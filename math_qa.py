import os
import json
import re
import requests
import pandas as pd
from dotenv import load_dotenv
from agenttools import BaseTool, ToolUsageExample
from mongo_utils import MongoVectorStore # Re-use the connection utility
from typing import Dict, Optional

# --- 1. CONFIGURATION ---
load_dotenv()
API_KEY = os.environ.get("JINA_API_KEY")
if not API_KEY:
    raise ValueError("JINA_API_KEY environment variable not set.")

MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME = "FredRag"

JINA_EMBED_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_RERANK_API_URL = "https://api.jina.ai/v1/rerank"
JINA_CHAT_API_URL = "https://api.jina.ai/v1/chat/completions"

# --- 2. RAG SYSTEM COMPONENTS ---

class JinaAIClient:
    """A client to interact with Jina AI APIs for embeddings and reranking."""
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def get_embeddings(self, texts):
        response = requests.post(
            JINA_EMBED_API_URL, headers=self.headers,
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

    def generate_chat_response(self, prompt):
        response = requests.post(
            JINA_CHAT_API_URL, headers=self.headers,
            json={"model": "jina-deepsearch-v1", "messages": [{"role": "user", "content": prompt}], "stream": False}
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

class MongoRAGManager:
    """A manager for the MathQA RAG system using MongoDB."""
    def __init__(self, jina_client, collection_name="math_problems"):
        self.vector_store = MongoVectorStore(MONGO_URI, DB_NAME, collection_name)
        self.jina_client = jina_client
        print(f"✅ MongoDB RAG collection '{collection_name}' is ready.")

    def count(self):
        return self.vector_store.collection.count_documents({})

    def add_documents(self, documents_df):
        print("Embedding documents for RAG with Jina AI...")
        embeddings = self.jina_client.get_embeddings(documents_df["text_for_embedding"].tolist())
        
        documents_to_insert = []
        for i, row in documents_df.iterrows():
            doc = {
                "_id": row["id"],
                "text": row["text_for_embedding"],
                "embedding": embeddings[i],
                "metadata": {
                    "tool": row["tool"],
                    "original_problem": row["original_problem"]
                }
            }
            documents_to_insert.append(doc)

        self.vector_store.collection.insert_many(documents_to_insert)
        print(f"✅ Added {len(documents_df)} documents to MongoDB RAG collection.")

    def query(self, user_query, n_results=3):
        query_embedding = self.jina_client.get_embeddings([user_query])[0]
        try:
            results = self.vector_store.search(query_embedding, num_results=n_results)
            if results:
                return results
        except Exception as exc:
            print(f"[!] Vector search failed ({exc}). Falling back to manual search.")

        print("[!] No vector-search results found; using manual cosine similarity.")
        return self.vector_store.search_manual(query_embedding, num_results=n_results)

class RAGSystem:
    """Orchestrates the RAG process using MongoDB."""
    def __init__(self, training_data, api_key):
        self.jina_client = JinaAIClient(api_key)
        self.db_manager = MongoRAGManager(self.jina_client)
        
        if self.db_manager.count() == 0:
            print("Math RAG collection is empty. Populating with new data...")
            processed_docs_df = self._load_and_preprocess_data(training_data)
            self.db_manager.add_documents(processed_docs_df)
        else:
            print("MathQA RAG system is already populated.")

    def _load_and_preprocess_data(self, training_data):
        # This function remains the same
        tool_mapping = {
            "Financial": "Financial_Calculator", "Percentage": "Financial_Calculator", "gain": "Financial_Calculator",
            "Mixture": "Algebraic_Problem_Solver", "Averages": "Algebraic_Problem_Solver", "Algebra": "Algebraic_Problem_Solver",
            "general": "Algebraic_Problem_Solver", "Work/Rate": "Work_Time_Analyzer", "physics": "Work_Time_Analyzer",
            "other": "General_Math_Tool",
        }
        processed_docs = []
        for i, item in enumerate(training_data):
            category = item.get("category", "other")
            tool_name = tool_mapping.get(category, "General_Math_Tool")
            text_for_embedding = (
                f"Problem: {item['Problem']} | Category: {category} | "
                f"Tool Used: {tool_name} | Rationale: {item['Rationale']}"
            )
            processed_docs.append({
                "id": str(i), "text_for_embedding": text_for_embedding,
                "tool": tool_name, "original_problem": item['Problem']
            })
        return pd.DataFrame(processed_docs)

    def answer_question(self, user_query, scenario: str = None):
        """Answers a user's question using the full RAG pipeline."""
        print(f"\n🔎 Querying RAG system for: '{user_query}'")
        retrieved_docs = self.db_manager.query(user_query, n_results=3)

        if retrieved_docs:
            docs_to_rerank = [doc.get('text', '') for doc in retrieved_docs]
            docs_to_rerank = [txt for txt in docs_to_rerank if txt]
            print(f"\n🔄 Reranking {len(docs_to_rerank)} documents for relevance...")
            reranked_results = self.jina_client.rerank_documents(user_query, docs_to_rerank)
            print("✅ Reranking complete.")

            context_chunks = []
            for i, doc in enumerate(reranked_results):
                relevance = doc.get('relevance_score', 0.0)
                document_payload = doc.get('document')
                if isinstance(document_payload, dict):
                    text = document_payload.get('text', '')
                else:
                    text = document_payload or doc.get('text', '') or ''
                context_chunks.append(
                    f"Example {i+1} (Relevance: {relevance:.2f}):\n{text}"
                )
            context_str = "\n---\n".join(context_chunks) if context_chunks else "No relevant examples found."
        else:
            print("[!] No relevant documents retrieved; proceeding with direct reasoning.")
            context_str = "No relevant examples were retrieved from the knowledge base."
        
        # Create a guidance sentence only if a scenario was provided
        scenario_guidance = ""
        if scenario:
            scenario_guidance = f"To solve this, use the analytical framework of '{scenario}'."

        prompt = f"""
            You are an expert Math AI, designed to solve complex word problems with precision and clarity.
            Your task is to solve the user's question. Analyze the provided context from a knowledge base, which contains examples of similar solved problems. {scenario_guidance}

        **Context: Examples from Knowledge Base**
        {context_str}

        **User's Question:**
        {user_query}

        **Your Response:**
        Provide a step-by-step rationale explaining your work, and conclude with the final answer in the strict format: 'Final Answer: [letter]'.
        Ensure your reasoning is thorough and that you double-check your final answer for accuracy.
        """
        print("\n🤖 Generating final answer with Jina DeepSearch...")
        return self.jina_client.generate_chat_response(prompt)

# --- 3. THE FINAL MATHQA TOOL FOR THE AGENT ---

class MathQATool(BaseTool):
    """
    A tool for solving mathematical word problems by leveraging a MongoDB-based RAG system.
    """
    def __init__(self):
        super().__init__("mathqa")
        self.description = "A tool for solving mathematical word problems using a Retrieval-Augmented Generation system."
        
        try:
            with open('train_new.json', 'r') as f:
                training_data = json.load(f)
            print("✅ Successfully loaded 'train_new.json' for MathQATool.")
            self.rag_system = RAGSystem(training_data, API_KEY)
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Could not initialize RAG system for MathQATool: {e}")
            self.rag_system = None

    def run(self, user_query: str, data_item: Optional[Dict] = None, recommended_scenario: str = None) -> ToolUsageExample:
        """
        Executes the math problem-solving logic by calling the internal RAG system.
        """
        if not self.rag_system:
            return self._create_error_response(user_query, "RAG system not initialized.")

        try:
            full_response_text = self.rag_system.answer_question(user_query, recommended_scenario)
            parsed_output = self._parse_llm_response(full_response_text)

            return ToolUsageExample(
                tool_name=self.name,
                user_query=user_query,
                raw_prompt="[Prompt managed by internal RAG system]",
                llm_response=full_response_text,
                parsed_output=parsed_output
            )
        except Exception as e:
            return self._create_error_response(user_query, f"An error occurred in the RAG system: {e}")

    def _parse_llm_response(self, response_text: str) -> dict:
        """A robust parser to find the final answer letter from the LLM's text."""
        match = re.search(
            r"(?:final\sanswer|answer\sis|the\sanswer\sis|correct\sanswer\sis|is:)\s*([a-e])\b",
            response_text,
            re.IGNORECASE | re.DOTALL
        )
        if match:
            return {"final_answer": match.group(1).lower()}
        else:
            return {"final_answer": None}

    def _create_error_response(self, user_query, error_message):
        """Helper to create a consistent error object."""
        print(f"❌ {error_message}")
        return ToolUsageExample(
            tool_name=self.name,
            user_query=user_query,
            raw_prompt="[Error occurred]",
            llm_response=error_message,
            parsed_output={"error": error_message, "final_answer": None}
        )
