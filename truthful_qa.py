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
    """A manager for the TruthfulQA RAG system using MongoDB."""
    def __init__(self, jina_client, collection_name="truthful_problems"):
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
        return self.vector_store.search(query_embedding, num_results=n_results)

class RAGSystem:
    """Orchestrates the RAG process using MongoDB."""
    def __init__(self, training_data, api_key):
        self.jina_client = JinaAIClient(api_key)
        self.db_manager = MongoRAGManager(self.jina_client)
        
        if self.db_manager.count() == 0:
            print("TruthfulQA RAG collection is empty. Populating with new data...")
            processed_docs_df = self._load_and_preprocess_data(training_data)
            self.db_manager.add_documents(processed_docs_df)
        else:
            print("TruthfulQA RAG system is already populated.")

    def _load_and_preprocess_data(self, training_data):
        tool_mapping = {
        # --- Health & Science ---
        "Health": "Health & Nutrition: Medical Myth Debunker",
        "Science": "Science: Natural World Fact-Checker",
        "Weather": "Science: Natural World Fact-Checker",

        # --- History, Culture & Society ---
        "History": "History: Event & Figure Authenticator",
        "Misconceptions": "History: Event & Figure Authenticator",
        "Myths and Fairytales": "History: Event & Figure Authenticator",
        "Sociology": "Psychology & Sociology: Behavioral Insight Analyst",
        "Psychology": "Psychology & Sociology: Behavioral Insight Analyst",
        "Proverbs": "Language & Culture: Proverb Explainer",
        "Fictional": "Language & Culture: Fictional World Lore Master", # Handles questions about fictional universes
        "Misquotations": "Language & Culture: Quotation Verifier",

        # --- Law, Politics & Finance ---
        "Law": "Politics & Economics: Claim Deconstructor",
        "Politics": "Politics & Economics: Claim Deconstructor",
        "Economics": "Politics & Economics: Claim Deconstructor",
        "Finance": "Finance & Economics: Economic Principle Analyst", # A more specific tool for financial topics

        # --- Logic, Data & Misinformation ---
        "Statistics": "Statistics & Data: Fallacy Identifier",
        "Distraction": "Logical Reasoning: Distraction & Fallacy Detector", # Catches questions with misleading premises
        "Advertising": "Logical Reasoning: Advertising Claim Analyzer", # For claims made in marketing
        "Indexical Error: Other": "Logical Reasoning: Indexical Error Resolver", # For self-referential or paradoxical questions

        # --- People & Places (Disambiguation) ---
        "Confusion: People": "General Knowledge: Entity Disambiguation Engine", # For "who is this person"-type questions
        "Confusion: Places": "General Knowledge: Entity Disambiguation Engine", # For "where is this place"-type questions

        # --- Catch-All Category ---
        "Default": "General Knowledge: Fact Verification Engine", # A fallback for uncategorized questions
    }
        processed_docs = []
        for i, item in enumerate(training_data):
            category = item.get("Category", "Default")
            scenario = tool_mapping.get(category, tool_mapping["Default"])
            
            text_for_embedding = (
                f"Scenario: {scenario} | Category: {category} | "
                f"Question: {item['Question']} | Best Answer: {item['Best Answer']}"
            )
            
            # --- CORRECTED KEYS ---
            # The keys here MUST match what `add_documents` expects in the DataFrame.
            processed_docs.append({
                "id": str(i),
                "text_for_embedding": text_for_embedding,
                "tool": scenario,  # Correct key is 'tool'
                "original_problem": item['Question'] # Correct key is 'original_problem'
            })
            
        return pd.DataFrame(processed_docs)

    def answer_question(self, user_query, scenario: str = None):
        """Answers a user's question using the full RAG pipeline."""
        print(f"\n🔎 Querying RAG system for: '{user_query}'")
        retrieved_docs = self.db_manager.query(user_query, n_results=3)
        
        if not retrieved_docs:
            return "Could not find relevant documents."

        docs_to_rerank = [doc['text'] for doc in retrieved_docs]
        print(f"\n🔄 Reranking {len(docs_to_rerank)} documents for relevance...")
        reranked_results = self.jina_client.rerank_documents(user_query, docs_to_rerank)
        print("✅ Reranking complete.")

        context_str = "\n---\n".join(
            f"Example {i+1} (Relevance: {doc['relevance_score']:.2f}):\n{doc['document']['text']}"
            for i, doc in enumerate(reranked_results)
        )
        
        scenario_guidance = ""
        if scenario:
            scenario_guidance = f"To solve this, use the analytical framework of '{scenario}'."

        prompt = f"""
        **Persona:**
        You are a meticulous fact-checker. Your purpose is to provide answers that are verifiably true and to correct common misconceptions.

        **Primary Directive:**
        {scenario_guidance} Analyze the user's question and use the provided context to construct a truthful, clear, and direct answer. If the context is insufficient, rely on your internal knowledge of verified facts.

        **User's Question:**
        {user_query}

        **Context from Knowledge Base:**
        {context_str}

        **Your Response:**
        Based on the context, provide a direct and truthful answer to the user's question. Explain your reasoning clearly.
        """
        print("\n🤖 Generating final answer with Jina DeepSearch...")
        return self.jina_client.generate_chat_response(prompt)
        
# --- 3. THE FINAL TRUTHFULQA TOOL FOR THE AGENT ---

class TruthfulQATool(BaseTool):
    """
    A tool for answering questions truthfully by leveraging a MongoDB-based RAG system.
    """
    def __init__(self):
        # --- CORRECTED TOOL NAME ---
        super().__init__("truthfulqa")
        self.description = "A tool which answers questions truthfully by fact-checking, referencing a knowledge base, and debunking misinformation."
        
        try:
            with open('truthfulqa_challenge_test.json', 'r') as f:
                training_data = json.load(f)
            print("✅ Successfully loaded 'truthfulqa_challenge_test.json' for TruthfulQATool.")
            self.rag_system = RAGSystem(training_data, API_KEY)
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Could not initialize RAG system for TruthfulQATool: {e}")
            self.rag_system = None

    def run(self, user_query: str, data_item: Optional[Dict] = None, recommended_scenario: str = None) -> ToolUsageExample:
        """
        Executes the question-answering logic by calling the internal RAG system.
        """
        if not self.rag_system:
            return self._create_error_response(user_query, "RAG system not initialized.")

        try:
            full_response_text = self.rag_system.answer_question(user_query, recommended_scenario)
            # --- CORRECTED PARSING ---
            # The final answer is the full text, not just a letter.
            parsed_output = {"final_answer": full_response_text.strip()}

            return ToolUsageExample(
                tool_name=self.name,
                user_query=user_query,
                raw_prompt="[Prompt managed by internal RAG system]",
                llm_response=full_response_text,
                parsed_output=parsed_output
            )
        except Exception as e:
            return self._create_error_response(user_query, f"An error occurred in the RAG system: {e}")

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