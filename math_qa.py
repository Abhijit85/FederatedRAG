import os
import json
import re
import requests
import pandas as pd
import chromadb
import time
from dotenv import load_dotenv
from agenttools import BaseTool, ToolUsageExample, CalculatorTool

# --- 1. CONFIGURATION ---
load_dotenv()
API_KEY = os.environ.get("JINA_API_KEY")
if not API_KEY:
    raise ValueError("JINA_API_KEY environment variable not set.")

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

class ChromaDBManager:
    """A wrapper for ChromaDB using Jina AI for embeddings."""
    def __init__(self, jina_client, collection_name="math_problems_jina"):
        self.client = chromadb.Client()
        self.jina_client = jina_client
        if collection_name in [c.name for c in self.client.list_collections()]:
            self.client.delete_collection(name=collection_name)
        self.collection = self.client.create_collection(name=collection_name)
        print(f"✅ ChromaDB collection '{collection_name}' created.")

    def add_documents(self, documents_df):
        print("Embedding documents with Jina AI...")
        embeddings = self.jina_client.get_embeddings(documents_df["text_for_embedding"].tolist())
        self.collection.add(
            ids=documents_df["id"].tolist(),
            embeddings=embeddings,
            metadatas=documents_df[['tool', 'original_problem']].to_dict('records'),
            documents=documents_df["text_for_embedding"].tolist()
        )
        print(f"✅ Added {len(documents_df)} documents to ChromaDB.")

    def query(self, user_query, n_results=3):
        query_embedding = self.jina_client.get_embeddings([user_query])
        return self.collection.query(query_embeddings=query_embedding, n_results=n_results)

class RAGSystem:
    """Orchestrates the RAG process: Retrieve -> Rerank -> Generate."""
    def __init__(self, training_data, api_key):
        self.jina_client = JinaAIClient(api_key)
        self.db_manager = ChromaDBManager(self.jina_client)
        
        processed_docs_df = self._load_and_preprocess_data(training_data)
        self.db_manager.add_documents(processed_docs_df)

    def _load_and_preprocess_data(self, training_data):
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

    def answer_question(self, user_query):
        """Answers a user's question using the full RAG pipeline."""
        print(f"\n🔎 Querying RAG system for: '{user_query}'")
        retrieved_docs = self.db_manager.query(user_query, n_results=3)
        
        if not retrieved_docs or not retrieved_docs.get('documents', [[]])[0]:
            return "Could not find relevant documents."

        docs_to_rerank = retrieved_docs['documents'][0]
        print(f"\n🔄 Reranking {len(docs_to_rerank)} documents for relevance...")
        reranked_results = self.jina_client.rerank_documents(user_query, docs_to_rerank)
        print("✅ Reranking complete.")

        context_str = "\n---\n".join(
            f"Example {i+1} (Relevance: {doc['relevance_score']:.2f}):\n{doc['document']}"
            for i, doc in enumerate(reranked_results)
        )
        prompt = f"""
        Analyze the user's question using the provided context from a knowledge base. Provide a step-by-step rationale and conclude with the final answer in the format 'Final Answer: [letter]'.

        **Context: Examples from Knowledge Base**
        {context_str}

        **User's Question:**
        {user_query}

        **Your Response:**
        """
        print("\n🤖 Generating final answer with Jina DeepSearch...")
        return self.jina_client.generate_chat_response(prompt)

# --- 3. THE FINAL MATHQA TOOL FOR THE AGENT ---

class MathQATool(BaseTool):
    """
    A tool for solving mathematical word problems by leveraging an internal RAG system.
    """
    def __init__(self):
        super().__init__("mathqa")
        self.description = "A tool for solving mathematical word problems using a Retrieval-Augmented Generation system."
        
        # Initialize the RAG system when the tool is created
        try:
            with open('train_new.json', 'r') as f:
                training_data = json.load(f)
            print("✅ Successfully loaded 'train_new.json' for MathQATool.")
            self.rag_system = RAGSystem(training_data, API_KEY)
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Could not initialize RAG system for MathQATool: {e}")
            self.rag_system = None

    def run(self, user_query: str, dynamic_prompt: str = None) -> ToolUsageExample:
        """
        Executes the math problem-solving logic by calling the internal RAG system.
        The dynamic_prompt from the agent is ignored here, as the RAG system builds its own.
        """
        if not self.rag_system:
            return self._create_error_response(user_query, "RAG system not initialized.")

        try:
            # --- The tool now calls its internal RAG system ---
            full_response_text = self.rag_system.answer_question(user_query)
            
            # The tool parses its own response to find the answer
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