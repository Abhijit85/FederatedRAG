import json
import os
import re
from typing import Dict, Optional, Sequence

import pandas as pd
import requests
from dotenv import load_dotenv

from agenttools import BaseTool, ToolUsageExample
from jina_key_manager import JinaAPIKeyRotator, get_available_jina_api_keys
from mongo_utils import MongoVectorStore  # Re-use the connection utility

# --- 1. CONFIGURATION ---
load_dotenv()
API_KEYS = get_available_jina_api_keys()
if not API_KEYS:
    raise ValueError("At least one JINA_API_KEY environment variable must be set.")

MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME = "FredRag"

JINA_EMBED_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_RERANK_API_URL = "https://api.jina.ai/v1/rerank"
JINA_CHAT_API_URL = "https://api.jina.ai/v1/chat/completions"

# --- 2. RAG SYSTEM COMPONENTS ---

class JinaAIClient:
    """A client to interact with Jina AI APIs for embeddings and reranking."""

    def __init__(self, api_keys: Sequence[str] | Sequence[tuple[str, str]] | None):
        self._rotator = JinaAPIKeyRotator(api_keys)

    @staticmethod
    def _build_headers(api_key: str) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def _post_json(self, url: str, payload: dict, *, timeout: float | None = None):
        return self._rotator.execute(
            lambda api_key: requests.post(
                url,
                headers=self._build_headers(api_key),
                json=payload,
                timeout=timeout,
            )
        )

    def get_embeddings(self, texts):
        response = self._post_json(
            JINA_EMBED_API_URL,
            {"model": "jina-embeddings-v2-base-en", "input": texts},
        )
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]

    def rerank_documents(self, query, documents):
        response = self._post_json(
            JINA_RERANK_API_URL,
            {
                "model": "jina-reranker-v2-base-multilingual",
                "query": query,
                "documents": documents,
                "top_n": len(documents),
            },
        )
        response.raise_for_status()
        return response.json()["results"]

    def generate_chat_response(self, prompt):
        response = self._post_json(
            JINA_CHAT_API_URL,
            {
                "model": "jina-deepsearch-v1",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

class MongoRAGManager:
    """A manager for the MMLUQA RAG system using MongoDB."""
    def __init__(self, jina_client, collection_name="mmlu_problems"):
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
        # Use the manual search for testing, as no Atlas index is set up for this collection
        return self.vector_store.search_manual(query_embedding, num_results=n_results)

class RAGSystem:
    """Orchestrates the RAG process using MongoDB."""

    def __init__(self, training_data, api_keys: Sequence[str]):
        self.jina_client = JinaAIClient(api_keys)
        self.db_manager = MongoRAGManager(self.jina_client)
        
        if self.db_manager.count() == 0:
            print("MMLU RAG collection is empty. Populating with new data...")
            processed_docs_df = self._load_and_preprocess_data(training_data)
            self.db_manager.add_documents(processed_docs_df)
        else:
            print("MMLUQA RAG system is already populated.")

    def _load_and_preprocess_data(self, training_data):
        # This function remains the same
        tool_mapping = {
                    # STEM
                    "Abstract Algebra": "STEM: Abstract & Higher Mathematics Solver",
                    "College Mathematics": "STEM: Abstract & Higher Mathematics Solver",
                    "Elementary Mathematics": "STEM: Foundational Mathematics Engine",
                    "High School Mathematics": "STEM: Foundational Mathematics Engine",
                    "High School Statistics": "STEM: Foundational Mathematics Engine",
                    "College Physics": "STEM: Physics Problem Calculator",
                    "High School Physics": "STEM: Physics Problem Calculator",
                    "Conceptual Physics": "STEM: Physics Problem Calculator",
                    "College Chemistry": "STEM: Chemistry Equation & Concept Analyzer",
                    "High School Chemistry": "STEM: Chemistry Equation & Concept Analyzer",
                    "College Biology": "STEM: Biological Systems Modeler",
                    "High School Biology": "STEM: Biological Systems Modeler",
                    "Anatomy": "STEM: Biological Systems Modeler",
                    "Virology": "STEM: Biological Systems Modeler",
                    "Astronomy": "STEM: Advanced Sciences & Engineering Module",
                    "Electrical Engineering": "STEM: Advanced Sciences & Engineering Module",
                    "Computer Security": "STEM: Advanced Sciences & Engineering Module",
                    "College Computer Science": "STEM: Advanced Sciences & Engineering Module",
                    "High School Computer Science": "STEM: Advanced Sciences & Engineering Module",
                    "Machine Learning": "STEM: Advanced Sciences & Engineering Module",

                    # Health & Medicine
                    "Clinical Knowledge": "Health & Medicine: Clinical Knowledge Base",
                    "College Medicine": "Health & Medicine: Clinical Knowledge Base",
                    "Professional Medicine": "Health & Medicine: Clinical Knowledge Base",
                    "Medical Genetics": "Health & Medicine: Clinical Knowledge Base",
                    "Human Aging": "Health & Medicine: Human Health & Wellness Advisor",
                    "Human Sexuality": "Health & Medicine: Human Health & Wellness Advisor",
                    "Nutrition": "Health & Medicine: Human Health & Wellness Advisor",

                    # Social Sciences
                    "Macroeconomics": "Social Sciences: Economic Modeler",
                    "Microeconomics": "Social Sciences: Economic Modeler",
                    "Econometrics": "Social Sciences: Economic Modeler",
                    "High School Psychology": "Social Sciences: Behavioral & Societal Analyzer",
                    "Professional Psychology": "Social Sciences: Behavioral & Societal Analyzer",
                    "Sociology": "Social Sciences: Behavioral & Societal Analyzer",

                    # Humanities
                    "Security Studies": "Humanities: Governance & Global Affairs Analyst",
                    "U.S. Foreign Policy": "Humanities: Governance & Global Affairs Analyst",
                    "International Law": "Humanities: Governance & Global Affairs Analyst",
                    "High School Government": "Humanities: Governance & Global Affairs Analyst",
                    "High School U.S. History": "Humanities: Historical Event Retriever",
                    "World History": "Humanities: Historical Event Retriever",
                    "European History": "Humanities: Historical Event Retriever",
                    "Prehistory": "Humanities: Historical Event Retriever",
                    "Philosophy": "Humanities: Philosophical & Moral Reasoner",
                    "World Religions": "Humanities: Philosophical & Moral Reasoner",
                    "Moral Disputes": "Humanities: Philosophical & Moral Reasoner",
                    "Moral Scenarios": "Humanities: Philosophical & Moral Reasoner",
                    "Jurisprudence": "Humanities: Philosophical & Moral Reasoner",
                    "Logical Fallacies": "Humanities: Philosophical & Moral Reasoner",
                    "Global Facts": "Humanities: Global Knowledge Navigator",

                    # Law & Professional
                    "Professional Law": "Law & Professional: Legal Case Analyzer",
                    "Management": "Professional Studies: Business Strategy Simulator",
                    "Marketing": "Professional Studies: Business Strategy Simulator",
                    "Public Relations": "Professional Studies: Business Strategy Simulator",
                    "Business Ethics": "Professional Studies: Business Strategy Simulator",
                    "Professional Accounting": "Professional Studies: Business Strategy Simulator",

                    # Miscellaneous
                    "Formal Logic": "Miscellaneous: Cross-Domain Problem Synthesizer",
                    "Other": "Miscellaneous: Cross-Domain Problem Synthesizer",
                }
        processed_docs = []
        for i, item in enumerate(training_data):
            # Use the 'subject' field as the category
            category = item.get("subject", "other")
            tool_name = tool_mapping.get(category, "General_MMLU_Tool")

            # --- CHANGES ARE HERE ---
            # Use 'question' instead of 'Problem'
            problem_text = item.get("question", "")
            # Use 'baseline_model_answer' instead of 'Rationale'
            rationale_text = item.get("baseline_model_answer", "")

            # Construct the text for embedding with the correct data
            text_for_embedding = (
                f"Problem: {problem_text} | Category: {category} | "
                f"Tool Used: {tool_name} | Rationale: {rationale_text}"
            )

            processed_docs.append({
                "id": str(i),
                "text_for_embedding": text_for_embedding,
                "tool": tool_name,
                # Use the corrected variable for the original problem
                "original_problem": problem_text
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
        # Create a guidance sentence only if a scenario was provided
        scenario_guidance = ""
        if scenario:
            scenario_guidance = f"To solve this, use the analytical framework of '{scenario}'."

        prompt = f"""
            You are a Polymath AI expert, an expert reasoning engine with deep knowledge across STEM, humanities, and professional studies with precision and clarity.
            Your task is to solve the following multiple-choice question. {scenario_guidance}
            Analyze the provided context from a knowledge base, which contains similar problems or relevant information.


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

# --- 3. THE FINAL MMLUQA TOOL FOR THE AGENT ---

class MMLUQATool(BaseTool):
    """
    A tool for solving mathematical word problems by leveraging a MongoDB-based RAG system.
    """
    def __init__(self):
        super().__init__("MMLUqa")
        self.description = "A tool for solving mathematical word problems using a Retrieval-Augmented Generation system."
        
        try:
            with open('mmlu_challenge_test_set.json', 'r') as f:
                training_data = json.load(f)
            print("✅ Successfully loaded 'mmlu_challenge_test_set.json' for MMLUQATool.")
            self.rag_system = RAGSystem(training_data, API_KEYS)
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Could not initialize RAG system for MMLUQATool: {e}")
            self.rag_system = None

    def run(self, user_query: str, data_item: Optional[Dict] = None, recommended_scenario: str = None) -> ToolUsageExample:
        """
        Executes the MMLU problem-solving logic by calling the internal RAG system.
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
            # --- IMPROVED ERROR HANDLING ---
            print(f"⚠️ Warning: Could not parse a final answer from the LLM response.")
            # Return a clear message instead of None
            return {"final_answer": "Could not be determined.", "error": "Failed to parse final answer from model output.", "raw_response": response_text}

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
