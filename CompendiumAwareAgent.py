import json
import os
import requests
from typing import Dict, Optional

from agenttools import BaseTool
from mongo_utils import MongoVectorStore

# --- CONFIGURATION ---
MODEL = "llama3.1-8b-instruct"
LAMBDA_API = "https://api.lambda.ai/v1/chat/completions"
JINA_EMBED_API_URL = "https://api.jina.ai/v1/embeddings"
MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME = "FredRag"
COLLECTION_NAME = "vectors"

class JinaAIClient:
    """A client to interact with Jina AI APIs for embeddings."""
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("JINA_API_KEY not found in environment.")
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def get_embedding(self, text):
        """Generates an embedding for a single text."""
        try:
            response = requests.post(
                JINA_EMBED_API_URL, headers=self.headers,
                json={"model": "jina-embeddings-v2-base-en", "input": [text]}
            )
            response.raise_for_status()
            return response.json()['data'][0]['embedding']
        except requests.exceptions.RequestException as e:
            print(f"Error getting embedding from Jina AI: {e}")
            return None

class CompendiumAwareAgent:
    """
    This agent uses a unified, persistent MongoDB vector store to intelligently route
    a query to the correct tool with the correct data payload.
    """

    def __init__(self, tools: Dict[str, BaseTool]):
        self.tools = tools
        self.vector_store = MongoVectorStore(MONGO_URI, DB_NAME, COLLECTION_NAME)

        lamda_api_key = os.environ.get("LAMDA_API_KEY")
        jina_api_key = os.environ.get("JINA_API_KEY")

        if not lamda_api_key:
            raise ValueError("LAMDA_API_KEY not found in environment.")
        
        self.jina_client = JinaAIClient(jina_api_key)
        self.headers = {
            "Authorization": f"Bearer {lamda_api_key}",
            "Content-Type": "application/json"
        }
        print("\n--- Compendium-Aware Agent Initialized ---")
        print(f"[✓] Connected to MongoDB vector store in database '{DB_NAME}'.")


    def _get_candidate_scenarios(self, query: str) -> str:
        """
        Uses vector search to find the most relevant tool scenarios from MongoDB.
        """
        print(f"Searching for scenarios related to: '{query[:50]}...'")
        try:
            query_vector = self.jina_client.get_embedding(query)
            if not query_vector:
                print("[!] Could not generate a query vector.")
                return "No relevant scenarios found."

            # Call the manual search function for testing purposes.
            results = self.vector_store.search_manual(query_vector, num_results=5)
            
            if not results:
                print("[!] No results from vector search.")
                return "No relevant scenarios found."

            scenarios = [res['text'] for res in results]
            print(f"✅ Found {len(scenarios)} candidate scenarios.")
            return "\n".join(f"- {s}" for s in scenarios)
        except Exception as e:
            print(f"[!] Vector search failed: {e}")
            return "No relevant scenarios found."

    def route_query(self, query: str, data_item: Optional[Dict] = None) -> Optional[Dict]:
        """
        The main query routing logic.
        """
        print(f"\n--- Routing Query: '{query[:50]}...' ---")
        
        candidate_scenarios = self._get_candidate_scenarios(query)
        
        coordination_plan = self._analyze_with_llm(query, candidate_scenarios)

        if not coordination_plan:
            print("[!] Could not determine an execution plan.")
            return None

        parent_tool_name = coordination_plan.get("primary_tool", {}).get("parent_tool_name")

        if parent_tool_name and parent_tool_name in self.tools:
            selected_tool = self.tools[parent_tool_name]
            print(f"--> Executing with tool: '{parent_tool_name}'")
            
            if data_item:
                return selected_tool.run(user_query=query, data_item=data_item)
            else:
                return selected_tool.run(user_query=query)
        else:
            print(f"[!] Tool '{parent_tool_name}' not found or invalid plan.")
            return None

    def _analyze_with_llm(self, query: str, candidate_descriptions: str) -> Optional[Dict]:
        """
        Uses an LLM to decide which tool to use based on the query and retrieved scenarios.
        """
        # --- THIS IS THE FIX ---
        # The prompt is now more explicit, telling the LLM to prioritize the retrieved scenarios.
        prompt = f"""
        You are an expert AI routing system. Your task is to analyze a user's query and a list of relevant tool scenarios to create an execution plan.

        **Critical Instruction:**
        You MUST give strong preference to the tool suggested by the most relevant 'Candidate Tools'. Do not default to a general tool if a specialized scenario is a good match. Your primary goal is to honor the context provided by the vector search.

        **User Query:**
        "{query}"

        **Candidate Tools (retrieved from a vector search):**
        {candidate_descriptions}

        Analyze the query and the candidate tools. Your primary task is to determine which main tool, 'mathqa' or 'scienceqa', is the correct one to handle this query based on the scenarios. Then, create a JSON object describing your plan.

        **JSON Response Format:**
        {{
          "plan_rationale": "A brief explanation of why you chose the parent tool, referencing the most relevant candidate scenario.",
          "primary_tool": {{
            "scenario_name": "The full name of the best matching tool scenario from the candidate list.",
            "parent_tool_name": "The final parent tool name. This value MUST be either 'mathqa' or 'scienceqa'."
          }}
        }}

        Respond with ONLY the valid JSON object.
        """
        
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "response_format": {"type": "json_object"} 
        }

        try:
            res = requests.post(LAMBDA_API, json=payload, headers=self.headers)
            res.raise_for_status()
            response_text = res.json()['choices'][0]['message']['content']
            return json.loads(response_text)
        except Exception as e:
            print(f"[!] LLM coordination analysis failed: {e}")
            return None
