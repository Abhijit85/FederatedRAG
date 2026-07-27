import json
import os
import re
from typing import Dict, Optional, List

from agenttools import BaseTool
from vector_search import VectorSearchFilter
from openrouter_client import chat_completion, get_openrouter_client

# --- CONFIGURATION ---
MODEL = os.environ.get("RERANK_MODEL", "llama3.1-8b-instruct")

class CompendiumAwareAgent:
    """
    An agent that uses a two-step LLM process (Rerank and Analyze) to route
    a user's query to the most appropriate tool.
    """
    def __init__(self, tools: List[BaseTool]):
        """
        Initializes the agent with a list of tools.
        """
        # Convert the list of tools into a dictionary for easy, name-based lookup.
        self.tools = {tool.name: tool for tool in tools}
        self.vector_search_filter = VectorSearchFilter()
        
        # Ensure the OpenRouter client can be constructed early.
        get_openrouter_client()

    def route_query(self, query: str, data_item: Optional[dict] = None):
        """
        Routes the user's query using a Retrieve -> Rerank -> Analyze chain.
        """
        # 1. RETRIEVE: Get the top candidate scenarios from the vector database.
        # We retrieve a few more than needed (e.g., 5) to give the LLM better context.
        candidate_docs = self.vector_search_filter.search(query, top_k=5)
        if not candidate_docs:
            print("❌ No relevant tools found in vector search.")
            return None

        # 2. RERANK: Use an LLM to find the single best matching scenario.
        # This is useful for precise logging and can be used for feedback loops.
        print(f"🔄 Reranking {len(candidate_docs)} scenarios for relevance...")
        best_scenario = self._llm_rerank_and_select(query, candidate_docs)
        if best_scenario:
            print(f"✅ Reranking complete. Best match: '{best_scenario[:100]}...'")
        else:
            print("[!] LLM reranker failed to select a scenario. Proceeding with analysis of all candidates.")

        # 3. ANALYZE: Use an LLM with the full list of candidates to create a robust execution plan.
        # This step makes the final decision on which parent tool to use.
        print(f"🔄 Analyzing scenarios to create an execution plan...")
        execution_plan = self._llm_analyze_for_coordination(query, candidate_docs)
        if not execution_plan:
            print("[!] LLM could not devise an execution plan. Aborting.")
            return None
        print("✅ LLM created an execution plan.")

        primary_tool_info = execution_plan.get("primary_tool")
        if not primary_tool_info:
            print("[!] Plan is missing a primary tool.")
            return None
            
        parent_tool_name = primary_tool_info.get("parent_tool_name")
        recommended_sub_tool = primary_tool_info.get("scenario_name")

        tool_to_run = self.tools.get(parent_tool_name)
        if not tool_to_run:
            print(f"[!] Tool '{parent_tool_name}' from plan not found in the agent's toolset.")
            return None

        print(f"[+] Executing plan. Routing to tool: '{parent_tool_name}' (Recommended scenario: '{recommended_sub_tool}')")

        # 4. EXECUTE: Run the selected tool and return its result and the recommended sub-tool.
        result = tool_to_run.run(user_query=query, data_item=data_item, recommended_scenario=recommended_sub_tool)
        return result, recommended_sub_tool

    def _llm_rerank_and_select(self, query: str, candidate_docs: list) -> Optional[str]:
        """Uses a focused LLM call to select the single best tool scenario from a list."""
        candidate_descriptions = "\n".join([f"- Tool Option {i+1}: \"{doc}\"" for i, doc in enumerate(candidate_docs)])
        
        prompt = f"""
        You are an expert tool router. Your task is to select the single most appropriate tool scenario for the given user query from the provided list of options.

        **User Query:**
        "{query}"

        **Candidate Tool Scenarios:**
        {candidate_descriptions}

        Analyze the query and the tool descriptions carefully. Respond with ONLY the number of the best tool option (e.g., "1", "2", "3").
        """
        
        try:
            completion = chat_completion(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
            )
            response_text = completion.choices[0].message.content.strip()
            
            match = re.search(r'\d+', response_text)
            if match:
                selected_index = int(match.group(0)) - 1
                if 0 <= selected_index < len(candidate_docs):
                    return candidate_docs[selected_index]
            return None
        except Exception as e:
            print(f"[!] LLM reranking failed: {e}")
            return None

    def _llm_analyze_for_coordination(self, query: str, candidate_docs: list) -> Optional[dict]:
        """Uses a broader LLM call to analyze all candidates and decide on the final parent tool."""
        candidate_descriptions = "\n".join([f"- \"{doc}\"" for doc in candidate_docs])
        
        # MODIFIED: Prompt is refined for clarity and better decision-making.
        prompt = f"""
        You are an AI mission planner. Your job is to analyze a user query and a list of retrieved tool scenarios to create a final execution plan.

        **User Query:**
        "{query}"

        **Candidate Tool Scenarios (retrieved from a vector search):**
        {candidate_descriptions}

        Your primary task is to determine which main tool, 'mathqa' or 'scienceqa' or 'mmluqa' or 'truthfulqa', is the correct one to handle this query, based on which tool owns the most relevant scenario from the candidate list. Then, create a JSON object describing your plan.

        **JSON Response Format:**
        {{
          "plan_rationale": "A brief explanation of why you chose the parent tool, referencing the most relevant candidate scenario.",
          "primary_tool": {{
            "scenario_name": "The full name of the best matching tool scenario from the candidate list.",
            "parent_tool_name": "The final parent tool name. This value MUST be 'mathqa' or 'scienceqa' or 'mmluqa' or 'truthfulqa'."
          }}
        }}

        Respond with ONLY the valid JSON object.
        """
        
        try:
            completion = chat_completion(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            response_text = completion.choices[0].message.content
            return json.loads(response_text)
        except Exception as e:
            print(f"[!] LLM coordination analysis failed: {e}")
            return None
