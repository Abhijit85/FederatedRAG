import json
import os
import re
import requests  # <-- THIS IS THE FIX
from typing import Dict, Optional

from agenttools import BaseTool
from CompendiumBuilder import CompendiumEntry, Precaution
from vector_search import VectorSearchFilter

# --- CONFIG ---
MODEL = "llama3.1-8b-instruct"
LAMBDA_API = "https://api.lambda.ai/v1/chat/completions"


class CompendiumAwareAgent:
    """
    This agent uses a unified, persistent ChromaDB vector store to intelligently route
    a query to the correct tool with the correct data payload.
    """

    def __init__(self, tools: Dict[str, BaseTool], final_compendium_path: str = "final_compendium.json"):
        self.tools = tools
        self.vector_search_filter = VectorSearchFilter()

        api_key = os.environ.get("LAMDA_API_KEY")
        if not api_key:
            raise ValueError("LAMDA_API_KEY not found in environment.")

        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        print("\n--- Initializing Agent with Final Merged Compendium ---")
        try:
            with open(final_compendium_path, 'r') as f:
                compendium_data = json.load(f)
                self.compendium = CompendiumEntry(**compendium_data)

            self.vector_search_filter.add_scenarios_from_compendium(self.compendium, "master_toolset")
            
            print("[✓] Unified vector store populated successfully from final compendium.")
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load or parse the final compendium at '{final_compendium_path}': {e}")

    def build_dynamic_prompt(self, query: str, data_item: Optional[dict], compendium: CompendiumEntry,
                             recommended_tool: str) -> str:
        textual = compendium.Textual_Compendium
        compendium_tool_name = compendium.Structured_Annex.Relations[0].source

        if data_item and 'question' in data_item:
            problem_part = data_item.get('question', "No question provided.")
            options_part = "Choices: " + json.dumps(data_item.get('choices', []))
            context_lecture = "Context: " + data_item.get('lecture', 'No context provided.')
            user_problem_section = f"""--- USER QUERY TO SOLVE ---\n{context_lecture}\nQuestion: \"{problem_part}\"\n{options_part}"""
        else:
            problem_part = query.split("\nOptions:")[0].strip()
            options_part = "Options:" + query.split("\nOptions:")[1].strip() if "\nOptions:" in query else "No options provided."
            user_problem_section = f"""--- USER QUERY TO SOLVE ---\nProblem: \"{problem_part}\"\n{options_part}"""

        prompt_intro = f"You are an expert AI assistant acting as the '{compendium_tool_name}', specializing as a '{recommended_tool}'. {textual.Tool_Description}"
        
        protocol_steps = ""
        if textual.Solving_Protocol:
            protocol_steps = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(textual.Solving_Protocol))
        prompt_protocol = f"Based on successful prior executions, you must follow this precise problem-solving protocol:\n{protocol_steps}"
        
        prompt_precautions = "CRITICAL PRECAUTIONS:\n" + "\n".join(f"- {p.precaution}" for p in textual.Precautions)

        final_prompt = f"""{prompt_intro}
--- SOLVING PROTOCOL ---
{prompt_protocol}
--- PRECAUTIONS & RULES ---
{prompt_precautions}
{user_problem_section}
--- YOUR RESPONSE ---
Provide a step-by-step breakdown of your reasoning and calculations.
After your reasoning, you MUST conclude with the final answer in the required format.
For Math Problems: 'Final Answer: [The correct option letter, e.g., a, b, c]'
For Science Problems: A JSON object with 'reasoning' and 'answer_index' keys.
        """
        return final_prompt.strip()

    def route_query(self, query: str, data_item: Optional[dict] = None):
        candidate_docs = self.vector_search_filter.search(query, top_k=3)
        if not candidate_docs:
            print("❌ No relevant tools found.")
            return None

        print(f"🔄 Analyzing {len(candidate_docs)} tools for multi-tool coordination...")
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
            print(f"[!] Tool '{parent_tool_name}' from plan not found.")
            return None

        print(f"[+] Executing plan. Primary tool: '{recommended_sub_tool}' routed to '{parent_tool_name}'")
        dynamic_prompt = self.build_dynamic_prompt(query, data_item, self.compendium, recommended_sub_tool)

        if parent_tool_name == 'scienceqa':
            return tool_to_run.run(data_item, dynamic_prompt)
        else:
            return tool_to_run.run(query, dynamic_prompt)
      
    def _llm_analyze_for_coordination(self, query: str, candidate_docs: list) -> Optional[dict]:
        """
        Uses an LLM to act as a mission planner. It analyzes the query and candidate
        tools, then determines the correct parent tool ('mathqa' or 'scienceqa') to execute.
        """
        candidate_descriptions = "\n".join([
            f"- {doc['scenario']} (From Parent Tool: {doc['tool_name']})" for doc in candidate_docs
        ])

        prompt = f"""
        You are an AI mission planner. Your job is to analyze a user query and a list of available specialized tools, then create a final execution plan.

        **User Query:**
        "{query}"

        **Candidate Tools (all part of a unified 'master_toolset'):**
        {candidate_descriptions}

        Analyze the query and the candidate tools. Your primary task is to determine which main tool, 'mathqa' or 'scienceqa', is the correct one to handle this query. Then, create a JSON object describing your plan.

        **JSON Response Format:**
        {{
          "plan_rationale": "A brief explanation of why you chose the parent tool based on the query.",
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
