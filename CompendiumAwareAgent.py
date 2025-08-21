from typing import List, Dict, Optional
from agenttools import BaseTool, ToolUsageExample
from CompendiumBuilder import CompendiumEntry
import re
import requests
import json
import os
from vector_search import VectorSearchFilter

# --- CONFIG ---
MODEL = "llama3.1-8b-instruct"
LAMBDA_API = "https://api.lambda.ai/v1/chat/completions"

class CompendiumAwareAgent:
    """
    This agent uses a unified vector store to intelligently route a query
    to the correct tool (e.g., MathQA or ScienceQA) with the correct data payload.
    """
    def __init__(self, tools: Dict[str, BaseTool], compendium_map: Dict[str, CompendiumEntry]):
        self.tools = tools
        self.compendium_map = compendium_map
        self.vector_search_filter = VectorSearchFilter()

        api_key = os.environ.get("LAMDA_API_KEY")
        if not api_key:
            raise ValueError("LAMDA_API_KEY not found in environment.")

        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Build the Unified Vector Store
        print("\n--- Populating Unified Vector Store from All Compendiums ---")
        for tool_name, compendium in self.compendium_map.items():
            self.vector_search_filter.add_scenarios_from_compendium(compendium, tool_name)
        print("[✓] Unified vector store populated successfully.")

    def build_dynamic_prompt(self, query: str, data_item: Optional[dict], compendium: CompendiumEntry, recommended_tool: str) -> str:
        textual = compendium.Textual_Compendium
        compendium_tool_name = compendium.Structured_Annex.Relations[0].source
        
        if data_item and 'question' in data_item:
            # ScienceQA Format
            problem_part = data_item.get('question', "No question provided.")
            options_part = "Choices: " + json.dumps(data_item.get('choices', []))
            context_lecture = "Context: " + data_item.get('lecture', 'No context provided.')
            user_problem_section = f"""--- USER QUERY TO SOLVE ---\n{context_lecture}\nQuestion: \"{problem_part}\"\n{options_part}"""
        else:
            # MathQA Format
            problem_part = query.split("\nOptions:")[0].strip()
            options_part = "Options:" + query.split("\nOptions:")[1].strip() if "\nOptions:" in query else "No options provided."
            user_problem_section = f"""--- USER QUERY TO SOLVE ---\nProblem: \"{problem_part}\"\n{options_part}"""

        prompt_intro = f"You are an expert AI assistant acting as the '{compendium_tool_name}', specializing as a '{recommended_tool}'. {textual.Tool_Description}"
        protocol_steps = "\n".join(f"{i+1}. {step}" for i, step in enumerate(textual.Solving_Protocol))
        prompt_protocol = f"Based on successful prior executions, you must follow this precise problem-solving protocol:\n{protocol_steps}"
        prompt_precautions = "CRITICAL PRECAUTIONS:\n" + "\n".join(f"- {p}" for p in textual.Precautions)

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
        """
        Routes the query using a Retrieve and Rerank chain.
        """
        # 1. RETRIEVE top-k candidates
        candidate_docs = self.vector_search_filter.search(query, top_k=3)
        if not candidate_docs:
            print("❌ No relevant tools found in vector search.")
            return None

        # 2. RERANK the candidates using an LLM
        print(f"🔄 Reranking {len(candidate_docs)} documents for relevance...")
        chosen_doc = self._llm_rerank_and_select(query, candidate_docs)
        if not chosen_doc:
            print("[!] LLM reranker failed to select a tool. Falling back to top vector search result.")
            chosen_doc = candidate_docs[0]
        
        print("✅ Reranking complete.")

        parent_tool_name = chosen_doc['tool_name']
        recommended_sub_tool = chosen_doc['scenario']

        tool_to_run = self.tools.get(parent_tool_name)
        chosen_tool_compendium = self.compendium_map.get(parent_tool_name)

        if not tool_to_run or not chosen_tool_compendium:
            print(f"[!] No matching tool or compendium found for '{parent_tool_name}'")
            return None
        
        compendium_tool_name = chosen_tool_compendium.Structured_Annex.Relations[0].source
        print(f"[+] Routing to tool: '{compendium_tool_name}' (normalized: '{parent_tool_name}')")
        print(f"🤔 Recommended Tool based on Reranking: {recommended_sub_tool}")

        dynamic_prompt = self.build_dynamic_prompt(query, data_item, chosen_tool_compendium, recommended_sub_tool)

        # Execute with the correct data payload
        if parent_tool_name == 'scienceqa':
            if not data_item:
                print(" [!] Routing error: A non-science query was routed to ScienceQATool. Aborting.")
                return None
            return tool_to_run.run(data_item, dynamic_prompt)
        else:
            return tool_to_run.run(query, dynamic_prompt)

    def _llm_rerank_and_select(self, query: str, candidate_docs: list) -> Optional[dict]:
        """Uses an LLM to select the single best tool from a list of candidates."""
        
        # Format the candidate tools for the prompt
        candidate_descriptions = "\n".join([
            f"- Tool Option {i+1}: \"{doc['scenario']}\""
            for i, doc in enumerate(candidate_docs)
        ])

        prompt = f"""
        You are an expert tool router. Your task is to select the single most appropriate tool for the given user query from the provided list of options.

        **User Query:**
        "{query}"

        **Candidate Tool Options:**
        {candidate_descriptions}

        Analyze the query and the tool descriptions carefully. Respond with ONLY the number of the best tool option (e.g., "1", "2", "3").
        """
        
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 5
        }
        
        try:
            res = requests.post(LAMBDA_API, json=payload, headers=self.headers)
            res.raise_for_status()
            response_text = res.json()['choices'][0]['message']['content'].strip()
            
            # Find the first number in the response
            match = re.search(r'\d+', response_text)
            if match:
                selected_index = int(match.group(0)) - 1
                if 0 <= selected_index < len(candidate_docs):
                    return candidate_docs[selected_index]
            return None
        except Exception as e:
            print(f"[!] LLM reranking failed: {e}")
            return None
      