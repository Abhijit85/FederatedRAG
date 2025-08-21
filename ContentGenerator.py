# ContentGenerator.py

from typing import List
import requests
import json
from agenttools import ToolUsageExample
import os
from dotenv import load_dotenv

load_dotenv()
# --- CONFIG ---
API_KEY =os.environ.get("LAMDA_API_KEY")
MODEL = "llama3.1-8b-instruct"
LAMBDA_API = "https://api.lambda.ai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

class ContentGenerator:
    """
    Step 1 in the compendium process.
    This class generates a rich, natural language draft of the compendium
    by synthesizing knowledge from tool usage examples.
    """

    def _build_generation_prompt(self, tool_name: str, examples: List[ToolUsageExample]) -> str:
        """Builds the prompt to generate the unstructured compendium content."""
        usage_blocks = "\n\n".join(
            f"User Query: {ex.user_query}\nAssistant's Solution:\n{ex.llm_response}" for ex in examples
        )

        # Extract the expert protocol from the first example's prompt for the LLM to learn from
        expert_protocol_example = examples[0].raw_prompt

        prompt = f"""
        You are an expert technical writer tasked with creating documentation for an AI tool named '{tool_name}'.
        Your documentation should be synthesized from the provided examples of an expert AI solving problems.

        --- EXPERT EXAMPLES ---
        {usage_blocks}
        --- END EXAMPLES ---

        --- EXPERT'S INTERNAL MONOLOGUE & PROTOCOL ---
        The expert AI that solved these problems was following a strict internal protocol. Here is the prompt that guided it:
        "{expert_protocol_example}"
        --- END PROTOCOL ---

        Based on all the information above, please generate a comprehensive, multi-part document in plain text. Do NOT use JSON.

        The document must have the following four sections:

        1.  **Tool Description:**
            A concise, one-paragraph description of the tool, its purpose, and primary capabilities based on the examples.

        2.  **Usage Scenarios:**
            A list of 3-5 distinct scenarios where this tool is applicable. Each scenario should be a full sentence.

        3.  **Precautions:**
            A list of critical precautions, limitations, or rules an agent must follow when using this tool. These should be derived from the error handling and specific instructions in the expert's protocol.

        4.  **Solving Protocol:**
            Synthesize the expert's methodology into a generic, step-by-step "Solving Protocol" that a new, naive agent could follow to solve similar problems. This should be a numbered list.
        """
        return prompt.strip()

    def generate_content(self, tool_name: str, examples: List[ToolUsageExample]) -> str:
        """
        Generates the raw textual content for the compendium.

        Returns:
            A single string containing the generated text.
        """
        prompt = self._build_generation_prompt(tool_name, examples)

        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are an expert technical writer and knowledge synthesis AI."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2000
        }

        try:
            res = requests.post(LAMBDA_API, json=payload, headers=HEADERS, timeout=120)
            res.raise_for_status()
            content = res.json()["choices"][0]["message"]["content"]
            return content
        except Exception as e:
            print(f"[!] Failed during content generation for tool '{tool_name}': {e}")
            return "" 