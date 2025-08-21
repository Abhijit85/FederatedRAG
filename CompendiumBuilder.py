# CompendiumBuilder.py

from typing import List
from pydantic import BaseModel
import requests
import json
import re
import os
from dotenv import load_dotenv

# --- FIX: Import VectorSearchFilter inside the class method where it's used ---
# This breaks the circular import loop at the module level.
# from vector_search import VectorSearchFilter # REMOVE THIS LINE

# --- CONFIG ---
load_dotenv()
API_KEY = os.environ.get("LAMDA_API_KEY")
MODEL = "llama3.1-8b-instruct"
LAMBDA_API = "https://api.lambda.ai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

class MultiToolCoordination(BaseModel):
    Description: str
    Examples: List[str]

class TextualCompendium(BaseModel):
    Tool_Description: str
    Usage_Scenarios: List[str]
    Precautions: List[str]
    Solving_Protocol: List[str]
    Multi_tool_Coordination: MultiToolCoordination

class StructuredRelation(BaseModel):
    source: str
    link: str
    target: str

class StructuredAnnex(BaseModel):
    Entities: List[str]
    Relations: List[StructuredRelation]

class CompendiumEntry(BaseModel):
    Textual_Compendium: TextualCompendium
    Structured_Annex: StructuredAnnex


class CompendiumBuilder:
    
    def __init__(self):
        # --- FIX: Import here as well ---
        from vector_search import VectorSearchFilter
        self.vector_search_filter = VectorSearchFilter()

    def structure_content(self, tool_name: str, raw_content: str) -> CompendiumEntry | None:
        """
        Takes raw textual content and structures it into the CompendiumEntry model
        using an LLM call.
        """
        prompt = self._get_structuring_prompt(tool_name, raw_content)
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
            "response_format": {"type": "json_object"}
        }

        try:
            res = requests.post(LAMBDA_API, json=payload, headers=HEADERS)
            res.raise_for_status()
            response_json = res.json()['choices'][0]['message']['content']
            
            # The API often returns JSON as a string, so we need to parse it.
            structured_data = json.loads(response_json)
            
            # Validate with Pydantic
            compendium_entry = CompendiumEntry(**structured_data)
            print(f"[✓] Successfully structured content for tool: '{tool_name}'")
            return compendium_entry
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            print(f"[!] Failed to structure content for '{tool_name}': {e}")
            return None
        except Exception as pydantic_error:
            print(f"[!] Pydantic validation failed for '{tool_name}': {pydantic_error}")
            return None

    def _get_structuring_prompt(self, tool_name: str, raw_content: str) -> str:
        """Generates the detailed prompt for the LLM to structure the compendium."""
        prompt = f"""
        You are a sophisticated AI tasked with creating a structured knowledge compendium entry for a tool named '{tool_name}'.
        Analyze the provided raw content and transform it into a valid JSON object that strictly adheres to the following Pydantic model schema.

        Raw Content to Analyze:
        ---
        {raw_content}
        ---

        Your output MUST be a single, valid JSON object matching this structure:
        {{
            "Textual_Compendium": {{
                "Tool_Description": "A comprehensive summary of the tool's purpose and capabilities.",
                "Usage_Scenarios": [
                    "A specific, concrete example of a problem this tool would solve.",
                    "Another distinct usage scenario or type of query it handles."
                ],
                "Precautions": [
                    "A critical limitation or rule to follow when using this tool (e.g., 'Do not use for non-mathematical logic puzzles').",
                    "Another important operational constraint."
                ],
                "Solving_Protocol": [
                    "Step 1 of the ideal process for using this tool.",
                    "Step 2, continuing the logical flow.",
                    "Step 3, leading to the final output."
                ],
                "Multi_tool_Coordination": {{
                    "Description": "How this tool works with other tools in the system.",
                    "Examples": ["An example of a multi-tool workflow involving '{tool_name}'."]
                }}
            }},
            "Structured_Annex": {{
                "Entities": [
                    "Primary concepts and objects related to the tool (e.g., 'Word_Problem', 'Mathematical_Formula')."
                ],
                "Relations": [
                    {{
                        "source": "{tool_name}",
                        "link": "performs",
                        "target": "Extract the primary action of the tool (e.g., 'mathematical calculations')."
                    }}
                ]
            }}
        }}

        Respond ONLY with the valid JSON object. Do not include any other text, markdown formatting, or explanations.
        """
        return prompt.strip()

    def structure_and_filter_content(self, tool_name: str, raw_content: str, existing_compendiums: List[CompendiumEntry], similarity_threshold=0.80):
        """
        Structures content and filters it based on similarity to existing compendiums.
        """
        # First, structure the content to get the tool description
        new_compendium_entry = self.structure_content(tool_name, raw_content)

        if not new_compendium_entry:
            return None

        # Now, check for similarity before adding
        new_tool_description = new_compendium_entry.Textual_Compendium.Tool_Description
        max_similarity = self.vector_search_filter.check_similarity(new_tool_description, existing_compendiums)

        print(f"Similarity score for '{tool_name}': {max_similarity:.2f}")

        if max_similarity >= similarity_threshold:
            print(f"[!] Tool '{tool_name}' is too similar to existing tools. Skipping.")
            return None
        
        print(f"[✓] Tool '{tool_name}' passed similarity check and will be added.")
        return new_compendium_entry