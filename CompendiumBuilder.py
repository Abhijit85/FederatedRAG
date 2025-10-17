# CompendiumBuilder.py

from typing import List, Optional
from pydantic import BaseModel
import json
import os
from dotenv import load_dotenv
from openrouter_client import chat_completion, get_openrouter_client

# --- CONFIG ---
load_dotenv()
get_openrouter_client()
MODEL = "llama3.1-8b-instruct"

# --- FIX: Define models for the nested objects to match the JSON structure ---
class Scenario(BaseModel):
    scenario: str
    context: str

class Precaution(BaseModel):
    precaution: str

class Example(BaseModel):
    example: str

class MultiToolCoordination(BaseModel):
    Description: str
    Examples: List[Example] # Use the new 'Example' model

class TextualCompendium(BaseModel):
    Tool_Description: str
    Usage_Scenarios: List[Scenario] # Use the new 'Scenario' model
    Precautions: List[Precaution] # Use the new 'Precaution' model
    # FIX: Make 'Solving_Protocol' optional, as it's missing from the merged JSON
    Solving_Protocol: Optional[List[str]] = None 
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
        # Lazily import to prevent circular dependencies
        from vector_search import VectorSearchFilter
        self.vector_search_filter = VectorSearchFilter()

    def structure_content(self, tool_name: str, raw_content: str) -> CompendiumEntry | None:
        """
        Takes raw textual content and structures it into the CompendiumEntry model
        using an LLM call.
        """
        prompt = self._get_structuring_prompt(tool_name, raw_content)
        try:
            response = chat_completion(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            response_json = response.choices[0].message.content
            
            structured_data = json.loads(response_json)
            compendium_entry = CompendiumEntry(**structured_data)

            print(f"[✓] Successfully structured content for tool: '{tool_name}'")
            return compendium_entry
        except Exception as e:
            print(f"[!] Failed to structure or validate content for '{tool_name}': {e}")
            return None

    def _get_structuring_prompt(self, tool_name: str, raw_content: str) -> str:
        """Generates the detailed prompt for the LLM to structure the compendium."""
        # Note: While this prompt asks for a simpler structure, the Pydantic models
        # are now robust enough to handle the more complex, correct structure.
        prompt = f"""
        You are an AI tasked with creating a structured knowledge compendium for a tool named '{tool_name}'.
        Analyze the provided raw content and transform it into a valid JSON object.

        Raw Content to Analyze:
        ---
        {raw_content}
        ---
        
        Your output MUST be a single, valid JSON object matching this structure:
        {{
            "Textual_Compendium": {{
                "Tool_Description": "...",
                "Usage_Scenarios": [{{ "scenario": "...", "context": "..." }}],
                "Precautions": [{{ "precaution": "..." }}],
                "Multi_tool_Coordination": {{
                    "Description": "...",
                    "Examples": [{{ "example": "..." }}]
                }}
            }},
            "Structured_Annex": {{
                "Entities": ["..."],
                "Relations": [{{ "source": "...", "link": "...", "target": "..." }}]
            }}
        }}
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
