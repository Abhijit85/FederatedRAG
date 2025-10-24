import os
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel

from openrouter_client import chat_completion, get_available_api_keys

# -------------- CONFIG --------------
load_dotenv()
if not get_available_api_keys(allow_empty=True) and not (
    os.environ.get("LAMBDA_API_KEY") or os.environ.get("LAMDA_API_KEY")
):
    raise ValueError("At least one API_KEY environment variable must be set.")
MODEL = "llama3.1-8b-instruct"

# -------------- PYDANTIC MODEL FOR TOOL USAGE LOGGING --------------
class ToolUsageExample(BaseModel):
    tool_name: str
    user_query: str
    raw_prompt: str
    llm_response: str
    parsed_output: Optional[dict] = None


# -------------- BASE TOOL WRAPPER --------------
class BaseTool:
    def __init__(self, name: str):
        self.name = name

    def run(self, user_query: str, dynamic_prompt: Optional[str]=None) -> ToolUsageExample:
        raise NotImplementedError


# -------------- WEATHER TOOL --------------
class WeatherTool(BaseTool):
    def __init__(self):
        super().__init__("weather")

    def run(self, user_query: str, dynamic_prompt: Optional[str]=None) -> ToolUsageExample:
        prompt_precautions = "If the city name isn't recognized or not given, you must ask the user for the specific region or a more precise location."
        prompt_multi_tool = (
            "Based on the weather conditions, if there's heavy rain or strong winds, recommend sending a severe weather alert. "
            "If travel could be impacted, suggest checking for alternative routes or advisories."
        )

        consolidated_prompt = f"""
            User Query: {user_query}

            Instructions for future queries and follow-up actions:
            - **Clarification:** {prompt_precautions}
            - **Alerts & Travel:** {prompt_multi_tool}

            Please provide the weather report first, and finally acknowledge the instructions for future reference.
        """

        completion = chat_completion(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a highly capable AI assistant. Always be helpful and follow all instructions precisely."},
                {"role": "user", "content": consolidated_prompt},
            ],
            max_tokens=1000,
        )
        response = completion.choices[0].message.content
        print(type(response), response)
        return ToolUsageExample(
            tool_name=self.name,
            user_query=user_query,
            raw_prompt=consolidated_prompt,
            llm_response=response
        )


# -------------- CALCULATOR TOOL --------------
class CalculatorTool(BaseTool):
    def __init__(self):
        super().__init__("calculator")

    def run(self, user_query: str, dynamic_prompt: Optional[str]=None) -> ToolUsageExample:
        prompt_precautions = """
            If the question cannot be solved as a well-defined math problem, you must follow these rules:
            - IF the problem is ambiguous or missing data, THEN respond: "Insufficient information. Please provide all necessary numbers and details."
            - IF the problem involves division by zero, THEN respond: "Error: Division by zero is not a valid operation."
            - IF the problem involves inconsistent or unclear units (e.g., mixing meters and feet without conversion), THEN ask for a conversion factor before proceeding.
            - IF the problem does not match a math word problem format (e.g., casual or unrelated text), THEN request a clear math-related query.
        """

        prompt_mathqa_protocol = """
            You are a math word problem solving assistant. Follow these steps strictly:
            1. Read the problem carefully, identify known values, unknowns, and relevant units.
            2. Translate the problem into a step-by-step solution plan, using operations of the following style :
            Add(a,b), Subtract(a,b), Multiply(a,b), Divide(a,b), Power(a,b), SquareRoot(a), Percentage(a,b), etc.
            3. Compute each step explicitly, showing intermediate results.
            4. Check that units are consistent across all operations.
            5. Independently verify the final result by recomputing it from scratch.
            6. Output format: Final Answer: [final numeric value or option letter]
            7. Do not skip any step, even if obvious.
            8. Avoid irrelevant or conversational text — focus solely on reasoning and answer.

            Always ensure the output format is very strictly followed. Do NOT output any text except for the final answer.
        """


        consolidated_prompt = f"""
            User Query: {user_query}
            
            Instructions:
            - **MathQA Problem Solving:** {prompt_mathqa_protocol}
        """
        if(dynamic_prompt):
           consolidated_prompt=dynamic_prompt
           print(dynamic_prompt)

        completion = chat_completion(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that functions as a calculator. You must follow all user instructions precisely, including the detailed protocols for handling errors and suggesting follow-up actions. Perform the calculation the user requests and provide the numerical answer."},
                {"role": "user", "content": consolidated_prompt},
            ],
            max_tokens=2000,
        )
        response = completion.choices[0].message.content
        print(type(response), response)
        return ToolUsageExample(
            tool_name=self.name,
            user_query=user_query,
            raw_prompt=consolidated_prompt,
            llm_response=response
        )
