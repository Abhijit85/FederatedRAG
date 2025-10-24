import json
import os
import re
import logging
import sys
import base64
from dotenv import load_dotenv

# Import the new manager and all necessary classes
# Note: Ensure these files (CompendiumManager.py, etc.) are in the same directory.
from CompendiumManager import CompendiumManager
from CompendiumAwareAgent import CompendiumAwareAgent
from math_qa import MathQATool
from science_qa import ScienceQATool
from CompendiumBuilder import CompendiumEntry
from openrouter_client import get_available_api_keys
from jina_key_manager import get_available_jina_api_keys

# Load environment variables from a .env file at the start
load_dotenv()

# --- 1. LOGGING SETUP ---
# This class will redirect all print() statements to the logger
class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message != '\n': # avoid logging empty lines
            self.level(message.strip())

    def flush(self):
        pass # Required for file-like object

def setup_logging(username):
    """Configures the logger to write to a user-specific file and the console."""
    log_formatter = logging.Formatter('%(message)s') # Keep the log clean
    logger = logging.getLogger()
    
    # Clear existing handlers to avoid duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.setLevel(logging.INFO)

    # User-specific file handler
    log_filename = f"evaluation_log_{username}.txt"
    file_handler = logging.FileHandler(log_filename, mode='w') # 'w' to overwrite file each run
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    # Redirect stdout and stderr to the logger
    sys.stdout = LoggerWriter(logger.info)
    sys.stderr = LoggerWriter(logger.error)
    
    print(f"[✓] Logging setup complete. Output will be saved to '{log_filename}'")
    return logger

def get_user_input(prompt):
    """Gets input from the user, bypassing the logger redirection for the prompt."""
    # Temporarily restore stdout to get raw input
    original_stdout = sys.stdout
    sys.stdout = sys.__stdout__
    try:
        user_input = input(prompt)
    finally:
        # Restore the logger
        sys.stdout = original_stdout
    return user_input

def encode_image_to_base64(image_path):
    """Encodes an image file to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded_string}"
    except FileNotFoundError:
        print(f"❌ Error: Image file not found at '{image_path}'.")
        return None
    except Exception as e:
        print(f"❌ Error encoding image: {e}")
        return None

def interactive_session(agent):
    """
    Handles the interactive user session for submitting and evaluating queries.
    """
    print("\n--- 3. STARTING INTERACTIVE AGENT SESSION ---")
    
    while True:
        query_type = get_user_input("Enter query type ('math' or 'science'), or 'exit' to quit: ").lower().strip()

        if query_type == 'exit':
            print("Exiting interactive session. Goodbye!")
            break
        
        query_data = {}
        query_text = ""
        data_payload = None

        if query_type == 'math':
            print("\n--- Building MathQA Query ---")
            problem = get_user_input("Enter the problem statement: ")
            options = get_user_input("Enter the options (e.g., a) 1, b) 2, ...): ")
            
            query_data = {
                "Problem": problem,
                "options": options,
                # Add other fields as needed, or leave them empty for inference
                "Rationale": "", "correct": "", "annotated_formula": "", 
                "linear_formula": "", "category": "general"
            }
            query_text = f"{problem}\nOptions: {options}"
            data_payload = None # Math tool doesn't need the extra payload

        elif query_type == 'science':
            print("\n--- Building ScienceQA Query ---")
            question = get_user_input("Enter the question: ")
            choices_str = get_user_input("Enter the choices, separated by a semicolon (;): ")
            choices = [choice.strip() for choice in choices_str.split(';')]
            hint = get_user_input("Enter a hint or context (optional, press Enter to skip): ")
            image_path = get_user_input("Enter the file path for an image (optional, press Enter to skip): ")

            image_data = None
            if image_path:
                image_data = encode_image_to_base64(image_path)

            query_data = {
                "question": question,
                "choices": choices,
                "hint": hint,
                "image": image_data if image_data else "",
                # Add other fields as needed
                "answer": -1, "task": "closed choice", "grade": "grade8",
                "subject": "natural science", "topic": "science-and-engineering-practices",
                "category": "Engineering practices", "skill": "Evaluate tests of engineering-design solutions",
                "lecture": "", "solution": ""
            }
            query_text = question
            data_payload = query_data
        
        else:
            print("❌ Invalid query type. Please enter 'math' or 'science'.")
            continue

        # Save the user-generated query to a file for review
        with open("last_user_query.json", "w", encoding='utf-8') as f:
            json.dump(query_data, f, indent=4)
        print("[✓] User query saved to 'last_user_query.json'")

        # --- Execute the agent's routing logic ---
        print(f"\n--- Processing {query_type.upper()} Query: '{query_text[:80]}...' ---")
        result = agent.route_query(query=query_text, data_item=data_payload)
        
        if result and result.llm_response:
            print("\n" + "="*25 + " AGENT REASONING & FINAL OUTPUT " + "="*25)
            print(result.llm_response)
            print("="*80)
            print("[✓] Query processed successfully.")
        else:
            print("[✗] Agent failed to produce a final result for this query.")

        print("\n" + "-"*80) # Separator for the next query

def main():
    """
    Main function to orchestrate the entire pipeline:
    1. Get user identity and set up logging.
    2. Manage Compendiums (Merge, Deduplicate).
    3. Initialize the Agent with the final knowledge base.
    4. Start the interactive evaluation session.
    """
    # --- PRE-INITIALIZATION ---
    username = get_user_input("Please enter your name for the session log: ").strip().replace(" ", "_")
    if not username:
        username = "default_user"
    
    logger = setup_logging(username)

    available_lambda_keys = get_available_api_keys(allow_empty=True)
    lambda_key = available_lambda_keys[0] if available_lambda_keys else None
    if not lambda_key:
        lambda_key = os.environ.get("LAMBDA_API_KEY") or os.environ.get("LAMDA_API_KEY")
        if lambda_key:
            print("⚠️ Detected deprecated env var for Lambda API key. Please rename it to API_KEY or API_KEY_<n>.")
    jina_keys = get_available_jina_api_keys(allow_empty=True)
    jina_key = jina_keys[0] if jina_keys else ""
    if jina_key:
        os.environ["JINA_API_KEY"] = jina_key
    if not lambda_key or not jina_key:
        print("❌ Error: API keys must be set in your .env file.")
        return
    
    # --- PHASE 1: COMPENDIUM MANAGEMENT ---
    print("\n--- 1. MANAGING KNOWLEDGE COMPENDIUMS ---")
    compendium_manager = CompendiumManager()
    
    source_files = ["mathqa_tools_compendium.json", "scienceqa_tools_compendium.json"]
    final_compendium_path = "final_compendium.json"
    final_compendium = compendium_manager.merge_compendiums(source_files, final_compendium_path)

    if not final_compendium:
        print("❌ Critical error: Could not create the final compendium. Aborting.")
        return
        
    # --- PHASE 2: AGENT INITIALIZATION ---
    print("\n--- 2. INITIALIZING COMPENDIUM-AWARE AGENT ---")
    tool_map = {"mathqa": MathQATool(), "scienceqa": ScienceQATool()}
    
    agent = CompendiumAwareAgent(tools=tool_map, final_compendium_path=final_compendium_path)

    # --- PHASE 3: INTERACTIVE EVALUATION ---
    interactive_session(agent)

if __name__ == "__main__":
    main()
