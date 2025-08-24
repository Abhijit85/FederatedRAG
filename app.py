import streamlit as st
import json
import os
import logging
import uuid
import base64
from dotenv import load_dotenv
from CompendiumManager import CompendiumManager
from CompendiumAwareAgent import CompendiumAwareAgent
from math_qa import MathQATool
from science_qa import ScienceQATool

# --- 1. APP CONFIGURATION & INITIALIZATION ---

# Load environment variables from a .env file
load_dotenv()

# Configure the Streamlit page for a professional and user-friendly layout
st.set_page_config(
    page_title="Compendium-Aware AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 2. LOGGING SETUP FOR MULTI-USER SUPPORT ---

def setup_user_logger(session_id):
    """
    Creates and configures a unique logger for each user session.
    This ensures that logs from concurrent users are saved to separate files.
    """
    log_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger = logging.getLogger(session_id)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    log_file = f"evaluation_log_{session_id}.txt"
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    
    return logger, log_file

# --- 3. AGENT INITIALIZATION (CACHED FOR EFFICIENCY) ---

@st.cache_resource
def initialize_agent():
    """
    Initializes the Compendium-Aware Agent. This function is cached
    to ensure the agent is loaded only once, improving performance.
    """
    if not os.environ.get("LAMDA_API_KEY") or not os.environ.get("JINA_API_KEY"):
        st.error("❌ Error: API keys must be set in your .env file.")
        return None

    compendium_manager = CompendiumManager()
    source_files = ["mathqa_tools_compendium.json", "scienceqa_tools_compendium.json"]
    final_compendium_path = "final_compendium.json"
    compendium_manager.merge_compendiums(source_files, final_compendium_path)

    tool_map = {"mathqa": MathQATool(), "scienceqa": ScienceQATool()}
    agent = CompendiumAwareAgent(tools=tool_map, final_compendium_path=final_compendium_path)
    
    return agent

# --- 4. STREAMLIT UI LAYOUT ---

st.title("🔬 Compendium-Aware AI Agent")
st.markdown("An intelligent system for solving complex math and science problems.")

agent = initialize_agent()

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

logger, log_file = setup_user_logger(st.session_state.session_id)

with st.sidebar:
    st.header("App Controls")
    st.markdown("Select the type of query you want to solve.")
    query_type = st.radio("Query Type", ["Math Problem", "Science Problem (with Image)"])
    st.info(f"📝 Your session log is being saved to: `{log_file}`")

if agent:
    if query_type == "Math Problem":
        st.header("🔢 Math Problem Solver")
        problem = st.text_area("Enter the problem statement:", height=100)
        options = st.text_area("Enter the options (e.g., a) 1, b) 2, ...):", height=100)
        
        if st.button("Solve Math Problem"):
            if problem and options:
                with st.spinner("Agent is thinking..."):
                    query_text = f"{problem}\nOptions: {options}"
                    logger.info(f"--- Processing MathQA Query: '{query_text[:50]}...' ---")
                    result = agent.route_query(query=query_text, data_item=None)
                    
                    st.subheader("Agent's Response:")
                    if result and result.llm_response:
                        st.text_area("Reasoning & Output", result.llm_response, height=300)
                    else:
                        st.error("The agent could not produce a result for this query.")
            else:
                st.warning("Please enter both a problem and its options.")

    elif query_type == "Science Problem (with Image)":
        st.header("⚛️ Science Problem Solver")
        question = st.text_area("Enter the science question:", height=100)
        choices_str = st.text_area("Enter the choices, separated by a semicolon (;):", height=100)
        uploaded_image = st.file_uploader("Upload an image for the science problem:", type=["png", "jpg", "jpeg"])

        if st.button("Solve Science Problem"):
            if question and choices_str:
                with st.spinner("Agent is analyzing the problem..."):
                    choices = [choice.strip() for choice in choices_str.split(';')]
                    image_data = None
                    if uploaded_image:
                        image_data = base64.b64encode(uploaded_image.getvalue()).decode('utf-8')
                        image_data = f"data:image/jpeg;base64,{image_data}"

                    data_payload = {
                        "question": question,
                        "choices": choices,
                        "image": image_data if image_data else "",
                        "hint": "", "answer": -1, "task": "closed choice", "grade": "grade8",
                        "subject": "natural science", "topic": "science-and-engineering-practices",
                        "category": "Engineering practices", "skill": "Evaluate tests of engineering-design solutions",
                        "lecture": "", "solution": ""
                    }
                    
                    logger.info(f"--- Processing ScienceQA Query: '{question[:50]}...' ---")
                    result = agent.route_query(query=question, data_item=data_payload)
                    
                    st.subheader("Agent's Response:")
                    if result and result.llm_response:
                        st.text_area("Reasoning & Output", result.llm_response, height=300)
                    else:
                        st.error("The agent could not produce a result for this query.")
            else:
                st.warning("Please provide a question and its choices.")
else:
    st.error("Agent could not be initialized. Please check your API keys and configuration.")
