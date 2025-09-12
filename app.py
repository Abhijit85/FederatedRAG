import streamlit as st
import json
import os
import uuid
import base64
from dotenv import load_dotenv
from CompendiumAwareAgent import CompendiumAwareAgent
from math_qa import MathQATool
from science_qa import ScienceQATool
from mmlu_qa import MMLUQATool
from mongo_utils import MongoLogger
from build_compendium import main as run_database_setup # Import the setup function

# --- 1. APP CONFIGURATION ---
load_dotenv()
st.set_page_config(page_title="FredRag", page_icon="🤖", layout="wide")

# --- 2. DATABASE SETUP (RUNS ONCE PER SESSION) ---
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False

if not st.session_state.db_initialized:
    st.cache_resource.clear()
    with st.spinner("Performing first-time setup: Building knowledge base..."):
        try:
            run_database_setup()
            st.session_state.db_initialized = True
            st.success("Knowledge base setup complete!")
        except Exception as e:
            st.error(f"Failed to build the knowledge base: {e}")
            st.stop()

# --- 3. SESSION & LOGGER SETUP ---
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME = "FredRag"
user_logger = MongoLogger(MONGO_URI, DB_NAME)
session_id = st.session_state.session_id

# --- 4. AGENT INITIALIZATION (CACHED) ---
@st.cache_resource
def initialize_agent():
    """Initializes the agent and its tools. Cached for performance."""
    try:
        agent = CompendiumAwareAgent(tools=[MathQATool(), ScienceQATool(), MMLUQATool()])
        print("✅ Agent initialized successfully.")
        return agent
    except Exception as e:
        st.error(f"Failed to initialize agent: {e}")
        return None

agent = initialize_agent()

# --- 5. STREAMLIT UI ---
st.title("FredRag 🤖")
st.caption("A multi-tool agent for advanced Math, Science and MMLU Q&A.")

if agent:
    tab1, tab2, tab3 = st.tabs(["🧮 MathQA", "🔬 ScienceQA", "🧮 MMLU"])

    # --- MathQA Tool Tab ---
    with tab1:
        st.header("Solve a Math Problem")
        math_question = st.text_input("Enter your math question:", key="math_q")
        
        if st.button("Get Math Answer", key="math_submit"):
            if math_question:
                with st.spinner("The agent is thinking..."):
                    log_data = {"query": math_question, "tool": "math_qa"}
                    log_id = user_logger.log_entry(session_id, log_data)
                    
                    # MODIFIED: Unpack the result and the recommended sub-tool from the agent's response
                    result, recommended_sub_tool = agent.route_query(query=math_question)
                    
                    st.subheader("Agent's Response:")
                    response_text = "The agent could not produce a result for this query."
                    if result and result.llm_response:
                        response_text = result.llm_response
                        st.text_area("Reasoning & Output", response_text, height=300)
                    else:
                        st.error(response_text)
                    
                    # MODIFIED: Log a structured dictionary containing the final response and the recommended tool
                    final_log_data = {
                        "llm_response": response_text,
                        "recommended_sub_tool": recommended_sub_tool
                    }
                    user_logger.log_exit(log_id, final_log_data)
            else:
                st.warning("Please enter a math question.")

    # --- ScienceQA Tool Tab ---
    with tab2:
        st.header("Solve a Science Problem")
        science_question = st.text_input("Enter your science question:", key="science_q")
        choices_input = st.text_input("Enter the choices (comma-separated):", key="science_c")
        uploaded_image = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"])

        if st.button("Get Science Answer", key="science_submit"):
            if science_question and choices_input:
                with st.spinner("The agent is thinking..."):
                    choices = [choice.strip() for choice in choices_input.split(',')]
                    
                    image_data = None
                    if uploaded_image:
                        image_data = f"data:image/jpeg;base64,{base64.b64encode(uploaded_image.getvalue()).decode('utf-8')}"

                    log_data_to_store = {
                        "query": science_question,
                        "tool": "science_qa",
                        "choices": choices,
                        "image_provided": uploaded_image is not None
                    }
                    log_id = user_logger.log_entry(session_id, log_data_to_store)
                    
                    data_payload = {"question": science_question, "choices": choices, "image": image_data or ""}
                    
                    # MODIFIED: Unpack the result and the recommended sub-tool from the agent's response
                    result, recommended_sub_tool = agent.route_query(query=science_question, data_item=data_payload)
                    
                    st.subheader("Agent's Response:")
                    response_text = "The agent could not produce a result for this query."
                    if result and result.llm_response:
                        response_text = result.llm_response
                        st.text_area("Reasoning & Output", response_text, height=300)
                    else:
                        st.error(response_text)
                        
                    # MODIFIED: Log a structured dictionary containing the final response and the recommended tool
                    final_log_data = {
                        "llm_response": response_text,
                        "recommended_sub_tool": recommended_sub_tool
                    }
                    user_logger.log_exit(log_id, final_log_data)
            else:
                st.warning("Please provide a question and its choices.")
    # --- MMLU Tool Tab ---
    with tab3:
        st.header("Solve a MMLU Problem")
        mmlu_question = st.text_input("Enter your mmlu question:", key="mmlu_q")
        
        if st.button("Get MMLU Answer", key="mmlu_submit"):
            if mmlu_question:
                with st.spinner("The agent is thinking..."):
                    log_data = {"query": mmlu_question, "tool": "mmlu_qa"}
                    log_id = user_logger.log_entry(session_id, log_data)
                    
                    # MODIFIED: Unpack the result and the recommended sub-tool from the agent's response
                    result, recommended_sub_tool = agent.route_query(query=mmlu_question)
                    
                    st.subheader("Agent's Response:")
                    response_text = "The agent could not produce a result for this query."
                    if result and result.llm_response:
                        response_text = result.llm_response
                        st.text_area("Reasoning & Output", response_text, height=300)
                    else:
                        st.error(response_text)
                    
                    # MODIFIED: Log a structured dictionary containing the final response and the recommended tool
                    final_log_data = {
                        "llm_response": response_text,
                        "recommended_sub_tool": recommended_sub_tool
                    }
                    user_logger.log_exit(log_id, final_log_data)
            else:
                st.warning("Please enter a mmlu question.")
else:
    st.error("Agent could not be initialized. Please check your API keys and configuration.")
