import streamlit as st
import json
import os
import uuid
import base64
from dotenv import load_dotenv
from CompendiumAwareAgent import CompendiumAwareAgent
from math_qa import MathQATool
from science_qa import ScienceQATool
from mongo_utils import MongoLogger
from build_compendium import main as run_database_setup # Import the setup function

# --- 1. APP CONFIGURATION & INITIALIZATION ---

# Load environment variables from a .env file
load_dotenv()

# Configure the Streamlit page
st.set_page_config(
    page_title="FredRag",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 2. ONE-TIME DATABASE SETUP PER SESSION---

# Use session state to ensure this runs only once for each new user session.
if 'db_initialized' not in st.session_state:
    # Clear the resource cache to ensure the agent is re-initialized with the new data.
    st.cache_resource.clear()
    with st.spinner("Performing first-time setup: Building knowledge base... This may take a moment."):
        try:
            run_database_setup()
            st.session_state.db_initialized = True
            st.success("Knowledge base setup complete!")
        except Exception as e:
            st.error(f"Failed to build the knowledge base: {e}")
            # Stop the app if setup fails
            st.stop()


# --- 3. LOGGING SETUP ---

# Initialize MongoDB logger
MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME = "FredRag"
logger = MongoLogger(MONGO_URI, DB_NAME)

def setup_user_logger(session_id):
    """Creates and configures a unique logger for each user session."""
    return logger, f"evaluation_log_{session_id}.txt"

# --- 4. AGENT INITIALIZATION (CACHED FOR EFFICIENCY) ---

@st.cache_resource
def initialize_agent():
    """
    Initializes all the necessary components for the agent.
    This function is cached to avoid re-initializing on every interaction.
    """
    try:
        # Initialize the tools
        math_tool = MathQATool()
        science_tool = ScienceQATool()
        tools = {"mathqa": math_tool, "scienceqa": science_tool}

        # Initialize the agent
        agent = CompendiumAwareAgent(tools=tools)
        return agent
    except Exception as e:
        st.error(f"Failed to initialize agent: {e}")
        return None

# --- 5. STREAMLIT UI LAYOUT ---

st.title("🧠 FredRag")
st.markdown("This agent uses a dynamically built knowledge base to answer questions about math and science.")

# Initialize or get the session ID
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

session_id = st.session_state.session_id
user_logger, log_file = setup_user_logger(session_id)

agent = initialize_agent()

if agent:
    # Create two columns for a clean layout
    col1, col2 = st.columns(2)

    with col1:
        st.header("Math QA")
        math_query = st.text_input("Enter your math question:")
        if st.button("Ask Math QA"):
            if math_query:
                log_id = user_logger.log_entry(session_id, f"Math QA Query: {math_query}")
                with st.spinner("The Math agent is thinking..."):
                    result = agent.route_query(query=math_query)
                    st.subheader("Agent's Response:")
                    if result and result.llm_response:
                        st.text_area("Reasoning & Output", result.llm_response, height=300)
                    else:
                        st.error("The agent could not produce a result for this query.")
                user_logger.log_exit(log_id)

    with col2:
        st.header("Science QA")
        science_question = st.text_input("Enter your science question:")
        uploaded_image = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"])
        choices_str = st.text_area("Enter choices (one per line):")

        if st.button("Ask Science QA"):
            choices = [choice.strip() for choice in choices_str.split('\n') if choice.strip()]
            if science_question and choices:
                log_id = user_logger.log_entry(session_id, f"Science QA Query: {science_question}")
                with st.spinner("The Science agent is thinking..."):
                    image_data = None
                    if uploaded_image:
                        image_data = base64.b64encode(uploaded_image.getvalue()).decode('utf-8')
                        image_data = f"data:image/jpeg;base64,{image_data}"

                    data_payload = {
                        "question": science_question,
                        "choices": choices,
                        "image": image_data if image_data else "",
                        "hint": "", "answer": -1, "task": "closed choice", "grade": "grade8",
                        "subject": "natural science", "topic": "science-and-engineering-practices",
                        "category": "Engineering practices", "skill": "Evaluate tests of engineering-design solutions",
                        "lecture": "", "solution": ""
                    }
                    
                    result = agent.route_query(query=science_question, data_item=data_payload)
                    
                    st.subheader("Agent's Response:")
                    if result and result.llm_response:
                        st.text_area("Reasoning & Output", result.llm_response, height=300)
                    else:
                        st.error("The agent could not produce a result for this query.")
                user_logger.log_exit(log_id)
            else:
                st.warning("Please provide a question and its choices.")
else:
    st.error("Agent could not be initialized. Please check your API keys and configuration.")
