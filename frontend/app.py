import os
import sys
import time
from pathlib import Path

import streamlit as st
import toml
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from openai import OpenAI

config_path = os.path.join(os.path.dirname(__file__), ".streamlit", "config.toml")

# Load the configuration settings
config = toml.load(config_path)

run_on_save = config["server"]["runOnSave"]
theme_base = config["theme"]["base"]
primary_color = config["theme"]["primaryColor"]

# Import functions directly from core module
from core import (
    create_retriever,
    initialize_rag,
    retrieve_context_per_question,
    show_context,
    text_wrap,
)

temperature = 0.4
max_tokens = 250
top_p = 1.0
presence_penalty = 0.0
frequency_penalty = 0.0

load_dotenv()

# Get the directory of the current script
current_dir = Path(__file__).parent
image_path = current_dir / "assets" / "images" / "logo.jpg"


def get_base64_image(image_path):
    import base64

    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


col1, col2 = st.columns([4, 1])
with col2:
    st.markdown(
        f"""
        <div style="text-align: right;">
            <img src="data:image/jpg;base64,{get_base64_image(str(image_path))}" 
                 style="width: 200px; height: 200px; object-fit: contain;">
        </div>
        """,
        unsafe_allow_html=True,
    )


st.title("Baymax T&C")

# Consolidated API key handling
st.sidebar.title("Powered by OpenAI")

# Try to get API key from environment first, then allow user input
# Try environment variable first, then fall back to sidebar input
api_key = os.getenv("OPENAI_API_KEY") or st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="Enter your OpenAI API key. This will not be stored permanently.",
)

# Initialize the OpenAI client only if we have a key
if not api_key:
    st.warning("Please enter your OpenAI API Key in the sidebar to continue.")
    st.stop()

# Store API key in session state if it's new or different
if "api_key" not in st.session_state or st.session_state.api_key != api_key:
    st.session_state.api_key = api_key
    # Clear the retriever if API key changes
    if "retriever" in st.session_state:
        del st.session_state.retriever

try:
    client = OpenAI(api_key=api_key)
    st.sidebar.success("API key provided successfully!")
except Exception as e:
    st.error(f"Error initializing OpenAI client: {str(e)}")
    st.stop()

# Initialize RAG system after API key verification
if "retriever" not in st.session_state:
    init_message = st.empty()
    try:
        os.environ["OPENAI_API_KEY"] = api_key  # Set API key for RAG system
        st.session_state.retriever = initialize_rag()
        init_message.success("RAG system initialized successfully!")
        time.sleep(3)
        init_message.empty()
    except Exception as e:
        init_message.error(
            f"Failed to initialize RAG system. Error: {str(e)}\nContinuing without RAG functionality."
        )
        print(
            f"RAG initialization error: {str(e)}", file=sys.stderr
        )  # Detailed logging
        # Don't stop the app, just continue without RAG
        st.session_state.retriever = None

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Model selection
model_options = ["gpt-4", "gpt-4-turbo-preview"]
selected_model = st.sidebar.selectbox("Select Model", model_options)
st.session_state["openai_model"] = selected_model

# Reset button
if st.sidebar.button("Reset Chat"):
    st.session_state.messages = []
    st.sidebar.success("Chat session reset!")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            formatted_content = (
                message["content"].replace("\n*", "\n\n*").replace("\n-", "\n\n-")
            )
            st.markdown(formatted_content)
        else:
            st.markdown(message["content"])

# System prompt
system_prompt = """You are a legal assistant explaining terms and conditions in plain English. 
Only use information from the provided context when responding. 
Provide a brief (less than 50 word) summary followed by 3-5 key bullet points that users should know, using everyday language. 
Explain any complex terms simply. 
Do not provide legal advice or information beyond what's in the context
Break down complex topics
Provide clear explanations
Use relevant examples
Maintain a conversational tone:"""

# Accept user input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get relevant context from RAG if available
    context = ""
    if "retriever" in st.session_state and st.session_state.retriever is not None:
        try:
            context = retrieve_context_per_question(prompt, st.session_state.retriever)
        except Exception as e:
            st.warning(f"RAG retrieval error: {str(e)}")

    start_time = time.time()
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            # Include context in the prompt if available
            system_prompt_with_context = system_prompt
            if context:
                if isinstance(context, list):
                    # Join list items with newlines if context is a list
                    context_str = "\n\n".join(str(item) for item in context)
                else:
                    context_str = str(context)
                system_prompt_with_context = (
                    f"{system_prompt}\n\nRelevant context:\n{context_str}"
                )

            # Create messages with proper formatting
            messages = [{"role": "system", "content": system_prompt_with_context}]
            messages.extend(
                [
                    {"role": str(m["role"]), "content": str(m["content"])}
                    for m in st.session_state.messages
                ]
            )

            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=messages,
                stream=True,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )

            # Updated streaming section with proper formatting
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    # Format response while streaming
                    formatted_response = full_response.replace("\n*", "\n\n*").replace(
                        "\n-", "\n\n-"
                    )
                    message_placeholder.markdown(formatted_response + "▌")
            # Format final response
            formatted_response = full_response.replace("\n*", "\n\n*").replace(
                "\n-", "\n\n-"
            )
            message_placeholder.markdown(formatted_response)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            full_response = (
                "I apologize, but I encountered an error while processing your request."
            )

    st.session_state.messages.append({"role": "assistant", "content": full_response})
