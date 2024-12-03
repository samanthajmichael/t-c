import streamlit as st
import toml
from openai import OpenAI
import time
from pathlib import Path
import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

config_path = os.path.join(os.path.dirname(__file__), ".streamlit", "config.toml")

# Load the configuration settings
config = toml.load(config_path)

run_on_save = config["server"]["runOnSave"]
theme_base = config["theme"]["base"]
primary_color = config["theme"]["primaryColor"]

# Import functions directly from core module
from core import (
    initialize_rag,
    create_retriever,
    show_context,
    text_wrap,
    retrieve_context_per_question
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
        unsafe_allow_html=True
    )


st.title("Baymax T&C")

if 'retriever' not in st.session_state:
    try:
        st.session_state.retriever = initialize_rag()
        st.success("RAG system initialized successfully!")
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")


# Get API key from environment variable or sidebar
api_key = os.getenv("OPENAI_API_KEY") or st.sidebar.text_input("OpenAI API Key", type="password")

# Initialize OpenAI client if API key is provided
if api_key:
    client = OpenAI(api_key=api_key)
else:
    st.warning("Please either set OPENAI_API_KEY environment variable or enter your API key in the sidebar.")


# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4"  # or "gpt-4-turbo-preview" for the latest version

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# # Sidebar configuration
st.sidebar.title("OpenAI API Key")

# Get API key from sidebar
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# Initialize OpenAI client if API key is provided
if api_key:
    client = OpenAI(api_key=api_key)
else:
    st.warning("Please enter your OpenAI API key in the sidebar to continue.")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4"  # or "gpt-4-turbo-preview" for the latest version


# # Model parameters -- for tuning the model
# temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
# max_tokens = st.sidebar.slider("Max Tokens", 50, 4000, 1000, 50)
# top_p = st.sidebar.slider("Top P", 0.0, 1.0, 1.0, 0.1)
# presence_penalty = st.sidebar.slider("Presence Penalty", -2.0, 2.0, 0.0, 0.1)
# frequency_penalty = st.sidebar.slider("Frequency Penalty", -2.0, 2.0, 0.0, 0.1)

# System prompt
system_prompt = (
    "System Prompt", 
    """You are a legal assistant explaining terms and conditions in plain English. 
    Only use information from the provided context when responding. 
    Provide a brief (less than 50 word) summary followed by 3-5 key bullet points that users should know, using everyday language. 
    Explain any complex terms simply. 
    Do not provide legal advice or information beyond what's in the context
    Break down complex topics
    Provide clear explanations
    Use relevant examples
    Maintain a conversational tone:"""
    )

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
        st.markdown(message["content"])

if api_key:
    # Accept user input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get relevant context from RAG if available
        context = ""
        if 'retriever' in st.session_state:
            try:
                context = retrieve_context_per_question(prompt, st.session_state.retriever)
                
                # Show retrieved context in expander
                # with st.expander("View Retrieved Context"):
                #     st.write(context)
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
                    system_prompt_with_context = f"{system_prompt}\n\nRelevant context:\n{context_str}"

                # Create messages with proper formatting
                messages = [
                    {"role": "system", "content": str(system_prompt_with_context)}
                ]
                messages.extend([
                    {"role": str(m["role"]), "content": str(m["content"])}
                    for m in st.session_state.messages
                ])

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
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

                # Add latency information -- for testing
                # end_time = time.time()
                # latency = end_time - start_time
                # tokens = len(full_response.split()) + 1
                # st.info(f"Latency: {tokens / latency:.2f} tokens per second")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                full_response = "I apologize, but I encountered an error while processing your request."
            
        st.session_state.messages.append({"role": "assistant", "content": full_response})