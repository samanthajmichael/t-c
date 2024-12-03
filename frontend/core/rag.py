import os

import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def setup_environment():
    """Setup environment variables and API key"""
    # Load environment variables
    load_dotenv()

    # Set OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        return True
    return False


@st.cache_resource
def encode_documents(path, chunk_size=1000, chunk_overlap=200):
    """
    Encodes all text files into a vector store using OpenAI embeddings.

    Args:
        path: The path to the directory of text files.
        chunk_size: The desired size of each text chunk.
        chunk_overlap: The amount of overlap between consecutive chunks.

    Returns:
        A FAISS vector store containing the encoded content of the files.
    """
    # Initialize text splitter and embeddings
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    embeddings = OpenAIEmbeddings()

    # Initialize an empty list to store all texts
    all_texts = []

    # Iterate through all files in the directory
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            file_path = os.path.join(path, filename)

            # Load text document
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            # Split text into chunks and add to all_texts
            texts = text_splitter.split_text(text)
            all_texts.extend(texts)

    # Create vector store from all texts
    vectorstore = FAISS.from_texts(all_texts, embeddings)
    return vectorstore


def create_retriever(vector_store, k=2):
    """Create a retriever from the vector store"""
    return vector_store.as_retriever(search_kwargs={"k": k})


@st.cache_resource
def initialize_rag(data_path="data/", chunk_size=1000, chunk_overlap=200, k=2):
    """Initialize the RAG system"""
    if not setup_environment():
        raise ValueError("Failed to load API key")

    try:
        # First create the vector store
        vector_store = encode_documents(data_path, chunk_size, chunk_overlap)
        # Then create the retriever with the vector store
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        return retriever
    except Exception as e:
        raise ValueError(f"Failed to initialize RAG system: {str(e)}")


def setup_environment():
    """Setup environment variables and API key"""
    # Load environment variables
    load_dotenv()

    # Set OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        return True
    return False


@st.cache_resource
def encode_documents(path, chunk_size=1000, chunk_overlap=200):
    """
    Encodes all text files into a vector store using OpenAI embeddings.

    Args:
        path: The path to the directory of text files.
        chunk_size: The desired size of each text chunk.
        chunk_overlap: The amount of overlap between consecutive chunks.

    Returns:
        A FAISS vector store containing the encoded content of the files.
    """
    # Initialize text splitter and embeddings
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    embeddings = OpenAIEmbeddings()

    # Initialize an empty list to store all texts
    all_texts = []

    # Iterate through all files in the directory
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            file_path = os.path.join(path, filename)

            # Load text document
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            # Split text into chunks and add to all_texts
            texts = text_splitter.split_text(text)
            all_texts.extend(texts)

    # Create vector store from all texts
    vectorstore = FAISS.from_texts(all_texts, embeddings)
    return vectorstore


def create_retriever(vector_store, k=2):
    """Create a retriever from the vector store"""
    return vector_store.as_retriever(search_kwargs={"k": k})


@st.cache_resource
def initialize_rag(data_path="data/", chunk_size=1000, chunk_overlap=200, k=2):
    """Initialize the RAG system"""
    if not setup_environment():
        raise ValueError("Failed to load API key")

    try:
        # First create the vector store
        vector_store = encode_documents(data_path, chunk_size, chunk_overlap)
        # Then create the retriever with the vector store
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        return retriever
    except Exception as e:
        raise ValueError(f"Failed to initialize RAG system: {str(e)}")
