import os

import streamlit as st
import json
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


def create_retriever(vector_store, k=2):
    """Create a retriever from the vector store"""
    return vector_store.as_retriever(search_kwargs={"k": k})


@st.cache_resource
def initialize_rag(data_path="frontend/data/", chunk_size=1000, chunk_overlap=200, k=2):
    """
    Initialize the RAG system with metadata.

    Args:
        data_path: Path to the data directory containing text files and metadata.json.
        chunk_size: The size of each text chunk.
        chunk_overlap: The overlap between chunks.
        k: Number of documents to retrieve.

    Returns:
        A retriever initialized with the vector store and metadata.
    """
    if not setup_environment():
        raise ValueError("Failed to load API key")

    try:
        # Encode documents with metadata
        vector_store = encode_documents(data_path, chunk_size, chunk_overlap)
        
        # Create retriever from the vector store
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        return retriever
    except Exception as e:
        raise ValueError(f"Failed to initialize RAG system: {str(e)}")


@st.cache_resource
def encode_documents(path, chunk_size=1000, chunk_overlap=200):
    """
    Encodes all text files into a vector store using OpenAI embeddings and includes metadata.

    Args:
        path: The path to the directory of text files.
        chunk_size: The desired size of each text chunk.
        chunk_overlap: The amount of overlap between consecutive chunks.

    Returns:
        A FAISS vector store containing the encoded content and metadata of the files.
    """
    # Load metadata
    metadata_path = os.path.join(path, "metadata.json")
    with open(metadata_path, "r") as meta_file:
        metadata_list = json.load(meta_file)

    # Initialize text splitter and embeddings
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    embeddings = OpenAIEmbeddings()

    # Store chunks and their metadata
    documents_with_metadata = []

    for meta in metadata_list:
        file_path = os.path.join(path, meta["filename"])

        # Load the text from the file
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        # Split the text into chunks
        chunks = text_splitter.create_documents([text])

        # Add metadata to each chunk
        for chunk in chunks:
            chunk.metadata = meta  # Attach metadata to each chunk
        documents_with_metadata.extend(chunks)

    # Create vector store with embeddings and metadata
    vectorstore = FAISS.from_documents(documents_with_metadata, embeddings)
    return vectorstore



def create_retriever(vector_store, k=2):
    """Create a retriever from the vector store"""
    return vector_store.as_retriever(search_kwargs={"k": k})

