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

def load_metadata(metadata_file):
    """
    Load metadata from a JSON file.

    Args:
        metadata_file (str): Path to the metadata.json file.

    Returns:
        list: A list of metadata dictionaries.
    """
    with open(metadata_file, "r") as f:
        return json.load(f)


def create_documents_with_metadata(metadata_list, data_folder, chunk_size=1000, chunk_overlap=200):
    """
    Create documents with metadata and content chunks.

    Args:
        metadata_list (list): List of metadata dictionaries.
        data_folder (str): Path to the folder containing text files.
        chunk_size (int): The size of each text chunk.
        chunk_overlap (int): Overlap between text chunks.

    Returns:
        list: A list of documents with metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    documents = []

    for meta in metadata_list:
        file_path = os.path.join(data_folder, meta["filename"])
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                chunks = text_splitter.create_documents([text])
                for chunk in chunks:
                    chunk.metadata = meta  # Attach metadata
                documents.extend(chunks)
        else:
            print(f"Warning: File {file_path} does not exist.")
    return documents


@st.cache_resource
def initialize_vectorstore_with_metadata(metadata_file, data_folder):
    """
    Initialize the vector store with metadata and content.

    Args:
        metadata_file (str): Path to the metadata.json file.
        data_folder (str): Path to the folder containing the text files.

    Returns:
        FAISS: A vector store initialized with documents and embeddings.
    """
    metadata_list = load_metadata(metadata_file)
    documents = create_documents_with_metadata(metadata_list, data_folder)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(documents, embeddings)



@st.cache_resource
def initialize_rag(metadata_file="data/metadata.json", data_folder="data/", k=2):
    """
    Initialize the RAG system with metadata.

    Args:
        metadata_file (str): Path to the metadata.json file.
        data_folder (str): Path to the data directory containing text files.
        k: Number of documents to retrieve.

    Returns:
        A retriever initialized with the vector store and metadata.
    """
    vectorstore = initialize_vectorstore_with_metadata(metadata_file, data_folder)
    return vectorstore.as_retriever(search_kwargs={"k": k})


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



