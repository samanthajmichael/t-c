# src/preprocess_data.py
import os
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Define paths
data_folder = Path("src/data")
index_file = "src/faiss_index.bin"
mapping_file = "src/filename_mapping.txt"  # To store filename mappings
embedding_model_name = "all-MiniLM-L6-v2"

# Load the embedding model
model = SentenceTransformer(embedding_model_name)


def load_documents(data_folder):
    documents = []
    filenames = []
    for file_path in data_folder.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read().strip()
            if content:  # Ensure non-empty content is added
                documents.append(content)
                filenames.append(file_path.name)
            else:
                print(f"Warning: '{file_path}' is empty.")
    if not documents:
        print(
            "No documents found. Please ensure there are valid `.txt` files in the 'data' directory."
        )
    else:
        print(f"Loaded {len(documents)} documents from {data_folder}")
    return documents, filenames


def embed_and_store(documents, filenames):
    if not documents:
        raise ValueError("No documents to embed. Please check the 'data' directory.")

    # Generate embeddings
    embeddings = model.encode(documents, convert_to_tensor=True, show_progress_bar=True)
    embeddings = embeddings.cpu().numpy()

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save index
    faiss.write_index(index, index_file)

    # Save filename mapping
    with open(mapping_file, "w", encoding="utf-8") as f:
        for idx, filename in enumerate(filenames):
            f.write(f"{idx}\t{filename}\n")

    print(f"FAISS index created and saved to {index_file}.")
    return index


if __name__ == "__main__":
    docs, filenames = load_documents(data_folder)
    faiss_index = embed_and_store(docs, filenames)
