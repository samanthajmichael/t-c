# src/preprocess_data.py
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Define paths
data_folders = [Path("src/cars"), Path("src/E-commerce"), Path("src/food"), Path("src/electronics")]
index_file = "src/faiss_index.bin"
mapping_file = "src/filename_mapping.txt"
embedding_model_name = "all-MiniLM-L6-v2"

# Load the embedding model
model = SentenceTransformer(embedding_model_name)

# Define patterns for section headers and sub-lists
section_pattern = r"^\d+\.\s+([A-Z].*)"  # Matches numbered section headers like "1. Title"
subsection_pattern = r"^\([a-z]+\)"  # Matches list items like "(a)", "(i)"

def load_and_chunk_documents(data_folders, chunk_size=120):
    """
    Load documents from multiple folders, split by sections, and sub-chunk longer sections.
    
    Args:
        data_folders (list of Path): List of paths to folders containing .txt documents.
        chunk_size (int): Maximum number of words per chunk.
        
    Returns:
        chunks (list): List of document chunks.
        metadata (list): Metadata for each chunk.
    """
    chunks = []
    metadata = []
    doc_id = 0

    for folder in data_folders:
        folder_name = folder.name  # Get the folder name (e.g., 'cars', 'E-commerce')
        for file_path in folder.glob("*.txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read().strip()
                if content:
                    # Split document by main section headers
                    sections = re.split(section_pattern, content, flags=re.MULTILINE)
                    
                    for i in range(1, len(sections), 2):  # Iterate over section titles and contents
                        section_title = sections[i].strip()
                        section_content = sections[i+1].strip()
                        
                        # Further split long sections into sub-chunks if they have list items
                        sub_chunks = re.split(subsection_pattern, section_content)
                        for j, sub_chunk in enumerate(sub_chunks):
                            words = sub_chunk.split()
                            for k in range(0, len(words), chunk_size):
                                chunk = " ".join(words[k:k + chunk_size])
                                chunks.append(chunk)
                                metadata.append({
                                    "Document ID": doc_id,
                                    "Section Title": section_title,
                                    "Subsection": f"{j}_{k // chunk_size}",
                                    "Folder": folder_name,  # Include the folder name in metadata
                                    "File Name": file_path.name,
                                    "Chunk ID": f"{doc_id}_{i}_{j}_{k // chunk_size}"
                                })
                    doc_id += 1
                else:
                    print(f"Warning: '{file_path}' is empty.")

    if not chunks:
        print("No document chunks found. Please ensure there are valid `.txt` files in the data folders.")
    else:
        print(f"Loaded {len(chunks)} chunks from documents in the provided data folders.")
    return chunks, metadata

def embed_and_store(chunks, metadata):
    if not chunks:
        raise ValueError("No document chunks to embed. Please check the 'data' directory.")
    
    # Generate embeddings
    embeddings = model.encode(chunks, convert_to_tensor=True, show_progress_bar=True)
    embeddings = embeddings.cpu().numpy()
    
    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    # Save index
    faiss.write_index(index, index_file)
    
    # Save chunk metadata
    with open(mapping_file, "w", encoding="utf-8") as f:
        for meta in metadata:
            f.write(f"{meta['Chunk ID']}\t{meta['Document ID']}\t{meta['Section Title']}\t{meta['Subsection']}\t{meta['Folder']}\t{meta['File Name']}\n")
    
    print(f"FAISS index created and saved to {index_file}. Metadata saved to {mapping_file}.")
    return index

if __name__ == "__main__":
    chunks, metadata = load_and_chunk_documents(data_folders)
    faiss_index = embed_and_store(chunks, metadata)
