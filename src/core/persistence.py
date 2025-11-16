# src/core/persistence.py

import faiss
import pickle
import os
from typing import List
from .data_models import Chunk

FAISS_DIR = "data/faiss_indexes"
CHUNKS_DIR = "data/chunks"


def save_rag_state(doc_id: int, index: faiss.Index, chunks: List[Chunk]):
    """
    Saves the FAISS index and chunks list to disk.

    Args:
        doc_id (int): The unique ID of the document session.
        index (faiss.Index): The FAISS vector index.
        chunks (List[Chunk]): The list of Chunk objects.

    Returns:
        A tuple of (faiss_path, chunks_path).
    """
    os.makedirs(FAISS_DIR, exist_ok=True)
    os.makedirs(CHUNKS_DIR, exist_ok=True)

    faiss_path = os.path.join(FAISS_DIR, f"index_{doc_id}.faiss")
    chunks_path = os.path.join(CHUNKS_DIR, f"chunks_{doc_id}.pkl")

    # Save the FAISS index
    faiss.write_index(index, faiss_path)
    print(f"FAISS index saved to {faiss_path}")

    # Save the chunks list using pickle
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Chunks list saved to {chunks_path}")

    return faiss_path, chunks_path


def load_rag_state(faiss_path: str, chunks_path: str):
    """
    Loads the FAISS index and chunks list from disk.

    Args:
        faiss_path (str): Path to the saved FAISS index file.
        chunks_path (str): Path to the saved chunks pickle file.

    Returns:
        A tuple of (loaded_index, loaded_chunks).
    """
    # Load the FAISS index
    index = faiss.read_index(faiss_path)
    print(f"FAISS index loaded from {faiss_path}")

    # Load the chunks list
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    print(f"Chunks list loaded from {chunks_path}")

    return index, chunks
