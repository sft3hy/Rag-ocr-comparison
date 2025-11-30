# src/core/persistence.py

import faiss
import pickle
import os
from typing import List, Tuple
from .data_models import Chunk

FAISS_DIR = "data/faiss_indexes"
CHUNKS_DIR = "data/chunks"


def save_rag_state(
    doc_id: int, index: faiss.Index, chunks: List[Chunk]
) -> Tuple[str, str]:
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
    print(f"✓ FAISS index saved to {faiss_path}")

    # Save the chunks list using pickle
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"✓ Chunks list saved to {chunks_path}")

    return faiss_path, chunks_path


def load_rag_state(
    faiss_path: str, chunks_path: str
) -> Tuple[faiss.Index, List[Chunk]]:
    """
    Loads the FAISS index and chunks list from disk.

    Args:
        faiss_path (str): Path to the saved FAISS index file.
        chunks_path (str): Path to the saved chunks pickle file.

    Returns:
        A tuple of (loaded_index, loaded_chunks).
    """
    if not os.path.exists(faiss_path):
        raise FileNotFoundError(f"FAISS index file not found: {faiss_path}")

    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    # Load the FAISS index
    index = faiss.read_index(faiss_path)
    print(f"✓ FAISS index loaded from {faiss_path}")

    # Load the chunks list
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    print(f"✓ Chunks list loaded from {chunks_path} ({len(chunks)} chunks)")

    return index, chunks


def delete_rag_state(doc_id: int) -> bool:
    """
    Deletes the saved FAISS index and chunks for a document.

    Args:
        doc_id (int): The unique ID of the document session.

    Returns:
        bool: True if files were deleted, False if files didn't exist.
    """
    faiss_path = os.path.join(FAISS_DIR, f"index_{doc_id}.faiss")
    chunks_path = os.path.join(CHUNKS_DIR, f"chunks_{doc_id}.pkl")

    deleted = False

    if os.path.exists(faiss_path):
        os.remove(faiss_path)
        print(f"✓ Deleted FAISS index: {faiss_path}")
        deleted = True

    if os.path.exists(chunks_path):
        os.remove(chunks_path)
        print(f"✓ Deleted chunks file: {chunks_path}")
        deleted = True

    return deleted


def get_state_size(doc_id: int) -> dict:
    """
    Gets the file sizes for a document's saved state.

    Args:
        doc_id (int): The unique ID of the document session.

    Returns:
        dict: Dictionary with 'faiss_size', 'chunks_size', and 'total_size' in bytes.
    """
    faiss_path = os.path.join(FAISS_DIR, f"index_{doc_id}.faiss")
    chunks_path = os.path.join(CHUNKS_DIR, f"chunks_{doc_id}.pkl")

    sizes = {"faiss_size": 0, "chunks_size": 0, "total_size": 0}

    if os.path.exists(faiss_path):
        sizes["faiss_size"] = os.path.getsize(faiss_path)

    if os.path.exists(chunks_path):
        sizes["chunks_size"] = os.path.getsize(chunks_path)

    sizes["total_size"] = sizes["faiss_size"] + sizes["chunks_size"]

    return sizes


def list_all_saved_states() -> List[int]:
    """
    Lists all document IDs that have saved state files.

    Returns:
        List[int]: List of document IDs with saved states.
    """
    doc_ids = set()

    # Check FAISS directory
    if os.path.exists(FAISS_DIR):
        for filename in os.listdir(FAISS_DIR):
            if filename.startswith("index_") and filename.endswith(".faiss"):
                doc_id = filename.replace("index_", "").replace(".faiss", "")
                try:
                    doc_ids.add(int(doc_id))
                except ValueError:
                    continue

    # Check chunks directory
    if os.path.exists(CHUNKS_DIR):
        for filename in os.listdir(CHUNKS_DIR):
            if filename.startswith("chunks_") and filename.endswith(".pkl"):
                doc_id = filename.replace("chunks_", "").replace(".pkl", "")
                try:
                    doc_ids.add(int(doc_id))
                except ValueError:
                    continue

    return sorted(list(doc_ids))


def cleanup_orphaned_states(valid_doc_ids: List[int]) -> int:
    """
    Removes saved state files for document IDs that are no longer in the database.

    Args:
        valid_doc_ids (List[int]): List of document IDs that should be kept.

    Returns:
        int: Number of orphaned states cleaned up.
    """
    all_saved = list_all_saved_states()
    orphaned = [doc_id for doc_id in all_saved if doc_id not in valid_doc_ids]

    cleaned = 0
    for doc_id in orphaned:
        if delete_rag_state(doc_id):
            cleaned += 1

    if cleaned > 0:
        print(f"✓ Cleaned up {cleaned} orphaned state file(s)")

    return cleaned
