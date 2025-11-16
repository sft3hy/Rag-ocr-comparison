# src/core/data_models.py

from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a text chunk with metadata"""

    text: str
    source: str
    page: int
    chunk_id: int
