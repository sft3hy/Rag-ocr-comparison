# src/core/data_models.py

from dataclasses import dataclass
from typing import Optional, Dict, Any
from dataclasses import field

@dataclass
class Chunk:
    """
    Updated Chunk model to support Parent-Child relationships.
    """
    text: str
    source: str
    page: int
    chunk_id: str
    parent_id: Optional[str] = None
    is_parent: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
