import re
from typing import List, Tuple, Dict
from src.core.data_models import Chunk
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4


class DocumentChunker:
    """
    Implements the Parent-Child chunking strategy.
    Splits document into large parent chunks (for context)
    and smaller child chunks (for embedding/retrieval).
    """

    def __init__(
        self,
        child_chunk_size: int = 400,
        child_chunk_overlap: int = 50,
        parent_chunk_size: int = 2000,
        parent_chunk_overlap: int = 200,
    ):
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        # using Recursive here as it's generally safer than Markdown only for generic text
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def process(self, text: str, source: str) -> Tuple[List[Chunk], Dict[str, Chunk]]:
        """
        Returns:
            - child_chunks: List of Chunk objects (to be embedded)
            - parent_map: Dict[parent_id, Chunk] (to be retrieved)
        """
        parent_docs = self.parent_splitter.create_documents([text])

        parent_map = {}
        child_chunks = []
        if "/" in source:
            source = source.split("/")[-1]

        for p_idx, p_doc in enumerate(parent_docs):
            # Create Parent Chunk
            parent_id = str(uuid4())
            parent_chunk = Chunk(
                text=p_doc.page_content,
                source=source,
                page=0,  # Page handling would require more complex parsing logic
                chunk_id=parent_id,
                is_parent=True,
                metadata={"index": p_idx},
            )
            parent_map[parent_id] = parent_chunk

            # Create Child Chunks from this Parent
            child_docs = self.child_splitter.create_documents([p_doc.page_content])

            for c_doc in child_docs:
                child_id = str(uuid4())
                child_chunk = Chunk(
                    text=c_doc.page_content,
                    source=source,
                    page=0,
                    chunk_id=child_id,
                    parent_id=parent_id,
                    is_parent=False,
                )
                child_chunks.append(child_chunk)

        return child_chunks, parent_map
