import os
import json
import uuid
import numpy as np
import faiss
import pickle
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from sentence_transformers import SentenceTransformer
from datetime import datetime

# Import LangChain splitters based on your provided snippet
from langchain_text_splitters import (
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.core.document_parser import DocumentParser
from src.services.groq_client import GroqClient
from src.services.sanctuary_client import SanctuaryClient
from src.vision.vision_models import VisionModelFactory
from src.core.persistence import save_rag_state, load_rag_state
from src.core.chunking import DocumentChunker
from src.core.data_models import Chunk

TEST = os.environ.get("TEST")


class SmartRAG:
    """
    Orchestrates the RAG pipeline using Parent-Child chunking.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dir: str = "potential_charts",
        vision_model_name: str = "Moondream2",
        verbose: bool = True,
        load_vision: bool = True,
    ):
        print("=" * 80)
        print("INITIALIZING SMART RAG SYSTEM (PARENT-CHILD CHUNKING ENABLED)")
        print("=" * 80)

        self.verbose = verbose
        self.steps_dir = "data/file_steps"
        if self.verbose:
            os.makedirs(self.steps_dir, exist_ok=True)

        # 1. Load llm
        print("\n[STEP 1] Loading llm client...")

        if TEST == "True":
            self.client = GroqClient()
            print("✓ Groq client loaded")
        else:
            self.client = SanctuaryClient()
            print("✓ Sanctuary client loaded")

        # 2. Load Embedding Model
        print("\n[STEP 2] Loading embedding model...")
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        print(f"✓ Embedding model loaded (dimension: {self.embedding_dim})")

        # 3. Setup Directories
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # 4. Load Vision
        self.vision_model_name = vision_model_name
        self.vision_model = None
        if load_vision:
            try:
                print(f"\n[STEP 4] Loading vision model: {vision_model_name}...")
                self.vision_model = VisionModelFactory.create_model(vision_model_name)
                print(f"✓ Vision model '{vision_model_name}' loaded successfully.")
            except Exception as e:
                print(f"✗ Failed to load vision model: {e}")

        # 5. Parser
        print("\n[STEP 5] Initializing document parser...")
        self.document_parser = DocumentParser(self.vision_model, self.output_dir)

        # 6. Initialize Chunker
        self.chunker = DocumentChunker(child_chunk_size=400, parent_chunk_size=2000)

        # Store State
        self.index = None
        self.child_chunks: List[Chunk] = []  # Only children are indexed
        self.parent_map: Dict[str, Chunk] = {}  # Parents are stored for retrieval
        self.chart_descriptions: Dict[str, str] = {}

        print("\n" + "=" * 80)
        print("INITIALIZATION COMPLETE")
        print("=" * 80 + "\n")

    def _save_step_example(self, step_name: str, data: any, file_format: str = "txt"):
        if not self.verbose:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(
            self.steps_dir, f"{step_name}_{timestamp}.{file_format}"
        )
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                if file_format == "json":
                    json.dump(data, f, indent=2, ensure_ascii=False)
                else:
                    f.write(str(data))
        except Exception as e:
            print(f"   ✗ Could not save example: {e}")

    def save_state(self, doc_id: int):
        """
        Custom save state to handle parent_map alongside index/chunks.
        """
        # Save standard components
        save_rag_state(doc_id, self.index, self.child_chunks)

        # Save parent map separately
        parent_path = f"data/chunks/{doc_id}_parents.pkl"
        with open(parent_path, "wb") as f:
            pickle.dump(self.parent_map, f)
        print(f"✓ Saved parent map to {parent_path}")

    def load_state(self, faiss_path: str, chunks_path: str, parents_path: str = None):
        """
        Loads index, child chunks, and parent map.
        """
        self.index, self.child_chunks = load_rag_state(faiss_path, chunks_path)

        # Determine parent path if not provided
        if parents_path is None:
            base_dir = os.path.dirname(chunks_path)
            filename = os.path.basename(chunks_path)
            name_without_ext = os.path.splitext(filename)[0]

            # Try to extract ID: assumes format is 'chunks_12' or '12_chunks'
            if name_without_ext.startswith("chunks_"):
                # If file is 'chunks_12.pkl', take the last part ('12')
                doc_id = name_without_ext.split("_")[-1]
            else:
                # If file is '12_chunks.pkl' or just '12.pkl', take the first part
                doc_id = name_without_ext.split("_")[0]

            parents_path = os.path.join(base_dir, f"{doc_id}_parents.pkl")

        print(f"   -> Looking for parent map at: {parents_path}")

        if os.path.exists(parents_path):
            with open(parents_path, "rb") as f:
                self.parent_map = pickle.load(f)
            print(f"✓ Loaded {len(self.parent_map)} parent chunks")
        else:
            print(f"✗ Warning: Parent map not found at {parents_path}")

    def index_document(self, file_path: str, progress_callback=None):
        """
        Processes document using Parent-Child strategy.
        """
        print("\n" + "=" * 80)
        print("STARTING DOCUMENT INDEXING (PARENT-CHILD)")
        print("=" * 80)

        # STEP 1: Parse
        print("\n[STEP 1] PARSING DOCUMENT")
        markdown_text = self.document_parser.parse(file_path, progress_callback)
        self.chart_descriptions = self.document_parser.chart_descriptions

        if self.vision_model:
            self.vision_model.offload_model()

        if self.verbose:
            self._save_step_example("step1_full_markdown", markdown_text, "md")

        # STEP 2: Parent-Child Chunking
        print("\n[STEP 2] CREATING PARENT & CHILD CHUNKS")
        print("-" * 80)

        # Use the specific DocumentChunker logic
        self.child_chunks, self.parent_map = self.chunker.process(
            markdown_text, file_path
        )

        if not self.child_chunks:
            print("   ✗ Warning: No chunks created.")
            return

        print(f"   ✓ Created {len(self.parent_map)} Parent Contexts")
        print(f"   ✓ Created {len(self.child_chunks)} Child Chunks (Indexable)")

        if self.verbose:
            # Save mapping statistics
            stats = {
                "total_parents": len(self.parent_map),
                "total_children": len(self.child_chunks),
                "avg_children_per_parent": len(self.child_chunks)
                / len(self.parent_map),
            }
            self._save_step_example("step2_chunk_stats", stats, "json")

        # STEP 3: Embed Children
        print("\n[STEP 3] GENERATING EMBEDDINGS (CHILDREN ONLY)")
        print("-" * 80)

        child_texts = [c.text for c in self.child_chunks]
        embeddings = self.embedding_model.encode(child_texts, show_progress_bar=True)

        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)

        # STEP 4: Index
        print("\n[STEP 4] BUILDING VECTOR INDEX")
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(np.array(embeddings).astype("float32"))
        print(f"   ✓ Indexed {self.index.ntotal} child vectors")

        print("\n" + "=" * 80)
        print("INDEXING COMPLETE")
        print("=" * 80 + "\n")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Retrieves Child chunks via similarity, maps them to Parent chunks,
        and returns the unique Parents.
        """
        if self.index is None or not self.child_chunks:
            return []

        if self.verbose:
            print("\n" + "=" * 80)
            print("RETRIEVAL (CHILD -> PARENT MAPPING)")
            print(f"Query: '{query}'")

        # 1. Embed Query
        query_embedding = self.embedding_model.encode([query])

        # 2. Search Index (finding Children)
        # We search for slightly more children (top_k * 3) to ensure we get good parent coverage
        search_k = top_k * 3
        distances, indices = self.index.search(
            np.array(query_embedding).astype("float32"), search_k
        )

        # 3. Map Children to Parents
        seen_parent_ids = set()
        unique_parents = []

        print(f"\n   • Found {len(indices[0])} child matches. Mapping to parents...")

        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.child_chunks):
                child = self.child_chunks[idx]
                parent_id = child.parent_id

                if parent_id and parent_id in self.parent_map:
                    if parent_id not in seen_parent_ids:
                        parent_chunk = self.parent_map[parent_id]
                        # We use the child's similarity score for the parent rank initially
                        unique_parents.append((parent_chunk, float(dist)))
                        seen_parent_ids.add(parent_id)

                        if self.verbose:
                            print(
                                f"     -> Child (sim {dist:.3f}) mapped to Parent {parent_id[:8]}..."
                            )

                if len(unique_parents) >= top_k:
                    break

        if self.verbose:
            print(f"   ✓ Retrieved {len(unique_parents)} unique parent contexts")

        return unique_parents

    def query(self, question: str, top_k: int = 5) -> Dict:
        """
        Full RAG pipeline using Parent Contexts.
        """
        print("\n" + "=" * 80)
        print("GENERATING ANSWER")
        print("=" * 80)

        # 1. Search (Returns Parents now)
        results = self.search(question, top_k)

        if not results:
            return {"response": "No relevant information found in the documents."}

        # 2. Build Context from PARENTS
        context_blocks = []
        for i, (chunk, score) in enumerate(results):
            context_blocks.append(
                f"[Context Block {i+1} | Source: {chunk.source}]\n{chunk.text}"
            )

        context = "\n\n---\n\n".join(context_blocks)

        # 3. Generate
        prompt = f"""Based on the provided context, answer the question accurately.
        
        Context:
        {context}

        Question: {question}
        
        Instructions:
        - The context provided describes larger sections of the document.
        - Answer solely based on this context.
        - If the answer is not found, state that.
        
        Answer:"""

        if self.verbose:
            self._save_step_example("query_parent_context", context, "txt")

        try:
            response = self.client.create_chat_completion(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=2048,
            )

            answer = response.choices[0].message.content
            print(f"✓ Response generated ({len(answer)} chars)")
            print(f"Answer: {answer[:200]}...")

            return {
                "question": question,
                "context": context,
                "response": answer,
                "retrieved_parents": len(results),
            }

        except Exception as e:
            print(f"✗ Error: {e}")
            return {"error": str(e)}

    def query_multiple(
        self, question: str, pipelines: List["SmartRAG"], top_k: int = 5
    ) -> Dict:
        # Implementation similar to original but leverages parent retrieval
        all_results = []
        for p in pipelines:
            all_results.extend(p.search(question, top_k))

        # Sort by best child-score (which is attached to the parent)
        all_results.sort(key=lambda x: x[1])
        top_results = all_results[:top_k]
        print(f"\n✓ Combined results from all documents: {len(top_results)} chunks")

        # Build context
        print("\n[BUILDING CONTEXT]")

        context = "\n\n---\n\n".join(
            [
                f"[Source: {chunk.source}, Page: {chunk.page}]\n{chunk.text}"
                for chunk, score in top_results
            ]
        )
        print(
            f"✓ Context built from {len(top_results)} chunks ({len(context):,} chars)"
        )

        # Create prompt
        prompt = f"""Based on the following context from multiple documents, answer the question accurately and concisely.
        Context:
        {context}

        Question: {question}
        
        Instructions:
        - Answer directly based ONLY on the context provided.
        - If information comes from different documents, mention which sources support which parts of your answer.
        - If chart data is mentioned, cite specific values and trends.
        - If page numbers are available in the source metadata, mention them.
        - If the answer isn't in the context, state that clearly.
        - Keep your answer focused and under 250 words.
        
        Answer:"""

        if self.verbose:
            context_info = {
                "question": question,
                "documents_queried": len(pipelines),
                "chunks_used": len(top_results),
                "context_length": len(context),
                "sources": [
                    {"source": chunk.source, "page": chunk.page, "score": float(score)}
                    for chunk, score in top_results
                ],
            }
            self._save_step_example("multi_query_context", context_info, "json")

        try:
            print("\n[GENERATING RESPONSE]")
            print("Calling Groq API with Llama model...")
            response = self.client.create_chat_completion(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise AI assistant. You answer questions based on provided context from multiple documents. Cite sources and page numbers when available. If information comes from different documents, clarify which sources support which parts of your answer.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=2048,
            )

            answer = response.choices[0].message.content
            print(f"✓ Response generated ({len(answer)} chars)")
            for chunk, score in top_results:
                print(chunk.source)

            result = {
                "question": question,
                "context": context,
                "results": [
                    {
                        "text": chunk.text,
                        "source": chunk.source,
                        "page": chunk.page,
                        "score": score,
                    }
                    for chunk, score in top_results
                ],
                "response": answer,
                "documents_queried": len(pipelines),
            }

            if self.verbose:
                self._save_step_example("multi_query_final_response", result, "json")
                print("\n" + "=" * 80)
                print("MULTI-DOCUMENT QUERY COMPLETE")
                print("=" * 80)
                print(f"Answer: {answer[:300]}...")
                print("=" * 80 + "\n")

            return result

        except Exception as e:
            print(f"✗ Error during Groq API call: {e}")
            error_result = {"error": str(e)}
            if self.verbose:
                self._save_step_example("multi_query_error", error_result, "json")
            return error_result
