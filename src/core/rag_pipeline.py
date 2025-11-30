import os
import json
import numpy as np
import faiss
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from datetime import datetime

from src.core.chunking import smart_chunk
from src.core.document_parser import DocumentParser
from src.services.groq_client import GroqClient
from src.vision.vision_models import VisionModelFactory
from src.core.persistence import save_rag_state, load_rag_state
from src.core.data_models import Chunk


class SmartRAG:
    """
    Orchestrates the entire RAG pipeline from document ingestion to querying.
    Now with verbose logging and examples at each step.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dir: str = "potential_charts",
        vision_model_name: str = "Moondream2",
        verbose: bool = True,
        load_vision: bool = True,
    ):
        """
        Initializes the RAG system, including models and services.

        Args:
            model_name (str): Sentence transformer model for embeddings.
            output_dir (str): Directory to save extracted charts.
            vision_model_name (str): The vision model to use for image analysis.
            verbose (bool): If True, saves detailed examples at each step.
        """
        print("=" * 80)
        print("INITIALIZING SMART RAG SYSTEM")
        print("=" * 80)
        print(f"Selected vision model: {vision_model_name}")
        print(f"Embedding model: {model_name}")
        print(f"Verbose mode: {verbose}")

        self.verbose = verbose

        # Create verbose output directory
        self.steps_dir = "data/file_steps"
        if self.verbose:
            os.makedirs(self.steps_dir, exist_ok=True)
            print(f"✓ Created steps directory: {self.steps_dir}")

        # Initialize external services and models
        print("\n[STEP 1] Loading Groq client...")
        self.groq_client = GroqClient()
        print("✓ Groq client loaded")

        print("\n[STEP 2] Loading embedding model...")
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        print(f"✓ Embedding model loaded (dimension: {self.embedding_dim})")

        # Set up output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"\n[STEP 3] Chart output directory: {self.output_dir}")

        # Load the specified vision model using the factory
        self.vision_model_name = vision_model_name
        self.vision_model = None
        if load_vision:
            try:
                print(f"\n[STEP 4] Loading vision model: {vision_model_name}...")
                self.vision_model = VisionModelFactory.create_model(vision_model_name)
                if self.vision_model:
                    print(f"✓ Vision model '{vision_model_name}' loaded successfully.")
                else:
                    print(
                        f"✗ Failed to load vision model '{vision_model_name}'. Continuing without vision capabilities."
                    )
            except Exception as e:
                print(
                    f"✗ Error loading vision model: {e}. Continuing without vision capabilities."
                )

            # Instantiate the document parser, passing the loaded vision model
            print("\n[STEP 5] Initializing document parser...")
            self.document_parser = DocumentParser(self.vision_model, self.output_dir)
            print("✓ Document parser ready")

        # Initialize vector store and data holders
        self.index = None
        self.chunks: List[Chunk] = []
        self.chart_descriptions: Dict[str, str] = {}

        print("\n" + "=" * 80)
        print("INITIALIZATION COMPLETE")
        print("=" * 80 + "\n")

    def _save_step_example(self, step_name: str, data: any, file_format: str = "txt"):
        """
        Saves an example of data at a specific processing step.

        Args:
            step_name (str): Name of the step (e.g., "1_raw_document")
            data (any): The data to save
            file_format (str): File extension (txt, json, md, etc.)
        """
        if not self.verbose:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{step_name}_{timestamp}.{file_format}"
        filepath = os.path.join(self.steps_dir, filename)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                if file_format == "json":
                    json.dump(data, f, indent=2, ensure_ascii=False)
                else:
                    f.write(str(data))
            print(f"   ✓ Saved example: {filepath}")
        except Exception as e:
            print(f"   ✗ Could not save example: {e}")

    def save_state(self, doc_id: int):
        """Saves the current index and chunks to disk for a given doc_id."""
        if self.index is None or not self.chunks:
            raise ValueError("Cannot save state without a built index and chunks.")
        return save_rag_state(doc_id, self.index, self.chunks)

    def load_state(self, faiss_path: str, chunks_path: str):
        """Loads an index and chunks from disk into the current object."""
        self.index, self.chunks = load_rag_state(faiss_path, chunks_path)

    def index_document(
        self,
        file_path: str,
        chunk_size: int = 500,
        overlap: int = 100,
        progress_callback=None,
    ):
        """
        Processes, chunks, embeds, and indexes a document with verbose logging.

        Args:
            file_path (str): The path to the document to be indexed.
            chunk_size (int): The target size for text chunks.
            overlap (int): The amount of overlap between consecutive chunks.
            progress_callback (callable, optional): A function to report progress.
        """
        print("\n" + "=" * 80)
        print("STARTING DOCUMENT INDEXING PIPELINE")
        print("=" * 80)
        print(f"Input file: {file_path}")
        print(f"Chunk size: {chunk_size} | Overlap: {overlap}")
        print("=" * 80 + "\n")

        # STEP 1: Save original file info
        if self.verbose:
            print("\n[STEP 1] ANALYZING ORIGINAL DOCUMENT")
            print("-" * 80)
            file_info = {
                "filename": os.path.basename(file_path),
                "filepath": file_path,
                "file_size_bytes": os.path.getsize(file_path),
                "file_extension": os.path.splitext(file_path)[1],
                "timestamp": datetime.now().isoformat(),
            }
            print(f"   • Filename: {file_info['filename']}")
            print(f"   • Size: {file_info['file_size_bytes']:,} bytes")
            print(f"   • Type: {file_info['file_extension']}")
            self._save_step_example("step1_original_file_info", file_info, "json")

        # STEP 2: Parse document to markdown
        print("\n[STEP 2] PARSING DOCUMENT TO MARKDOWN")
        print("-" * 80)
        print("   Extracting text, images, tables, and structure...")
        markdown_text = self.document_parser.parse(file_path, progress_callback)
        self.chart_descriptions = self.document_parser.chart_descriptions

        if self.verbose:
            print(f"   • Markdown length: {len(markdown_text):,} characters")
            print(f"   • Charts detected: {len(self.chart_descriptions)}")

            # Save full markdown
            self._save_step_example("step2_full_markdown", markdown_text, "md")

            # Save a preview
            preview = (
                markdown_text[:2000]
                + "\n\n... [truncated] ...\n\n"
                + markdown_text[-1000:]
                if len(markdown_text) > 3000
                else markdown_text
            )
            markdown_info = {
                "total_length": len(markdown_text),
                "total_lines": markdown_text.count("\n"),
                "preview": preview,
                "charts_found": len(self.chart_descriptions),
                "chart_list": list(self.chart_descriptions.keys()),
            }
            self._save_step_example("step2_markdown_preview", markdown_info, "json")

            # Save chart descriptions
            if self.chart_descriptions:
                print("\n   CHART DESCRIPTIONS:")
                for chart_name, description in self.chart_descriptions.items():
                    print(f"   • {chart_name}: {description[:100]}...")
                self._save_step_example(
                    "step2_chart_descriptions", self.chart_descriptions, "json"
                )
        self.vision_model.offload_model()

        # STEP 3: Chunk the document
        print("\n[STEP 3] CHUNKING DOCUMENT")
        print("-" * 80)
        print(f"   Creating chunks with size={chunk_size}, overlap={overlap}...")
        self.chunks = smart_chunk(markdown_text, file_path, chunk_size, overlap)

        if not self.chunks:
            print("   ✗ Warning: No chunks were created from the document.")
            return

        print(f"   ✓ Created {len(self.chunks)} chunks")

        if self.verbose:
            # Save chunk statistics
            chunk_lengths = [len(chunk.text) for chunk in self.chunks]
            chunk_stats = {
                "total_chunks": len(self.chunks),
                "avg_chunk_length": np.mean(chunk_lengths),
                "min_chunk_length": np.min(chunk_lengths),
                "max_chunk_length": np.max(chunk_lengths),
                "chunk_size_setting": chunk_size,
                "overlap_setting": overlap,
            }
            print(
                f"   • Average chunk length: {chunk_stats['avg_chunk_length']:.0f} chars"
            )
            print(
                f"   • Min/Max: {chunk_stats['min_chunk_length']}/{chunk_stats['max_chunk_length']} chars"
            )
            self._save_step_example("step3_chunk_statistics", chunk_stats, "json")

            # Save first 5 chunks as examples
            example_chunks = []
            for i, chunk in enumerate(self.chunks[:5]):
                example_chunks.append(
                    {
                        "chunk_id": i,
                        "source": chunk.source,
                        "page": chunk.page,
                        "text": chunk.text,
                        "length": len(chunk.text),
                    }
                )
            self._save_step_example("step3_example_chunks", example_chunks, "json")

            print(f"\n   EXAMPLE CHUNK #{0}:")
            print(f"   Source: {self.chunks[0].source} | Page: {self.chunks[0].page}")
            print(f"   Text preview: {self.chunks[0].text[:200]}...")

        # STEP 4: Generate embeddings
        print("\n[STEP 4] GENERATING EMBEDDINGS")
        print("-" * 80)
        print(
            f"   Converting {len(self.chunks)} chunks to {self.embedding_dim}-dimensional vectors..."
        )

        chunk_texts = [chunk.text for chunk in self.chunks]
        embeddings = self.embedding_model.encode(chunk_texts, show_progress_bar=True)

        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)

        print(f"   ✓ Generated embeddings with shape: {embeddings.shape}")

        if self.verbose:
            # Save embedding statistics and examples
            embedding_info = {
                "embedding_shape": list(embeddings.shape),
                "embedding_dimension": self.embedding_dim,
                "total_vectors": embeddings.shape[0],
                "model_name": self.embedding_model._modules[
                    "0"
                ].auto_model.name_or_path,
            }
            print(f"   • Vectors created: {embeddings.shape[0]}")
            print(f"   • Vector dimension: {self.embedding_dim}")
            self._save_step_example("step4_embedding_info", embedding_info, "json")

            # Save first 3 embedding vectors as examples
            example_embeddings = {
                "chunk_0": {
                    "text_preview": chunk_texts[0][:100],
                    "vector_preview": embeddings[0][:20].tolist(),
                    "vector_stats": {
                        "mean": float(np.mean(embeddings[0])),
                        "std": float(np.std(embeddings[0])),
                        "min": float(np.min(embeddings[0])),
                        "max": float(np.max(embeddings[0])),
                    },
                }
            }
            if len(embeddings) > 1:
                example_embeddings["chunk_1"] = {
                    "text_preview": chunk_texts[1][:100],
                    "vector_preview": embeddings[1][:20].tolist(),
                    "vector_stats": {
                        "mean": float(np.mean(embeddings[1])),
                        "std": float(np.std(embeddings[1])),
                        "min": float(np.min(embeddings[1])),
                        "max": float(np.max(embeddings[1])),
                    },
                }
            self._save_step_example(
                "step4_example_embeddings", example_embeddings, "json"
            )

            print(f"\n   EMBEDDING VECTOR EXAMPLE (first 10 dimensions):")
            print(f"   {embeddings[0][:10]}")

        # STEP 5: Build FAISS index
        print("\n[STEP 5] BUILDING VECTOR INDEX")
        print("-" * 80)

        if self.index is None:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            print(f"   Created new FAISS index (L2 distance)")

        self.index.add(np.array(embeddings).astype("float32"))
        print(f"   ✓ Added {embeddings.shape[0]} vectors to index")
        print(f"   ✓ Total vectors in index: {self.index.ntotal}")

        if self.verbose:
            index_info = {
                "index_type": "IndexFlatL2",
                "dimension": self.embedding_dim,
                "total_vectors": self.index.ntotal,
                "is_trained": self.index.is_trained,
                "metric_type": "L2 (Euclidean distance)",
            }
            self._save_step_example("step5_index_info", index_info, "json")

            # Test search to show similarity
            test_query = chunk_texts[0]
            query_embedding = self.embedding_model.encode([test_query])
            distances, indices = self.index.search(
                np.array(query_embedding).astype("float32"), 3
            )

            search_test = {
                "test_query": test_query[:150],
                "top_3_matches": [
                    {
                        "rank": i + 1,
                        "chunk_id": int(idx),
                        "distance": float(dist),
                        "text_preview": chunk_texts[idx][:150],
                    }
                    for i, (dist, idx) in enumerate(zip(distances[0], indices[0]))
                ],
            }
            self._save_step_example("step5_search_test", search_test, "json")

        # STEP 6: Summary
        print("\n" + "=" * 80)
        print("DOCUMENT INDEXING COMPLETE")
        print("=" * 80)
        print(f"✓ Parsed document to markdown ({len(markdown_text):,} chars)")
        print(f"✓ Created {len(self.chunks)} chunks")
        print(
            f"✓ Generated {embeddings.shape[0]} embedding vectors ({self.embedding_dim}D)"
        )
        print(f"✓ Built searchable index with {self.index.ntotal} vectors")
        print(f"✓ Detected {len(self.chart_descriptions)} charts/images")

        if self.verbose:
            print(f"\n📁 All processing examples saved to: {self.steps_dir}/")

            # Save final pipeline summary
            pipeline_summary = {
                "timestamp": datetime.now().isoformat(),
                "input_file": file_path,
                "pipeline_steps": {
                    "1_document_analysis": "Analyzed file metadata and properties",
                    "2_markdown_conversion": f"Extracted {len(markdown_text):,} characters",
                    "3_chunking": f"Created {len(self.chunks)} chunks",
                    "4_embedding": f"Generated {embeddings.shape[0]}x{self.embedding_dim} embeddings",
                    "5_indexing": f"Built index with {self.index.ntotal} vectors",
                },
                "results": {
                    "total_chunks": len(self.chunks),
                    "total_vectors": self.index.ntotal,
                    "charts_detected": len(self.chart_descriptions),
                    "markdown_length": len(markdown_text),
                },
            }
            self._save_step_example("step6_pipeline_summary", pipeline_summary, "json")

        print("=" * 80 + "\n")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Searches the vector store for the most relevant chunks to a query.

        Args:
            query (str): The user's query.
            top_k (int): The number of results to return.

        Returns:
            A list of tuples, each containing a Chunk and its relevance score.
        """
        if self.index is None or not self.chunks:
            print("Error: Search attempted before indexing a document.")
            return []

        if self.verbose:
            print("\n" + "=" * 80)
            print("PERFORMING VECTOR SEARCH")
            print("=" * 80)
            print(f"Query: '{query}'")
            print(f"Searching for top {top_k} matches...")

        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(
            np.array(query_embedding).astype("float32"), top_k
        )

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(dist)))

        if self.verbose:
            print(f"✓ Found {len(results)} matches")
            for i, (chunk, score) in enumerate(results, 1):
                print(
                    f"   {i}. Score: {score:.4f} | Source: {chunk.source} | Page: {chunk.page}"
                )

            search_results = {
                "query": query,
                "top_k": top_k,
                "results": [
                    {
                        "rank": i + 1,
                        "score": float(score),
                        "source": chunk.source,
                        "page": chunk.page,
                        "text_preview": chunk.text[:200],
                    }
                    for i, (chunk, score) in enumerate(results)
                ],
            }
            self._save_step_example("search_results", search_results, "json")
            print("=" * 80 + "\n")

        return results

    def query(self, question: str, top_k: int = 5) -> Dict:
        """
        Performs a full RAG query: search, context-building, and generation.

        Args:
            question (str): The user's question.
            top_k (int): The number of chunks to use as context.

        Returns:
            A dictionary containing the response and source documents.
        """
        print("\n" + "=" * 80)
        print("FULL RAG QUERY PIPELINE")
        print("=" * 80)
        print(f"Question: '{question}'")
        print("=" * 80 + "\n")

        # Search for relevant chunks
        results = self.search(question, top_k)

        # Build context
        print("\n[BUILDING CONTEXT]")
        context = "\n\n---\n\n".join(
            [
                f"[Source: {chunk.source}, Page: {chunk.page}]\n{chunk.text}"
                for chunk, score in results
            ]
        )
        print(f"✓ Context built from {len(results)} chunks ({len(context):,} chars)")

        # Create prompt
        prompt = f"""Based on the following context from a document, answer the question accurately and concisely.
        Context:
        {context}

        Question: {question}
        
        Instructions:
        - Answer directly based ONLY on the context provided.
        - If chart data is mentioned, cite specific values and trends.
        - If page numbers are available in the source metadata, mention them.
        - If the answer isn't in the context, state that clearly.
        - Keep your answer focused and under 200 words.
        
        Answer:"""

        if self.verbose:
            context_info = {
                "question": question,
                "chunks_used": len(results),
                "context_length": len(context),
                "sources": [
                    {"source": chunk.source, "page": chunk.page, "score": float(score)}
                    for chunk, score in results
                ],
            }
            self._save_step_example("query_context", context_info, "json")
            self._save_step_example("query_full_prompt", prompt, "txt")

        try:
            print("\n[GENERATING RESPONSE]")
            print("Calling Groq API with Llama model...")
            response = self.groq_client.create_chat_completion(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise AI assistant. You answer questions strictly based on the provided document context. Cite page numbers and specific data points when available. If the information is not in the context, you must say so.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=2048,
            )

            answer = response.choices[0].message.content
            print(f"✓ Response generated ({len(answer)} chars)")

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
                    for chunk, score in results
                ],
                "response": answer,
            }

            if self.verbose:
                self._save_step_example("query_final_response", result, "json")
                print("\n" + "=" * 80)
                print("QUERY COMPLETE")
                print("=" * 80)
                print(f"Answer: {answer[:300]}...")
                print("=" * 80 + "\n")

            return result

        except Exception as e:
            print(f"✗ Error during Groq API call: {e}")
            error_result = {"error": str(e)}
            if self.verbose:
                self._save_step_example("query_error", error_result, "json")
            return error_result

    def query_multiple(
        self, question: str, pipelines: List["SmartRAG"], top_k: int = 5
    ) -> Dict:
        """
        Performs a RAG query across multiple document pipelines.

        Args:
            question (str): The user's question.
            pipelines (List[SmartRAG]): List of SmartRAG pipelines to query.
            top_k (int): The number of chunks to retrieve from each document.

        Returns:
            A dictionary containing the response and source documents.
        """
        print("\n" + "=" * 80)
        print("MULTI-DOCUMENT RAG QUERY PIPELINE")
        print("=" * 80)
        print(f"Question: '{question}'")
        print(f"Querying {len(pipelines)} document(s)")
        print("=" * 80 + "\n")

        all_results = []

        # Search across all pipelines
        for idx, pipeline in enumerate(pipelines, 1):
            print(f"\n[SEARCHING DOCUMENT {idx}/{len(pipelines)}]")
            results = pipeline.search(question, top_k)
            all_results.extend(results)

        # Sort by relevance score and take top results
        all_results.sort(key=lambda x: x[1])  # Sort by distance (lower is better)
        top_results = all_results[: top_k * 2]  # Get more results from combined set

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
            response = self.groq_client.create_chat_completion(
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
