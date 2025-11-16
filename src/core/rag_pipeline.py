import os
import numpy as np
import faiss
from dataclasses import dataclass
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

from src.core.chunking import smart_chunk
from src.core.document_parser import DocumentParser
from src.services.groq_client import GroqClient
from src.vision.vision_models import VisionModelFactory

from src.core.data_models import Chunk


class SmartRAG:
    """
    Orchestrates the entire RAG pipeline from document ingestion to querying.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dir: str = "potential_charts",
        vision_model_name: str = "Moondream2",
    ):
        """
        Initializes the RAG system, including models and services.

        Args:
            model_name (str): Sentence transformer model for embeddings.
            output_dir (str): Directory to save extracted charts.
            vision_model_name (str): The vision model to use for image analysis.
        """
        print("Initializing Smart RAG system...")
        print(f"Selected vision model: {vision_model_name}")

        # Initialize external services and models
        self.groq_client = GroqClient()
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # Set up output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Chart output directory: {self.output_dir}")

        # Load the specified vision model using the factory
        self.vision_model_name = vision_model_name
        self.vision_model = None
        try:
            print(f"Loading vision model: {vision_model_name}...")
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
        self.document_parser = DocumentParser(self.vision_model, self.output_dir)

        # Initialize vector store and data holders
        self.index = None
        self.chunks: List[Chunk] = []
        self.chart_descriptions: Dict[str, str] = {}

    def index_document(
        self,
        file_path: str,
        chunk_size: int = 500,
        overlap: int = 100,
        progress_callback=None,
    ):
        """
        Processes, chunks, embeds, and indexes a document.

        Args:
            file_path (str): The path to the document to be indexed.
            chunk_size (int): The target size for text chunks.
            overlap (int): The amount of overlap between consecutive chunks.
            progress_callback (callable, optional): A function to report progress.
        """
        print(f"Starting document indexing for: {file_path}")
        # Delegate parsing to the DocumentParser
        markdown_text = self.document_parser.parse(file_path, progress_callback)
        self.chart_descriptions = self.document_parser.chart_descriptions

        # Chunk the resulting text
        print("Chunking document content...")
        self.chunks = smart_chunk(markdown_text, file_path, chunk_size, overlap)
        if not self.chunks:
            print("Warning: No chunks were created from the document.")
            return

        print(f"Created {len(self.chunks)} chunks. Generating embeddings...")
        chunk_texts = [chunk.text for chunk in self.chunks]
        embeddings = self.embedding_model.encode(chunk_texts, show_progress_bar=True)

        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)

        # Initialize FAISS index and add embeddings
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.embedding_dim)

        self.index.add(np.array(embeddings).astype("float32"))
        print("Document indexing complete. Vector store is ready.")

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

        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(
            np.array(query_embedding).astype("float32"), top_k
        )

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(dist)))
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
        print(f"Received query: '{question}'")
        results = self.search(question, top_k)

        context = "\n\n---\n\n".join(
            [
                f"[Source: {chunk.source}, Page: {chunk.page}]\n{chunk.text}"
                for chunk, score in results
            ]
        )

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

        try:
            print("Generating response with Groq...")
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

            return {
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
                "response": response.choices[0].message.content,
            }
        except Exception as e:
            print(f"Error during Groq API call: {e}")
            return {"error": str(e)}
