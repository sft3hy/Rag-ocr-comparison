import os
import tempfile
from typing import List, Dict, Tuple, Set
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import pytesseract
from PIL import Image
import re

# Core document conversion libraries
from markitdown import MarkItDown
from pdf2image import convert_from_path


class RAGPipeline:
    def __init__(self, groq_api_key: str, use_ocr: bool = False):
        self.groq_client = Groq(api_key=groq_api_key)
        self.use_ocr = use_ocr
        self.chunks = []
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.chunk_vectors = None
        self.full_text = ""

    def _enrich_markdown_with_ocr(
        self, markdown_content: str, image_path_map: Dict[str, str]
    ) -> Tuple[str, Set[str]]:
        """Replaces known image placeholders in Markdown with their OCR'd text."""
        enriched_markdown = markdown_content
        processed_images = set()

        for md_img_name, img_path in image_path_map.items():
            escaped_img_name = re.escape(md_img_name)
            pattern = re.compile(r"!\[.*?\]\(" + escaped_img_name + r"\)")

            if not pattern.search(enriched_markdown):
                continue

            try:
                img = Image.open(img_path)
                ocr_text = pytesseract.image_to_string(img)
                replacement_block = (
                    f"\n> **[OCR content from image: {md_img_name}]**\n"
                    f"> {ocr_text.strip() if ocr_text.strip() else '*No text found in image.*'}\n"
                )
                enriched_markdown = pattern.sub(replacement_block, enriched_markdown)
                processed_images.add(md_img_name)
            except Exception as e:
                # Silently fail for now, fallback will catch it
                print(f"Inline OCR for {md_img_name} failed: {e}")

        return enriched_markdown, processed_images

    def extract_text_from_image(self, img_path: str) -> str:
        """Extracts text from a standalone image file using OCR."""
        if not self.use_ocr:
            return "OCR is disabled."
        try:
            return pytesseract.image_to_string(Image.open(img_path))
        except Exception as e:
            return f"Error during image OCR: {e}"

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Splits text into chunks of a specified word count."""
        words = text.split()
        return [
            " ".join(words[i : i + chunk_size])
            for i in range(0, len(words), chunk_size)
            if " ".join(words[i : i + chunk_size]).strip()
        ]

    def process_document(self, uploaded_file) -> str:
        """
        Processes a document with a robust three-stage approach to ensure all
        text and image content is captured and returned as Markdown.
        """
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        is_image_file = file_ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            if is_image_file:
                ocr_text = self.extract_text_from_image(tmp_path)
                final_markdown = (
                    f"# Document: {os.path.basename(uploaded_file.name)}\n\n"
                    f"> **[OCR content from image]**\n> {ocr_text}\n"
                )
            else:
                # STAGE 1 & 2: Use MarkItDown for primary conversion and initial OCR
                with tempfile.TemporaryDirectory() as img_dir:
                    md_converter = MarkItDown(image_output_dir=img_dir)
                    result = md_converter.convert(tmp_path)
                    markdown_content = result.text_content
                    final_markdown = markdown_content

                    if self.use_ocr:
                        extracted_image_names = os.listdir(img_dir)
                        if extracted_image_names:
                            image_path_map = {
                                name: os.path.join(img_dir, name)
                                for name in extracted_image_names
                            }
                            final_markdown, _ = self._enrich_markdown_with_ocr(
                                markdown_content, image_path_map
                            )

                # STAGE 3: Full-Page Visual OCR Fallback for PDFs to find missed content
                if self.use_ocr and file_ext == ".pdf":
                    # Get a set of simplified lines from the text we already have
                    # This helps us avoid duplicating text
                    known_text_lines = set()
                    for line in final_markdown.split("\n"):
                        # Normalize line for better matching
                        cleaned_line = "".join(filter(str.isalnum, line)).lower()
                        if (
                            len(cleaned_line) > 10
                        ):  # Only consider reasonably long lines
                            known_text_lines.add(cleaned_line)

                    missed_text_blocks = []
                    # Convert PDF pages to images and OCR them
                    page_images = convert_from_path(tmp_path)
                    for i, page_image in enumerate(page_images):
                        page_ocr_text = pytesseract.image_to_string(page_image)

                        # Check if this new text is already known
                        newly_found_lines = []
                        for line in page_ocr_text.split("\n"):
                            cleaned_line = "".join(filter(str.isalnum, line)).lower()
                            if (
                                len(cleaned_line) > 10
                                and cleaned_line not in known_text_lines
                            ):
                                newly_found_lines.append(line)

                        if newly_found_lines:
                            missed_text_blocks.append(
                                f"--- Page {i+1} ---\n" + "\n".join(newly_found_lines)
                            )

                    # Append any truly new text to the end
                    if missed_text_blocks:
                        final_markdown += "\n\n---\n\n## Additional Content Captured via Full-Page OCR\n"
                        final_markdown += "\n".join(missed_text_blocks)

            # Finalize, chunk, and vectorize the result
            self.full_text = final_markdown
            if self.full_text:
                self.chunks = self.chunk_text(self.full_text)
                if self.chunks:
                    self.chunk_vectors = self.vectorizer.fit_transform(self.chunks)
        finally:
            os.unlink(tmp_path)

        return self.full_text

    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieves the most relevant text chunks for a given query."""
        if not self.chunks or self.chunk_vectors is None:
            return []

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.chunk_vectors)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.chunks[i] for i in top_indices]

    def query(self, question: str) -> str:
        """Queries the RAG pipeline using the processed document."""
        relevant_chunks = self.retrieve_relevant_chunks(question)

        if not relevant_chunks:
            return "No relevant information was found in the document to answer the question."

        context = "\n\n".join(relevant_chunks)
        prompt = f"""Based on the following context, answer the question. The context is in Markdown format. If the answer cannot be found in the context, state that clearly.

Context:
{context}

Question: {question}

Answer:"""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on the provided context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1000,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"An error occurred while generating the answer: {str(e)}"
