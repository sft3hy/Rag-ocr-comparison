"""
Smart RAG Implementation with OCR, Vision Model, and Advanced Chunking

Features:
- Converts documents to Markdown
- Uses Tesseract OCR for non-native text
- Uses Moondream2 for chart/image understanding
- Smart semantic chunking and vectorization
- FAISS vector store for efficient retrieval
- MODIFIED: Pluggable ML-based chart detection (YOLOv8, TATR, or Heuristics)

Requirements:
pip install pymupdf pytesseract pillow torch transformers sentence-transformers faiss-cpu numpy groq
Also requires tesseract-ocr installed on system

For ML Chart Detectors:
# For YOLOv8
pip install ultralytics
# For Table Transformer (TATR)
pip install timm
"""

import os
import re
import io
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoImageProcessor,
    TableTransformerForObjectDetection,
)
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from groq import Groq
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- ML-based Chart Detector Classes ---


class ChartDetector:
    """Base class for chart detection models."""

    def __init__(self, *args, **kwargs):
        # Define the device FIRST
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"ChartDetector using device: {self.device}")

        # THEN load the model
        self.model = self.load_model(*args, **kwargs)

    def load_model(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")

    def detect(self, page_image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """
        Detects chart regions in a given page image.
        Returns a list of bounding boxes [x1, y1, x2, y2].
        """
        raise NotImplementedError("Subclasses must implement this method.")


class HeuristicDetector(ChartDetector):
    """
    The original heuristic-based chart detection logic.
    This is kept as a fallback or for comparison.
    """

    def load_model(self, *args, **kwargs):
        # No model to load for heuristics
        print("Using HeuristicDetector. No ML model loaded.")
        return None

    def detect(self, page_image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """
        Tries to find charts by dividing the page into a grid.
        This is a simplified version of the original logic, applied to a full-page image.
        """
        boxes = []
        page_width, page_height = page_image.size

        # Divide page into 2x2 grid and check each section
        grid_divisions = 2
        section_width = page_width // grid_divisions
        section_height = page_height // grid_divisions

        for row in range(grid_divisions):
            for col in range(grid_divisions):
                x1 = col * section_width
                y1 = row * section_height
                x2 = (col + 1) * section_width
                y2 = (row + 1) * section_height

                section_image = page_image.crop((x1, y1, x2, y2))

                # Check if the section looks like a chart
                if self._looks_like_chart(section_image):
                    boxes.append((x1, y1, x2, y2))
                    print(
                        f"    Heuristic found potential chart in section ({row},{col})"
                    )
        return boxes

    def _looks_like_chart(self, image: Image.Image) -> bool:
        """Heuristic to determine if an image region contains a chart."""
        try:
            width, height = image.size
            if width < 150 or height < 150:
                return False

            ocr_text = pytesseract.image_to_string(image).strip()
            text_density = (len(ocr_text) / (width * height)) * 1000

            # Reject if it's mostly dense text
            if text_density > 0.4:
                return False

            # Check for visual complexity (variance)
            img_array = np.array(image.convert("L"))
            variance = np.var(img_array)
            return variance > 500
        except:
            return False


class YOLOv8Detector(ChartDetector):
    """
    Chart detection using a YOLOv8 model.
    NOTE: The user should provide a model fine-tuned for charts.
    We use a table-detection model here as a placeholder/example.
    """

    def load_model(self, model_path: str = "foduucom/table-detection-and-extraction"):
        try:
            from ultralytics import YOLO

            print(f"Loading YOLOv8 model from: {model_path}")
            # This can be a path to a local .pt file or a HuggingFace repo
            print("here")
            model = YOLO("foduucom/table-detection-and-extraction")
            print("here 2")
            model.to(self.device)
            return model
        except ImportError:
            raise ImportError(
                "YOLOv8 requires 'ultralytics' to be installed. Please run 'pip install ultralytics'"
            )
        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")
            return None

    def detect(self, page_image: Image.Image) -> List[Tuple[int, int, int, int]]:
        if not self.model:
            return []

        results = self.model(
            page_image, verbose=False
        )  # verbose=False to reduce console spam
        boxes = []
        for result in results:
            # Note: You might need to adjust this based on your model's classes
            # For this table model, we assume class 0 is 'table' and class 1 is 'table rotated'
            for box in result.boxes:
                # if int(box.cls) in [0, 1]: # Filter for specific classes if needed
                coords = box.xyxy[0].cpu().numpy().astype(int)
                boxes.append(tuple(coords))
        print(f"    YOLOv8 detected {len(boxes)} potential charts/tables.")
        return boxes


class TATRDetector(ChartDetector):
    """
    Chart/Table detection using Microsoft's Table Transformer (TATR).
    This model is specifically for tables but the architecture is suitable for charts.
    """

    def load_model(self, model_path: str = "microsoft/table-transformer-detection"):
        try:
            print(f"Loading Table Transformer model: {model_path}")
            self.image_processor = AutoImageProcessor.from_pretrained(model_path)
            model = TableTransformerForObjectDetection.from_pretrained(model_path)
            model.to(self.device)
            return model
        except ImportError:
            raise ImportError(
                "Table Transformer requires 'timm' to be installed. Please run 'pip install timm'"
            )
        except Exception as e:
            print(f"Error loading TATR model: {e}")
            return None

    def detect(self, page_image: Image.Image) -> List[Tuple[int, int, int, int]]:
        if not self.model:
            return []

        inputs = self.image_processor(images=page_image, return_tensors="pt").to(
            self.device
        )
        outputs = self.model(**inputs)

        # Post-process to get bounding boxes
        target_sizes = torch.tensor([page_image.size[::-1]]).to(self.device)
        results = self.image_processor.post_process_object_detection(
            outputs, threshold=0.8, target_sizes=target_sizes
        )[0]

        boxes = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            # Note: you may want to filter by label if your model detects multiple object types
            box = [round(i, 2) for i in box.tolist()]
            boxes.append(tuple(map(int, box)))

        print(f"    Table Transformer detected {len(boxes)} potential charts/tables.")
        return boxes


@dataclass
class Chunk:
    """Represents a text chunk with metadata"""

    text: str
    source: str
    page: int
    chunk_id: int


class SmartRAG:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dir: str = "potential_charts",
        chart_detector_model: str = "heuristic",  # New: "heuristic", "yolo", or "tatr"
    ):
        """
        Initialize the RAG system.
        Args:
            model_name (str): Name of the sentence transformer model.
            output_dir (str): Directory to save extracted charts.
            chart_detector_model (str): The chart detection model to use.
                                        Options: 'heuristic', 'yolo', 'tatr'.
        """
        print("Initializing Smart RAG system...")

        self.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

        # Embedding model for vectorization
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # Create output directory for charts
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Chart output directory: {self.output_dir}")

        # --- NEW: Initialize the selected chart detector ---
        if chart_detector_model.lower() == "yolo":
            # You can point this to a local .pt file for your custom model
            # e.g., YOLOv8Detector("path/to/your/best.pt")
            self.chart_detector = YOLOv8Detector()
        elif chart_detector_model.lower() == "tatr":
            self.chart_detector = TATRDetector()
        elif chart_detector_model.lower() == "heuristic":
            self.chart_detector = HeuristicDetector()
        else:
            raise ValueError(
                "Unsupported chart_detector_model. Choose from 'heuristic', 'yolo', 'tatr'."
            )

        # Vision model for chart understanding
        self.use_vision = False
        try:
            print("Loading Moondream2 vision model...")
            device = (
                "mps"
                if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available() else "cpu"
            )

            model_id = "vikhyatk/moondream2"
            self.vision_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                revision="2025-06-21",
                trust_remote_code=True,
                device_map={"": "mps"},  # Enable GPU acceleration using MPS
            )

            self.vision_tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True
            )
            self.vision_device = device
            self.use_vision = True
            print(f"Vision model loaded successfully on {device}")
        except Exception as e:
            print(f"Error loading vision model: {e}")
            self.use_vision = False

        # Vector store
        self.index = None
        self.chunks: List[Chunk] = []
        self.chart_descriptions = {}  # Store chart descriptions by filename

    def extract_charts_as_images(
        self, page, file_output_dir: str, page_num: int
    ) -> List[Image.Image]:
        """
        REFACTORED: Extracts charts from a PDF page using the selected ML detector.
        """
        charts = []
        try:
            # 1. Render the entire page to an image at high resolution
            mat = fitz.Matrix(3.0, 3.0)  # 3x zoom for better quality
            pix = page.get_pixmap(matrix=mat)
            page_image = Image.open(io.BytesIO(pix.tobytes("png")))

            # 2. Use the selected detector to find chart bounding boxes
            chart_boxes = self.chart_detector.detect(page_image)

            # 3. Crop chart images from the page image using the bounding boxes
            for i, box in enumerate(chart_boxes):
                # The box coordinates are relative to the high-res page image
                chart_image = page_image.crop(box)

                if chart_image.width < 150 or chart_image.height < 150:
                    print(f"    Skipping detected region {i+1} - too small.")
                    continue

                charts.append(chart_image)
                print(f"    Captured chart region {i + 1}: {chart_image.size}")

                # Save the chart to disk
                chart_filename = f"page{page_num}_chart{i + 1}.png"
                chart_path = os.path.join(self.output_dir, chart_filename)
                chart_image.save(chart_path)
                print(f"    Saved chart to: {chart_path}")

        except Exception as e:
            print(f"    Error during ML chart detection on page {page_num}: {e}")

        return charts

    def _looks_like_chart(self, image: Image.Image) -> bool:
        """
        Heuristic to determine if an image region contains a chart.
        Checks for visual complexity and non-text patterns.
        """
        try:
            # First do the text density check
            width, height = image.size
            ocr_text = pytesseract.image_to_string(image).strip()
            text_length = len(ocr_text)
            image_area = width * height
            text_density = (text_length / image_area) * 1000

            # Reject if too much text
            if text_density > 0.4:
                return False

            # Then check visual complexity
            img_array = np.array(image.convert("L"))
            variance = np.var(img_array)

            # Check aspect ratio
            # if not self._has_chart_aspect_ratio(image):
            #     print(f"      Rejecting: Unusual aspect ratio - likely not a chart")
            #     return False
            # Charts have varied pixel values but low text
            return variance > 500

        except:
            # Fallback to simple variance check
            img_array = np.array(image.convert("L"))
            variance = np.var(img_array)
            # Check aspect ratio
            # if not self._has_chart_aspect_ratio(image):
            #     print(f"      Rejecting: Unusual aspect ratio - likely not a chart")
            #     return False
            return variance > 500

    def _has_chart_aspect_ratio(self, image: Image.Image) -> bool:
        """Charts typically have reasonable aspect ratios (not super wide or tall)"""
        width, height = image.size
        aspect_ratio = width / height if height > 0 else 0

        # Charts are usually between 1:3 and 3:1 ratio
        # Text screenshots often have extreme ratios
        return 0.33 <= aspect_ratio <= 3.0

    def extract_text_from_pdf(
        self, pdf_path: str, file_output_dir: str, progress_callback=None
    ) -> str:
        """Extract text from PDF with OCR fallback and image understanding"""
        print(f"Processing PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        full_text = []
        total_pages = len(doc)

        for page_num, page in enumerate(doc):
            print(f"\nProcessing page {page_num + 1}/{len(doc)}...")
            if progress_callback:
                progress_callback(page_num + 1, total_pages)

            page_text = []
            text = page.get_text()
            if text.strip():
                page_text.append(text)

            # Method 1: Extract embedded images
            image_list = page.get_images(full=True)
            print(f"  Found {len(image_list)} embedded images on page {page_num + 1}")

            for img_idx, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))

                    if self._is_likely_chart(image):
                        print(
                            f"    Embedded image {img_idx + 1} is likely a chart (size: {image.size})"
                        )
                        if self.use_vision:
                            print(f"    Analyzing chart with vision model...")
                            description = self._describe_image(image)
                            print(f"    Chart description (FULL):\n    {description}")
                            page_text.append(f"\n[CHART DESCRIPTION: {description}]\n")

                            chart_filename = (
                                f"page{page_num + 1}_embedded{img_idx + 1}.png"
                            )
                            chart_path = os.path.join(file_output_dir, chart_filename)
                            image.save(chart_path)
                            print(f"    Saved embedded image to: {chart_path}")
                            self.chart_descriptions[chart_filename] = description
                    else:
                        print(
                            f"    Embedded image {img_idx + 1} is too small for chart analysis (size: {image.size})"
                        )

                    ocr_text = pytesseract.image_to_string(image)
                    if ocr_text.strip():
                        print(f"    Extracted {len(ocr_text)} characters via OCR")
                        page_text.append(f"\n[OCR from image: {ocr_text}]\n")

                except Exception as e:
                    print(f"    Error processing embedded image {img_idx}: {str(e)}")

            # Method 2: Use the new ML-based chart extractor for vector graphics
            print(
                f"  Searching for charts using {self.chart_detector.__class__.__name__}..."
            )
            chart_images = self.extract_charts_as_images(
                page=page, page_num=page_num + 1, file_output_dir=file_output_dir
            )

            for chart_idx, chart_image in enumerate(chart_images):
                try:
                    if self.use_vision:
                        print(f"\n    ===== Analyzing Chart {chart_idx + 1} =====")
                        description = self._describe_image(chart_image)
                        print(f"    FULL DESCRIPTION:\n    {description}")
                        print(
                            f"    ===== End Chart {chart_idx + 1} Description =====\n"
                        )
                        page_text.append(f"\n[CHART DESCRIPTION: {description}]\n")

                        chart_filename = f"page{page_num + 1}_chart{chart_idx + 1}.png"
                        self.chart_descriptions[chart_filename] = description

                    ocr_text = pytesseract.image_to_string(chart_image)
                    if ocr_text.strip():
                        print(
                            f"    Extracted {len(ocr_text)} characters via OCR from chart region"
                        )
                        page_text.append(f"\n[OCR from chart region: {ocr_text}]\n")

                except Exception as e:
                    print(f"    Error analyzing chart {chart_idx + 1}: {str(e)}")

            page_markdown = self._convert_to_markdown(
                "\n".join(page_text), page_num + 1
            )
            full_text.append(page_markdown)

        doc.close()
        return "\n\n".join(full_text)

    def _is_likely_chart(self, image: Image.Image) -> bool:
        """Heuristic to determine if image is likely a chart/diagram"""
        width, height = image.size
        if width < 200 or height < 200:
            return False
        try:
            ocr_text = pytesseract.image_to_string(image).strip()
            text_density = (
                (len(ocr_text) / (width * height)) * 1000 if (width * height) > 0 else 0
            )
            if text_density > 0.7:
                print("text density too large: ", text_density)
                return False
            return True
        except Exception:
            return True

    def _describe_image(self, image: Image.Image) -> str:
        """Use Moondream2 to describe charts and diagrams"""
        try:
            enc_image = self.vision_model.encode_image(image)
            prompt = """
Describe this chart or diagram in complete detail. Include ALL data points, trends, labels, axis information, and key insights. List specific numbers, percentages, years, and values shown. Describe the complete trend from beginning to end.
"""
            description = self.vision_model.answer_question(
                enc_image, prompt, self.vision_tokenizer
            )
            return description
        except Exception as e:
            return f"[Image analysis failed: {str(e)}]"

    def _convert_to_markdown(self, text: str, page_num: int) -> str:
        markdown = f"## Page {page_num}\n\n"
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" +", " ", text)
        lines = text.split("\n")
        formatted_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append("")
                continue
            if len(line) < 100 and line.isupper() and len(line) > 3:
                formatted_lines.append(f"### {line.title()}\n")
            else:
                formatted_lines.append(line)
        markdown += "\n".join(formatted_lines)
        return markdown

    def smart_chunk(
        self, text: str, source: str, chunk_size: int = 500, overlap: int = 100
    ) -> List[Chunk]:
        chunks = []
        # Use a regex that captures chart descriptions as a whole
        text_and_charts = re.split(
            r"(\[CHART DESCRIPTION:.*?\])", text, flags=re.DOTALL
        )

        for piece in text_and_charts:
            if piece.startswith("[CHART DESCRIPTION:"):
                # Add the entire chart description as a single chunk
                chunks.append(
                    Chunk(
                        text=piece,
                        source=source,
                        # You might need to parse the page number differently here
                        page=self._get_page_from_context(text, piece),
                        chunk_id=len(chunks),
                    )
                )
            else:
                # Process the regular text as before
                pages = re.split(r"## Page (\d+)", piece)
                current_page = 1
                for i in range(1, len(pages), 2):
                    if i < len(pages):
                        current_page = int(pages[i])
                        page_text = pages[i + 1] if i + 1 < len(pages) else ""
                    else:
                        page_text = pages[i]
                    sentences = re.split(r"(?<=[.!?])\s+", page_text)
                    current_chunk = []
                    current_length = 0
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                        sentence_length = len(sentence)
                        if (
                            current_length + sentence_length > chunk_size
                            and current_chunk
                        ):
                            chunk_text = " ".join(current_chunk)
                            chunks.append(
                                Chunk(
                                    text=chunk_text,
                                    source=source,
                                    page=current_page,
                                    chunk_id=len(chunks),
                                )
                            )
                            # ... (rest of the original chunking logic for overlap)
                            overlap_sentences = []
                            overlap_length = 0
                            for s in reversed(current_chunk):
                                if overlap_length + len(s) <= overlap:
                                    overlap_sentences.insert(0, s)
                                    overlap_length += len(s)
                                else:
                                    break
                            current_chunk = overlap_sentences
                            current_length = overlap_length
                        current_chunk.append(sentence)
                        current_length += sentence_length
                    if current_chunk:
                        chunk_text = " ".join(current_chunk)
                        chunks.append(
                            Chunk(
                                text=chunk_text,
                                source=source,
                                page=current_page,
                                chunk_id=len(chunks),
                            )
                        )
        return chunks

    # Helper function to get the page number
    def _get_page_from_context(self, full_text, chart_description):
        # This is a simplified example; you might need a more robust way to track pages
        # if chart descriptions can span page breaks in the text.
        try:
            preceding_text = full_text.split(chart_description)[0]
            page_matches = re.findall(r"## Page (\d+)", preceding_text)
            if page_matches:
                return int(page_matches[-1])
        except:
            pass
        return 0  # Default page number

    def index_document(
        self,
        pdf_path: str,
        chunk_size: int = 500,
        overlap: int = 100,
        progress_callback=None,
    ):
        # Use the output_dir directly without creating subdirectories
        # The calling code (Streamlit) already creates a unique directory
        file_output_dir = self.output_dir

        print(f"Saving charts to: {file_output_dir}")

        markdown_text = self.extract_text_from_pdf(
            pdf_path, file_output_dir, progress_callback
        )
        print("\nChunking document...")
        chunks = self.smart_chunk(markdown_text, pdf_path, chunk_size, overlap)
        self.chunks.extend(chunks)
        print("Vectorizing chunks...")
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.encode(chunk_texts, show_progress_bar=True)
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(np.array(embeddings).astype("float32"))
        print(f"Indexed {len(chunks)} chunks from {pdf_path}\n")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        if self.index is None or len(self.chunks) == 0:
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
        results = self.search(question, top_k)
        context = "\n\n---\n\n".join(
            [f"[Page: {chunk.page}]\n{chunk.text}" for chunk, score in results]
        )
        prompt = f"""Based on the following context from a document, answer the question accurately and concisely.
        Context:
        {context}
        Question: {question}
        Instructions:
        - Answer directly based on the context provided
        - If chart data is mentioned, cite specific values and trends
        - If the answer isn't in the context, say so
        - Keep your answer focused and under 150 words
        Answer:"""
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise assistant that answers questions based strictly on the provided context. When chart data is available, cite specific numbers and trends. Be concise and accurate. If page numbers are in the metadata, mention them.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=500,
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
            print(e)
            return {"error": str(e)}


# --- Example Usage ---
if __name__ == "__main__":
    # CHOOSE YOUR DETECTOR: "heuristic", "yolo", or "tatr"
    # Note: "yolo" and "tatr" will download pre-trained models on first run.
    # The models used here are trained on tables, so they may detect tables
    # as charts. For best results, fine-tune a model on your specific chart data.
    CHART_DETECTOR_TO_USE = "yolo"  # <-- CHANGE THIS TO EXPERIMENT

    try:
        # Initialize RAG system with the chosen detector
        rag = SmartRAG(chart_detector_model=CHART_DETECTOR_TO_USE)

        pdf_path = (
            "/path/to/your/document.pdf"  # <--- IMPORTANT: REPLACE with your PDF path
        )

        if os.path.exists(pdf_path):
            rag.index_document(pdf_path, chunk_size=500, overlap=100)

            response = rag.query(
                "What is the trend in percentage of high intensity armed conflicts over the last decade?",
                top_k=3,
            )

            print("=" * 80)
            print("QUERY RESULTS")
            print("=" * 80)
            if "error" in response:
                print(f"An error occurred: {response['error']}")
            else:
                print(f"\nQuestion: {response['question']}\n")
                print("Retrieved Context:")
                print(response["context"])
                print(f"\nAnswer from groq:\n{response['response']}")
        else:
            print(f"PDF file not found: {pdf_path}")
            print("Please provide a valid PDF path to test the RAG system.")

    except (ImportError, ValueError) as e:
        print(f"\nERROR: {e}")
        print(
            "Please ensure you have installed the required libraries for the selected model."
        )
