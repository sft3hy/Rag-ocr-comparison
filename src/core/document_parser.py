import os
import re
import io
import fitz  # PyMuPDF
import pytesseract
import subprocess
from PIL import Image
from docx import Document
from pptx import Presentation
from typing import List, Tuple, Dict, Optional
import numpy as np

from src.utils.chart_detection import TATRDetector, HeuristicDetector
from src.vision.vision_models import VisionModel


class DocumentParser:
    """
    Handles the parsing of different document formats (PDF, DOCX, PPTX),
    including text extraction, chart detection, and vision-based analysis.
    """

    def __init__(self, vision_model: Optional[VisionModel], output_dir: str):
        """
        Initializes the parser with necessary models and configurations.

        Args:
            vision_model (VisionModel, optional): An instance of a loaded vision model.
            output_dir (str): Directory to save extracted chart images.
        """
        self.vision_model = vision_model
        self.use_vision = vision_model is not None
        self.output_dir = output_dir
        self.chart_descriptions: Dict[str, str] = {}

        # Initialize chart detection models
        print("Initializing chart detectors...")
        self.chart_detector_tatr = TATRDetector()
        self.chart_detector_heuristic = HeuristicDetector()

    def parse(self, file_path: str, progress_callback=None) -> str:
        """
        Public method to parse a document based on its file extension.

        Args:
            file_path (str): The path to the document.
            progress_callback (callable, optional): A function to report progress.

        Returns:
            A string containing the document content in Markdown format.
        """
        file_ext = os.path.splitext(file_path)[1].lower()

        # Create file-specific output directory
        file_output_dir = os.path.join(
            self.output_dir, os.path.splitext(os.path.basename(file_path))[0]
        )
        os.makedirs(file_output_dir, exist_ok=True)

        if file_ext == ".pdf":
            return self._extract_text_from_pdf(
                file_path, file_output_dir, progress_callback
            )
        elif file_ext == ".docx":
            return self._extract_text_from_docx(
                file_path, file_output_dir, progress_callback
            )
        elif file_ext == ".pptx":
            return self._extract_text_from_pptx(
                file_path, file_output_dir, progress_callback
            )
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

    def _describe_image(self, image: Image.Image) -> str:
        """Use selected vision model to describe charts and diagrams"""
        if not self.use_vision:
            return "[Vision model not available]"
        prompt = "Describe this chart or diagram in detail. Include data points, trends, labels, and key insights."
        return self.vision_model.describe_image(image, prompt)

    def _describe_slide(self, image: Image.Image) -> str:
        """Use selected vision model to describe full slide"""
        if not self.use_vision:
            return "[Vision model not available]"
        prompt = "Describe this presentation slide in detail. Include any charts, diagrams, text, and visual elements."
        return self.vision_model.describe_image(image, prompt)

    def _extract_text_from_pptx(
        self, pptx_path: str, file_output_dir: str, progress_callback=None
    ) -> str:
        print(f"Processing PPTX: {pptx_path}")
        prs = Presentation(pptx_path)
        full_text = []
        total_slides = len(prs.slides)
        slide_images = self._convert_pptx_to_images(pptx_path, file_output_dir)

        for slide_idx, slide in enumerate(prs.slides):
            print(f"\nProcessing slide {slide_idx + 1}/{total_slides}...")
            if progress_callback:
                progress_callback(slide_idx + 1, total_slides)

            slide_text = [f"## Slide {slide_idx + 1}\n"]
            text_content = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    text_content.append(shape.text.strip())

            if text_content:
                slide_text.append("Text content:\n" + "\n".join(text_content) + "\n")

            slide_image = None
            if slide_idx < len(slide_images):
                slide_image = slide_images[slide_idx]

            if slide_image and self.use_vision:
                print(f"  Analyzing full slide with vision model...")
                slide_filename = f"slide{slide_idx + 1}_full.png"
                slide_path = os.path.join(file_output_dir, slide_filename)
                slide_image.save(slide_path)
                description = self._describe_slide(slide_image)
                slide_text.append(f"\n[SLIDE VISUAL DESCRIPTION: {description}]\n")
                self.chart_descriptions[slide_filename] = description

            full_text.append("\n".join(slide_text))

        return "\n\n".join(full_text)

    def _convert_pptx_to_images(
        self, pptx_path: str, output_dir: str
    ) -> List[Image.Image]:
        images = []
        try:
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                pdf_path = os.path.join(tmpdir, "presentation.pdf")
                cmd = [
                    "soffice",
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    tmpdir,
                    pptx_path,
                ]
                result = subprocess.run(
                    cmd, capture_output=True, timeout=120, text=True
                )
                if result.returncode == 0:
                    pdf_files = [f for f in os.listdir(tmpdir) if f.endswith(".pdf")]
                    if pdf_files:
                        pdf_path = os.path.join(tmpdir, pdf_files[0])
                        doc = fitz.open(pdf_path)
                        for page_num in range(len(doc)):
                            page = doc[page_num]
                            mat = fitz.Matrix(2.0, 2.0)
                            pix = page.get_pixmap(matrix=mat)
                            img_bytes = pix.tobytes("png")
                            img = Image.open(io.BytesIO(img_bytes))
                            images.append(img)
                        doc.close()
        except Exception as e:
            print(f"  Error converting PPTX to images: {e}")
        return images

    def _extract_text_from_docx(
        self, docx_path: str, file_output_dir: str, progress_callback=None
    ) -> str:
        print(f"Processing DOCX: {docx_path}")
        doc = Document(docx_path)
        full_text = []

        for para_idx, para in enumerate(doc.paragraphs):
            if para.text.strip():
                full_text.append(para.text)
            if progress_callback and para_idx % 10 == 0:
                progress_callback(para_idx, len(doc.paragraphs))

        for rel_idx, rel in enumerate(doc.part.rels.values()):
            if "image" in rel.target_ref:
                try:
                    image_data = rel.target_part.blob
                    image = Image.open(io.BytesIO(image_data))
                    if self._is_likely_chart(image):
                        if self.use_vision:
                            description = self._describe_image(image)
                            full_text.append(f"\n[CHART DESCRIPTION: {description}]\n")
                            chart_filename = f"docx_chart{rel_idx + 1}.png"
                            chart_path = os.path.join(file_output_dir, chart_filename)
                            image.save(chart_path)
                            self.chart_descriptions[chart_filename] = description
                    ocr_text = pytesseract.image_to_string(image)
                    if ocr_text.strip():
                        full_text.append(f"\n[OCR from image: {ocr_text}]\n")
                except Exception as e:
                    print(f"  Error processing image {rel_idx}: {e}")

        return "\n\n".join(full_text)

    def extract_charts_as_images(
        self, page, file_output_dir: str, page_num: int
    ) -> List[Image.Image]:
        charts = []
        try:
            mat = fitz.Matrix(4.0, 4.0)
            pix = page.get_pixmap(matrix=mat)
            page_image = Image.open(io.BytesIO(pix.tobytes("png")))

            print(f"  Running chart detection on page {page_num}...")
            chart_boxes_tatr = self.chart_detector_tatr.detect(page_image)
            chart_boxes_heuristic = self.chart_detector_heuristic.detect(page_image)

            print(
                f"  Combining {len(chart_boxes_tatr)} TATR + {len(chart_boxes_heuristic)} heuristic boxes..."
            )
            chart_boxes_combined = self._combine_and_deduplicate_boxes(
                chart_boxes_tatr, chart_boxes_heuristic, iou_threshold=0.5
            )
            print(
                f"  Final: {len(chart_boxes_combined)} unique chart regions after deduplication."
            )

            for i, box in enumerate(chart_boxes_combined):
                box = self._expand_box(box, page_image.width, page_image.height)
                chart_image = page_image.crop(box)
                if chart_image.width < 150 or chart_image.height < 150:
                    continue
                charts.append(chart_image)
                chart_filename = f"page{page_num}_chart{i + 1}.png"
                chart_path = os.path.join(file_output_dir, chart_filename)
                chart_image.save(chart_path)

        except Exception as e:
            print(f"    Error during chart detection on page {page_num}: {e}")
        return charts

    def _looks_like_chart(self, image: Image.Image) -> bool:
        try:
            width, height = image.size
            ocr_text = pytesseract.image_to_string(image).strip()
            text_density = (len(ocr_text) / (width * height)) * 1000
            if text_density > 0.4:
                return False
            img_array = np.array(image.convert("L"))
            variance = np.var(img_array)
            return variance > 500
        except:
            img_array = np.array(image.convert("L"))
            variance = np.var(img_array)
            return variance > 500

    def _extract_text_from_pdf(
        self, pdf_path: str, file_output_dir: str, progress_callback=None
    ) -> str:
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

            image_list = page.get_images(full=True)
            for img_idx, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))

                    if self._is_likely_chart(image):
                        if self.use_vision:
                            description = self._describe_image(image)
                            page_text.append(f"\n[CHART DESCRIPTION: {description}]\n")
                            chart_filename = (
                                f"page{page_num + 1}_embedded{img_idx + 1}.png"
                            )
                            chart_path = os.path.join(file_output_dir, chart_filename)
                            image.save(chart_path)
                            self.chart_descriptions[chart_filename] = description

                    ocr_text = pytesseract.image_to_string(image)
                    if ocr_text.strip():
                        page_text.append(f"\n[OCR from image: {ocr_text}]\n")

                except Exception as e:
                    print(f"    Error processing embedded image {img_idx}: {str(e)}")

            chart_images = self.extract_charts_as_images(
                page=page, page_num=page_num + 1, file_output_dir=file_output_dir
            )

            for chart_idx, chart_image in enumerate(chart_images):
                try:
                    if self.use_vision:
                        description = self._describe_image(chart_image)
                        page_text.append(f"\n[CHART DESCRIPTION: {description}]\n")
                        chart_filename = f"page{page_num + 1}_chart{chart_idx + 1}.png"
                        self.chart_descriptions[chart_filename] = description

                    ocr_text = pytesseract.image_to_string(chart_image)
                    if ocr_text.strip():
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
        width, height = image.size
        if width < 200 or height < 200:
            return False
        try:
            ocr_text = pytesseract.image_to_string(image).strip()
            text_density = (
                (len(ocr_text) / (width * height)) * 1000 if (width * height) > 0 else 0
            )
            if text_density > 0.7:
                return False
            return True
        except:
            return True

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

    # --- Chart Detection Utility Methods ---

    def _expand_box(
        self,
        box: Tuple[int, int, int, int],
        page_width: int,
        page_height: int,
        padding_percent: float = 0.05,
    ) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        pad_x = int((x2 - x1) * padding_percent)
        pad_y = int((y2 - y1) * padding_percent)
        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(page_width, x2 + pad_x),
            min(page_height, y2 + pad_y),
        )

    def _calculate_iou(
        self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]
    ) -> float:
        x1_inter, y1_inter = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2_inter, y2_inter = min(box1[2], box2[2]), min(box1[3], box2[3])

        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _combine_and_deduplicate_boxes(
        self, boxes1: List, boxes2: List, iou_threshold: float = 0.7
    ) -> List:
        all_boxes = list(set(boxes1 + boxes2))
        unique_boxes = []
        all_boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)

        for box in all_boxes:
            is_duplicate = any(
                self._calculate_iou(box, unique_box) > iou_threshold
                for unique_box in unique_boxes
            )
            if not is_duplicate:
                unique_boxes.append(box)
        return unique_boxes
