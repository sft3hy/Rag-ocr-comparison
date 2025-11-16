import os

# Fix tokenizer warning - MUST be before any transformers imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
import pytesseract
from PIL import Image
from typing import List, Tuple
from transformers import AutoImageProcessor, TableTransformerForObjectDetection


class ChartDetector:
    """
    Base class for chart detection models.
    Handles device selection automatically.
    """

    def __init__(self, *args, **kwargs):
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"ChartDetector using device: {self.device}")
        self.model = self.load_model(*args, **kwargs)

    def load_model(self, *args, **kwargs):
        """Abstract method to load the specific detection model."""
        raise NotImplementedError("Subclasses must implement the load_model method.")

    def detect(self, page_image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """Abstract method to detect charts in an image and return bounding boxes."""
        raise NotImplementedError("Subclasses must implement the detect method.")


class HeuristicDetector(ChartDetector):
    """
    A heuristic-based chart detector that analyzes image sections for chart-like properties.
    It looks for regions with low text density and high pixel variance.
    """

    def load_model(self, *args, **kwargs):
        """No model to load for heuristic detection."""
        print("HeuristicDetector initialized")
        return None

    def detect(self, page_image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """
        Divides the page into a grid and checks each section for chart characteristics.

        Args:
            page_image (Image.Image): The image of the document page.

        Returns:
            A list of bounding box tuples (x1, y1, x2, y2) for potential charts.
        """
        boxes = []
        page_width, page_height = page_image.size
        grid_divisions = 4  # Increased to 4x4 for finer granularity
        section_width = page_width // grid_divisions
        section_height = page_height // grid_divisions

        print(f"    Heuristic: Scanning {grid_divisions}x{grid_divisions} grid...")

        for row in range(grid_divisions):
            for col in range(grid_divisions):
                x1, y1 = col * section_width, row * section_height
                x2, y2 = (col + 1) * section_width, (row + 1) * section_height
                section_image = page_image.crop((x1, y1, x2, y2))

                if self._looks_like_chart(section_image):
                    # Add padding to avoid cutting off charts (10% on each side)
                    padding = int(min(section_width, section_height) * 0.3)
                    x1_padded = max(0, x1 - padding)
                    y1_padded = max(0, y1 - padding)
                    x2_padded = min(page_width, x2 + padding)
                    y2_padded = min(page_height, y2 + padding)
                    boxes.append((x1_padded, y1_padded, x2_padded, y2_padded))

        # Merge overlapping boxes
        boxes = self._merge_overlapping_boxes(boxes)
        print(f"    Heuristic detected {len(boxes)} potential chart regions.")
        return boxes

    def _looks_like_chart(self, image: Image.Image) -> bool:
        """Determines if an image section is likely a chart."""
        try:
            width, height = image.size
            if width < 80 or height < 80:  # Reduced minimum size
                return False

            # Convert to grayscale for analysis
            img_array = np.array(image.convert("L"))

            # Calculate variance (charts have patterns/structure)
            variance = np.var(img_array)

            # Calculate edge density (charts have lines/axes)
            edges_y = (
                np.abs(np.diff(img_array, axis=0)).sum()
                if img_array.shape[0] > 1
                else 0
            )
            edges_x = (
                np.abs(np.diff(img_array, axis=1)).sum()
                if img_array.shape[1] > 1
                else 0
            )
            edge_density = (edges_x + edges_y) / (width * height)

            # Check color variance (charts often have distinct colors)
            if image.mode == "RGB":
                color_std = np.std(np.array(image), axis=(0, 1)).mean()
            else:
                color_std = 0

            # Charts typically have low text density
            try:
                ocr_text = pytesseract.image_to_string(image, timeout=3).strip()
                text_density = (len(ocr_text) / (width * height)) * 1000
            except:
                text_density = 0

            # Multiple criteria for chart detection
            has_high_variance = variance > 300  # Relaxed from 400
            has_edges = edge_density > 40  # Relaxed from 50
            has_color = color_std > 20
            low_text = text_density < 1.5  # Relaxed from 1.0

            # Consider it a chart if it meets multiple criteria
            score = sum([has_high_variance, has_edges, has_color, low_text])
            return score >= 2  # At least 2 criteria must be met

        except Exception as e:
            print(f"      Heuristic error: {e}")
            return False

    def _merge_overlapping_boxes(
        self, boxes: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """Merge overlapping bounding boxes."""
        if not boxes:
            return []

        merged = []
        boxes = sorted(boxes, key=lambda x: (x[0], x[1]))

        for box in boxes:
            if not merged:
                merged.append(box)
                continue

            x1, y1, x2, y2 = box
            overlap = False

            for i, (mx1, my1, mx2, my2) in enumerate(merged):
                # Check if boxes overlap
                if not (x2 < mx1 or x1 > mx2 or y2 < my1 or y1 > my2):
                    # Merge boxes
                    merged[i] = (min(x1, mx1), min(y1, my1), max(x2, mx2), max(y2, my2))
                    overlap = True
                    break

            if not overlap:
                merged.append(box)

        return merged


class TATRDetector(ChartDetector):
    """
    A chart detector using the Microsoft Table Transformer (TATR) model.
    Excellent for detecting tables and objects with clear boundaries.
    """

    def load_model(self, model_path: str = "microsoft/table-transformer-detection"):
        """
        Loads the Table Transformer model and processor from Hugging Face.
        """
        try:
            print(f"Loading Table Transformer model: {model_path}")
            self.image_processor = AutoImageProcessor.from_pretrained(model_path)
            model = TableTransformerForObjectDetection.from_pretrained(model_path)
            model.to(self.device)
            print("✓ Table Transformer model loaded successfully.")
            return model
        except Exception as e:
            print(f"✗ Error loading TATR model: {e}. This detector will be disabled.")
            return None

    def detect(self, page_image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """
        Performs object detection using the TATR model.

        Args:
            page_image (Image.Image): The image of the document page.

        Returns:
            A list of bounding box tuples (x1, y1, x2, y2) for detected objects.
        """
        if not self.model:
            print("    TATR: Model not loaded, returning empty list")
            return []

        try:
            print(f"    TATR: Processing image size {page_image.size}")
            inputs = self.image_processor(images=page_image, return_tensors="pt").to(
                self.device
            )

            with torch.no_grad():
                outputs = self.model(**inputs)

            print(f"    TATR: Model inference complete, processing results...")
            target_sizes = torch.tensor([page_image.size[::-1]]).to(self.device)

            # Try multiple thresholds to see what we get
            thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
            best_results = []
            best_threshold = 0.3

            for threshold in thresholds:
                results = self.image_processor.post_process_object_detection(
                    outputs,
                    threshold=threshold,
                    target_sizes=target_sizes,
                )[0]

                num_detections = len(results["boxes"])
                print(
                    f"    TATR threshold {threshold:.1f}: {num_detections} detections"
                )

                if num_detections > len(best_results):
                    best_results = results["boxes"]
                    best_threshold = threshold

            # Use a low threshold (0.2) to catch more charts
            final_threshold = 0.2
            results = self.image_processor.post_process_object_detection(
                outputs,
                threshold=final_threshold,
                target_sizes=target_sizes,
            )[0]

            # Add padding to bounding boxes to avoid cutting off charts
            page_width, page_height = page_image.size
            boxes = []
            for box in results["boxes"]:
                x1, y1, x2, y2 = map(int, box.tolist())

                # Add 8% padding on each side
                width, height = x2 - x1, y2 - y1
                padding_x = int(width * 0.08)
                padding_y = int(height * 0.08)

                x1 = max(0, x1 - padding_x)
                y1 = max(0, y1 - padding_y)
                x2 = min(page_width, x2 + padding_x)
                y2 = min(page_height, y2 + padding_y)

                boxes.append((x1, y1, x2, y2))

            print(
                f"    TATR FINAL: {len(boxes)} charts/tables at threshold {final_threshold}"
            )
            return boxes

        except Exception as e:
            print(f"    TATR ERROR: {e}")
            import traceback

            traceback.print_exc()
            return []
