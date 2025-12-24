"""
Document Chart Extractor - PubLayNet Fixed Version with Padding
Converts documents (PDF, Word, PowerPoint) to images and uses PubLayNet (via Detectron2)
to detect charts, figures, and tables with added context padding.
"""

import os
import sys
from pathlib import Path
from typing import List
import tempfile
import shutil
import requests

# Document conversion
from pdf2image import convert_from_path
from PIL import Image

# ML detection
import torch
import cv2
import numpy as np

# Detectron2 imports
try:
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2 import model_zoo
except ImportError:
    print("Error: Detectron2 is not installed.")
    print(
        "Please install it using: pip install 'git+https://github.com/facebookresearch/detectron2.git'"
    )
    sys.exit(1)


class DocumentChartExtractor:
    def __init__(self, confidence_threshold=0.5, use_ml=True, padding=50):
        """
        Initialize the chart extractor with a layout detection model (PubLayNet).

        Args:
            confidence_threshold (float): Minimum score to accept a detection.
            use_ml (bool): Whether to use Detectron2 (True) or CV fallback (False).
            padding (int): Pixels of context to add around the detected chart.
        """
        print(
            f"Initializing Document Chart Extractor (ML={use_ml}, Padding={padding}px)..."
        )

        self.confidence_threshold = confidence_threshold
        self.use_cv_fallback = not use_ml
        self.padding = padding

        # PubLayNet Label Map
        # Note: The specific weights used have 6 classes (plus background).
        self.label_map = {
            0: "Text",
            1: "Title",
            2: "List",
            3: "Table",
            4: "Figure",
            5: "Other",
        }

        if use_ml:
            self._setup_ml_model()
        else:
            print("Using CV-only mode (heuristics only)")

    def _setup_ml_model(self):
        """Configure and load the Detectron2 model with PubLayNet weights."""
        print("Setting up PubLayNet model...")
        try:
            # 1. Setup Base Configuration (Faster R-CNN ResNet50 FPN 3x)
            base_config = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
            self.cfg = get_cfg()
            self.cfg.merge_from_file(model_zoo.get_config_file(base_config))

            # 2. Modify Config to match the Pre-trained Weights (6 classes)
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold

            # Force cpu if CUDA not available
            if not torch.cuda.is_available():
                self.cfg.MODEL.DEVICE = "cpu"
                print("CUDA not available. Running on cpu.")
            else:
                print(f"Running on GPU: {torch.cuda.get_device_name(0)}")

            # 3. Download and Set Weights
            self._ensure_publaynet_weights()

            # 4. Create Predictor
            self.predictor = DefaultPredictor(self.cfg)
            print("ML model loaded successfully!")

        except Exception as e:
            print(f"ML model setup failed: {e}")
            print("Falling back to CV-only mode")
            self.use_cv_fallback = True

    def _ensure_publaynet_weights(self):
        """Download PubLayNet weights from Layout Parser's repository."""
        model_url = "https://www.dropbox.com/s/dgy9c10wykk4lq4/model_final.pth?dl=1"

        cache_dir = Path.home() / ".torch" / "detectron2_models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / "publaynet_faster_rcnn_R_50_FPN_3x.pth"

        if not model_path.exists():
            print(f"Downloading PubLayNet weights (~330MB)...")
            try:
                response = requests.get(model_url, stream=True, allow_redirects=True)
                response.raise_for_status()

                with open(model_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                print("\nDownload complete.")
            except Exception as e:
                print(f"\nError downloading model: {e}")
                if model_path.exists():
                    model_path.unlink()
                raise e

        self.cfg.MODEL.WEIGHTS = str(model_path)

    def document_to_images(self, file_path: str, output_dir: str = None) -> List[str]:
        """Convert PDF/DOCX/PPTX to a list of page images."""
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        else:
            os.makedirs(output_dir, exist_ok=True)

        print(f"Converting {file_path.name} to images...")

        if ext == ".pdf":
            return self._pdf_to_images(file_path, output_dir)
        elif ext in [".pptx", ".ppt"]:
            return self._office_to_images(file_path, output_dir, "pptx")
        elif ext in [".docx", ".doc"]:
            return self._office_to_images(file_path, output_dir, "docx")
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _pdf_to_images(self, pdf_path: Path, output_dir: str) -> List[str]:
        """Convert PDF to images using pdf2image."""
        try:
            images = convert_from_path(str(pdf_path), dpi=200)
            image_paths = []

            for i, img in enumerate(images):
                img_path = os.path.join(output_dir, f"page_{i+1:03d}.png")
                img.save(img_path, "PNG")
                image_paths.append(img_path)

            return image_paths
        except Exception as e:
            print(f"Error converting PDF: {e}")
            print("Ensure 'poppler-utils' is installed on your system.")
            raise

    def _office_to_images(
        self, file_path: Path, output_dir: str, file_type: str
    ) -> List[str]:
        """Convert Office docs to images via LibreOffice headless."""
        try:
            import subprocess

            cmd = [
                "soffice",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                output_dir,
                str(file_path),
            ]
            subprocess.run(cmd, capture_output=True, timeout=60)

            pdf_path = Path(output_dir) / (file_path.stem + ".pdf")
            if pdf_path.exists():
                return self._pdf_to_images(pdf_path, output_dir)
            else:
                raise FileNotFoundError("Intermediate PDF not found.")
        except Exception as e:
            print(f"Conversion error: {e}")
            raise

    def detect_elements(self, image_path: str) -> List[dict]:
        """
        Detect non-text elements (Charts, Tables, Figures) in a page image.
        """
        img = cv2.imread(image_path)
        if img is None:
            return []

        if self.use_cv_fallback:
            return self._detect_charts_cv(img)

        try:
            outputs = self.predictor(img)

            instances = outputs["instances"].to("cpu")
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()

            detections = []
            img_h, img_w = img.shape[:2]

            for box, score, cls_id in zip(boxes, scores, classes):
                cls_name = self.label_map.get(int(cls_id), "Unknown")

                # Filter: Keep Figures and Tables
                if cls_name in ["Figure", "Table"]:
                    x1, y1, x2, y2 = map(int, box)

                    # Apply Padding
                    x1 = max(0, x1 - self.padding)
                    y1 = max(0, y1 - self.padding)
                    x2 = min(img_w, x2 + self.padding)
                    y2 = min(img_h, y2 + self.padding)

                    cropped = img[y1:y2, x1:x2]
                    if cropped.size == 0:
                        continue

                    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

                    detections.append(
                        {
                            "type": cls_name,
                            "bbox": (x1, y1, x2, y2),
                            "confidence": float(score),
                            "cropped_image": cropped_rgb,
                        }
                    )

            return detections

        except Exception as e:
            print(f"Prediction error: {e}")
            return self._detect_charts_cv(img)

    def _detect_charts_cv(self, img: np.ndarray) -> List[dict]:
        """Fallback heuristics for chart detection."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_h, img_w = img.shape[:2]

        edges = cv2.Canny(gray, 50, 150)
        dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=3)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 100 or h < 100:
                continue
            if w / h > 5 or h / w > 5:
                continue

            area_pct = (w * h) / (img_w * img_h)
            if area_pct < 0.05 or area_pct > 0.9:
                continue

            # Apply Padding
            x_pad = max(0, x - self.padding)
            y_pad = max(0, y - self.padding)
            w_pad = min(img_w - x_pad, w + 2 * self.padding)
            h_pad = min(img_h - y_pad, h + 2 * self.padding)

            cropped = img[y_pad : y_pad + h_pad, x_pad : x_pad + w_pad]
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

            detections.append(
                {
                    "type": "Figure_CV",
                    "bbox": (x_pad, y_pad, x_pad + w_pad, y_pad + h_pad),
                    "confidence": 0.5,
                    "cropped_image": cropped_rgb,
                }
            )

        return detections

    def process_document(self, file_path: str, output_dir: str = "extracted_charts"):
        """Main execution flow."""
        try:
            temp_dir = tempfile.mkdtemp()
            image_paths = self.document_to_images(file_path, temp_dir)

            os.makedirs(output_dir, exist_ok=True)
            total_charts = 0

            print(f"\nProcessing {len(image_paths)} pages...")

            for page_idx, img_path in enumerate(image_paths, 1):
                print(f"  Analyzing Page {page_idx}...", end="", flush=True)

                detections = self.detect_elements(img_path)
                print(f" Found {len(detections)} elements.")

                for i, det in enumerate(detections, 1):
                    out_name = f"p{page_idx:03d}_item{i}_{det['type']}.png"
                    out_path = os.path.join(output_dir, out_name)
                    Image.fromarray(det["cropped_image"]).save(out_path)
                    total_charts += 1

            print(f"\nDone! Extracted {total_charts} elements to '{output_dir}/'")

        except Exception as e:
            print(f"\nCritical Error: {e}")
        finally:
            if "temp_dir" in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to document")
    parser.add_argument("--out", default="output", help="Output directory")
    parser.add_argument(
        "--padding", type=int, default=75, help="Padding around charts (pixels)"
    )

    # Simulate arguments if running directly
    if len(sys.argv) == 1:
        print("Usage: python script.py <file_path> [--padding 50]")
        return

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
        return

    # Pass padding from arguments
    extractor = DocumentChartExtractor(
        confidence_threshold=0.5, use_ml=True, padding=args.padding
    )
    extractor.process_document(args.file, args.out)


if __name__ == "__main__":
    main()
