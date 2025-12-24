"""
src/utils/chart_detection.py

Chart detection module using Detectron2 with PubLayNet weights.
Includes fallback to Computer Vision heuristics if ML fails or is disabled.
"""

import os
import sys
import shutil
import requests
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# Image processing
from PIL import Image
import cv2
import numpy as np
import torch
import gc

# Detectron2 imports (Guarded)
try:
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2 import model_zoo

    _DETECTRON2_AVAILABLE = True
except ImportError:
    _DETECTRON2_AVAILABLE = False


class ChartDetector:
    """Base interface for chart detection."""

    def detect(self, page_image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """
        Detect charts/figures in a page image.
        Returns: List of bounding boxes (x1, y1, x2, y2)
        """
        return []

    def offload_model(self):
        """Free up resources."""
        pass


class PubLayNetDetector(ChartDetector):
    """
    Detects Charts, Tables, and Figures using Faster R-CNN (ResNet50)
    trained on the PubLayNet dataset.
    """

    def __init__(self, confidence_threshold=0.5, padding=75):
        self.confidence_threshold = confidence_threshold
        self.padding = padding
        self.predictor = None
        self.cfg = None
        self._is_loaded = False

        # PubLayNet Label Map (6 classes)
        self.label_map = {
            0: "Text",
            1: "Title",
            2: "List",
            3: "Table",
            4: "Figure",
            5: "Other",
        }

        # Target classes to extract (We usually want Figures and Tables for charts)
        self.target_classes = ["Figure", "Table"]

    def load_model(self):
        """Initialize the Detectron2 model."""
        if self._is_loaded:
            return

        if not _DETECTRON2_AVAILABLE:
            print("Warning: Detectron2 not installed. Falling back to CV heuristics.")
            return

        print("Loading PubLayNet Detector...")
        try:
            # 1. Setup Base Config
            base_config = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
            self.cfg = get_cfg()
            self.cfg.merge_from_file(model_zoo.get_config_file(base_config))

            # 2. Modify Config
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold

            # Device selection
            if torch.cuda.is_available():
                self.cfg.MODEL.DEVICE = "cuda"
            elif torch.backends.mps.is_available():
                self.cfg.MODEL.DEVICE = "cpu"
            else:
                self.cfg.MODEL.DEVICE = "cpu"

            # 3. Ensure Weights
            self._ensure_publaynet_weights()

            # 4. Create Predictor
            self.predictor = DefaultPredictor(self.cfg)
            self._is_loaded = True
            print(f"✓ PubLayNet loaded on {self.cfg.MODEL.DEVICE}")

        except Exception as e:
            print(f"✗ Failed to load PubLayNet model: {e}")
            print("Falling back to CV heuristics.")

    def _ensure_publaynet_weights(self):
        """Download weights if not present."""
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
                print("Download complete.")
            except Exception as e:
                if model_path.exists():
                    model_path.unlink()
                raise e

        self.cfg.MODEL.WEIGHTS = str(model_path)

    def detect(self, page_image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """
        Runs detection on a PIL Image.
        Returns list of bboxes: (x1, y1, x2, y2)
        """
        # Ensure model is loaded
        if not self._is_loaded and _DETECTRON2_AVAILABLE:
            self.load_model()

        # Convert PIL to CV2 (BGR)
        img_np = np.array(page_image)
        if img_np.shape[-1] == 4:  # Handle RGBA
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        detections = []

        # 1. Try ML Detection
        if self._is_loaded and self.predictor:
            try:
                outputs = self.predictor(img_np)
                instances = outputs["instances"].to("cpu")
                boxes = instances.pred_boxes.tensor.numpy()
                scores = instances.scores.numpy()
                classes = instances.pred_classes.numpy()
                img_h, img_w = img_np.shape[:2]

                for box, score, cls_id in zip(boxes, scores, classes):
                    cls_name = self.label_map.get(int(cls_id), "Unknown")

                    if cls_name in self.target_classes:
                        x1, y1, x2, y2 = map(int, box)

                        # Apply Padding
                        x1 = max(0, x1 - self.padding)
                        y1 = max(0, y1 - self.padding)
                        x2 = min(img_w, x2 + self.padding)
                        y2 = min(img_h, y2 + self.padding)

                        detections.append((x1, y1, x2, y2))

                # If ML found something, return it. If not, try fallback?
                # Usually if ML runs but finds nothing, there is nothing.
                # But we can be aggressive and try CV if ML yields 0.
                if detections:
                    return detections

            except Exception as e:
                print(f"Prediction error: {e}")

        # 2. Fallback CV Heuristics
        print("Using CV fallback for chart detection...")
        return self._detect_cv_fallback(img_np)

    def _detect_cv_fallback(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Heuristic detection using Canny edges and contours."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_h, img_w = img.shape[:2]

        # Edge detection + Dilation to merge text blocks
        edges = cv2.Canny(gray, 50, 150)
        dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=3)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        bboxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Filter noise
            if w < 100 or h < 100:
                continue
            if w / h > 5 or h / w > 5:
                continue  # Ignore extreme aspect ratios (lines)

            area_pct = (w * h) / (img_w * img_h)
            if area_pct < 0.05 or area_pct > 0.9:
                continue

            # Apply Padding
            x_pad = max(0, x - self.padding)
            y_pad = max(0, y - self.padding)
            w_pad = min(img_w - x_pad, w + 2 * self.padding)
            h_pad = min(img_h - y_pad, h + 2 * self.padding)

            bboxes.append((x_pad, y_pad, x_pad + w_pad, y_pad + h_pad))

        return bboxes

    def offload_model(self):
        """Release resources to free up GPU/RAM."""
        if not self._is_loaded:
            return

        print("Offloading PubLayNet Detector...")
        self.predictor = None
        self.cfg = None
        self._is_loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

        print("✓ PubLayNet offloaded.")


# Backward compatibility / Helper Factory
def get_detector() -> ChartDetector:
    """Returns the best available detector."""
    return PubLayNetDetector()


# Deprecated classes kept for interface compatibility if needed
class HeuristicDetector(ChartDetector):
    def detect(self, page_image: Image.Image) -> List[Tuple[int, int, int, int]]:
        det = PubLayNetDetector()
        # Force fallback behavior
        det._is_loaded = False
        img_np = np.array(page_image)
        if img_np.shape[-1] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        return det._detect_cv_fallback(img_np)
