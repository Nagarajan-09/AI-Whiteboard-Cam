import logging
import os
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

from app.models.whiteboard import BoundingBox, DetectionResult, WhiteboardElement

logger = logging.getLogger(__name__)


class YOLODetector:
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "auto",
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model_path = model_path
        self.device = self._get_device(device)
        self.model = None
        self.class_names: Dict[int, str] = {}
        self._load_model()

    def _get_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"

        return device

    def _load_model(self) -> None:
        if not self.model_path or not os.path.exists(self.model_path):
            logger.info(
                "No custom whiteboard-trained YOLO model found. "
                "Skipping YOLO and using contour-based shape detection instead."
            )
            self.model = None
            return

        try:
            from ultralytics import YOLO

            self.model = YOLO(self.model_path)
            self.model.conf = self.conf_threshold
            self.model.iou = self.iou_threshold
            self.class_names = self.model.names if hasattr(self.model, "names") else {}
            logger.info(f"Custom YOLO model loaded on {self.device}: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load custom YOLO model: {e}")
            self.model = None

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        if self.model is None:
            self._load_model()
            if self.model is None:
                return self._fallback_detection(image)

        results = self.model(image, verbose=False)
        detections: List[DetectionResult] = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = self.class_names.get(cls, str(cls))

                    if conf >= self.conf_threshold:
                        bbox = BoundingBox(
                            x=float(x1) / image.shape[1],
                            y=float(y1) / image.shape[0],
                            width=float(x2 - x1) / image.shape[1],
                            height=float(y2 - y1) / image.shape[0],
                        )
                        detections.append(
                            DetectionResult(
                                class_id=cls,
                                class_name=class_name,
                                confidence=float(conf),
                                bbox=bbox,
                            )
                        )

        if not detections:
            logger.info(
                "YOLO (COCO-pretrained) found nothing whiteboard-relevant. "
                "Falling back to contour detection."
            )
            return self._fallback_detection(image)

        return detections

    def _fallback_detection(self, image: np.ndarray) -> List[DetectionResult]:
        """Fallback to traditional computer vision if YOLO fails or finds nothing."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        logger.info(f"Fallback contour detection found {len(contours)} raw contours")

        detections: List[DetectionResult] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:
                continue

            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            x, y, w, h = cv2.boundingRect(contour)
            shape_type = self._classify_shape(approx, w, h)

            bbox = BoundingBox(
                x=float(x) / image.shape[1],
                y=float(y) / image.shape[0],
                width=float(w) / image.shape[1],
                height=float(h) / image.shape[0],
            )
            detections.append(
                DetectionResult(
                    class_id=0,
                    class_name=shape_type,
                    confidence=0.8,
                    bbox=bbox,
                )
            )

        logger.info(f"Fallback contour detection retained {len(detections)} shapes")
        return detections

    def _classify_shape(self, approx: np.ndarray, w: int, h: int) -> str:
        """Simple shape classification based on contour approximation."""
        vertices = len(approx)
        aspect_ratio = w / h if h > 0 else 1

        if vertices == 3:
            return "triangle"
        if vertices == 4:
            if 0.8 <= aspect_ratio <= 1.2:
                return "square" if abs(w - h) < 10 else "rectangle"
            return "rectangle"
        if vertices == 5:
            return "pentagon"
        if vertices == 6:
            return "hexagon"
        if vertices > 6:
            if 0.8 <= aspect_ratio <= 1.2:
                return "circle"
            return "ellipse"

        return "rectangle"

    def detect_shapes(self, image: np.ndarray) -> List[WhiteboardElement]:
        detections = self.detect(image)
        elements: List[WhiteboardElement] = []

        for i, det in enumerate(detections):
            element = WhiteboardElement(
                id=f"elem_{i}",
                type=det.class_name,
                bbox=det.bbox,
                text=None,
                confidence=det.confidence,
                style=self._get_default_style(det.class_name),
            )
            elements.append(element)

        return elements

    def _get_default_style(self, class_name: str) -> dict:
        styles = {
            "rectangle": {"stroke": "#1f77b4", "fill": "transparent", "strokeWidth": 2},
            "rounded_rectangle": {
                "stroke": "#1f77b4",
                "fill": "transparent",
                "strokeWidth": 2,
                "rx": 10,
            },
            "circle": {"stroke": "#ff7f0e", "fill": "transparent", "strokeWidth": 2},
            "ellipse": {"stroke": "#ff7f0e", "fill": "transparent", "strokeWidth": 2},
            "triangle": {"stroke": "#2ca02c", "fill": "transparent", "strokeWidth": 2},
            "pentagon": {"stroke": "#ff7f0e", "fill": "transparent", "strokeWidth": 2},
            "hexagon": {"stroke": "#ff7f0e", "fill": "transparent", "strokeWidth": 2},
            "diamond": {"stroke": "#2ca02c", "fill": "transparent", "strokeWidth": 2},
            "arrow": {
                "stroke": "#d62728",
                "fill": "#d62728",
                "strokeWidth": 2,
                "endArrow": "block",
            },
            "line": {"stroke": "#9467bd", "strokeWidth": 2},
            "sticky_note": {"stroke": "#f0e68c", "fill": "#f0e68c", "strokeWidth": 1},
            "text_box": {"stroke": "#8c564b", "fill": "transparent", "strokeWidth": 1},
            "decision": {"stroke": "#2ca02c", "fill": "transparent", "strokeWidth": 2},
            "process": {"stroke": "#1f77b4", "fill": "transparent", "strokeWidth": 2},
            "terminator": {
                "stroke": "#e377c2",
                "fill": "transparent",
                "strokeWidth": 2,
            },
            "data": {"stroke": "#1f77b4", "fill": "transparent", "strokeWidth": 2},
            "database": {"stroke": "#1f77b4", "fill": "transparent", "strokeWidth": 2},
            "document": {"stroke": "#1f77b4", "fill": "transparent", "strokeWidth": 2},
        }
        return styles.get(class_name, styles["rectangle"])


class WhiteboardYOLODetector(YOLODetector):
    WHITEBOARD_CLASSES = {
        0: "rectangle",
        1: "circle",
        2: "triangle",
        3: "diamond",
        4: "arrow",
        5: "line",
        6: "ellipse",
        7: "hexagon",
        8: "pentagon",
        9: "sticky_note",
        10: "text_box",
        11: "connector",
        12: "decision",
        13: "process",
        14: "terminator",
        15: "data",
        16: "database",
        17: "document",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_names = self.WHITEBOARD_CLASSES

    def detect_elements(self, image: np.ndarray) -> List[WhiteboardElement]:
        detections = self.detect(image)
        elements: List[WhiteboardElement] = []

        for i, det in enumerate(detections):
            element_type = self.WHITEBOARD_CLASSES.get(det.class_id, det.class_name)
            element = WhiteboardElement(
                id=f"elem_{i}",
                type=element_type,
                bbox=det.bbox,
                text=None,
                confidence=det.confidence,
                style=self._get_default_style(element_type),
            )
            elements.append(element)

        return elements
