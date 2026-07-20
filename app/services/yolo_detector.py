from typing import List, Dict, Any
import os

import cv2
import numpy as np

from app.core.config import settings


class WhiteboardYOLODetector:
    def __init__(self):
        self.model = None

        self.class_names = [
            "rectangle",
            "rounded_rectangle",
            "circle",
            "ellipse",
            "diamond",
            "hexagon",
            "parallelogram",
            "cylinder",
            "sticky_note",
            "text_box",
            "arrow",
            "line",
            "dashed_line",
            "dotted_line",
            "decision",
            "process",
            "terminator",
            "data",
            "database",
            "document",
            "cloud",
            "actor",
            "use_case",
            "package",
            "component",
        ]

    def load_model(self):
        model_path = settings.yolo_model_path

        if not model_path or not os.path.exists(model_path):
            print(
                "Custom whiteboard YOLO model not found. "
                "Using OpenCV fallback detection."
            )
            self.model = None
            return

        try:
            from ultralytics import YOLO

            self.model = YOLO(model_path)
            print(f"Whiteboard YOLO model loaded: {model_path}")

        except Exception as exc:
            print(f"Failed to load YOLO model: {exc}")
            self.model = None