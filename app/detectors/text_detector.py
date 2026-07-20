import cv2
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging

from app.models.whiteboard import BoundingBox, TextDetection, WhiteboardElement

logger = logging.getLogger(__name__)


class TextDetector:
    def __init__(
        self,
        use_paddle: bool = True,
        lang: str = 'en',
        use_gpu: bool = False,
        det_model_dir: Optional[str] = None,
        rec_model_dir: Optional[str] = None
    ):
        self.use_paddle = use_paddle
        self.lang = lang
        self.use_gpu = use_gpu
        self.det_model_dir = det_model_dir
        self.rec_model_dir = rec_model_dir
        self.ocr = None
        self.tesseract = None
    
    def load(self):
        if self.use_paddle:
            self._load_paddle()
        else:
            self._load_tesseract()
    
    def _load_paddle(self):
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang,
                use_gpu=self.use_gpu,
                det_model_dir=self.det_model_dir,
                rec_model_dir=self.rec_model_dir,
                show_log=False
            )
            logger.info("PaddleOCR loaded successfully")
        except ImportError:
            logger.warning("PaddleOCR not available, falling back to Tesseract")
            self.use_paddle = False
            self._load_tesseract()
        except Exception as e:
            logger.error(f"Failed to load PaddleOCR: {e}")
            self.use_paddle = False
            self._load_tesseract()
    
    def _load_tesseract(self):
        try:
            import pytesseract
            self.tesseract = pytesseract
            logger.info("Tesseract loaded successfully")
        except ImportError:
            logger.error("Neither PaddleOCR nor Tesseract available")
            raise
    
    def detect(self, image: np.ndarray) -> List[TextDetection]:
        if self.use_paddle and self.ocr is not None:
            return self._detect_paddle(image)
        elif self.tesseract is not None:
            return self._detect_tesseract(image)
        else:
            self.load()
            return self.detect(image)
    
    def _detect_paddle(self, image: np.ndarray) -> List[TextDetection]:
        results = self.ocr.ocr(image, cls=True)
        detections = []
        
        if results and results[0]:
            for line in results[0]:
                bbox_points = line[0]
                text = line[1][0]
                confidence = line[1][1]
                
                xs = [p[0] for p in bbox_points]
                ys = [p[1] for p in bbox_points]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
                
                bbox = BoundingBox(
                    x=float(x1),
                    y=float(y1),
                    width=float(x2 - x1),
                    height=float(y2 - y1),
                    confidence=float(confidence),
                    class_id=0,
                    class_name='text'
                )
                
                detections.append(TextDetection(
                    text=text,
                    bbox=bbox,
                    confidence=float(confidence),
                    language=self.lang
                ))
        
        return detections
    
    def _detect_tesseract(self, image: np.ndarray) -> List[TextDetection]:
        try:
            data = self.tesseract.image_to_data(image, output_type=self.tesseract.Output.DICT)
            detections = []
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = data['conf'][i]
                
                if text and conf > 30:
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    bbox = BoundingBox(
                        x=float(x),
                        y=float(y),
                        width=float(w),
                        height=float(h),
                        confidence=conf / 100.0,
                        class_id=0,
                        class_name='text'
                    )
                    detections.append(TextDetection(
                        text=text,
                        bbox=bbox,
                        confidence=conf / 100.0,
                        language=self.lang
                    ))
            
            return detections
        except Exception as e:
            logger.error(f"Tesseract detection failed: {e}")
            return []
    
    def associate_text_with_elements(
        self,
        texts: List[TextDetection],
        elements: List[WhiteboardElement],
        max_distance: float = 50.0
    ) -> List[WhiteboardElement]:
        for elem in elements:
            elem_center = elem.bbox.center()
            best_text = None
            best_distance = float('inf')
            
            for text in texts:
                text_center = text.bbox.center()
                distance = np.sqrt(
                    (elem_center[0] - text_center[0]) ** 2 +
                    (elem_center[1] - text_center[1]) ** 2
                )
                
                if distance < best_distance and distance < max_distance:
                    best_distance = distance
                    best_text = text
            
            if best_text:
                elem.text = best_text.text
                elem.text_confidence = best_text.confidence
        
        return elements


class WhiteboardTextDetector(TextDetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_text_size = 10
        self.max_text_size = 500
    
    def detect_and_associate(
        self,
        image: np.ndarray,
        elements: List[WhiteboardElement]
    ) -> List[WhiteboardElement]:
        texts = self.detect(image)
        texts = [t for t in texts if self.min_text_size <= t.bbox.width <= self.max_text_size]
        return self.associate_text_with_elements(texts, elements)
    
    def detect_all_text(self, image: np.ndarray) -> List[TextDetection]:
        return self.detect(image)