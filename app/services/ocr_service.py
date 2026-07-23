import logging
from typing import Any, Dict, List, Optional
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class OCRService:
    def __init__(self):
        self.ocr = None
        self._load_model()

    def _load_model(self):
        if not getattr(settings, "ocr_enabled", True):
            return

        try:
            from paddleocr import PaddleOCR
            lang = settings.ocr_languages[0] if getattr(settings, "ocr_languages", None) else 'en'
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=lang,
                use_gpu=False,
                show_log=False
            )
            logger.info("PaddleOCR model loaded successfully")
        except ImportError:
            logger.info("PaddleOCR not available, attempting Tesseract fallback")
            self._load_tesseract()
        except Exception as e:
            logger.warning(f"Failed to load PaddleOCR: {e}. Attempting Tesseract fallback")
            self._load_tesseract()

    def _load_tesseract(self):
        try:
            import pytesseract
            self.ocr = 'tesseract'
            logger.info("Tesseract loaded as secondary OCR engine")
        except ImportError:
            logger.warning("Neither PaddleOCR nor PyTesseract could be loaded")
            self.ocr = None

    def detect_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if self.ocr is None or image is None or image.size == 0:
            return []

        if hasattr(self.ocr, 'ocr'):
            return self._detect_paddle(image)
        elif self.ocr == 'tesseract':
            return self._detect_tesseract(image)

        return []

    def _detect_paddle(self, image: np.ndarray) -> List[Dict[str, Any]]:
        results = self.ocr.ocr(image, cls=True)
        texts = []

        img_h, img_w = image.shape[:2]
        if img_h == 0 or img_w == 0:
            return []

        if results and results[0]:
            for line in results[0]:
                bbox = line[0]
                text = line[1][0]
                confidence = line[1][1]

                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)

                texts.append({
                    'text': text.strip(),
                    'bbox': [
                        float(x1 / img_w),
                        float(y1 / img_h),
                        float((x2 - x1) / img_w),
                        float((y2 - y1) / img_h)
                    ],
                    'confidence': float(confidence),
                    'language': settings.ocr_languages[0] if getattr(settings, "ocr_languages", None) else 'en'
                })

        return texts

    def _detect_tesseract(self, image: np.ndarray) -> List[Dict[str, Any]]:
        import pytesseract
        from pytesseract import Output

        img_h, img_w = image.shape[:2]
        if img_h == 0 or img_w == 0:
            return []

        data = pytesseract.image_to_data(image, output_type=Output.DICT)
        texts = []

        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = data['conf'][i]

            if text and conf > 30:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

                texts.append({
                    'text': text,
                    'bbox': [
                        float(x / img_w),
                        float(y / img_h),
                        float(w / img_w),
                        float(h / img_h)
                    ],
                    'confidence': float(conf / 100.0),
                    'language': 'en'
                })

        return texts


class TextAssociator:
    @staticmethod
    def associate_text_with_elements(
        texts: List[Dict[str, Any]],
        elements: List[Dict[str, Any]],
        max_distance: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Associates extracted OCR text bounding boxes with diagram elements
        using minimum center distance and spatial containment matching.
        """
        for elem in elements:
            if not elem.get('bbox'):
                continue

            elem_bbox = elem['bbox']
            # Compute center of element
            elem_center = (
                elem_bbox[0] + elem_bbox[2] / 2,
                elem_bbox[1] + elem_bbox[3] / 2
            )

            best_text = None
            best_dist = float('inf')

            for text in texts:
                text_bbox = text['bbox']
                text_center = (
                    text_bbox[0] + text_bbox[2] / 2,
                    text_bbox[1] + text_bbox[3] / 2
                )

                dist = np.sqrt(
                    (elem_center[0] - text_center[0]) ** 2 +
                    (elem_center[1] - text_center[1]) ** 2
                )

                if dist < best_dist and dist < max_distance:
                    best_dist = dist
                    best_text = text

            if best_text:
                elem['text'] = best_text['text']
                elem['text_confidence'] = best_text['confidence']

        return elements

    @staticmethod
    def find_orphan_texts(
        texts: List[Dict[str, Any]],
        elements: List[Dict[str, Any]],
        max_distance: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Returns text elements that do not fall within any detected shape element.
        """
        orphan_texts = []

        for text in texts:
            if not text.get('bbox'):
                continue

            text_bbox = text['bbox']
            text_center = (
                text_bbox[0] + text_bbox[2] / 2,
                text_bbox[1] + text_bbox[3] / 2
            )

            is_associated = False
            for elem in elements:
                if not elem.get('bbox'):
                    continue

                elem_bbox = elem['bbox']
                elem_center = (
                    elem_bbox[0] + elem_bbox[2] / 2,
                    elem_bbox[1] + elem_bbox[3] / 2
                )

                dist = np.sqrt(
                    (elem_center[0] - text_center[0]) ** 2 +
                    (elem_center[1] - text_center[1]) ** 2
                )

                if dist < max_distance:
                    is_associated = True
                    break

            if not is_associated:
                orphan_texts.append(text)

        return orphan_texts