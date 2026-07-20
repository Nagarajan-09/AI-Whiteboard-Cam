from typing import List, Dict, Any, Optional
import numpy as np

from app.core.config import settings


class OCRService:
    def __init__(self):
        self.ocr = None
        self._load_model()
    
    def _load_model(self):
        if not settings.ocr_enabled:
            return
        
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=settings.ocr_languages[0] if settings.ocr_languages else 'en',
                use_gpu=False,
                show_log=False
            )
        except ImportError:
            print("PaddleOCR not available, trying Tesseract")
            self._load_tesseract()
        except Exception as e:
            print(f"Failed to load PaddleOCR: {e}")
            self._load_tesseract()
    
    def _load_tesseract(self):
        try:
            import pytesseract
            self.ocr = 'tesseract'
        except ImportError:
            self.ocr = None
    
    def detect_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if self.ocr is None:
            return []
        
        if hasattr(self.ocr, 'ocr'):
            return self._detect_paddle(image)
        elif self.ocr == 'tesseract':
            return self._detect_tesseract(image)
        
        return []
    
    def _detect_paddle(self, image: np.ndarray) -> List[Dict[str, Any]]:
        results = self.ocr.ocr(image, cls=True)
        texts = []
        
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
                    'text': text,
                    'bbox': [
                        x1 / image.shape[1],
                        y1 / image.shape[0],
                        (x2 - x1) / image.shape[1],
                        (y2 - y1) / image.shape[0]
                    ],
                    'confidence': float(confidence),
                    'language': settings.ocr_languages[0] if settings.ocr_languages else 'en'
                })
        
        return texts
    
    def _detect_tesseract(self, image: np.ndarray) -> List[Dict[str, Any]]:
        import pytesseract
        from pytesseract import Output
        
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
                        x / image.shape[1],
                        y / image.shape[0],
                        w / image.shape[1],
                        h / image.shape[0]
                    ],
                    'confidence': conf / 100.0,
                    'language': 'en'
                })
        
        return texts


class TextAssociator:
    @staticmethod
    def associate_text_with_elements(
        texts: List[Dict[str, Any]],
        elements: List[Dict[str, Any]],
        max_distance: float = 0.05
    ) -> List[Dict[str, Any]]:
        for elem in elements:
            elem_center = (
                elem['bbox'][0] + elem['bbox'][2] / 2,
                elem['bbox'][1] + elem['bbox'][3] / 2
            )
            
            best_text = None
            best_dist = float('inf')
            
            for text in texts:
                text_center = (
                    text['bbox'][0] + text['bbox'][2] / 2,
                    text['bbox'][1] + text['bbox'][3] / 2
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
        max_distance: float = 0.05
    ) -> List[Dict[str, Any]]:
        orphan_texts = []
        
        for text in texts:
            text_center = (
                text['bbox'][0] + text['bbox'][2] / 2,
                text['bbox'][1] + text['bbox'][3] / 2
            )
            
            is_associated = False
            for elem in elements:
                elem_center = (
                    elem['bbox'][0] + elem['bbox'][2] / 2,
                    elem['bbox'][1] + elem['bbox'][3] / 2
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