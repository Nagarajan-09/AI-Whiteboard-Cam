import cv2
import importlib.util
import logging
import math
import numpy as np
from typing import Any, Dict

logger = logging.getLogger(__name__)

def _import_easyocr():
    spec = importlib.util.find_spec("easyocr")
    if spec is None:
        return None
    return importlib.import_module("easyocr")

easyocr = _import_easyocr()

class HybridGraphExtractor:
    def __init__(self):
        if easyocr is None:
            raise ImportError(
                "easyocr is required for HybridGraphExtractor. "
                "Install it with 'pip install easyocr' before using this service."
            )
        logger.info("Initializing EasyOCR (this may take a moment on first run to download models)...")
        # Initialize OCR for English. gpu=False ensures it runs on CPU if CUDA isn't available.
        self.reader = easyocr.Reader(['en'], gpu=False)

    def process_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Extracts nodes and connections using highly robust CV and OCR.
        """
        # 1. Decode & Resize for consistent OpenCV parameter scaling
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image bytes.")
            
        # Scale image to a standard width (e.g., 800px) so kernel sizes work universally
        h, w = img.shape[:2]
        scale = 800.0 / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        # --- OCR PHASE ---
        # Boost contrast heavily to help EasyOCR read text inside small boxes
        logger.info("Running EasyOCR...")
        ocr_gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=-30)
        ocr_results = self.reader.readtext(ocr_gray)
        
        extracted_nodes = []
        for i, (bbox, text, conf) in enumerate(ocr_results):
            if conf < 0.2:  # Skip highly uncertain text
                continue
                
            tl, tr, br, bl = bbox
            center_x = int((tl[0] + br[0]) / 2)
            center_y = int((tl[1] + br[1]) / 2)
            
            # Convert back to percentages for the LLM refinement stage
            pct_x = round((center_x / width) * 100, 2)
            pct_y = round((center_y / height) * 100, 2)
            
            extracted_nodes.append({
                "id": f"N{i+1}",
                "text": text,
                "x": pct_x,
                "y": pct_y,
                "pixel_center": (center_x, center_y)
            })

        # --- LINE DETECTION PHASE ---
        logger.info("Running OpenCV Line Detection...")
        
        # 1. Adaptive Thresholding: Defeats shadows by analyzing local contrast
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
        )
        
        # 2. Morphological closing to bridge gaps in hand-drawn pen strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 3. Highly sensitive Hough Lines to catch faint, short arrows
        lines = cv2.HoughLinesP(
            closed, 
            rho=1, 
            theta=np.pi/180, 
            threshold=25,       # Lowered to catch faint lines
            minLineLength=20,   # Lowered to catch short branch lines
            maxLineGap=25       # Increased to bridge pen skips
        )

        extracted_connections = []
        seen_edges = set()

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Expand search radius for lines that don't perfectly touch the text box
                start_node = self._get_closest_node((x1, y1), extracted_nodes, max_dist=120)
                end_node = self._get_closest_node((x2, y2), extracted_nodes, max_dist=120)
                
                if start_node and end_node and start_node['id'] != end_node['id']:
                    edge_sig = tuple(sorted([start_node['id'], end_node['id']]))
                    if edge_sig not in seen_edges:
                        seen_edges.add(edge_sig)
                        
                        # Top-to-bottom directional heuristic
                        if start_node['pixel_center'][1] <= end_node['pixel_center'][1]:
                            from_id, to_id = start_node['id'], end_node['id']
                        else:
                            from_id, to_id = end_node['id'], start_node['id']
                            
                        extracted_connections.append({
                            "from_id": from_id,
                            "to_id": to_id,
                            "label": ""
                        })

        for node in extracted_nodes:
            del node['pixel_center']

        return {
            "elements": extracted_nodes,
            "connections": extracted_connections
        }

    def _get_closest_node(self, point: tuple, nodes: list, max_dist: int = 150) -> dict:
        """Finds the closest node to a given (x,y) point within a maximum pixel distance."""
        best_node = None
        min_dist = float('inf')
        px, py = point
        
        for node in nodes:
            nx, ny = node['pixel_center']
            dist = math.hypot(nx - px, ny - py)
            if dist < min_dist and dist < max_dist:
                min_dist = dist
                best_node = node
                
        return best_node