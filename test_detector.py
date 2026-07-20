import cv2
from app.detectors.yolo_detector import WhiteboardYOLODetector

img = cv2.imread("test1.png")
print("Image loaded:", img is not None, img.shape if img is not None else None)

detector = WhiteboardYOLODetector()
print("Has detect_elements method:", hasattr(detector, "detect_elements"))

elements = detector.detect_elements(img)
print(f"Got {len(elements)} elements")
for e in elements:
    print(e.type, e.bbox)