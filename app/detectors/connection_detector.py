import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from app.models.whiteboard import WhiteboardElement, Connection, BoundingBox

logger = logging.getLogger(__name__)


@dataclass
class LineSegment:
    start: Tuple[float, float]
    end: Tuple[float, float]
    angle: float
    length: float


class ConnectionDetector:
    def __init__(
        self,
        min_line_length: float = 30,
        max_line_gap: float = 10,
        arrow_threshold: float = 0.3,
        connection_threshold: float = 20.0
    ):
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.arrow_threshold = arrow_threshold
        self.connection_threshold = connection_threshold
    
    def detect(self, image: np.ndarray, elements: List[WhiteboardElement]) -> List[Connection]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        if lines is None:
            return []
        
        line_segments = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            angle = np.arctan2(y2 - y1, x2 - x1)
            line_segments.append(LineSegment(
                start=(x1, y1),
                end=(x2, y2),
                angle=angle,
                length=length
            ))
        
        connections = self._match_connections(line_segments, elements, image)
        return connections
    
    def _match_connections(
        self,
        lines: List[LineSegment],
        elements: List[WhiteboardElement],
        image: np.ndarray
    ) -> List[Connection]:
        connections = []
        element_centers = {elem.id: elem.bbox.center() for elem in elements}
        
        for line in lines:
            start_elem = self._find_closest_element(line.start, element_centers)
            end_elem = self._find_closest_element(line.end, element_centers)
            
            if start_elem and end_elem and start_elem != end_elem:
                conn_type = self._detect_arrow_type(line, image)
                connection = Connection(
                    start_id=start_elem,
                    end_id=end_elem,
                    type=conn_type,
                    style=self._get_connection_style(conn_type)
                )
                connections.append(connection)
        
        return self._deduplicate_connections(connections)
    
    def _find_closest_element(
        self,
        point: Tuple[float, float],
        element_centers: Dict[str, Tuple[float, float]]
    ) -> Optional[str]:
        min_dist = float('inf')
        closest_id = None
        
        for elem_id, center in element_centers.items():
            dist = np.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2)
            if dist < min_dist and dist < self.connection_threshold:
                min_dist = dist
                closest_id = elem_id
        
        return closest_id
    
    def _detect_arrow_type(self, line: LineSegment, image: np.ndarray) -> str:
        x1, y1 = map(int, line.start)
        x2, y2 = map(int, line.end)
        
        if 0 <= x1 < image.shape[1] and 0 <= y1 < image.shape[0] and \
           0 <= x2 < image.shape[1] and 0 <= y2 < image.shape[0]:
            roi_size = 15
            start_roi = image[max(0, y1-roi_size):min(image.shape[0], y1+roi_size),
                            max(0, x1-roi_size):min(image.shape[1], x1+roi_size)]
            end_roi = image[max(0, y2-roi_size):min(image.shape[0], y2+roi_size),
                          max(0, x2-roi_size):min(image.shape[1], x2+roi_size)]
            
            if self._has_arrowhead(end_roi, line.angle):
                return "arrow"
            elif self._has_arrowhead(start_roi, line.angle + np.pi):
                return "reverse_arrow"
        
        return "line"
    
    def _has_arrowhead(self, roi: np.ndarray, angle: float) -> bool:
        if roi.size == 0:
            return False
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 10 < area < 500:
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) >= 3:
                    return True
        
        return False
    
    def _get_connection_style(self, conn_type: str) -> Dict[str, Any]:
        styles = {
            'arrow': {'stroke': '#000000', 'strokeWidth': 2, 'endArrow': 'block'},
            'reverse_arrow': {'stroke': '#000000', 'strokeWidth': 2, 'startArrow': 'block'},
            'line': {'stroke': '#666666', 'strokeWidth': 1, 'strokeDasharray': '5,5'},
            'dashed': {'stroke': '#666666', 'strokeWidth': 1, 'strokeDasharray': '5,5'},
            'dotted': {'stroke': '#666666', 'strokeWidth': 1, 'strokeDasharray': '2,4'},
        }
        return styles.get(conn_type, styles['line'])
    
    def _deduplicate_connections(self, connections: List[Connection]) -> List[Connection]:
        unique = {}
        for conn in connections:
            key = tuple(sorted([conn.start_id, conn.end_id]))
            if key not in unique or conn.type == 'arrow':
                unique[key] = conn
        return list(unique.values())


class ArrowDetector:
    ARROW_TEMPLATES = {
        'arrow_right': np.array([
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=np.uint8),
    }
    
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
    
    def detect_arrows(self, image: np.ndarray) -> List[Dict[str, Any]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        arrows = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 20 < area < 500:
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) >= 3:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    if 0.5 < aspect_ratio < 3:
                        arrows.append({
                            'bbox': (x, y, w, h),
                            'center': (x + w/2, y + h/2),
                            'angle': self._estimate_arrow_angle(contour),
                            'confidence': min(area / 100, 1.0)
                        })
        
        return arrows
    
    def _estimate_arrow_angle(self, contour: np.ndarray) -> float:
        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            return 0
        
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        
        max_dist = 0
        farthest = None
        for point in contour:
            px, py = point[0]
            dist = (px - cx) ** 2 + (py - cy) ** 2
            if dist > max_dist:
                max_dist = dist
                farthest = (px, py)
        
        if farthest:
            return np.arctan2(farthest[1] - cy, farthest[0] - cx)
        return 0


class ShapeConnectionAnalyzer:
    def __init__(self, connection_threshold: float = 30.0):
        self.connection_threshold = connection_threshold
    
    def analyze(
        self,
        elements: List[WhiteboardElement],
        connections: List[Connection]
    ) -> Dict[str, Any]:
        element_map = {elem.id: elem for elem in elements}
        
        graph = {elem.id: [] for elem in elements}
        for conn in connections:
            if conn.start_id in graph and conn.end_id in graph:
                graph[conn.start_id].append({
                    'target': conn.end_id,
                    'type': conn.type,
                    'style': conn.style
                })
        
        diagram_type = self._classify_diagram(elements, connections)
        clusters = self._find_clusters(elements, graph)
        hierarchy = self._build_hierarchy(elements, graph)
        
        return {
            'graph': graph,
            'diagram_type': diagram_type,
            'clusters': clusters,
            'hierarchy': hierarchy,
            'element_count': len(elements),
            'connection_count': len(connections)
        }
    
    def _classify_diagram(
        self,
        elements: List[WhiteboardElement],
        connections: List[Connection]
    ) -> str:
        type_counts = {}
        for elem in elements:
            type_counts[elem.type] = type_counts.get(elem.type, 0) + 1
        
        if type_counts.get('diamond', 0) > 0 or type_counts.get('decision', 0) > 0:
            return 'flowchart'
        elif type_counts.get('database', 0) > 0 or type_counts.get('data', 0) > 0:
            return 'er_diagram'
        elif type_counts.get('rectangle', 0) > 2:
            return 'flowchart'
        elif type_counts.get('sticky_note', 0) > 2:
            return 'mindmap'
        elif len(connections) > len(elements):
            return 'network'
        else:
            return 'general'
    
    def _find_clusters(
        self,
        elements: List[WhiteboardElement],
        graph: Dict[str, List]
    ) -> List[List[str]]:
        visited = set()
        clusters = []
        
        def dfs(node: str, cluster: List[str]):
            visited.add(node)
            cluster.append(node)
            for edge in graph.get(node, []):
                if edge['target'] not in visited:
                    dfs(edge['target'], cluster)
        
        for elem in elements:
            if elem.id not in visited:
                cluster = []
                dfs(elem.id, cluster)
                if cluster:
                    clusters.append(cluster)
        
        return clusters
    
    def _build_hierarchy(
        self,
        elements: List[WhiteboardElement],
        graph: Dict[str, List]
    ) -> Dict[str, Any]:
        indegree = {elem.id: 0 for elem in elements}
        for node, edges in graph.items():
            for edge in edges:
                if edge['target'] in indegree:
                    indegree[edge['target']] += 1
        
        roots = [node for node, deg in indegree.items() if deg == 0]
        
        hierarchy = {'roots': roots, 'levels': {}}
        visited = set()
        
        def build_level(node: str, level: int):
            if node in visited:
                return
            visited.add(node)
            if level not in hierarchy['levels']:
                hierarchy['levels'][level] = []
            hierarchy['levels'][level].append(node)
            
            for edge in graph.get(node, []):
                build_level(edge['target'], level + 1)
        
        for root in roots:
            build_level(root, 0)
        
        return hierarchy