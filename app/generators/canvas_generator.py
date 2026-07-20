from typing import Dict, Any, List, Optional
import json

from app.models.whiteboard import (
    ProcessedWhiteboard,
    WhiteboardElement,
    Connection,
    BoundingBox
)


class CanvasGenerator:
    def __init__(self):
        pass
    
    def generate(self, whiteboard: ProcessedWhiteboard) -> Dict[str, Any]:
        objects = []
        
        for elem in whiteboard.elements:
            obj = self._element_to_canvas_object(elem, whiteboard.image.width, whiteboard.image.height)
            objects.append(obj)
        
        for conn in whiteboard.connections:
            start_elem = next((e for e in whiteboard.elements if e.id == conn.start_id), None)
            end_elem = next((e for e in whiteboard.elements if e.id == conn.end_id), None)
            
            if start_elem and end_elem:
                obj = self._connection_to_canvas_object(
                    conn, start_elem, end_elem,
                    whiteboard.image.width, whiteboard.image.height
                )
                objects.append(obj)
        
        return {
            "version": "6.0.0",
            "objects": objects,
            "background": "#ffffff",
            "width": whiteboard.image.width,
            "height": whiteboard.image.height
        }
    
    def _element_to_canvas_object(
        self,
        elem: WhiteboardElement,
        canvas_width: int,
        canvas_height: int
    ) -> Dict[str, Any]:
        x = elem.bbox.x * canvas_width
        y = elem.bbox.y * canvas_height
        width = elem.bbox.width * canvas_width
        height = elem.bbox.height * canvas_height
        
        base_obj = {
            "type": self._map_element_type(elem.type),
            "version": "6.0.0",
            "originX": "left",
            "originY": "top",
            "left": x,
            "top": y,
            "width": width,
            "height": height,
            "fill": elem.style.get('fill', '#ffffff'),
            "stroke": elem.style.get('stroke', '#000000'),
            "strokeWidth": elem.style.get('strokeWidth', 2),
            "strokeDashArray": elem.style.get('strokeDashArray', None),
            "strokeLineCap": "round",
            "strokeLineJoin": "round",
            "strokeMiterLimit": 4,
            "scaleX": 1,
            "scaleY": 1,
            "angle": 0,
            "flipX": False,
            "flipY": False,
            "opacity": 1,
            "shadow": None,
            "visible": True,
            "backgroundColor": "",
            "fillRule": "nonzero",
            "paintFirst": "fill",
            "globalCompositeOperation": "source-over",
            "skewX": 0,
            "skewY": 0,
        }
        
        if elem.type in ['rectangle', 'rounded_rectangle', 'process', 'decision', 'data', 'database', 'document']:
            base_obj.update({
                "rx": elem.style.get('rx', 10 if 'rounded' in elem.type else 0),
                "ry": elem.style.get('ry', 10 if 'rounded' in elem.type else 0),
            })
        
        if elem.type == 'circle' or elem.type == 'ellipse':
            base_obj["type"] = "circle"
            base_obj["radius"] = min(width, height) / 2
            base_obj["left"] = x + width / 2
            base_obj["top"] = y + height / 2
        
        if elem.type == 'diamond':
            base_obj["type"] = "path"
            base_obj["path"] = self._diamond_path(width, height)
            base_obj["left"] = x
            base_obj["top"] = y
        
        if elem.type == 'hexagon':
            base_obj["type"] = "path"
            base_obj["path"] = self._hexagon_path(width, height)
            base_obj["left"] = x
            base_obj["top"] = y
        
        if elem.type == 'parallelogram':
            base_obj["type"] = "path"
            base_obj["path"] = self._parallelogram_path(width, height)
            base_obj["left"] = x
            base_obj["top"] = y
        
        if elem.type == 'cylinder':
            base_obj["type"] = "path"
            base_obj["path"] = self._cylinder_path(width, height)
            base_obj["left"] = x
            base_obj["top"] = y
        
        if elem.text:
            text_obj = self._create_text_object(elem, x, y, width, height)
            return {
                "type": "group",
                "objects": [base_obj, text_obj],
                "left": x,
                "top": y,
                "width": width,
                "height": height,
            }
        
        return base_obj
    
    def _create_text_object(
        self,
        elem: WhiteboardElement,
        x: float,
        y: float,
        width: float,
        height: float
    ) -> Dict[str, Any]:
        return {
            "type": "textbox",
            "version": "6.0.0",
            "originX": "center",
            "originY": "center",
            "left": x + width / 2,
            "top": y + height / 2,
            "width": width * 0.9,
            "height": height * 0.9,
            "fill": elem.style.get('textFill', '#000000'),
            "fontFamily": elem.style.get('fontFamily', 'Arial'),
            "fontSize": elem.style.get('fontSize', 14),
            "fontWeight": elem.style.get('fontWeight', 'normal'),
            "fontStyle": elem.style.get('fontStyle', 'normal'),
            "textAlign": "center",
            "textBackgroundColor": "",
            "charSpacing": 0,
            "lineHeight": 1.2,
            "text": elem.text,
            "styles": {},
            "path": None,
            "pathStartOffset": 0,
            "pathSide": "start",
            "pathAlign": "center",
            "minWidth": 20,
            "splitByGrapheme": False,
        }
    
    def _connection_to_canvas_object(
        self,
        conn: Connection,
        start_elem: WhiteboardElement,
        end_elem: WhiteboardElement,
        canvas_width: int,
        canvas_height: int
    ) -> Dict[str, Any]:
        start_x = start_elem.bbox.center_x * canvas_width
        start_y = start_elem.bbox.center_y * canvas_height
        end_x = end_elem.bbox.center_x * canvas_width
        end_y = end_elem.bbox.center_y * canvas_height
        
        path = self._create_arrow_path(start_x, start_y, end_x, end_y, conn.type)
        
        return {
            "type": "path",
            "version": "6.0.0",
            "originX": "center",
            "originY": "center",
            "left": (start_x + end_x) / 2,
            "top": (start_y + end_y) / 2,
            "width": abs(end_x - start_x) + 20,
            "height": abs(end_y - start_y) + 20,
            "fill": "",
            "stroke": conn.style.get('stroke', '#000000'),
            "strokeWidth": conn.style.get('strokeWidth', 2),
            "strokeDashArray": conn.style.get('strokeDashArray', None),
            "strokeLineCap": "round",
            "strokeLineJoin": "round",
            "strokeMiterLimit": 4,
            "scaleX": 1,
            "scaleY": 1,
            "angle": 0,
            "flipX": False,
            "flipY": False,
            "opacity": 1,
            "shadow": None,
            "visible": True,
            "backgroundColor": "",
            "fillRule": "nonzero",
            "paintFirst": "fill",
            "globalCompositeOperation": "source-over",
            "skewX": 0,
            "skewY": 0,
            "path": path,
            "pathOffset": {
                "x": start_x,
                "y": start_y
            }
        }
    
    def _create_arrow_path(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        arrow_type: str
    ) -> List[List]:
        dx = x2 - x1
        dy = y2 - y1
        length = (dx ** 2 + dy ** 2) ** 0.5
        
        if length == 0:
            return [["M", x1, y1], ["L", x1, y1]]
        
        ux = dx / length
        uy = dy / length
        
        arrow_size = 10
        arrow_angle = 0.5
        
        path = [["M", x1, y1], ["L", x2, y2]]
        
        if arrow_type in ['arrow', 'bold', 'thick']:
            ax = x2 - arrow_size * (ux * 0.866 - uy * 0.5)
            ay = y2 - arrow_size * (uy * 0.866 + ux * 0.5)
            bx = x2 - arrow_size * (ux * 0.866 + uy * 0.5)
            by = y2 - arrow_size * (uy * 0.866 - ux * 0.5)
            path.extend([["M", ax, ay], ["L", x2, y2], ["L", bx, by]])
        
        elif arrow_type == 'reverse_arrow':
            ax = x1 + arrow_size * (ux * 0.866 - uy * 0.5)
            ay = y1 + arrow_size * (uy * 0.866 + ux * 0.5)
            bx = x1 + arrow_size * (ux * 0.866 + uy * 0.5)
            by = y1 + arrow_size * (uy * 0.866 - ux * 0.5)
            path.extend([["M", ax, ay], ["L", x1, y1], ["L", bx, by]])
        
        elif arrow_type == 'bidirectional':
            ax = x2 - arrow_size * (ux * 0.866 - uy * 0.5)
            ay = y2 - arrow_size * (uy * 0.866 + ux * 0.5)
            bx = x2 - arrow_size * (ux * 0.866 + uy * 0.5)
            by = y2 - arrow_size * (uy * 0.866 - ux * 0.5)
            path.extend([["M", ax, ay], ["L", x2, y2], ["L", bx, by]])
            
            ax = x1 + arrow_size * (ux * 0.866 - uy * 0.5)
            ay = y1 + arrow_size * (uy * 0.866 + ux * 0.5)
            bx = x1 + arrow_size * (ux * 0.866 + uy * 0.5)
            by = y1 + arrow_size * (uy * 0.866 - ux * 0.5)
            path.extend([["M", ax, ay], ["L", x1, y1], ["L", bx, by]])
        
        return path
    
    def _diamond_path(self, width: float, height: float) -> List[List]:
        hw = width / 2
        hh = height / 2
        return [
            ["M", hw, 0],
            ["L", width, hh],
            ["L", hw, height],
            ["L", 0, hh],
            ["Z"]
        ]
    
    def _hexagon_path(self, width: float, height: float) -> List[List]:
        hw = width / 2
        hh = height / 2
        return [
            ["M", hw * 0.5, 0],
            ["L", width, 0],
            ["L", width, hh],
            ["L", hw * 0.5, height],
            ["L", 0, height],
            ["L", 0, hh],
            ["Z"]
        ]
    
    def _parallelogram_path(self, width: float, height: float) -> List[List]:
        offset = width * 0.2
        return [
            ["M", offset, 0],
            ["L", width, 0],
            ["L", width - offset, height],
            ["L", 0, height],
            ["Z"]
        ]
    
    def _cylinder_path(self, width: float, height: float) -> List[List]:
        return [
            ["M", 0, height * 0.15],
            ["C", 0, 0, width, 0, width, height * 0.15],
            ["L", width, height],
            ["C", width, height * 0.85, 0, height * 0.85, 0, height],
            ["Z"],
            ["M", 0, height * 0.15],
            ["C", 0, 0, width, 0, width, height * 0.15]
        ]
    
    def _map_element_type(self, element_type: str) -> str:
        type_map = {
            'rectangle': 'rect',
            'rounded_rectangle': 'rect',
            'circle': 'circle',
            'ellipse': 'circle',
            'diamond': 'path',
            'hexagon': 'path',
            'parallelogram': 'path',
            'cylinder': 'path',
            'sticky_note': 'rect',
            'text_box': 'rect',
            'decision': 'path',
            'process': 'rect',
            'terminator': 'circle',
            'data': 'path',
            'database': 'path',
            'document': 'rect',
        }
        return type_map.get(element_type, 'rect')