from typing import List, Dict, Any, Optional
from enum import Enum

from app.models.whiteboard import ProcessedWhiteboard, WhiteboardElement, Connection, BoundingBox


class DiagramType(str, Enum):
    FLOWCHART = "flowchart"
    SEQUENCE = "sequence"
    CLASS = "class"
    STATE = "state"
    ER = "er"
    MIND_MAP = "mindmap"
    TIMELINE = "timeline"
    GENERAL = "graph"


class MermaidGenerator:
    NODE_SHAPES = {
        'rectangle': ('[', ']'),
        'rounded_rectangle': ('(', ')'),
        'circle': ('((', '))'),
        'diamond': ('{', '}'),
        'hexagon': ('{{', '}}'),
        'parallelogram': ('[/', '/]'),
        'cylinder': ('[(', ')]'),
        'sticky_note': ('[', ']'),
        'text_box': ('[', ']'),
        'decision': ('{', '}'),
        'process': ('[', ']'),
        'terminator': ('(', ')'),
        'data': ('[/', '/]'),
        'database': ('[(', ')]'),
        'document': ('[', ']'),
        'ellipse': ('((', '))'),
        'arrow': ('[', ']'),
        'line': ('[', ']'),
    }
    
    ARROW_TYPES = {
        'arrow': '-->',
        'line': '---',
        'dashed': '-.-',
        'dotted': '-..-',
        'bold': '==>',
        'thick': '==>',
        'reverse_arrow': '<--',
        'bidirectional': '<-->',
    }
    
    def __init__(self, theme: str = "default"):
        self.theme = theme
        self.node_counter = 0
        self.node_map = {}
    
    def generate(self, whiteboard: ProcessedWhiteboard, diagram_type: Optional[DiagramType] = None) -> str:
        if diagram_type is None:
            diagram_type = self._detect_diagram_type(whiteboard)
        
        if diagram_type == DiagramType.FLOWCHART:
            return self._generate_flowchart(whiteboard)
        elif diagram_type == DiagramType.SEQUENCE:
            return self._generate_sequence(whiteboard)
        elif diagram_type == DiagramType.CLASS:
            return self._generate_class(whiteboard)
        elif diagram_type == DiagramType.STATE:
            return self._generate_state(whiteboard)
        elif diagram_type == DiagramType.ER:
            return self._generate_er(whiteboard)
        elif diagram_type == DiagramType.MIND_MAP:
            return self._generate_mindmap(whiteboard)
        elif diagram_type == DiagramType.TIMELINE:
            return self._generate_timeline(whiteboard)
        else:
            return self._generate_general(whiteboard)
    
    def _detect_diagram_type(self, whiteboard: ProcessedWhiteboard) -> DiagramType:
        types = [e.type for e in whiteboard.elements]
        
        if any(t in types for t in ['decision', 'diamond']):
            return DiagramType.FLOWCHART
        elif any(t in types for t in ['database', 'data', 'cylinder']):
            return DiagramType.ER
        elif any(t in types for t in ['sticky_note']) and len(types) > 3:
            return DiagramType.MIND_MAP
        elif len(whiteboard.connections) > len(whiteboard.elements) * 1.5:
            return DiagramType.FLOWCHART
        else:
            return DiagramType.GENERAL
    
    def _get_node_id(self, element: WhiteboardElement) -> str:
        if element.id not in self.node_map:
            self.node_counter += 1
            clean_id = ''.join(c if c.isalnum() or c == '_' else '_' for c in element.id)
            if clean_id[0].isdigit():
                clean_id = f"n{clean_id}"
            self.node_map[element.id] = f"{clean_id}_{self.node_counter}"
        return self.node_map[element.id]
    
    def _get_node_shape(self, element_type: str) -> tuple:
        return self.NODE_SHAPES.get(element_type, ('[', ']'))
    
    def _escape_text(self, text: str) -> str:
        text = text.replace('"', '\\"')
        text = text.replace('[', '\\[').replace(']', '\\]')
        text = text.replace('(', '\\(').replace(')', '\\)')
        text = text.replace('{', '\\{').replace('}', '\\}')
        return text
    
    def _generate_flowchart(self, whiteboard: ProcessedWhiteboard) -> str:
        lines = ["```mermaid", "flowchart TD"]
        
        for elem in whiteboard.elements:
            node_id = self._get_node_id(elem)
            open_shape, close_shape = self._get_node_shape(elem.type)
            label = self._escape_text(elem.text) if elem.text else elem.type
            lines.append(f"    {node_id}{open_shape}{label}{close_shape}")
        
        for conn in whiteboard.connections:
            if conn.start_id in self.node_map and conn.end_id in self.node_map:
                start_id = self.node_map[conn.start_id]
                end_id = self.node_map[conn.end_id]
                arrow = self.ARROW_TYPES.get(conn.type, '-->')
                lines.append(f"    {start_id} {arrow} {end_id}")
        
        if self.theme != "default":
            lines.append(f"    %%{{init: {{'theme': '{self.theme}'}}}}%%")
        
        lines.append("```")
        return "\n".join(lines)
    
    def _generate_sequence(self, whiteboard: ProcessedWhiteboard) -> str:
        lines = ["```mermaid", "sequenceDiagram"]
        
        participants = {}
        for elem in whiteboard.elements:
            if elem.type in ['rectangle', 'process', 'actor']:
                pid = self._get_node_id(elem)
                label = self._escape_text(elem.text) if elem.text else elem.type
                participants[pid] = label
                lines.append(f"    participant {pid} as {label}")
        
        for conn in whiteboard.connections:
            if conn.start_id in self.node_map and conn.end_id in self.node_map:
                start = self.node_map[conn.start_id]
                end = self.node_map[conn.end_id]
                label = ""
                if conn.type == 'arrow':
                    lines.append(f"    {start}->>{end}: {label}")
                else:
                    lines.append(f"    {start}-->>{end}: {label}")
        
        lines.append("```")
        return "\n".join(lines)
    
    def _generate_class(self, whiteboard: ProcessedWhiteboard) -> str:
        lines = ["```mermaid", "classDiagram"]
        
        for elem in whiteboard.elements:
            if elem.type in ['rectangle', 'class', 'process']:
                node_id = self._get_node_id(elem)
                label = self._escape_text(elem.text) if elem.text else elem.type
                lines.append(f"    class {node_id} {{")
                lines.append(f"        +{label}")
                lines.append("    }")
        
        for conn in whiteboard.connections:
            if conn.start_id in self.node_map and conn.end_id in self.node_map:
                start = self.node_map[conn.start_id]
                end = self.node_map[conn.end_id]
                lines.append(f"    {start} --|> {end}")
        
        lines.append("```")
        return "\n".join(lines)
    
    def _generate_state(self, whiteboard: ProcessedWhiteboard) -> str:
        lines = ["```mermaid", "stateDiagram-v2"]
        
        for elem in whiteboard.elements:
            node_id = self._get_node_id(elem)
            label = self._escape_text(elem.text) if elem.text else elem.type
            
            if elem.type in ['circle', 'terminator']:
                lines.append(f"    [*] --> {node_id}")
                lines.append(f"    {node_id} : {label}")
            else:
                lines.append(f"    state {node_id} {{")
                lines.append(f"        {label}")
                lines.append("    }")
        
        for conn in whiteboard.connections:
            if conn.start_id in self.node_map and conn.end_id in self.node_map:
                start = self.node_map[conn.start_id]
                end = self.node_map[conn.end_id]
                lines.append(f"    {start} --> {end}")
        
        lines.append("```")
        return "\n".join(lines)
    
    def _generate_er(self, whiteboard: ProcessedWhiteboard) -> str:
        lines = ["```mermaid", "erDiagram"]
        
        for elem in whiteboard.elements:
            node_id = self._get_node_id(elem)
            label = self._escape_text(elem.text) if elem.text else elem.type
            
            if elem.type in ['database', 'cylinder', 'data']:
                lines.append(f"    {node_id} {{")
                lines.append(f"        string {label}")
                lines.append("    }")
            else:
                lines.append(f"    {node_id} {{")
                lines.append(f"        string {label}")
                lines.append("    }")
        
        for conn in whiteboard.connections:
            if conn.start_id in self.node_map and conn.end_id in self.node_map:
                start = self.node_map[conn.start_id]
                end = self.node_map[conn.end_id]
                lines.append(f"    {start} ||--o{{ {end}")
        
        lines.append("```")
        return "\n".join(lines)
    
    def _generate_mindmap(self, whiteboard: ProcessedWhiteboard) -> str:
        lines = ["```mermaid", "mindmap"]
        
        root = None
        for elem in whiteboard.elements:
            if elem.type == 'sticky_note' or root is None:
                root = self._get_node_id(elem)
                label = self._escape_text(elem.text) if elem.text else elem.type
                lines.append(f"  root({label})")
                break
        
        if not root:
            root = "root"
            lines.append(f"  root(Central Idea)")
        
        for elem in whiteboard.elements:
            if elem.id == self.node_map.get(root):
                continue
            node_id = self._get_node_id(elem)
            label = self._escape_text(elem.text) if elem.text else elem.type
            lines.append(f"    {node_id}({label})")
        
        for conn in whiteboard.connections:
            if conn.start_id in self.node_map and conn.end_id in self.node_map:
                start = self.node_map[conn.start_id]
                end = self.node_map[conn.end_id]
                if start == root:
                    lines.append(f"    {end}")
                else:
                    lines.append(f"      {end}")
        
        lines.append("```")
        return "\n".join(lines)
    
    def _generate_timeline(self, whiteboard: ProcessedWhiteboard) -> str:
        lines = ["```mermaid", "timeline"]
        
        for elem in whiteboard.elements:
            label = self._escape_text(elem.text) if elem.text else elem.type
            lines.append(f"    section {label}")
            lines.append(f"        {label}")
        
        lines.append("```")
        return "\n".join(lines)
    
    def _generate_general(self, whiteboard: ProcessedWhiteboard) -> str:
        return self._generate_flowchart(whiteboard)