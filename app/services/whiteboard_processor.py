import asyncio
import io
import logging
from typing import Any, Dict, Optional
from PIL import Image

from app.models.whiteboard import (
    AnalysisResult,
    Connection,
    ProcessedWhiteboard,
    WhiteboardElement,
    WhiteboardImage,
)
from app.services.nvidia_service import NvidiaService

logger = logging.getLogger(__name__)


class WhiteboardProcessor:

    def __init__(self):
        self.nvidia_service: Optional[NvidiaService] = None
        try:
            self.nvidia_service = NvidiaService()
            logger.info("NVIDIA vision service initialized successfully")
        except Exception as exc:
            logger.warning("NVIDIA service initialization warning: %s", exc)

    async def initialize(self):
        logger.info("Whiteboard processor ready")

    async def process_image(
        self,
        image_data: bytes,
        filename: str,
        content_type: str,
        diagram_type: Optional[str] = None,
        output_format: str = "mermaid",
    ) -> ProcessedWhiteboard:

        width, height = self._get_image_dimensions(image_data)

        image_obj = WhiteboardImage(
            id="img_1",
            filename=filename or "whiteboard.jpg",
            content_type=content_type or "image/jpeg",
            size=len(image_data),
            width=width,
            height=height,
        )

        extracted_data = {"elements": [], "connections": []}
        api_error_message = None

        if self.nvidia_service:
            try:
                raw_data = await asyncio.to_thread(
                    self.nvidia_service.extract_whiteboard_data,
                    image_data,
                    content_type,
                )
                # Stage 2: Sanitize raw VLM extraction
                extracted_data = self._sanitize_extracted_data(raw_data)
            except Exception as e:
                api_error_message = str(e)
                logger.error("Processor caught NVIDIA API Error: %s", e)

        # Build clean Mermaid code
        if api_error_message:
            mermaid_code = f'flowchart TD\n    N1["API Error: {api_error_message[:40]}..."]'
        else:
            mermaid_code = self._build_mermaid_code(extracted_data)

        canvas_json = self._build_canvas_json(extracted_data)

        elements_list = [
            WhiteboardElement(
                id=str(e.get("id") or f"N{i+1}"),
                type=str(e.get("type") or "rectangle"),
                text=str(e.get("text") or "Shape"),
            )
            for i, e in enumerate(extracted_data.get("elements", []))
        ]

        connections_list = [
            Connection(
                start_id=str(c.get("from_id") or ""),
                end_id=str(c.get("to_id") or ""),
                label=str(c.get("label") or ""),
            )
            for c in extracted_data.get("connections", [])
            if c.get("from_id") and c.get("to_id")
        ]

        return ProcessedWhiteboard(
            image=image_obj,
            elements=elements_list,
            connections=connections_list,
            mermaid_code=mermaid_code,
            canvas_json=canvas_json,
            analysis=AnalysisResult(
                diagram_type=str(diagram_type or "flowchart"),
                element_count=len(elements_list),
                connection_count=len(connections_list),
            ),
        )

    @staticmethod
    def _sanitize_extracted_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 2 Graph Sanitizer:
        Deduplicates identical nodes and converts orphan floating words (Yes/No)
        mistakenly recognized as shapes into edge labels.
        """
        raw_elements = data.get("elements", [])
        raw_connections = data.get("connections", [])

        EDGE_LABEL_KEYWORDS = {"yes", "no", "true", "false", "ok", "cancel"}

        sanitized_elements = []
        element_text_to_id = {}
        id_redirection = {}

        for elem in raw_elements:
            raw_id = str(elem.get("id") or "")
            text = str(elem.get("text") or "").strip()
            clean_text_lower = text.lower()

            # Rule 1: Skip shapes that are actually floating line labels
            if clean_text_lower in EDGE_LABEL_KEYWORDS:
                continue

            # Rule 2: Deduplicate nodes with identical text
            if clean_text_lower in element_text_to_id:
                id_redirection[raw_id] = element_text_to_id[clean_text_lower]
            else:
                element_text_to_id[clean_text_lower] = raw_id
                sanitized_elements.append(elem)

        # Rule 3: Re-map connections to use merged node IDs
        sanitized_connections = []
        for conn in raw_connections:
            from_id = id_redirection.get(conn.get("from_id"), conn.get("from_id"))
            to_id = id_redirection.get(conn.get("to_id"), conn.get("to_id"))

            if from_id and to_id and from_id != to_id:
                sanitized_connections.append({
                    "from_id": from_id,
                    "to_id": to_id,
                    "label": str(conn.get("label") or "").strip()
                })

        return {
            "elements": sanitized_elements,
            "connections": sanitized_connections
        }

    @staticmethod
    def _build_mermaid_code(data: Dict[str, Any]) -> str:
        """Generates publication-quality Mermaid diagrams with spatial ordering and CSS class definitions."""
        elements = data.get("elements", [])
        connections = data.get("connections", [])

        if not elements:
            return 'flowchart TD\n    N1["No diagram elements detected"]'

        # Sort elements spatially by 'y' coordinate (top-to-bottom) for optimal Dagre layout ranking
        sorted_elements = sorted(
            elements,
            key=lambda e: float(e.get("y") if e.get("y") is not None else 0),
        )

        lines = ["flowchart TD"]

        # Color Palette Tokens
        lines.append("    classDef startEnd fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d;")
        lines.append("    classDef decision fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#78350f;")
        lines.append("    classDef process fill:#eff6ff,stroke:#2563eb,stroke-width:2px,color:#1e3a8a;")

        # Render Nodes with Class Assignments
        for i, elem in enumerate(sorted_elements):
            node_id = str(elem.get("id") or f"N{i+1}").replace("-", "_")
            text = str(elem.get("text") or "Shape").replace('"', "'")
            shape_type = str(elem.get("type") or "rectangle").lower()

            if "circle" in shape_type or "oval" in shape_type or text.lower() in ["start", "stop", "end"]:
                lines.append(f'    {node_id}(("{text}")):::startEnd')
            elif "diamond" in shape_type or "decision" in shape_type:
                lines.append(f'    {node_id}{{"{text}"}}:::decision')
            else:
                lines.append(f'    {node_id}["{text}"]:::process')

        # Render Connections
        for conn in connections:
            from_id = str(conn.get("from_id") or "").replace("-", "_")
            to_id = str(conn.get("to_id") or "").replace("-", "_")
            label = str(conn.get("label") or "").strip().replace('"', "'")

            if from_id and to_id:
                if label:
                    lines.append(f'    {from_id} -->|"{label}"| {to_id}')
                else:
                    lines.append(f'    {from_id} --> {to_id}')

        return "\n".join(lines)

    @staticmethod
    def _build_canvas_json(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates Fabric.js canvas JSON with node IDs and connection endpoints
        to enable real-time dynamic line tracking in the frontend.
        """
        objects = []
        elements = data.get("elements", [])
        connections = data.get("connections", [])

        canvas_width = 600
        canvas_height = 500

        # Store node coordinates mapped to node_id
        node_coords: Dict[str, tuple[int, int]] = {}

        # 1. Build Node Groups with explicit `node_id`
        for i, elem in enumerate(elements):
            node_id = str(elem.get("id") or f"N{i+1}").replace("-", "_")
            text = str(elem.get("text") or "Node")

            rel_x = float(elem.get("x") if elem.get("x") is not None else 50)
            rel_y = float(elem.get("y") if elem.get("y") is not None else (i + 1) * 20)

            # Convert percentages to canvas pixel offsets
            center_x = int((rel_x / 100.0) * (canvas_width - 140)) + 70
            center_y = int((rel_y / 100.0) * (canvas_height - 80)) + 40

            node_coords[node_id] = (center_x, center_y)

            top_left_x = center_x - 55
            top_left_y = center_y - 22

            fill_color = "#dcfce7" if text.lower() in ["start", "stop", "end"] else "#eff6ff"
            stroke_color = "#16a34a" if text.lower() in ["start", "stop", "end"] else "#2563eb"
            text_color = "#14532d" if text.lower() in ["start", "stop", "end"] else "#1e3a8a"

            group_objects = [
                {
                    "type": "rect",
                    "left": top_left_x,
                    "top": top_left_y,
                    "width": 110,
                    "height": 44,
                    "fill": fill_color,
                    "stroke": stroke_color,
                    "strokeWidth": 2,
                    "rx": 8,
                    "ry": 8,
                },
                {
                    "type": "textbox",
                    "left": top_left_x + 5,
                    "top": top_left_y + 12,
                    "text": text,
                    "fontSize": 14,
                    "fontFamily": "system-ui, sans-serif",
                    "fontWeight": "bold",
                    "fill": text_color,
                    "width": 100,
                    "textAlign": "center",
                }
            ]

            # Critical: Include `node_id` so frontend can match moving shapes to lines
            objects.append({
                "type": "group",
                "left": top_left_x,
                "top": top_left_y,
                "node_id": node_id,
                "objects": group_objects
            })

        # 2. Build Lines with `from_id` and `to_id`
        for conn in connections:
            from_id = str(conn.get("from_id") or "").replace("-", "_")
            to_id = str(conn.get("to_id") or "").replace("-", "_")

            if from_id in node_coords and to_id in node_coords:
                x1, y1 = node_coords[from_id]
                x2, y2 = node_coords[to_id]

                # Critical: Include `from_id` and `to_id`
                objects.append({
                    "type": "line",
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "stroke": "#64748b",
                    "strokeWidth": 2,
                    "from_id": from_id,
                    "to_id": to_id
                })

        return {
            "version": "5.3.0",
            "width": canvas_width,
            "height": canvas_height,
            "objects": objects,
        }

    @staticmethod
    def _get_image_dimensions(image_data: bytes) -> tuple[int, int]:
        try:
            with Image.open(io.BytesIO(image_data)) as img:
                return img.width, img.height
        except Exception:
            return 800, 600