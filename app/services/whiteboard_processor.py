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
                extracted_data = await asyncio.to_thread(
                    self.nvidia_service.extract_whiteboard_data,
                    image_data,
                    content_type,
                )
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

        # Professional Color Palette Tokens
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
        """Generates Fabric.js canvas layout using relative spatial coordinates (x, y)."""
        objects = []
        elements = data.get("elements", [])

        canvas_width = 600
        canvas_height = 500

        for i, elem in enumerate(elements):
            text = str(elem.get("text") or "Node")

            rel_x = float(elem.get("x") if elem.get("x") is not None else 50)
            rel_y = float(elem.get("y") if elem.get("y") is not None else (i + 1) * 20)

            pixel_x = int((rel_x / 100.0) * (canvas_width - 140)) + 20
            pixel_y = int((rel_y / 100.0) * (canvas_height - 80)) + 20

            # Add Shape Rectangle
            objects.append({
                "type": "rect",
                "left": pixel_x,
                "top": pixel_y,
                "width": 110,
                "height": 45,
                "fill": "#eff6ff",
                "stroke": "#2563eb",
                "strokeWidth": 2,
                "rx": 6,
                "ry": 6,
            })
            # Add Centered Text Inside Shape
            objects.append({
                "type": "textbox",
                "left": pixel_x + 10,
                "top": pixel_y + 12,
                "text": text,
                "fontSize": 14,
                "fill": "#1e3a8a",
                "width": 90,
                "textAlign": "center",
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