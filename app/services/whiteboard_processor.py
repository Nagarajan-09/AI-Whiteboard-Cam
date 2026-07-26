import asyncio
import io
import logging
import math
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

# Node box dimensions (must match the frontend's BOX_HALF_W / BOX_HALF_H in index.html)
NODE_HALF_W = 55
NODE_HALF_H = 22
ARROW_SIZE = 12

# Graph layout spacing
COL_SPACING = 160
ROW_SPACING = 110
MARGIN_X = 80
MARGIN_Y = 60


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
        Deduplicates identical nodes while preserving visually drawn shapes (like 'Yes'/'No' boxes).
        """
        raw_elements = data.get("elements", [])
        raw_connections = data.get("connections", [])

        sanitized_elements = []
        seen_texts = {}
        id_redirection = {}

        for elem in raw_elements:
            raw_id = str(elem.get("id") or "")
            text = str(elem.get("text") or "").strip()
            clean_key = text.lower()

            # Merge true duplicate nodes (e.g., if model sees two 'Button' nodes at same place)
            if clean_key in seen_texts:
                id_redirection[raw_id] = seen_texts[clean_key]
            else:
                seen_texts[clean_key] = raw_id
                sanitized_elements.append(elem)

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
        node_ids_in_order = []
        for i, elem in enumerate(sorted_elements):
            node_id = str(elem.get("id") or f"N{i+1}").replace("-", "_")
            text = str(elem.get("text") or "Shape").replace('"', "'")
            shape_type = str(elem.get("type") or "rectangle").lower()
            node_ids_in_order.append(node_id)

            if "circle" in shape_type or "oval" in shape_type or text.lower() in ["start", "stop", "end"]:
                lines.append(f'    {node_id}(("{text}")):::startEnd')
            elif "diamond" in shape_type or "decision" in shape_type:
                lines.append(f'    {node_id}{{"{text}"}}:::decision')
            else:
                lines.append(f'    {node_id}["{text}"]:::process')

        # Render real connections, tracking edge declaration order (Mermaid indexes
        # linkStyle by the order edges appear in the source, real or invisible)
        edge_index = 0
        connected_ids = set()
        for conn in connections:
            from_id = str(conn.get("from_id") or "").replace("-", "_")
            to_id = str(conn.get("to_id") or "").replace("-", "_")
            label = str(conn.get("label") or "").strip().replace('"', "'")

            if from_id and to_id:
                if label:
                    lines.append(f'    {from_id} -->|"{label}"| {to_id}')
                else:
                    lines.append(f'    {from_id} --> {to_id}')
                connected_ids.add(from_id)
                connected_ids.add(to_id)
                edge_index += 1

        # Pin isolated nodes (no connections at all) to their spatial rank using an
        # invisible edge from the previous node in top-to-bottom order. Without this,
        # dagre floats edge-less nodes to the top rank regardless of where they were
        # actually drawn on the whiteboard.
        invisible_edge_indices = []
        prev_node_id = None
        for node_id in node_ids_in_order:
            if node_id not in connected_ids and prev_node_id is not None:
                lines.append(f'    {prev_node_id} --> {node_id}')
                invisible_edge_indices.append(edge_index)
                edge_index += 1
            prev_node_id = node_id

        for idx in invisible_edge_indices:
            lines.append(f'    linkStyle {idx} display:none;')

        return "\n".join(lines)

    @staticmethod
    def _clip_point_to_box(cx: float, cy: float, ox: float, oy: float,
                            half_w: float = NODE_HALF_W, half_h: float = NODE_HALF_H) -> tuple[float, float]:
        """
        Returns the point where a line from (cx,cy) toward (ox,oy) crosses the
        boundary of a half_w x half_h box centered at (cx,cy). Used to stop
        connector lines at each node's edge instead of drawing into its center.
        """
        dx = ox - cx
        dy = oy - cy
        if dx == 0 and dy == 0:
            return cx, cy

        scales = []
        if dx != 0:
            scales.append(half_w / abs(dx))
        if dy != 0:
            scales.append(half_h / abs(dy))

        t = min(scales) if scales else 0.0
        t = max(0.0, min(t, 1.0))
        return cx + dx * t, cy + dy * t

    @staticmethod
    def _compute_layout(elements: list, connections: list):
        """
        Graph-based auto-layout: assigns each node a (level, column) position
        derived from connection structure -- not the vision model's raw x/y
        guesses -- so branches always spread out cleanly the same way
        Mermaid's dagre engine lays out the vector diagram. This avoids the
        "connector line cuts through unrelated boxes" problem that raw
        model-estimated coordinates produced for branching flowcharts.

        Level = longest path from a root (topological, Kahn's algorithm).
        Column within a level = barycenter of parent columns, so a node's
        siblings end up visually aligned under their shared parent.
        """
        ids = []
        id_to_elem: Dict[str, Any] = {}
        for i, elem in enumerate(elements):
            node_id = str(elem.get("id") or f"N{i+1}").replace("-", "_")
            ids.append(node_id)
            id_to_elem[node_id] = elem

        valid_ids = set(ids)
        children: Dict[str, list] = {nid: [] for nid in ids}
        parents: Dict[str, list] = {nid: [] for nid in ids}

        seen_edges = set()
        for conn in connections:
            from_id = str(conn.get("from_id") or "").replace("-", "_")
            to_id = str(conn.get("to_id") or "").replace("-", "_")
            if from_id in valid_ids and to_id in valid_ids and from_id != to_id:
                edge = (from_id, to_id)
                if edge not in seen_edges:
                    seen_edges.add(edge)
                    children[from_id].append(to_id)
                    parents[to_id].append(from_id)

        # Kahn's topological sort, assigning level = 1 + max(parent levels)
        in_degree = {nid: len(parents[nid]) for nid in ids}
        level: Dict[str, int] = {}
        queue = [nid for nid in ids if in_degree[nid] == 0]
        for nid in queue:
            level[nid] = 0

        processed = set(queue)
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in children[u]:
                level[v] = max(level.get(v, 0), level[u] + 1)
                in_degree[v] -= 1
                if in_degree[v] <= 0 and v not in processed:
                    processed.add(v)
                    queue.append(v)

        # Any leftover nodes (shouldn't normally happen post-sanitizer, but
        # guards against a cyclic edge) get pushed to a final row
        remaining = [nid for nid in ids if nid not in level]
        next_level = (max(level.values()) + 1) if level else 0
        for nid in remaining:
            level[nid] = next_level

        # Group nodes by level, preserving original order as the initial tiebreaker
        levels_map: Dict[int, list] = {}
        for nid in ids:
            levels_map.setdefault(level[nid], []).append(nid)

        col_index: Dict[str, int] = {}

        for lvl in sorted(levels_map.keys()):
            row = levels_map[lvl]
            if lvl == 0:
                # Root row: order by the vision model's original x estimate
                row.sort(key=lambda nid: float(id_to_elem[nid].get("x") or 0))
            else:
                # Barycenter method: order by the average column of each
                # node's parents so connected nodes stay visually aligned
                # under their source, and unrelated branches separate cleanly
                def barycenter(nid, _parents=parents, _col_index=col_index, _id_to_elem=id_to_elem):
                    p = _parents[nid]
                    if not p:
                        return float(_id_to_elem[nid].get("x") or 0)
                    return sum(_col_index.get(pid, 0) for pid in p) / len(p)

                row.sort(key=barycenter)

            for i, nid in enumerate(row):
                col_index[nid] = i
            levels_map[lvl] = row

        max_row_len = max((len(row) for row in levels_map.values()), default=1)
        num_levels = max(levels_map.keys(), default=0) + 1

        canvas_width = max(600, MARGIN_X * 2 + (max_row_len - 1) * COL_SPACING + 2 * NODE_HALF_W)
        canvas_height = max(500, MARGIN_Y * 2 + (num_levels - 1) * ROW_SPACING + 2 * NODE_HALF_H)

        positions: Dict[str, tuple[int, int]] = {}
        for lvl, row in levels_map.items():
            row_width = (len(row) - 1) * COL_SPACING
            start_x = (canvas_width - row_width) / 2
            y = MARGIN_Y + lvl * ROW_SPACING
            for i, nid in enumerate(row):
                x = start_x + i * COL_SPACING
                positions[nid] = (int(x), int(y))

        return positions, int(canvas_width), int(canvas_height)

    @staticmethod
    def _build_canvas_json(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates Fabric.js canvas JSON with node IDs and connection endpoints
        to enable real-time dynamic line tracking in the frontend. Node
        positions come from the graph-based auto-layout (not raw model x/y),
        lines are clipped to each node's box edge, and each connection carries
        an arrowhead triangle so direction is visible on the interactive canvas.
        """
        objects = []
        elements = data.get("elements", [])
        connections = data.get("connections", [])

        if not elements:
            return {"version": "5.3.0", "width": 600, "height": 500, "objects": []}

        positions, canvas_width, canvas_height = WhiteboardProcessor._compute_layout(elements, connections)

        # Store node coordinates mapped to node_id
        node_coords: Dict[str, tuple[int, int]] = {}

        # 1. Build Node Groups with explicit `node_id`
        for i, elem in enumerate(elements):
            node_id = str(elem.get("id") or f"N{i+1}").replace("-", "_")
            text = str(elem.get("text") or "Node")

            center_x, center_y = positions.get(node_id, (canvas_width // 2, canvas_height // 2))
            node_coords[node_id] = (center_x, center_y)

            top_left_x = center_x - NODE_HALF_W
            top_left_y = center_y - NODE_HALF_H

            fill_color = "#dcfce7" if text.lower() in ["start", "stop", "end"] else "#eff6ff"
            stroke_color = "#16a34a" if text.lower() in ["start", "stop", "end"] else "#2563eb"
            text_color = "#14532d" if text.lower() in ["start", "stop", "end"] else "#1e3a8a"

            group_objects = [
                {
                    "type": "rect",
                    "left": top_left_x,
                    "top": top_left_y,
                    "width": NODE_HALF_W * 2,
                    "height": NODE_HALF_H * 2,
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
                    "width": NODE_HALF_W * 2 - 10,
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

        # 2. Build Lines + Arrowheads with `from_id` and `to_id`
        for conn in connections:
            from_id = str(conn.get("from_id") or "").replace("-", "_")
            to_id = str(conn.get("to_id") or "").replace("-", "_")

            if from_id in node_coords and to_id in node_coords:
                x1, y1 = node_coords[from_id]
                x2, y2 = node_coords[to_id]

                # Clip so the line starts/ends at each node's box edge, not its center
                start_x, start_y = WhiteboardProcessor._clip_point_to_box(x1, y1, x2, y2)
                end_x, end_y = WhiteboardProcessor._clip_point_to_box(x2, y2, x1, y1)

                objects.append({
                    "type": "line",
                    "x1": start_x,
                    "y1": start_y,
                    "x2": end_x,
                    "y2": end_y,
                    "stroke": "#64748b",
                    "strokeWidth": 2,
                    "from_id": from_id,
                    "to_id": to_id
                })

                # Arrowhead pointing along the line direction, tip at the target's box edge
                angle_deg = math.degrees(math.atan2(end_y - start_y, end_x - start_x)) + 90
                objects.append({
                    "type": "triangle",
                    "left": end_x,
                    "top": end_y,
                    "width": ARROW_SIZE,
                    "height": ARROW_SIZE,
                    "angle": angle_deg,
                    "fill": "#64748b",
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