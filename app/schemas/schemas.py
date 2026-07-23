from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class DiagramType(str, Enum):
    FLOWCHART = "flowchart"
    SEQUENCE = "sequence"
    CLASS = "class"
    STATE = "state"
    ENTITY_RELATIONSHIP = "er"
    JOURNEY = "journey"
    GANTT = "gantt"
    PIE = "pie"
    QUADRANT = "quadrant"
    REQUIREMENT = "requirement"
    GITGRAPH = "gitgraph"
    MINDMAP = "mindmap"
    TIMELINE = "timeline"
    ZENUM = "zenuml"
    SANKEY = "sankey"
    BLOCK = "block"
    PACKET = "packet"
    QUADRANT_CHART = "quadrantChart"
    ARCHITECTURE = "architecture"


class OutputFormat(str, Enum):
    MERMAID = "mermaid"
    MERMAID_JS = "mermaid-js"
    SVG = "svg"
    PNG = "png"
    CANVAS_JSON = "canvas-json"


class ShapeType(str, Enum):
    RECTANGLE = "rectangle"
    ROUNDED_RECTANGLE = "rounded_rectangle"
    CIRCLE = "circle"
    DIAMOND = "diamond"
    TRIANGLE = "triangle"
    HEXAGON = "hexagon"
    PARALLELOGRAM = "parallelogram"
    CYLINDER = "cylinder"
    ARROW = "arrow"
    LINE = "line"
    CURVE = "curve"
    TEXT = "text"
    IMAGE = "image"
    STICKY_NOTE = "sticky_note"
    CONNECTOR = "connector"


class DetectedElement(BaseModel):
    type: ShapeType
    bbox: List[float]  # [x, y, width, height] normalized 0-1
    confidence: float
    text: Optional[str] = None
    text_confidence: Optional[float] = None
    connections: List[Dict[str, Any]] = []
    style: Dict[str, Any] = {}


class DetectedText(BaseModel):
    text: str
    bbox: List[float]  # [x, y, width, height] normalized
    confidence: float
    language: Optional[str] = None


class WhiteboardImage(BaseModel):
    id: str
    filename: str
    content_type: str
    size: int
    width: int
    height: int
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)


class DetectedElements(BaseModel):
    elements: List[DetectedElement]
    texts: List[DetectedText]
    connections: List[Dict[str, Any]]
    image_size: List[int]  # [width, height]


class MermaidDiagram(BaseModel):
    diagram_type: DiagramType
    mermaid_code: str
    elements: List[DetectedElement]
    texts: List[DetectedText]
    connections: List[Dict[str, Any]]
    metadata: Dict[str, Any] = {}


class CanvasData(BaseModel):
    width: int
    height: int
    elements: List[Dict[str, Any]]
    connections: List[Dict[str, Any]]
    background: str = "#ffffff"


class ProcessingResult(BaseModel):
    success: bool
    image_id: str
    diagram: Optional[MermaidDiagram] = None
    canvas: Optional[CanvasData] = None
    mermaid_code: Optional[str] = None
    svg_output: Optional[str] = None
    png_base64: Optional[str] = None
    processing_time: float
    error: Optional[str] = None


class UploadResponse(BaseModel):
    success: bool
    image_id: str
    message: str


class ProcessRequest(BaseModel):
    image_id: str
    diagram_type: Optional[DiagramType] = None
    output_format: OutputFormat = OutputFormat.MERMAID
    mermaid_theme: str = "default"
    auto_detect_type: bool = True
    enhance_text: bool = True
    detect_connections: bool = True
    enhance_shapes: bool = True


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool