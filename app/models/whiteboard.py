from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, root_validator
import uuid


class BoundingBox(BaseModel):
    x: float = Field(default=0.0, ge=0, le=1, description="Normalized x coordinate (0-1)")
    y: float = Field(default=0.0, ge=0, le=1, description="Normalized y coordinate (0-1)")
    width: float = Field(default=0.0, ge=0, le=1, description="Normalized width (0-1)")
    height: float = Field(default=0.0, ge=0, le=1, description="Normalized height (0-1)")
    
    @property
    def center_x(self) -> float:
        return self.x + self.width / 2
    
    @property
    def center_y(self) -> float:
        return self.y + self.height / 2
    
    def center(self) -> tuple:
        return (self.center_x, self.center_y)
    
    def to_absolute(self, img_width: int, img_height: int) -> tuple:
        return (
            int(self.x * img_width),
            int(self.y * img_height),
            int(self.width * img_width),
            int(self.height * img_height)
        )


class WhiteboardElement(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str = Field(default="rectangle")
    bbox: Optional[BoundingBox] = None
    text: Optional[str] = ""
    text_confidence: Optional[float] = 1.0
    confidence: float = Field(ge=0, le=1, default=1.0)
    style: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def center_x(self) -> float:
        return self.bbox.center_x if self.bbox else 0.0
    
    @property
    def center_y(self) -> float:
        return self.bbox.center_y if self.bbox else 0.0


class Connection(BaseModel):
    start_id: str
    end_id: str
    type: str = "arrow"
    label: Optional[str] = ""
    style: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0, le=1, default=1.0)

    @root_validator(pre=True)
    def handle_id_aliases(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Support from_id -> start_id and to_id -> end_id aliases from LLM responses."""
        if "from_id" in values and "start_id" not in values:
            values["start_id"] = values["from_id"]
        if "to_id" in values and "end_id" not in values:
            values["end_id"] = values["to_id"]
        return values


class WhiteboardImage(BaseModel):
    id: str
    filename: str
    content_type: str
    size: int
    width: int
    height: int


class AnalysisResult(BaseModel):
    graph: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    diagram_type: str = "general"
    clusters: List[List[str]] = Field(default_factory=list)
    hierarchy: Dict[str, Any] = Field(default_factory=dict)
    element_count: int = 0
    connection_count: int = 0


class ProcessedWhiteboard(BaseModel):
    image: WhiteboardImage
    elements: List[WhiteboardElement] = Field(default_factory=list)
    connections: List[Connection] = Field(default_factory=list)
    analysis: AnalysisResult = Field(default_factory=AnalysisResult)
    mermaid_code: Optional[str] = None
    canvas_json: Optional[Dict[str, Any]] = None
    svg: Optional[str] = None


class TextDetection(BaseModel):
    text: str
    confidence: float
    bbox: BoundingBox


class DetectionResult(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    style: Dict[str, Any] = Field(default_factory=dict)


class ProcessRequest(BaseModel):
    image_id: str
    diagram_type: Optional[str] = None
    output_format: str = "mermaid"
    ocr_enabled: bool = True
    connection_detection_enabled: bool = True


class ProcessResponse(BaseModel):
    success: bool
    whiteboard_id: Optional[str] = None
    processed: Optional[ProcessedWhiteboard] = None
    mermaid_code: Optional[str] = None
    canvas_json: Optional[Dict[str, Any]] = None
    svg: Optional[str] = None
    error: Optional[str] = None


class UploadResponse(BaseModel):
    success: bool
    image_id: Optional[str] = None
    filename: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    error: Optional[str] = None