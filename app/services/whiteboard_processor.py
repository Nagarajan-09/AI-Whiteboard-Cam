import asyncio
import logging
from typing import Optional, Any

import cv2
import numpy as np

from app.models.whiteboard import (
    WhiteboardImage,
    ProcessedWhiteboard,
    AnalysisResult,
)
from app.detectors.yolo_detector import WhiteboardYOLODetector
from app.detectors.text_detector import WhiteboardTextDetector
from app.detectors.connection_detector import ConnectionDetector
from app.generators.mermaid_generator import MermaidGenerator
from app.generators.canvas_generator import CanvasGenerator
from app.services.nvidia_service import NvidiaService
from app.core.config import settings


logger = logging.getLogger(__name__)


class WhiteboardProcessor:

    def __init__(self):
        self.yolo_detector = WhiteboardYOLODetector(
            model_path=settings.yolo_model_path,
            conf_threshold=settings.yolo_conf_threshold,
            iou_threshold=settings.yolo_iou_threshold,
        )

        self.text_detector = WhiteboardTextDetector(
            use_paddle=settings.ocr_enabled,
            lang=(
                settings.ocr_languages[0]
                if settings.ocr_languages
                else "en"
            ),
        )

        self.connection_detector = ConnectionDetector()

        # Existing rule-based Mermaid generator.
        # This is used as a fallback when NVIDIA fails.
        self.mermaid_generator = MermaidGenerator(
            theme=settings.mermaid_theme
        )

        self.canvas_generator = CanvasGenerator()

        # NVIDIA service is optional.
        # The application can still start if NVIDIA initialization fails.
        self.nvidia_service: Optional[NvidiaService] = None

        try:
            self.nvidia_service = NvidiaService()
            logger.info("NVIDIA Mermaid service initialized")
        except Exception as exc:
            logger.warning(
                "NVIDIA service could not be initialized. "
                "The local Mermaid generator will be used. Error: %s",
                exc,
            )

    async def initialize(self):
        self.text_detector.load()
        logger.info("Whiteboard processor initialized")

    async def process_image(
        self,
        image_data: bytes,
        filename: str,
        content_type: str,
        diagram_type: Optional[str] = None,
        output_format: str = "mermaid",
    ) -> ProcessedWhiteboard:

        image = self._load_image(image_data)

        if image is None:
            raise ValueError("Failed to load image")

        height, width = image.shape[:2]

        logger.info(
            "Processing image: %s, width=%s, height=%s",
            filename,
            width,
            height,
        )

        # Step 1: Detect shapes.
        elements = self.yolo_detector.detect_elements(image)

        logger.info("Detected %s whiteboard elements", len(elements))

        # Step 2: Detect and associate OCR text.
        if settings.ocr_enabled:
            elements = self.text_detector.detect_and_associate(
                image,
                elements,
            )

        # Step 3: Assign stable IDs before detecting connections.
        for index, element in enumerate(elements):
            element.id = f"elem_{index}"

        # Step 4: Detect lines and arrows.
        connections = self.connection_detector.detect(
            image,
            elements,
        )

        # Normalize connection IDs.
        for connection in connections:
            if connection.start_id is not None:
                connection.start_id = self._normalize_element_id(
                    connection.start_id
                )

            if connection.end_id is not None:
                connection.end_id = self._normalize_element_id(
                    connection.end_id
                )

        image_obj = WhiteboardImage(
            id="img_1",
            filename=filename,
            content_type=content_type,
            size=len(image_data),
            width=width,
            height=height,
        )

        processed = ProcessedWhiteboard(
            image=image_obj,
            elements=elements,
            connections=connections,
        )

        processed.analysis = self._analyze(
            elements,
            connections,
        )

        # Step 5: Generate Mermaid.
        if output_format in ["mermaid", "all"]:
            processed.mermaid_code = await self._generate_mermaid(
                processed
            )

        # Step 6: Generate canvas JSON.
        if output_format in ["canvas", "all"]:
            processed.canvas_json = self.canvas_generator.generate(
                processed
            )

        # Step 7: Generate SVG.
        if (
            output_format in ["svg", "all"]
            and processed.mermaid_code
        ):
            processed.svg = self._mermaid_to_svg(
                processed.mermaid_code
            )

        return processed

    async def _generate_mermaid(
        self,
        processed: ProcessedWhiteboard,
    ) -> str:
        """
        Generate Mermaid using NVIDIA.

        If NVIDIA fails or is unavailable, use the existing
        rule-based Mermaid generator.
        """

        if self.nvidia_service is None:
            logger.info(
                "NVIDIA service unavailable. "
                "Using local Mermaid generator."
            )

            return self.mermaid_generator.generate(processed)

        elements_data = [
            self._model_to_dict(element)
            for element in processed.elements
        ]

        connections_data = [
            self._connection_to_dict(connection)
            for connection in processed.connections
        ]

        try:
            logger.info(
                "Generating Mermaid with NVIDIA for "
                "%s elements and %s connections",
                len(elements_data),
                len(connections_data),
            )

            mermaid_code = await asyncio.to_thread(
                self.nvidia_service.generate_mermaid,
                elements_data,
                connections_data,
            )

            logger.info(
                "Mermaid successfully generated using NVIDIA"
            )

            return mermaid_code

        except Exception as exc:
            logger.exception(
                "NVIDIA Mermaid generation failed. "
                "Using local generator instead. Error: %s",
                exc,
            )

            return self.mermaid_generator.generate(processed)

    @staticmethod
    def _model_to_dict(model: Any) -> dict:
        """
        Convert a Pydantic model or regular object into a dictionary.
        """

        if hasattr(model, "model_dump"):
            return model.model_dump()

        if hasattr(model, "dict"):
            return model.dict()

        if hasattr(model, "__dict__"):
            return {
                key: value
                for key, value in vars(model).items()
                if not key.startswith("_")
            }

        raise TypeError(
            f"Cannot convert object of type "
            f"{type(model).__name__} to dictionary"
        )

    def _connection_to_dict(self, connection: Any) -> dict:
        """
        Convert a connection object to the format expected
        by NvidiaService.

        NvidiaService expects source and target, while the current
        connection model uses start_id and end_id.
        """

        connection_data = self._model_to_dict(connection)

        return {
            "source": connection_data.get("start_id"),
            "target": connection_data.get("end_id"),
            "type": connection_data.get("type"),
            "label": connection_data.get("label"),
            "confidence": connection_data.get("confidence"),
        }

    @staticmethod
    def _normalize_element_id(element_id: Any) -> str:
        """
        Convert IDs such as 0, '0' or 'elem_0' into 'elem_0'.
        """

        element_id = str(element_id)

        if element_id.startswith("elem_"):
            return element_id

        return f"elem_{element_id}"

    def _load_image(
        self,
        image_data: bytes,
    ) -> Optional[np.ndarray]:

        if not image_data:
            return None

        image_array = np.frombuffer(
            image_data,
            dtype=np.uint8,
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR,
        )

        return image

    def _analyze(
        self,
        elements,
        connections,
    ) -> AnalysisResult:

        type_counts = {}

        for element in elements:
            type_counts[element.type] = (
                type_counts.get(element.type, 0) + 1
            )

        connection_types = {}

        for connection in connections:
            connection_types[connection.type] = (
                connection_types.get(connection.type, 0) + 1
            )

        diagram_type = "flowchart"

        if (
            type_counts.get("diamond", 0) > 0
            or type_counts.get("decision", 0) > 0
        ):
            diagram_type = "flowchart"

        elif type_counts.get("sticky_note", 0) > 2:
            diagram_type = "mindmap"

        elif (
            type_counts.get("database", 0) > 0
            or type_counts.get("cylinder", 0) > 0
        ):
            diagram_type = "er"

        elif len(connections) > len(elements) * 1.5:
            diagram_type = "network"

        return AnalysisResult(
            diagram_type=diagram_type,
            element_counts=type_counts,
            connection_counts=connection_types,
            element_count=len(elements),
            connection_count=len(connections),
        )

    def _mermaid_to_svg(
        self,
        mermaid_code: str,
    ) -> str:

        escaped_mermaid = (
            mermaid_code
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        lines = escaped_mermaid.splitlines()

        tspans = "".join(
            (
                f'<tspan x="10" dy="1.2em">'
                f"{line}"
                f"</tspan>"
            )
            for line in lines
        )

        return f"""
<svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 800 600"
>
    <text
        x="10"
        y="20"
        font-family="monospace"
        font-size="12"
    >
        {tspans}
    </text>
</svg>
""".strip()