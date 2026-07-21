import logging
from typing import Optional

from app.models.whiteboard import WhiteboardImage, ProcessedWhiteboard, AnalysisResult
from app.services.nvidia_service import NvidiaService
from app.core.config import settings


logger = logging.getLogger(__name__)


class WhiteboardProcessor:

    def __init__(self):

        self.nvidia_service: Optional[NvidiaService] = None

        try:
            self.nvidia_service = NvidiaService()
            logger.info("NVIDIA vision service initialized")
        except Exception as exc:
            logger.warning(
                "NVIDIA vision service could not be initialized. "
                "A minimal placeholder diagram will be used instead. Error: %s",
                exc,
            )

    async def initialize(self):
        logger.info("Whiteboard processor initialized")

    async def process_image(
        self,
        image_data: bytes,
        filename: str,
        content_type: str,
        diagram_type: Optional[str] = None,
        output_format: str = "mermaid",
    ) -> ProcessedWhiteboard:

        width, height = self._get_image_dimensions(image_data)

        logger.info(
            "Processing image: %s, width=%s, height=%s",
            filename,
            width,
            height,
        )

        image_obj = WhiteboardImage(
            id="img_1",
            filename=filename,
            content_type=content_type,
            size=len(image_data),
            width=width,
            height=height,
        )

        mermaid_code = await self._generate_mermaid(image_data, content_type)

        processed = ProcessedWhiteboard(
            image=image_obj,
            elements=[],
            connections=[],
        )

        processed.mermaid_code = mermaid_code

        processed.analysis = AnalysisResult(
            diagram_type="flowchart",
            element_count=mermaid_code.count("\n") if mermaid_code else 0,
            connection_count=mermaid_code.count("-->") if mermaid_code else 0,
        )

        if output_format in ["svg", "all"] and mermaid_code:
            processed.svg = self._mermaid_to_svg(mermaid_code)

        return processed

    async def _generate_mermaid(
        self,
        image_data: bytes,
        content_type: str,
    ) -> str:
        """
        Generate Mermaid directly from the image using the NVIDIA
        vision-language model. Falls back to a minimal placeholder
        diagram if the NVIDIA call fails or is unavailable.
        """

        if self.nvidia_service is None:
            logger.info("NVIDIA vision service unavailable. Using placeholder diagram.")
            return "flowchart TD\n    N1[Unable to process image]"

        try:
            logger.info("Generating Mermaid from image using NVIDIA vision model")

            import asyncio
            mermaid_code = await asyncio.to_thread(
                self.nvidia_service.generate_mermaid_from_image,
                image_data,
                content_type,
            )

            logger.info("Mermaid successfully generated using NVIDIA vision model")

            return mermaid_code

        except Exception as exc:
            logger.exception(
                "NVIDIA vision Mermaid generation failed. Using placeholder diagram. Error: %s",
                exc,
            )
            return "flowchart TD\n    N1[Diagram generation failed]"

    @staticmethod
    def _get_image_dimensions(image_data: bytes) -> tuple:
        """
        Read image width/height without a full OpenCV decode, using PIL
        (lighter dependency than pulling in OpenCV just for this).
        """
        from PIL import Image
        import io

        with Image.open(io.BytesIO(image_data)) as img:
            return img.width, img.height

    def _mermaid_to_svg(self, mermaid_code: str) -> str:
        escaped_mermaid = (
            mermaid_code
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        lines = escaped_mermaid.splitlines()

        tspans = "".join(
            f'<tspan x="10" dy="1.2em">{line}</tspan>'
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