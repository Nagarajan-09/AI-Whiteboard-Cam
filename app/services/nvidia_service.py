import json
import logging
from typing import Any

from openai import OpenAI

from app.core.config import settings


logger = logging.getLogger(__name__)


class NvidiaService:
    """
    Uses NVIDIA Nemotron to convert detected whiteboard information
    into valid Mermaid flowchart code.
    """

    def __init__(self) -> None:
        if not settings.nvidia_api_key:
            raise ValueError("NVIDIA_API_KEY is missing from the .env file")

        if not settings.nvidia_base_url:
            raise ValueError("NVIDIA_BASE_URL is missing from the .env file")

        if not settings.nvidia_model:
            raise ValueError("NVIDIA_MODEL is missing from the .env file")

        self.client = OpenAI(
            api_key=settings.nvidia_api_key,
            base_url=settings.nvidia_base_url,
        )

        self.model = settings.nvidia_model

        logger.info("NVIDIA service initialized with model: %s", self.model)

    def generate_mermaid(
        self,
        elements: list[dict[str, Any]],
        connections: list[dict[str, Any]],
    ) -> str:
        """
        Generate Mermaid flowchart code from detected elements and connections.

        Args:
            elements: Detected whiteboard shapes and OCR text.
            connections: Detected arrows or lines between elements.

        Returns:
            Mermaid flowchart code without Markdown code fences.
        """

        detection_data = {
            "elements": elements,
            "connections": connections,
        }

        prompt = self._build_prompt(detection_data)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You convert structured whiteboard detection data "
                            "into valid Mermaid flowchart syntax."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0,
                max_tokens=1500,
            )

            mermaid_code = response.choices[0].message.content

            if not mermaid_code:
                raise ValueError("NVIDIA returned an empty response")

            cleaned_code = self._clean_mermaid_code(mermaid_code)

            if not self._is_valid_mermaid(cleaned_code):
                raise ValueError(
                    f"NVIDIA returned invalid Mermaid code: {cleaned_code}"
                )

            return cleaned_code

        except Exception:
            logger.exception("NVIDIA Mermaid generation failed")
            raise

    def _build_prompt(self, detection_data: dict[str, Any]) -> str:
        data_json = json.dumps(
            detection_data,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

        return f"""
Convert the whiteboard detection data below into a Mermaid flowchart.

Detection data:

{data_json}

Rules:

1. Return only Mermaid code.
2. Do not use Markdown code fences.
3. Start with: flowchart TD
4. Create one node for every detected element.
5. Use readable node IDs such as N1, N2, N3.
6. Use the detected text as the node label.
7. When text is missing, use the element type as the label.
8. Use arrows for detected connections.
9. Do not invent unnecessary nodes.
10. Escape characters that could break Mermaid syntax.
11. Keep the diagram simple and syntactically valid.

Shape mapping:

- rectangle -> N1[Label]
- rounded_rectangle -> N1(Label)
- circle -> N1((Label))
- ellipse -> N1([Label])
- diamond -> N1{{Label}}
- decision -> N1{{Label}}
- process -> N1[Label]
- start -> N1([Start])
- end -> N1([End])

Return only the final Mermaid flowchart.
""".strip()

    @staticmethod
    def _clean_mermaid_code(content: str) -> str:
        """
        Remove Markdown code fences and unnecessary surrounding text.
        """

        cleaned = content.strip()

        if cleaned.startswith("```mermaid"):
            cleaned = cleaned[len("```mermaid"):].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:].strip()

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        mermaid_start = cleaned.find("flowchart")

        if mermaid_start == -1:
            mermaid_start = cleaned.find("graph")

        if mermaid_start > 0:
            cleaned = cleaned[mermaid_start:]

        return cleaned.strip()

    @staticmethod
    def _is_valid_mermaid(code: str) -> bool:
        """
        Perform basic Mermaid validation.

        Full Mermaid validation will happen in the frontend renderer.
        """

        if not code:
            return False

        first_line = code.splitlines()[0].strip().lower()

        return (
            first_line.startswith("flowchart")
            or first_line.startswith("graph")
        )