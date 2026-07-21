import base64
import logging
from typing import Optional

from openai import OpenAI

from app.core.config import settings


logger = logging.getLogger(__name__)


class NvidiaService:
    """
    Uses an NVIDIA-hosted vision-language model to read a whiteboard
    image directly and produce valid Mermaid flowchart code.

    This replaces the previous multi-stage pipeline (YOLO shape
    detection + OCR + OpenCV connection detection + LLM-from-JSON)
    with a single vision-capable LLM call.
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

        logger.info("NVIDIA vision service initialized with model: %s", self.model)

    def generate_mermaid_from_image(
        self,
        image_bytes: bytes,
        content_type: str = "image/png",
    ) -> str:
        """
        Send the whiteboard image directly to the vision-language model
        and get back Mermaid flowchart code.

        Args:
            image_bytes: Raw bytes of the uploaded image.
            content_type: MIME type of the image (e.g. "image/png", "image/jpeg").

        Returns:
            Mermaid flowchart code without Markdown code fences.
        """

        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{content_type};base64,{base64_image}"

        prompt = self._build_prompt()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert at reading whiteboard photos "
                            "and converting them into valid Mermaid flowchart syntax."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": data_url,
                                },
                            },
                        ],
                    },
                ],
                temperature=0,
                max_tokens=2500,
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
            logger.exception("NVIDIA vision Mermaid generation failed")
            raise

    @staticmethod
    def _build_prompt() -> str:
        return """
Look at this whiteboard/diagram photo carefully.

STEP 0 - COMPLEXITY CHECK (do this first, always):

Count the distinct shapes visible in the image.

- If there are 3 or fewer shapes AND you cannot see any arrows or
  lines connecting them, STOP HERE. Do not analyze further, do not
  invent structure. Output exactly one node per shape, with zero
  connections, then skip directly to the final Mermaid output. Do
  not create more nodes than there are shapes, under any
  circumstances.

- If there are 3 or fewer shapes AND there IS at least one visible
  connecting arrow, output only those shapes and only that arrow
  (or arrows) — nothing more.

- Only if there are 4 or more shapes with visible connections between
  them, proceed to the full analysis in Step 1 below.

STEP 1 - ANALYZE (only for diagrams with 4+ connected shapes):

a) List every distinct shape, numbering them in reading order (top to
   bottom, left to right). Note its shape type and exact visible text.

b) List every arrow/line. For each: which shape it starts at, which
   shape it ends at, direction, and any label.

   Pay attention to diamonds (usually mean 2+ outgoing arrows),
   branches, merges, loops, and horizontal arrows — these are common
   in real diagrams and easy to miss.

c) After listing, COUNT your total nodes and connections. If your
   count is more than double the number of shapes you can visually
   count in the image, you have likely duplicated something — go back
   and remove the duplicates before writing the final output. Each
   shape in the image should appear as exactly ONE node, never more.

STEP 2 - OUTPUT:

Grounding rules (apply always, regardless of which path above you took):
1. Node count must exactly equal the number of shapes actually visible
   in the image. Never invent extra nodes. Never repeat the same shape
   as multiple nodes.
2. Connection count and direction must exactly match arrows actually
   visible in the image. Never invent a connection that isn't drawn.
3. If text is illegible, use the shape type as a fallback label.

Formatting rules:
4. Return only Mermaid code, no explanation, no Markdown fences.
5. Start with: flowchart TD
6. Node IDs: N1, N2, N3... in reading order.
7. Shape mapping: rectangle -> N1[Label], rounded rectangle ->
   N1(Label), circle/ellipse -> N1((Label)), diamond -> N1{Label},
   start/end oval -> N1([Label]).
8. Connections: N1 --> N2, or N1 -->|label| N2 if labeled.

If you used Step 1, first write your analysis as plain text, then a
new line with "MERMAID:", then the final code. If you used the Step 0
shortcut, skip straight to "MERMAID:" and the code — do not write an
analysis for simple images.
""".strip()

    @staticmethod
    def _clean_mermaid_code(content: str) -> str:
        """
        Remove Markdown code fences, analysis text, and any content
        before the actual Mermaid diagram.
        """

        cleaned = content.strip()

        # Strip everything up to and including a "MERMAID:" marker, if present.
        marker_index = cleaned.upper().find("MERMAID:")
        if marker_index != -1:
            cleaned = cleaned[marker_index + len("MERMAID:"):].strip()

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