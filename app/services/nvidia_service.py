import base64
import io
import json
import logging
import re
from typing import Any, Dict
import httpx
from PIL import Image, ImageEnhance, ImageOps

from openai import OpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)


class NvidiaService:
    def __init__(self) -> None:
        if not settings.nvidia_api_key or not settings.nvidia_base_url or not settings.nvidia_model:
            raise ValueError("NVIDIA API configuration missing in settings/.env file")

        self.api_key = str(settings.nvidia_api_key).strip().strip("'\"")
        self.base_url = str(settings.nvidia_base_url).strip().strip("'\"")
        self.model = str(settings.nvidia_model).strip().strip("'\"")

        http_client = httpx.Client(verify=False)

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=http_client
        )

        logger.info("NVIDIA Vision Service initialized with Model: %s", self.model)

    def extract_whiteboard_data(
        self,
        image_bytes: bytes,
        content_type: str = "image/jpeg",
    ) -> Dict[str, Any]:

        optimized_bytes = self._prepare_image_bytes(image_bytes)
        base64_image = base64.b64encode(optimized_bytes).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{base64_image}"

        prompt = """
You are a diagram-scanning OCR engine, not a business analyst. You extract only what
is visibly drawn. You never infer, complete, or "correct" a workflow's logic.

TASK: Extract every shape, handwritten text, and connector line visible in this image
into the JSON schema below. Nothing more, nothing less.

JSON SCHEMA:
{
  "elements": [
    {
      "id": "N1",
      "type": "rectangle|oval|diamond|parallelogram",
      "text": "exact text as written, or [UNCLEAR] if illegible",
      "x": 50,
      "y": 10,
      "confidence": "high|low"
    }
  ],
  "connections": [
    {
      "from_id": "N1",
      "to_id": "N2",
      "label": "text on the line if any, else empty string"
    }
  ]
}

STRICT VISUAL GROUNDING RULES:
1. Every element you output must correspond to a shape you can actually see drawn
   (a closed rectangle, oval, diamond, or parallelogram) with text inside or beside it.
   Do not invent a node to make the flow "make sense."
2. Only extract a connection if you can see a drawn line or arrow between two shapes.
   Never add a connection because it would be the logical next step -- if there's no
   visible line, it does not exist. A branch with no outgoing arrow (e.g. a dead-end
   "No" box) must be output with no connection from it.
3. Preserve exact handwritten text, exact connector direction (arrowhead determines
   from_id -> to_id), and exact branch structure as drawn. Do not redesign, optimize,
   merge, or "clean up" the workflow's logic.
4. If text or a shape boundary is ambiguous, set "text" to "[UNCLEAR]" and
   "confidence" to "low" rather than guessing. It is always better to mark something
   unclear than to hallucinate a plausible-sounding value.
5. x/y are the shape's spatial center as a percentage of image width/height
   (0=left/top, 100=right/bottom) -- estimate these from the shape's actual position,
   don't default them to evenly-spaced guesses.
6. Before finalizing: re-check that every element and connection you're about to
   output has a visible shape/line backing it in the image. Drop anything you can't
   visually justify.

Return ONLY valid raw JSON bounded strictly by { and }. No preamble, no markdown fences.
""".strip()

        try:
            logger.info("Executing API request to NVIDIA model: %s", self.model)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a spatial computer vision API. Output strictly valid JSON matching the requested schema.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            },
                        ],
                    },
                ],
                temperature=0.0,
                max_tokens=1024,
            )

            raw_content = response.choices[0].message.content
            logger.info("NVIDIA API RAW RESPONSE:\n%s", raw_content)

            if not raw_content:
                raise ValueError("NVIDIA Vision API returned empty text content.")

            return self._parse_json(raw_content)

        except Exception as exc:
            logger.error("!!! NVIDIA API CALL FAILED !!! Error: %s", exc, exc_info=True)
            raise exc

    @staticmethod
    def _prepare_image_bytes(image_bytes: bytes, max_size: int = 1024) -> bytes:
        """Resizes high-res images and applies autocontrast to sharpen marker lines."""
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                img = img.convert("RGB")

                # Contrast enhancement for phone photos/shadows
                img = ImageOps.autocontrast(img, cutoff=2)
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(1.4)

                w, h = img.size
                if max(w, h) > max_size:
                    scale = max_size / float(max(w, h))
                    img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=88)
                return buffer.getvalue()
        except Exception:
            return image_bytes

    @staticmethod
    def _parse_json(content: str) -> Dict[str, Any]:
        cleaned = content.strip()
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE).strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start != -1 and end != -1:
            cleaned = cleaned[start : end + 1]

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as err:
            logger.error("Failed to parse JSON string: %s. Raw: %s", err, cleaned)
            return {"elements": [], "connections": []}