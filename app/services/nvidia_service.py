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
Analyze this handwritten diagram photo and extract every shape and connection into valid JSON.

JSON SCHEMA:
{
  "elements": [
    {
      "id": "N1", 
      "type": "rectangle|oval|diamond|parallelogram", 
      "text": "Extracted text inside shape",
      "x": 50,  // Horizontal spatial center percentage (0=left, 100=right)
      "y": 10   // Vertical spatial center percentage (0=top, 100=bottom)
    }
  ],
  "connections": [
    {
      "from_id": "N1", 
      "to_id": "N2", 
      "label": "text on line if any"
    }
  ]
}

STRICT VISUAL TRUTH RULES:
1. EXPLICIT SHAPES: Every text enclosed in a drawn box (rectangle, oval, parallelogram) MUST be an element in "elements".
   - In this image, 'Start', 'Button', 'Yes', 'No', and 'Result' are ALL enclosed in distinct drawn boxes! Extract all 5 as separate elements.

2. CONNECTIONS (ARROWS ONLY):
   - Extract ONLY drawn arrows.
   - 'Start' points to 'Button'.
   - 'Button' branches to 'Yes' AND 'No'.
   - ONLY 'Yes' has an arrow pointing to 'Result'. 'No' has NO outgoing arrow—do NOT connect 'No' to 'Result'!

3. Return ONLY valid raw JSON bounded strictly by { and }.
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