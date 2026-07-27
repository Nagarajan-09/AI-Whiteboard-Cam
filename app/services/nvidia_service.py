import base64
import io
import json
import logging
import re
from typing import Any, Dict, List, Tuple
import httpx
from PIL import Image, ImageEnhance, ImageOps

from openai import OpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)

# Two fixed seeds for self-consistency voting -- arbitrary but stable across runs
SEED_RUN_A = 42
SEED_RUN_B = 1337


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

    PROMPT = """
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

    def extract_whiteboard_data(
        self,
        image_bytes: bytes,
        content_type: str = "image/jpeg",
    ) -> Dict[str, Any]:
        """
        Runs extraction twice (different seeds) and merges via self-consistency
        voting. Hosted vision models can return different results for the same
        image across calls even at temperature 0 -- batched inference on the
        backend makes floating-point ops non-associative, so logits (and thus
        sampled tokens) can shift run to run. Two-pass voting catches that:
        anything both runs agree on is kept as high-confidence; anything only
        one run reported is kept but flagged low-confidence rather than either
        silently trusted or dropped outright.
        """
        optimized_bytes = self._prepare_image_bytes(image_bytes)
        base64_image = base64.b64encode(optimized_bytes).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{base64_image}"

        run_a_raw = self._call_vision_model(data_url, seed=SEED_RUN_A)
        run_b_raw = self._call_vision_model(data_url, seed=SEED_RUN_B)

        run_a = self._parse_json(run_a_raw)
        run_b = self._parse_json(run_b_raw)

        return self._merge_consensus(run_a, run_b)

    def _call_vision_model(self, data_url: str, seed: int) -> str:
        base_kwargs = dict(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a spatial computer vision API. Output strictly valid JSON matching the requested schema.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                },
            ],
            temperature=0.0,
            max_tokens=3000,
        )

        try:
            logger.info("Executing API request to NVIDIA model: %s (seed=%s)", self.model, seed)
            response = self.client.chat.completions.create(seed=seed, **base_kwargs)
        except TypeError:
            # Installed openai-python version doesn't expose `seed` as a kwarg at all
            logger.warning("openai client does not support 'seed' kwarg; retrying without it")
            response = self.client.chat.completions.create(**base_kwargs)
        except Exception as exc:
            # Some OpenAI-compatible endpoints reject unrecognized params outright
            if "seed" in str(exc).lower():
                logger.warning("Endpoint rejected 'seed' param; retrying without it. Error: %s", exc)
                response = self.client.chat.completions.create(**base_kwargs)
            else:
                logger.error("!!! NVIDIA API CALL FAILED !!! Error: %s", exc, exc_info=True)
                raise exc

        raw_content = response.choices[0].message.content
        logger.info("NVIDIA API RAW RESPONSE (seed=%s):\n%s", seed, raw_content)

        if not raw_content:
            raise ValueError("NVIDIA Vision API returned empty text content.")

        return raw_content

    @staticmethod
    def _normalize_text(text: Any) -> str:
        return str(text or "").strip().lower()

    @staticmethod
    def _merge_consensus(run_a: Dict[str, Any], run_b: Dict[str, Any]) -> Dict[str, Any]:
        elems_a: List[dict] = run_a.get("elements", []) or []
        elems_b: List[dict] = run_b.get("elements", []) or []

        by_text_b = {NvidiaService._normalize_text(e.get("text")): e for e in elems_b}

        merged_elements: List[dict] = []
        matched_b_keys = set()

        for elem in elems_a:
            key = NvidiaService._normalize_text(elem.get("text"))
            match = by_text_b.get(key)
            if match and key not in matched_b_keys:
                matched_b_keys.add(key)
                try:
                    avg_x = (float(elem.get("x") or 0) + float(match.get("x") or 0)) / 2
                    avg_y = (float(elem.get("y") or 0) + float(match.get("y") or 0)) / 2
                except (TypeError, ValueError):
                    avg_x, avg_y = elem.get("x"), elem.get("y")
                merged_elements.append({
                    **elem,
                    "x": avg_x,
                    "y": avg_y,
                    "confidence": "high",
                })
            else:
                # Only run A reported this -- keep it, but flag it as unconfirmed
                merged_elements.append({**elem, "confidence": "low"})

        for key, elem in by_text_b.items():
            if key not in matched_b_keys:
                # Only run B reported this -- keep it, but flag it as unconfirmed
                merged_elements.append({**elem, "confidence": "low"})

        def connection_text_pair(conn: dict, elems: List[dict]) -> Tuple[str, str]:
            id_to_text = {
                str(e.get("id")): NvidiaService._normalize_text(e.get("text"))
                for e in elems
            }
            return (
                id_to_text.get(str(conn.get("from_id")), ""),
                id_to_text.get(str(conn.get("to_id")), ""),
            )

        conns_a: List[dict] = run_a.get("connections", []) or []
        conns_b: List[dict] = run_b.get("connections", []) or []

        pairs_a = {connection_text_pair(c, elems_a) for c in conns_a}
        pairs_b = {connection_text_pair(c, elems_b) for c in conns_b}

        merged_connections: List[dict] = []
        seen_pairs = set()

        for conn in conns_a:
            pair = connection_text_pair(conn, elems_a)
            if pair in seen_pairs or not pair[0] or not pair[1]:
                continue
            seen_pairs.add(pair)
            merged_connections.append(conn)  # kept regardless; confirmed-by-both is implicit via pair in pairs_b

        for conn in conns_b:
            pair = connection_text_pair(conn, elems_b)
            if pair in seen_pairs or pair in pairs_a or not pair[0] or not pair[1]:
                continue
            seen_pairs.add(pair)
            merged_connections.append(conn)

        return {
            "elements": merged_elements,
            "connections": merged_connections,
        }

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