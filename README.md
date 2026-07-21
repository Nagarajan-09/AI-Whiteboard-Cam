# AI Whiteboard Cam

An AI-powered application that converts whiteboard photos into clean Mermaid diagrams — in a single vision-LLM call.

## Features

- **Image Upload**: Drag & drop or click to upload whiteboard images
- **Vision-LLM Understanding**: A single call to an NVIDIA-hosted vision-language model reads shapes, handwritten/printed text, and connections directly from the photo — no separate detection, OCR, or connection-detection stages
- **Grounded Output**: A carefully tuned prompt keeps the model from hallucinating structure that isn't in the image, and from missing structure that is
- **Mermaid Output**: Clean, renderable Mermaid flowchart code
- **SVG Export**: A simple SVG rendering of the generated Mermaid code
- **Web Interface**: Clean, responsive frontend for easy use

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────────────┐
│  Frontend   │────▶│  FastAPI     │────▶│  NVIDIA Vision-LLM    │
│  (HTML/JS)  │     │  Backend     │     │  (single call: image  │
└─────────────┘     └──────────────┘     │   in, Mermaid out)    │
                                          └───────────────────────┘
                                                     │
                                                     ▼
                                          ┌───────────────────────┐
                                          │  Mermaid Code +       │
                                          │  SVG Rendering        │
                                          └───────────────────────┘
```

This replaces an earlier multi-stage pipeline (YOLO shape detection →
OCR → OpenCV connection detection → LLM-from-JSON generation). That
pipeline is no longer used: it required a custom-trained YOLO model
(never available), depended on OCR engines that were unreliable to
install locally, and needed a separate detection stage for every
element type. A single vision-capable LLM call now handles shape
recognition, text reading, and connection inference together, with
far fewer moving parts and no local model/OCR dependencies.

## Quick Start

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure your .env (see Configuration below)

# Run server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/upload` | Upload whiteboard image |
| POST | `/api/v1/process` | Process image into a Mermaid diagram |
| GET | `/health` | Health check |

### Example Request

```bash
# Upload
curl -X POST -F "file=@whiteboard.jpg" http://localhost:8000/api/v1/upload

# Process
curl -X POST http://localhost:8000/api/v1/process \
  -H "Content-Type: application/json" \
  -d '{"image_id": "uuid", "output_format": "all"}'
```

### Response Format

```json
{
  "success": true,
  "whiteboard_id": "uuid",
  "mermaid_code": "flowchart TD\n    N1[Start]\n    N2[Process]\n    N1 --> N2",
  "svg": "<svg>...</svg>"
}
```

## Configuration

Key settings in `.env`:

```env
NVIDIA_API_KEY=your_key_here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=meta/llama-3.2-11b-vision-instruct   # vision-capable model
MAX_UPLOAD_SIZE=10485760          # 10MB max
MERMAID_THEME=default             # default, dark, forest, neutral
```

**Note:** `NVIDIA_MODEL` must be a vision-capable (multimodal) model —
it needs to accept an image as input, not just text. Check
[build.nvidia.com](https://build.nvidia.com) for the current catalog
of available vision-language models if you want to try a different
one.

## Frontend Usage

1. Open http://localhost:3000
2. Drag & drop or click to upload a whiteboard photo
3. Click "Process Image"
4. View the rendered Mermaid diagram and its underlying code

## Project Structure

```
ai-whiteboard-cam/
├── app/
│   ├── api/routes.py                # FastAPI routes
│   ├── core/config.py               # Configuration
│   ├── models/whiteboard.py         # Pydantic models
│   ├── services/
│   │   ├── nvidia_service.py        # Vision-LLM call + prompt + Mermaid cleanup
│   │   └── whiteboard_processor.py  # Main pipeline (upload -> vision call -> Mermaid/SVG)
├── frontend/index.html              # Web interface
├── main.py                          # FastAPI app entry
├── requirements.txt                 # Python dependencies
└── .env.example                     # Configuration template
```

## Prompt Design Notes

Getting reliable, non-hallucinated output from a vision LLM on sparse
or ambiguous whiteboard photos took some iteration. The current prompt
in `nvidia_service.py` handles two failure modes explicitly:

- **Over-generation (hallucination)**: on simple images (1-3
  unconnected shapes), earlier prompt versions would invent a full
  flowchart structure (fake "Start"/"Decision"/"End" nodes) that
  wasn't in the image at all. The prompt now includes an explicit
  "complexity check" step that short-circuits straight to a minimal,
  literal output for simple images, skipping the more elaborate
  analysis reasoning that was causing the invention.
- **Under-generation (missed structure)**: on more complex, multi-shape
  diagrams with real branches/decisions, the model can also miss
  actual shapes or connections (e.g. dropping a "Start"/"End" node it
  wasn't confident about). The prompt asks the model to explicitly
  enumerate every shape and every arrow before writing any Mermaid
  code, and to self-check its counts against what's visibly in the
  image.

If you're tuning this further, test against at least three image
types before trusting a prompt change: (1) a single simple shape, (2)
a small linear chain (3-5 shapes), and (3) a diagram with a real
decision/branch. A fix for one case has repeatedly caused a regression
in another during development, so all three need to be re-checked
together.

## Requirements

- Python 3.11+
- An NVIDIA API key with access to a vision-capable model (free tier
  available at [build.nvidia.com](https://build.nvidia.com))

## License

MIT License - Feel free to use and modify!