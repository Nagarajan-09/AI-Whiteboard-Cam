# AI Whiteboard Cam

An AI-powered application that converts whiteboard photos into clean Mermaid diagrams and interactive canvases.

## Features

- **Image Upload**: Drag & drop or click to upload whiteboard images
- **Object Detection**: YOLO-based detection of shapes, arrows, text boxes, sticky notes
- **OCR Text Extraction**: PaddleOCR/Tesseract for extracting handwritten/printed text
- **Connection Detection**: Automatic detection of arrows and lines between elements
- **Multiple Diagram Types**: Flowchart, Sequence, Class, State, ER, Mind Map, Timeline
- **Multiple Output Formats**: 
  - Mermaid.js code (renderable in any Markdown viewer)
  - Interactive Fabric.js canvas
  - SVG vector graphics
  - Excalidraw-compatible JSON
- **Web Interface**: Clean, responsive frontend for easy use

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Frontend   │────▶│  FastAPI     │────▶│  YOLOv8     │
│  (HTML/JS)  │     │  Backend     │     │  Detector   │
└─────────────┘     └──────────────┘     └─────────────┘
                           │                     │
                           ▼                     ▼
                    ┌──────────────┐     ┌─────────────┐
                    │  OCR Service │     │  Connection │
                    │  (PaddleOCR) │     │  Detector   │
                    └──────────────┘     └─────────────┘
                           │                     │
                           ▼                     ▼
                    ┌─────────────────────────────────┐
                    │    Mermaid Generator            │
                    │  (Flowchart, Sequence, Class,   │
                    │   State, ER, Mindmap, Timeline) │
                    └─────────────────────────────────┘
                           │
                           ▼
                    ┌─────────────────────────────────┐
                    │    Canvas Generator             │
                    │  (Fabric.js, Excalidraw, SVG)   │
                    └─────────────────────────────────┘
```

## Quick Start

### Using Docker (Recommended)

```bash
# Clone and navigate
cd ai-whiteboard-cam

# Copy environment file
cp .env.example .env

# Start with Docker Compose
docker-compose up --build
```

Access:
- Frontend: http://localhost:3000
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download YOLO model (first run auto-downloads)
# Or manually: wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt

# Run server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/upload` | Upload whiteboard image |
| POST | `/api/v1/process` | Process image to diagram |
| GET | `/api/v1/image/{id}` | Get image info |
| DELETE | `/api/v1/image/{id}` | Delete image |
| GET | `/health` | Health check |

### Example Request

```bash
# Upload
curl -X POST -F "file=@whiteboard.jpg" http://localhost:8000/api/v1/upload

# Process
curl -X POST http://localhost:8000/api/v1/process \
  -H "Content-Type: application/json" \
  -d '{"image_id": "uuid", "diagram_type": "flowchart", "output_format": "all"}'
```

### Response Format

```json
{
  "success": true,
  "whiteboard_id": "uuid",
  "mermaid_code": "```mermaid\nflowchart TD\nA[Start] --> B[Process]\n```",
  "canvas_json": {
    "version": "6.0.0",
    "objects": [...],
    "width": 800,
    "height": 600
  },
  "svg": "<svg>...</svg>"
}
```

## Supported Diagram Types

| Type | Mermaid Keyword | Best For |
|------|-----------------|----------|
| Flowchart | `flowchart` | Processes, decisions, flows |
| Sequence | `sequenceDiagram` | API calls, interactions |
| Class | `classDiagram` | Object-oriented design |
| State | `stateDiagram-v2` | State machines |
| ER | `erDiagram` | Database schemas |
| Mindmap | `mindmap` | Brainstorming, concepts |
| Timeline | `timeline` | Project timelines |

## Supported Shapes

- **Basic**: Rectangle, Circle, Ellipse, Diamond, Triangle, Hexagon
- **Flowchart**: Process, Decision, Terminator, Data, Document
- **Database**: Cylinder, Database
- **UI**: Sticky Note, Text Box, Arrow, Line (solid/dashed/dotted)

## Configuration

Key settings in `.env`:

```env
YOLO_MODEL_PATH=yolo11n.pt        # Custom model path
YOLO_CONF_THRESHOLD=0.25          # Detection confidence
OCR_ENABLED=true                  # Enable/disable OCR
OCR_LANGUAGES=["en", "zh"]        # OCR languages
MAX_UPLOAD_SIZE=10485760          # 10MB max
MERMAID_THEME=default             # default, dark, forest, neutral
```

## Frontend Usage

1. Open http://localhost:3000
2. Drag & drop or click to upload a whiteboard photo
3. Select diagram type (or Auto Detect)
4. Choose output format
5. Click "Process Image"
6. View results in tabs:
   - **Mermaid**: Rendered diagram + raw code
   - **Canvas**: Interactive Fabric.js editor
   - **JSON**: Canvas JSON for embedding
   - **SVG**: Vector graphic
   - **Excalidraw**: Import into Excalidraw

## Project Structure

```
ai-whiteboard-cam/
├── app/
│   ├── api/routes.py           # FastAPI routes
│   ├── core/config.py          # Configuration
│   ├── models/whiteboard.py    # Pydantic models
│   ├── services/
│   │   ├── yolo_detector.py    # YOLO object detection
│   │   ├── ocr_service.py      # PaddleOCR/Tesseract
│   │   └── whiteboard_processor.py  # Main pipeline
│   ├── detectors/
│   │   └── connection_detector.py  # Arrow/line detection
│   └── generators/
│       ├── mermaid_generator.py    # Mermaid code generation
│       └── canvas_generator.py     # Fabric.js/Excalidraw JSON
├── frontend/index.html         # Web interface
├── main.py                     # FastAPI app entry
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container build
├── docker-compose.yml          # Multi-service deployment
└── .env.example                # Configuration template
```

## Requirements

- Python 3.11+
- OpenCV dependencies (libglib2.0-0, libsm6, libxext6, libxrender-dev, libgl1-mesa-glx)
- 2GB+ RAM for model inference
- Optional: GPU (CUDA) for faster processing

## License

MIT License - Feel free to use and modify!