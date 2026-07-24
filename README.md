# AI Whiteboard Cam

An AI-powered web application that converts whiteboard photos and handwritten sketches into publication-ready, interactive Mermaid.js diagrams and editable Fabric.js canvas layouts.

## Key Features

- **Vision AI Processing**: Leverages NVIDIA LLaMA 3.2 Vision models for high-accuracy handwritten text extraction and shape recognition.
- **Two-Stage Graph Sanitization**: Built-in Python graph sanitizer that automatically deduplicates overlapping node detections and resolves floating branch labels (like "Yes/No") into edge arrows.
- **Publication-Ready Mermaid Diagrams**: Injects modern design tokens, smooth curves (`curve: 'basis'`), and automatic node color-coding (Start/Stop in green, Decisions in amber, Processes in blue).
- **High-DPI Vector Exports**: One-click client-side export functions for **Retina PNGs (2x resolution)**, **SVG vectors**, and direct source code copying.
- **Interactive Fabric.js Canvas**: Converts spatial percentage coordinates (x, y) into interactive canvas elements for manual adjustments.
- **Responsive Web Interface**: Modern UI with real-time image drag-and-drop, image contrast optimization, and synchronized tab navigation.

---

## Architecture Overview

```text
┌─────────────────┐       ┌──────────────────────┐       ┌────────────────────────┐
│  Frontend UI    │ ────> │   FastAPI Backend    │ ────> │ NVIDIA Vision API      │
│  (HTML/JS/CSS)  │ <──── │  (Port 8001 / async) │ <──── │ (LLaMA 3.2 Vision)     │
└─────────────────┘       └──────────────────────┘       └────────────────────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │ Stage-2 Graph        │
                          │ Sanitizer (Python)   │
                          └──────────────────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │  Mermaid & Fabric.js │
                          │  Code Generators     │
                          └──────────────────────┘


AI-Whiteboard-Cam/
├── app/
│   ├── api/
│   │   └── routes.py              # FastAPI endpoint routing
│   ├── core/
│   │   └── config.py              # Environment configuration settings
│   ├── models/
│   │   └── whiteboard.py          # Pydantic schema data models
│   └── services/
│       ├── nvidia_service.py      # NVIDIA LLaMA Vision API integration & image preprocessing
│       └── whiteboard_processor.py# Main pipeline & Stage-2 Graph Sanitization
├── frontend/
│   └── index.html                 # Modern web interface with high-DPI export handlers
├── main.py                        # FastAPI application entry point
├── requirements.txt               # Dependencies (httpx, openai, Pillow, fastapi, uvicorn)
└── .env                           # Environment variables & API credentials

Quick Start Guide
1. Prerequisites
Python 3.10+ installed

NVIDIA API Key (obtainable from build.nvidia.com)

2. Environment Configuration
Create or update your .env file in the project root:

NVIDIA_API_KEY=your_nvidia_api_key_here
NVIDIA_BASE_URL=[https://integrate.api.nvidia.com/v1](https://integrate.api.nvidia.com/v1)
NVIDIA_MODEL=meta/llama-3.2-11b-vision-instruct

# Set UTF-8 encoding (Windows Command Prompt)
set PYTHONUTF8=1
set PYTHONPATH=.

# Start FastAPI server via Uvicorn
uvicorn app.main:app --reload --host 127.0.0.1 --port 8001

