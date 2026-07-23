from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.core.config import settings
from app.services.whiteboard_processor import WhiteboardProcessor

logger = logging.getLogger(__name__)

processor = WhiteboardProcessor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Initializing AI Whiteboard Cam application dependencies...")
    await processor.initialize()
    yield
    # Shutdown logic
    logger.info("Shutting down AI Whiteboard Cam application...")


app = FastAPI(
    title=getattr(settings, "app_name", "AI Whiteboard Cam"),
    description="AI-powered whiteboard image to diagram converter",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local development and canvas rendering
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router, prefix="/api/v1")

# Mount static frontend files safely
frontend_dir = "frontend"
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    logger.warning(f"Static directory '{frontend_dir}' not found. Serving API endpoints only.")