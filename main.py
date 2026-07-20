from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.core.config import settings
from app.services.whiteboard_processor import WhiteboardProcessor


processor = WhiteboardProcessor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await processor.initialize()
    yield


app = FastAPI(
    title=settings.app_name,
    description="AI-powered whiteboard image to diagram converter",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

# Mount the frontend at the root
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
