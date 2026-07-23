import io
import logging
import uuid
from typing import Dict, Any, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from app.core.config import settings
from app.models.whiteboard import (
    ProcessRequest,
    ProcessResponse,
    ProcessedWhiteboard,
    UploadResponse,
)
from app.services.whiteboard_processor import WhiteboardProcessor

logger = logging.getLogger(__name__)

router = APIRouter()
processor = WhiteboardProcessor()

# In-memory storage for uploaded image metadata and bytes
uploaded_images: Dict[str, Dict[str, Any]] = {}


@router.post("/upload", response_model=UploadResponse)
async def upload_image(
    file: UploadFile = File(...),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be a valid image")

    image_data = await file.read()
    
    if len(image_data) > settings.max_upload_size:
        raise HTTPException(status_code=400, detail="File size exceeds maximum limit")

    ext = file.filename.split(".")[-1].lower() if file.filename and "." in file.filename else ""
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension. Allowed: {', '.join(settings.allowed_extensions)}"
        )

    # Extract actual dimensions on upload
    width, height = 800, 600
    try:
        with Image.open(io.BytesIO(image_data)) as img:
            width, height = img.width, img.height
    except Exception as err:
        logger.warning(f"Could not read image metadata using PIL: {err}")

    image_id = str(uuid.uuid4())

    uploaded_images[image_id] = {
        "data": image_data,
        "filename": file.filename or f"{image_id}.png",
        "content_type": file.content_type,
        "size": len(image_data),
        "width": width,
        "height": height
    }

    logger.info(f"Image uploaded successfully: ID={image_id}, Name={file.filename}")

    return UploadResponse(
        success=True,
        image_id=image_id,
        filename=file.filename,
        width=width,
        height=height
    )


@router.post("/process", response_model=ProcessResponse)
async def process_whiteboard(request: ProcessRequest):
    if request.image_id not in uploaded_images:
        raise HTTPException(status_code=404, detail="Uploaded image ID not found")

    image_info = uploaded_images[request.image_id]

    try:
        processed = await processor.process_image(
            image_data=image_info["data"],
            filename=image_info["filename"],
            content_type=image_info["content_type"],
            diagram_type=request.diagram_type,
            output_format=request.output_format
        )

        return ProcessResponse(
            success=True,
            whiteboard_id=request.image_id,
            processed=processed,
            mermaid_code=processed.mermaid_code,
            canvas_json=processed.canvas_json,
            svg=processed.svg
        )
    except Exception as e:
        logger.exception(f"Error processing whiteboard image ID {request.image_id}")
        return ProcessResponse(
            success=False,
            whiteboard_id=request.image_id,
            error=str(e)
        )


@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": getattr(settings, "app_name", "AI Whiteboard Cam")}


@router.get("/image/{image_id}")
async def get_image_info(image_id: str):
    if image_id not in uploaded_images:
        raise HTTPException(status_code=404, detail="Image not found")

    info = uploaded_images[image_id]
    return {
        "image_id": image_id,
        "filename": info["filename"],
        "content_type": info["content_type"],
        "size": info["size"],
        "width": info.get("width"),
        "height": info.get("height")
    }


@router.delete("/image/{image_id}")
async def delete_image(image_id: str):
    if image_id not in uploaded_images:
        raise HTTPException(status_code=404, detail="Image not found")

    del uploaded_images[image_id]
    return {"success": True}