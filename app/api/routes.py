from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional
import uuid
import io

from app.models.whiteboard import (
    ProcessRequest,
    ProcessResponse,
    UploadResponse,
    ProcessedWhiteboard
)
from app.services.whiteboard_processor import WhiteboardProcessor
from app.core.config import settings

router = APIRouter()
processor = WhiteboardProcessor()

uploaded_images = {}


@router.post("/upload", response_model=UploadResponse)
async def upload_image(
    file: UploadFile = File(...),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if file.size and file.size > settings.max_upload_size:
        raise HTTPException(status_code=400, detail="File too large")
    
    ext = file.filename.split(".")[-1].lower() if file.filename else ""
    if ext not in settings.allowed_extensions:
        raise HTTPException(status_code=400, detail="Invalid file extension")
    
    image_data = await file.read()
    image_id = str(uuid.uuid4())
    
    uploaded_images[image_id] = {
        "data": image_data,
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(image_data)
    }
    
    return UploadResponse(
        success=True,
        image_id=image_id,
        filename=file.filename,
        width=0,
        height=0
    )


@router.post("/process", response_model=ProcessResponse)
async def process_whiteboard(request: ProcessRequest):
    if request.image_id not in uploaded_images:
        raise HTTPException(status_code=404, detail="Image not found")
    
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
        return ProcessResponse(
            success=False,
            whiteboard_id=request.image_id,
            error=str(e)
        )


@router.get("/health")
async def health_check():
    return {"status": "healthy", "version": settings.app_name}


@router.get("/image/{image_id}")
async def get_image_info(image_id: str):
    if image_id not in uploaded_images:
        raise HTTPException(status_code=404, detail="Image not found")
    
    info = uploaded_images[image_id]
    return {
        "image_id": image_id,
        "filename": info["filename"],
        "content_type": info["content_type"],
        "size": info["size"]
    }


@router.delete("/image/{image_id}")
async def delete_image(image_id: str):
    if image_id not in uploaded_images:
        raise HTTPException(status_code=404, detail="Image not found")
    
    del uploaded_images[image_id]
    return {"success": True}