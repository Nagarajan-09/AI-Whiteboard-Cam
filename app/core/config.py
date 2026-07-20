from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application settings
    app_name: str = "AI Whiteboard Cam"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000

    # YOLO settings
    yolo_model_path: str = "yolov8n.pt"
    yolo_conf_threshold: float = 0.25
    yolo_iou_threshold: float = 0.45

    # OCR settings
    ocr_enabled: bool = True
    ocr_languages: list[str] = Field(
        default_factory=lambda: ["en"]
    )

    # Mermaid settings
    mermaid_theme: str = "default"
    mermaid_theme_variables: dict[str, Any] = Field(
        default_factory=dict
    )

    # Upload settings
    max_upload_size: int = 10 * 1024 * 1024
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [
            "jpg",
            "jpeg",
            "png",
            "webp",
            "bmp",
        ]
    )

    # Output settings
    output_format: str = "mermaid"

    # NVIDIA settings
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()