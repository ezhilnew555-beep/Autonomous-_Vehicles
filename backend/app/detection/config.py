"""
Detection Configuration — Phase 2.

Exposes configurable hyperparameters for YOLO detection:
- MODEL_NAME
- CONFIDENCE_THRESHOLD
- IMAGE_SIZE
- DEVICE (CPU / CUDA auto-fallback)
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class DetectionConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    MODEL_NAME: str = "models/yolo11n.pt"
    CONFIDENCE_THRESHOLD: float = 0.40
    IMAGE_SIZE: int = 640
    DEVICE: str = "cpu"


_detection_config: DetectionConfig | None = None


def get_detection_config() -> DetectionConfig:
    global _detection_config
    if _detection_config is None:
        _detection_config = DetectionConfig()
    return _detection_config
