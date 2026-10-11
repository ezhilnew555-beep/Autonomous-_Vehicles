"""
Model Manager — Phase 2.

Manages loading, caching, device resolution, and verification for YOLO models.
Automatically falls back to CPU if CUDA GPU is unavailable.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from app.detection.config import DetectionConfig, get_detection_config
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ModelManager:
    """
    Singleton manager for loading, caching, and serving YOLO models.
    """

    _instance: Optional[ModelManager] = None
    _loaded_models: dict[str, YOLO] = {}

    def __new__(cls) -> ModelManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def resolve_device(cls, requested_device: str) -> str:
        """
        Validates the requested device. If CUDA is requested but not available,
        gracefully falls back to CPU with a log warning.
        """
        device_lower = str(requested_device).lower().strip()
        if "cuda" in device_lower or device_lower in ("0", "1"):
            try:
                import torch
                if torch.cuda.is_available():
                    return requested_device
                else:
                    logger.warning(
                        f"CUDA device '{requested_device}' requested but torch.cuda.is_available() is False. "
                        f"Automatically falling back to CPU."
                    )
                    return "cpu"
            except ImportError:
                logger.warning("torch not available for CUDA check; falling back to CPU.")
                return "cpu"
        return "cpu"

    @classmethod
    def load_model(
        cls,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> YOLO:
        """
        Loads and returns a YOLO model instance, reusing cached model if already loaded.
        """
        config = get_detection_config()
        path = model_path or config.MODEL_NAME
        req_device = device or config.DEVICE
        resolved_dev = cls.resolve_device(req_device)

        cache_key = f"{path}_{resolved_dev}"
        if cache_key in cls._loaded_models:
            return cls._loaded_models[cache_key]

        logger.info(f"ModelManager: Loading YOLO model from '{path}' on device '{resolved_dev}'...")
        try:
            # Ultralytics will auto-download from CDN if weights file does not exist locally
            model = YOLO(path)
            model.to(resolved_dev)
            cls._loaded_models[cache_key] = model
            logger.info(f"ModelManager: Model '{path}' successfully loaded on '{resolved_dev}'.")
            return model
        except Exception as exc:
            logger.error(f"ModelManager: Failed to load YOLO model from '{path}': {exc}")
            raise
