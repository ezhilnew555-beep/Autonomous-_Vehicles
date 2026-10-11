"""
Eareyes 2k26 — FastAPI application entry point.

Phase 1: Infrastructure
Phase 2: YOLO Object Detection
Phase 3: ByteTrack Tracking + Motion Estimation
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_v1_router
from app.config import get_settings
from app.utils.file_utils import ensure_dir
from app.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create storage directories and log configuration summary."""
    settings = get_settings()
    ensure_dir(settings.UPLOAD_DIR)
    ensure_dir(settings.OUTPUT_DIR)
    ensure_dir("models")

    logger.info("=" * 60)
    logger.info("  Eareyes 2k26 — Road Object Detection API  v0.3.0")
    logger.info("=" * 60)
    logger.info(f"  Upload dir  : {settings.UPLOAD_DIR}")
    logger.info(f"  Output dir  : {settings.OUTPUT_DIR}")
    logger.info(f"  YOLO model  : {settings.YOLO_MODEL_PATH}")
    logger.info(f"  Device      : {settings.YOLO_DEVICE}")
    logger.info(f"  Confidence  : {settings.YOLO_CONFIDENCE}")
    logger.info(f"  Image size  : {settings.YOLO_IMG_SIZE}")
    logger.info(f"  Track buffer: {settings.TRACK_BUFFER} frames")
    logger.info("=" * 60)

    yield

    logger.info("Eareyes 2k26 backend shutting down.")


# ─── Application ──────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Eareyes 2k26 — Autonomous Road Perception & Adaptive Path Planning API",
    description = (
        "Adaptive Path Planning and Collision Avoidance for Autonomous Vehicles "
        "on Unstructured Indian Roads.\n\n"
        "**Full Pipeline**: Detection + Tracking + Motion Prediction + Collision Risk & TTC + "
        "Autonomous Decision Making + Adaptive Path Planning + Road Context Awareness + System Evaluation."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)

# CORS — allow Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins     = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.include_router(api_v1_router, prefix="/api/v1")


# ─── Health check ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
@app.get("/api/v1/health", tags=["Health"])
async def health():
    return {
        "status":   "ok",
        "service":  "eareyes-2k26",
        "version":  "1.0.0",
        "pipeline": "Full Phase 1-9: Detection, Tracking, Prediction, Collision, Decision, Adaptive Planning & Context",
    }
