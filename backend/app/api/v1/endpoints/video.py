"""
Video API endpoints — upload, process, status, result, original.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, Query
from fastapi.responses import FileResponse, StreamingResponse

from app.config import Settings, get_settings
from app.models.schemas import (
    JobStatus,
    ProcessResponse,
    StatusResponse,
    UploadResponse,
)
from app.services.video_service import get_job, list_jobs, process_video, register_job
from app.utils.file_utils import (
    generate_video_id,
    get_output_path,
    get_safe_extension,
    get_upload_path,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


# ─── POST /upload ─────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_video(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
):
    """
    Validate and save an uploaded video file.
    Returns a unique video_id for use in subsequent API calls.
    """
    # Validate extension
    try:
        ext = get_safe_extension(file.filename or "")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    video_id  = generate_video_id()
    save_path = get_upload_path(settings.UPLOAD_DIR, video_id, ext)

    # Stream-write to disk (1 MB chunks — never loads full video in RAM)
    total_bytes = 0
    try:
        with open(save_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                total_bytes += len(chunk)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}")

    if total_bytes == 0:
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(
            status_code=400,
            detail="Uploaded video file is empty (0 bytes). Please select a valid video file.",
        )

    output_path = get_output_path(settings.OUTPUT_DIR, video_id)

    register_job(
        video_id          = video_id,
        original_filename = file.filename or "video",
        input_path        = str(save_path),
        output_path       = str(output_path),
        size_bytes        = total_bytes,
    )

    logger.info(f"Uploaded '{file.filename}' → video_id={video_id}, {total_bytes:,} bytes")
    return UploadResponse(
        video_id   = video_id,
        filename   = file.filename or "video",
        size_bytes = total_bytes,
        message    = "Upload successful. POST to /process/{video_id} to start.",
    )


# ─── POST /process/{video_id} ─────────────────────────────────────────────────

@router.post("/process/{video_id}", response_model=ProcessResponse)
async def start_processing(
    video_id:         str,
    background_tasks: BackgroundTasks,
    mode:             Optional[str] = Query(None, description="'prediction' for Phase 4, 'tracking' for Phase 3, 'detection' for Phase 2, or 'full' for Phases 3-9"),
    settings:         Settings = Depends(get_settings),
):
    """Kick off the background pipeline for a given video."""
    job = get_job(video_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Video ID '{video_id}' not found.")
    if job.status in (JobStatus.PROCESSING, JobStatus.DONE):
        raise HTTPException(
            status_code=409,
            detail=f"Job is already '{job.status.value}'. Upload a new video to re-process.",
        )

    pipeline_mode = (mode or settings.PIPELINE_MODE).lower().strip()
    background_tasks.add_task(process_video, video_id, settings, pipeline_mode)
    logger.info(f"Processing started for video_id={video_id} (mode={pipeline_mode})")
    return ProcessResponse(
        video_id=video_id,
        message=f"Processing started in background (mode={pipeline_mode}).",
    )


# ─── GET /status/{video_id} ───────────────────────────────────────────────────

@router.get("/status/{video_id}", response_model=StatusResponse)
async def get_status(video_id: str):
    """Poll job status, progress (0–100), and final stats when done."""
    job = get_job(video_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Video ID '{video_id}' not found.")

    return StatusResponse(
        video_id = video_id,
        status   = job.status,
        progress = round(job.progress, 1),
        message  = job.message,
        stats    = job.stats,
    )


# ─── GET /result/{video_id} ───────────────────────────────────────────────────

@router.get("/result/{video_id}")
async def get_result_video(video_id: str):
    """Stream the processed annotated video."""
    job = get_job(video_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Video ID not found.")
    if job.status != JobStatus.DONE:
        raise HTTPException(status_code=425, detail="Processing not yet complete.")

    output_path = Path(job.output_path)
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file missing — processing may have failed.")

    return FileResponse(
        path=str(output_path),
        media_type="video/mp4",
        filename=f"{video_id}_processed.mp4",
    )


# ─── GET /original/{video_id} ─────────────────────────────────────────────────

@router.get("/original/{video_id}")
async def get_original_video(video_id: str):
    """Stream the original uploaded video."""
    job = get_job(video_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Video ID not found.")

    input_path = Path(job.input_path)
    if not input_path.exists():
        raise HTTPException(status_code=404, detail="Original file not found.")

    return FileResponse(
        path=str(input_path),
        media_type="video/mp4",
        filename=f"original_{video_id}.mp4",
    )


# ─── GET /jobs ────────────────────────────────────────────────────────────────

@router.get("/jobs")
async def list_all_jobs():
    """List all jobs in the current session (for debugging)."""
    return [
        {
            "video_id":  j.video_id,
            "filename":  j.original_filename,
            "status":    j.status,
            "progress":  j.progress,
            "size_bytes": j.size_bytes,
        }
        for j in list_jobs()
    ]
