"""
Analysis API Endpoints — Unified road analysis interface.
"""
from __future__ import annotations
from typing import List, Optional
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, UploadFile, File
from fastapi.responses import FileResponse

from app.config import Settings, get_settings
from app.services.video_service import get_job, register_job
from app.analysis.analysis_result import AnalysisResult, analysis_store
from app.analysis.analysis_orchestrator import AnalysisOrchestrator
from app.utils.file_utils import generate_video_id, get_safe_extension, get_upload_path, get_output_path
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/upload", status_code=201)
async def upload_analysis_video(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
):
    """
    Direct upload endpoint for unified road analysis.
    Returns analysis_id and video_id ready for analysis pipeline.
    """
    try:
        ext = get_safe_extension(file.filename or "")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    video_id = generate_video_id()
    save_path = get_upload_path(settings.UPLOAD_DIR, video_id, ext)

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
        raise HTTPException(status_code=400, detail="Uploaded video file is empty.")

    output_path = get_output_path(settings.OUTPUT_DIR, video_id)
    register_job(
        video_id=video_id,
        original_filename=file.filename or "video",
        input_path=str(save_path),
        output_path=str(output_path),
        size_bytes=total_bytes,
    )

    result = AnalysisOrchestrator.start_analysis(video_id, settings)
    return {
        "analysis_id": result.analysis_id,
        "video_id": video_id,
        "filename": file.filename or "video",
        "size_bytes": total_bytes,
        "status": result.status,
        "progress": result.progress,
        "message": "Upload successful. POST to /api/v1/analysis/{analysis_id}/start to begin.",
    }


@router.post("/start/{video_id}", status_code=202)
async def start_analysis(
    video_id: str,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
):
    """
    Start the unified autonomous road analysis for an uploaded video.
    Returns analysis_id and initial processing status.
    """
    job = get_job(video_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Video ID '{video_id}' not found.")

    # Check if an analysis already exists for this video
    existing = analysis_store.get_by_video_id(video_id)
    if existing and existing.status in ("PROCESSING", "COMPLETED"):
        return {
            "analysis_id": existing.analysis_id,
            "video_id": video_id,
            "status": existing.status,
            "progress": existing.progress,
            "message": f"Analysis already {existing.status.lower()}.",
        }

    # Initialize analysis record
    result = AnalysisOrchestrator.start_analysis(video_id, settings)
    # Launch in background
    background_tasks.add_task(AnalysisOrchestrator.run_pipeline, result.analysis_id, settings)
    logger.info(f"Analysis started for video_id={video_id} (analysis_id={result.analysis_id})")

    return {
        "analysis_id": result.analysis_id,
        "video_id": video_id,
        "status": result.status,
        "progress": result.progress,
        "current_step": result.current_step,
        "message": "Autonomous road analysis initiated.",
    }


@router.get("/status/{analysis_id}")
async def get_analysis_status(analysis_id: str):
    """
    Poll live progress and current checklist step for an active analysis.
    """
    result = analysis_store.get(analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Analysis ID '{analysis_id}' not found.")

    return {
        "analysis_id": result.analysis_id,
        "video_id": result.video_id,
        "status": result.status,
        "progress": result.progress,
        "current_step": result.current_step,
        "road_status": result.road_status,
        "vehicle_action": result.vehicle_action,
        "error": result.error,
    }


@router.get("/{analysis_id}", response_model=AnalysisResult)
async def get_analysis_result(analysis_id: str):
    """
    Retrieve the full unified analysis result.
    """
    result = analysis_store.get(analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Analysis ID '{analysis_id}' not found.")

    return result


@router.get("/video/{video_id}", response_model=AnalysisResult)
async def get_analysis_by_video(video_id: str):
    """
    Retrieve the analysis result by video ID.
    """
    result = analysis_store.get_by_video_id(video_id)
    if not result:
        job = get_job(video_id)
        if job and job.stats:
            result = AnalysisOrchestrator.create_from_job(job)

    if not result:
        raise HTTPException(status_code=404, detail=f"No analysis found for video ID '{video_id}'.")

    return result


@router.post("/{analysis_id}/start", status_code=202)
async def start_analysis_by_id(
    analysis_id: str,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
):
    """
    Start analysis using analysis_id (or video_id fallback).
    """
    result = analysis_store.get(analysis_id)
    if not result:
        result = analysis_store.get_by_video_id(analysis_id)
        if not result:
            job = get_job(analysis_id)
            if not job:
                raise HTTPException(status_code=404, detail=f"Analysis or Video ID '{analysis_id}' not found.")
            result = AnalysisOrchestrator.start_analysis(analysis_id, settings)

    if result.status in ("PROCESSING", "COMPLETED"):
        return {
            "analysis_id": result.analysis_id,
            "video_id": result.video_id,
            "status": result.status,
            "progress": result.progress,
            "message": f"Analysis already {result.status.lower()}.",
        }

    background_tasks.add_task(AnalysisOrchestrator.run_pipeline, result.analysis_id, settings)
    return {
        "analysis_id": result.analysis_id,
        "video_id": result.video_id,
        "status": result.status,
        "progress": result.progress,
        "current_step": result.current_step,
        "message": "Autonomous road analysis initiated.",
    }


@router.get("/{analysis_id}/events")
async def get_analysis_events(analysis_id: str):
    """
    Retrieve chronological timeline of safety events, decisions, and path replans.
    """
    result = analysis_store.get(analysis_id) or analysis_store.get_by_video_id(analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Analysis ID '{analysis_id}' not found.")
    return {"analysis_id": result.analysis_id, "video_id": result.video_id, "events": result.timeline}


@router.get("/{analysis_id}/metrics")
async def get_analysis_metrics(analysis_id: str):
    """
    Retrieve live or final performance metrics.
    """
    result = analysis_store.get(analysis_id) or analysis_store.get_by_video_id(analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Analysis ID '{analysis_id}' not found.")
    return {"analysis_id": result.analysis_id, "performance": result.performance}


@router.get("/{analysis_id}/video")
async def get_analysis_video(analysis_id: str):
    """
    Stream or download the processed annotated video for this analysis.
    """
    result = analysis_store.get(analysis_id) or analysis_store.get_by_video_id(analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Analysis ID '{analysis_id}' not found.")
    job = get_job(result.video_id)
    if not job or not job.output_path:
        raise HTTPException(status_code=404, detail="Video file not found.")
    out_file = Path(job.output_path)
    if not out_file.exists():
        raise HTTPException(status_code=404, detail="Annotated video not ready yet.")
    return FileResponse(
        path=str(out_file),
        media_type="video/mp4",
        filename=f"analysis_{result.video_id}.mp4",
    )


@router.get("/history/all", response_model=List[AnalysisResult])
async def list_analysis_history():
    """
    List all analyses performed in this session.
    """
    return analysis_store.list_all()
