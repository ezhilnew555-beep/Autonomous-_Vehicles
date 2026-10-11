"""
AnalysisResult — standardized data models and in-memory store for the unified road analysis.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import time


class VideoInfoModel(BaseModel):
    filename: str = "road_video.mp4"
    duration_sec: float = 0.0
    video_fps: float = 25.0
    total_frames: int = 0
    status: str = "Ready"


class ObjectCountsModel(BaseModel):
    current_frame: Dict[str, int] = Field(default_factory=dict)
    total_unique: Dict[str, int] = Field(default_factory=dict)
    total_unique_all: int = 0
    active_frame_total: int = 0


class SafetyAnalysisModel(BaseModel):
    risk_level: str = "SAFE"
    threat_object: str = "None"
    distance_m: Optional[float] = None
    ttc_sec: Optional[float] = None
    movement_direction: str = "Stationary"
    prediction_confidence: str = "Not available"


class RoadInformationModel(BaseModel):
    road_type: str = "Unknown"
    traffic_signal: str = "Not detected"
    intersection: str = "Not detected"
    speed_limit: str = "Not detected"
    detected_signs: List[str] = Field(default_factory=lambda: ["Not detected"])


class PathInfoModel(BaseModel):
    status: str = "SAFE"  # SAFE, REPLANNING, BLOCKED, UNKNOWN
    clearance_meters: float = 3.5
    is_replanned: bool = False
    path_points: List[List[float]] = Field(default_factory=list)
    blocked_regions: List[Dict[str, Any]] = Field(default_factory=list)


class PerformanceMetricsModel(BaseModel):
    fps: float = 0.0
    replanning_latency_ms: float = 0.0
    path_smoothness: float = 1.0
    collision_count: int = 0
    total_frames: int = 0
    processing_time_sec: float = 0.0
    total_replans: int = 0
    total_collision_events: int = 0
    scenario_completion_rate: str = "Not measured"
    collision_free_rate: str = "100%"


class SummaryModel(BaseModel):
    executive_summary: str = ""
    traffic_breakdown: str = ""
    hazard_assessment: str = ""
    navigation_assessment: str = ""
    safety_verdict: str = ""


class TechnicalDetailsModel(BaseModel):
    yolo_detections: List[Dict[str, Any]] = Field(default_factory=list)
    tracked_objects: List[Dict[str, Any]] = Field(default_factory=list)
    kalman_predictions: List[Dict[str, Any]] = Field(default_factory=list)
    ttc_records: List[Dict[str, Any]] = Field(default_factory=list)
    occupancy_cost: float = 0.0
    environment_rules: List[str] = Field(default_factory=list)
    active_signals: List[Dict[str, Any]] = Field(default_factory=list)
    active_signs: List[Dict[str, Any]] = Field(default_factory=list)
    # Architecture stack specifications
    object_detection: str = "YOLO"
    tracking: str = "ByteTrack"
    prediction: str = "Kalman Filter"
    risk: str = "Trajectory Conflict + TTC"
    decision: str = "Rule-Based Decision Engine"
    path_planning: str = "Occupancy Grid + Candidate Paths"
    replanning: str = "Dynamic Replanning"
    video: str = "OpenCV"
    backend: str = "FastAPI"
    frontend: str = "React + Vite"


class AnalysisResult(BaseModel):
    analysis_id: str
    video_id: str
    status: str = "PENDING"  # "PENDING", "PROCESSING", "COMPLETED", "FAILED"
    progress: float = 0.0
    current_step: str = "Reading video"
    road_status: str = "SAFE"
    road_status_description: str = ""
    current_danger: str = "No immediate danger"
    vehicle_action: str = "CONTINUE"
    video_info: VideoInfoModel = Field(default_factory=VideoInfoModel)
    safety_analysis: SafetyAnalysisModel = Field(default_factory=SafetyAnalysisModel)
    road_information: RoadInformationModel = Field(default_factory=RoadInformationModel)
    object_counts: ObjectCountsModel = Field(default_factory=ObjectCountsModel)
    path_info: PathInfoModel = Field(default_factory=PathInfoModel)
    timeline: List[Dict[str, Any]] = Field(default_factory=list)
    summary: SummaryModel = Field(default_factory=SummaryModel)
    performance: PerformanceMetricsModel = Field(default_factory=PerformanceMetricsModel)
    technical_details: TechnicalDetailsModel = Field(default_factory=TechnicalDetailsModel)
    annotated_video_url: Optional[str] = None
    error: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    completed_at: Optional[float] = None


class AnalysisResultStore:
    """In-memory and disk-backed store for durable retrieval of analysis results."""

    def __init__(self):
        self._store: Dict[str, AnalysisResult] = {}
        # Map video_id -> analysis_id
        self._video_to_analysis: Dict[str, str] = {}

    def put(self, result: AnalysisResult) -> None:
        self._store[result.analysis_id] = result
        self._video_to_analysis[result.video_id] = result.analysis_id
        # Persist to disk
        try:
            from app.config import get_settings
            from pathlib import Path
            import json
            settings = get_settings()
            out_dir = Path(settings.OUTPUT_DIR)
            out_dir.mkdir(parents=True, exist_ok=True)
            json_file = out_dir / f"{result.video_id}_analysis.json"
            json_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        except Exception:
            pass

    def get(self, analysis_id: str) -> Optional[AnalysisResult]:
        if analysis_id in self._store:
            return self._store[analysis_id]
        # Attempt recovery from disk
        try:
            from app.config import get_settings
            from pathlib import Path
            settings = get_settings()
            out_dir = Path(settings.OUTPUT_DIR)
            for f in out_dir.glob("*_analysis.json"):
                try:
                    res = AnalysisResult.model_validate_json(f.read_text(encoding="utf-8"))
                    if res.analysis_id == analysis_id:
                        self.put(res)
                        return res
                except Exception:
                    continue
        except Exception:
            pass
        return None

    def get_by_video_id(self, video_id: str) -> Optional[AnalysisResult]:
        aid = self._video_to_analysis.get(video_id)
        if aid and aid in self._store:
            return self._store[aid]
        # Attempt recovery from disk
        try:
            from app.config import get_settings
            from pathlib import Path
            settings = get_settings()
            out_dir = Path(settings.OUTPUT_DIR)
            json_file = out_dir / f"{video_id}_analysis.json"
            if json_file.exists():
                res = AnalysisResult.model_validate_json(json_file.read_text(encoding="utf-8"))
                self.put(res)
                return res
        except Exception:
            pass
        return None

    def list_all(self) -> List[AnalysisResult]:
        return list(self._store.values())


# Global singleton store
analysis_store = AnalysisResultStore()
