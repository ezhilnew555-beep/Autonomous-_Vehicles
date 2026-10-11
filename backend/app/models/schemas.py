"""
Pydantic schemas for API request/response models.
Internal dataclasses (Detection, TrackedObject) live in their own modules.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel


# ─── Enums ────────────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    DONE       = "done"
    ERROR      = "error"


# ─── Upload ───────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    video_id: str
    filename: str
    size_bytes: int
    message: str


# ─── Process ──────────────────────────────────────────────────────────────────

class ProcessResponse(BaseModel):
    video_id: str
    message: str


# ─── Status ───────────────────────────────────────────────────────────────────

class MotionInfo(BaseModel):
    dx: float
    dy: float
    pixel_speed: float          # pixels/frame — NOT km/h (no calibration)
    direction_angle: float      # degrees, 0° = rightward
    direction_label: str        # compass label: N, NE, E, …


class RiskEventItem(BaseModel):
    timestamp_sec: float
    frame_idx: int
    object_id: int
    object_class: str
    distance_px: float
    ttc_sec: Optional[float] = None
    risk_level: str
    reason: str


class ReplanningEventItem(BaseModel):
    timestamp_sec: float
    frame_idx: int
    trigger_reason: str
    previous_status: str
    new_status: str
    planning_time_ms: float
    obstacle_id: Optional[int] = None
    obstacle_class: Optional[str] = None


class DecisionItem(BaseModel):
    frame_idx: int
    timestamp_sec: float
    action: str                       # CONTINUE | SLOW_DOWN | STOP | AVOID | REPLAN | EMERGENCY_STOP
    target_object_id: Optional[int] = None
    target_class: Optional[str] = None
    risk_level: Optional[str] = None
    ttc_sec: Optional[float] = None
    avoidance_direction: str = "NONE" # LEFT | RIGHT | NONE
    target_speed_factor: float = 1.0  # 0.0 to 1.0
    reason: str


class ActiveTrackItem(BaseModel):
    id: int
    class_name: str
    confidence: float
    direction: str                     # FORWARD | BACKWARD | LEFT | RIGHT | DIAGONAL | STATIONARY | UNKNOWN
    direction_symbol: str = "?"        # ↑ | ↓ | ← | → | ↗ | ■ | ?
    speed_px_per_frame: float = 0.0    # Estimated Pixel Speed (px/f)
    speed_px_per_sec: float = 0.0      # Estimated Pixel Speed (px/s)
    center: list[float] = []           # [cx, cy]
    bbox: list[float] = []             # [x1, y1, x2, y2]
    frames_seen: int = 1


class ActivePredictionItem(BaseModel):
    id: int
    class_name: str
    direction: str                     # FORWARD, BACKWARD, LEFT, etc.
    direction_symbol: str = "?"
    speed_px_per_frame: float = 0.0    # px/frame
    speed_px_per_sec: float = 0.0      # px/sec
    horizon_sec: float = 2.0           # 2.0s
    confidence_pct: float = 0.0        # e.g. 85.0 (%)
    is_available: bool = True
    unavailability_reason: Optional[str] = None
    prediction_method: str = "kalman"  # "kalman" | "linear" | "none"
    waypoints_count: int = 0
    waypoints: list[list[float]] = []  # [[x, y], ...]
    current_center: list[float] = []   # [cx, cy]


class TrackedObjectSummary(BaseModel):
    track_id: int
    class_name: str
    avg_confidence: float
    frame_count: int
    avg_pixel_speed: float
    last_direction: str
    # ── Phase 4: prediction summary ───────────────────────────────────────────
    has_prediction:      bool = False
    prediction_method:   str  = "none"    # "kalman" | "linear" | "none"
    predicted_direction: str  = "—"       # compass label of overall predicted path
    # ── Phase 5: collision risk summary ───────────────────────────────────────
    max_risk_level:      str  = "LOW"     # Highest risk level observed (LOW, MEDIUM, HIGH, CRITICAL)
    min_ttc_sec:         Optional[float] = None # Shortest valid TTC observed
    last_risk_reason:    str  = "—"       # Latest evaluated risk rationale


class DetectionStats(BaseModel):
    total_frames: int
    processed_frames: int
    processing_fps: float       # frames/second of the processing pipeline
    video_fps: float
    duration_seconds: float
    unique_track_ids: int = 0
    objects_by_class: dict[str, int]
    tracked_objects: list[TrackedObjectSummary] = []
    pipeline_mode: str = "tracking" # "detection" (Phase 2) | "tracking" (Phase 3) | "prediction" (Phase 4) | "full" (Phases 3-9)
    total_objects_detected: int = 0
    total_frame_detections: int = 0
    peak_concurrent_objects: int = 0
    frame_detections_by_class: dict[str, int] = {}
    processing_time_sec: float = 0.0
    # ── Phase 3: Tracking Specifics ───────────────────────────────────────────
    active_tracks: list[ActiveTrackItem] = []
    active_tracks_count: int = 0
    total_tracks_count: int = 0
    tracking_duration_sec: float = 0.0
    # ── Phase 4: Motion Prediction Specifics ──────────────────────────────────
    objects_with_predictions: int = 0   # tracks that had reliable predictions
    prediction_horizon_frames: int = 0  # horizon used (for display)
    prediction_horizon_sec: float = 2.0
    prediction_interval_sec: float = 0.2
    active_predictions: list[ActivePredictionItem] = []
    active_predictions_count: int = 0
    avg_prediction_confidence: float = 0.0
    # ── Phase 5: Collision risk summary ───────────────────────────────────────
    risk_counts: dict[str, int] = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    critical_events_count: int = 0
    high_events_count: int = 0
    risk_events: list[RiskEventItem] = []
    # ── Phase 6: Autonomous Decision-Making ───────────────────────────────────
    action_counts: dict[str, int] = {}
    dominant_action: str = "CONTINUE"
    decision_history: list[DecisionItem] = []
    # ── Phase 7: Adaptive Path Planning ───────────────────────────────────────
    replanning_count: int = 0
    avg_planning_time_ms: float = 0.0
    final_path_status: str = "CONTINUE_ORIGINAL"
    replanning_events: list[ReplanningEventItem] = []
    # ── Phase 8: Road-Context Awareness ───────────────────────────────────────
    active_traffic_signal: Optional[dict] = None
    traffic_signals_detected: list[dict] = []
    road_signs_detected: list[dict] = []
    active_map_profile: Optional[dict] = None


class StatusResponse(BaseModel):
    video_id: str
    status: JobStatus
    progress: float             # 0.0 – 100.0
    message: str = ""
    stats: Optional[DetectionStats] = None
