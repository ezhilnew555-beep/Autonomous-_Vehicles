"""
ObjectTracker — wraps ByteTrackTracker (trackers package) for persistent multi-object tracking.

supervision >= 0.28 deprecated sv.ByteTrack in favour of ByteTrackTracker from the
standalone `trackers` package.  We use ByteTrackTracker with a graceful fallback.

Pipeline position:
    YOLODetector.detect(frame)  →  ObjectTracker.update(detections)  →  list[TrackedObject]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import supervision as sv

# ── Tracker import — prefer new API, fall back to deprecated sv.ByteTrack ─────
try:
    from trackers import ByteTrackTracker as _ByteTracker
    _USE_NEW_API = True
except ImportError:
    try:
        # supervision < 0.28 compatibility shim
        from supervision import ByteTrack as _ByteTracker   # type: ignore
        _USE_NEW_API = False
    except ImportError:
        raise ImportError(
            "No ByteTrack implementation found. "
            "Run:  pip install trackers>=0.4.0"
        )

from app.detection.detector import Detection
from app.tracking.motion import MotionEstimator, MotionData
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ─── Data structure ───────────────────────────────────────────────────────────

@dataclass
class TrackedObject:
    """
    A detection assigned a persistent tracking ID by ByteTrack.

    Contains positional data, motion estimates, and the full trajectory
    (list of historical centre points) for annotation.
    """
    track_id:   int
    class_id:   int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    motion:     Optional[MotionData]             = None
    trajectory: list[tuple[float, float]]        = field(default_factory=list)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def label(self) -> str:
        return f"#{self.track_id} {self.class_name.title()} {self.confidence:.0%}"


# ─── Tracker ──────────────────────────────────────────────────────────────────

class ObjectTracker:
    """
    Combines ByteTrackTracker with MotionEstimator.

    Args:
        track_activation_threshold: Min confidence to start a new track.
        lost_track_buffer: Frames to keep a track alive without a matching detection.
        minimum_matching_threshold: IoU threshold for detection-to-track matching.
        frame_rate: Video FPS for Kalman filter tuning.
        history_length: Centre-point positions to keep per track (trajectory length).
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer:          int   = 30,
        minimum_matching_threshold: float = 0.80,
        frame_rate:                 int   = 30,
        history_length:             int   = 50,
    ) -> None:
        if _USE_NEW_API:
            # trackers package API (supervision >= 0.28)
            self._byte_tracker = _ByteTracker(
                track_activation_threshold = track_activation_threshold,
                lost_track_buffer          = lost_track_buffer,
                minimum_iou_threshold      = minimum_matching_threshold,  # renamed param
                frame_rate                 = frame_rate,
            )
            logger.info(
                f"ObjectTracker: ByteTrackTracker (trackers pkg) "
                f"thresh={track_activation_threshold} buffer={lost_track_buffer} "
                f"min_iou={minimum_matching_threshold} fps={frame_rate}"
            )
        else:
            # sv.ByteTrack legacy (supervision < 0.28)
            self._byte_tracker = _ByteTracker(
                track_activation_threshold = track_activation_threshold,
                lost_track_buffer          = lost_track_buffer,
                minimum_matching_threshold = minimum_matching_threshold,
                frame_rate                 = frame_rate,
            )
            logger.info("ObjectTracker: sv.ByteTrack (legacy, supervision < 0.28)")

        self._motion           = MotionEstimator(history_length=history_length)
        self._class_name_map: dict[int, str] = {}

    # ─── Public API ───────────────────────────────────────────────────────────

    def update(self, detections: list[Detection]) -> list[TrackedObject]:
        """
        Update tracker with detections from the current frame.

        Returns:
            List of TrackedObject with persistent IDs and motion data.
        """
        # Update class name lookup
        for d in detections:
            self._class_name_map[d.class_id] = d.class_name

        # Build supervision Detections object
        if detections:
            xyxy       = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float32)
            confidence = np.array([d.confidence for d in detections], dtype=np.float32)
            class_id   = np.array([d.class_id   for d in detections], dtype=int)
            sv_dets    = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        else:
            sv_dets = sv.Detections(
                xyxy       = np.empty((0, 4), dtype=np.float32),
                confidence = np.array([], dtype=np.float32),
                class_id   = np.array([], dtype=int),
            )

        # Run tracker — new API uses .update(), legacy uses .update_with_detections()
        if _USE_NEW_API:
            tracked_sv = self._byte_tracker.update(sv_dets)
        else:
            tracked_sv = self._byte_tracker.update_with_detections(sv_dets)

        results:    list[TrackedObject] = []
        active_ids: set[int]            = set()

        n = len(tracked_sv)
        for i in range(n):
            x1, y1, x2, y2 = (float(v) for v in tracked_sv.xyxy[i])

            conf     = float(tracked_sv.confidence[i]) if tracked_sv.confidence  is not None else 0.0
            cls_id   = int(tracked_sv.class_id[i])     if tracked_sv.class_id    is not None else 0
            raw_tid  = tracked_sv.tracker_id[i]        if tracked_sv.tracker_id  is not None else None
            if raw_tid is None or int(raw_tid) < 0:
                continue
            track_id = int(raw_tid)

            class_name = self._class_name_map.get(cls_id, f"class_{cls_id}")
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            self._motion.update(track_id, (cx, cy))
            motion_data = self._motion.get_motion(track_id)
            trajectory  = self._motion.get_trajectory(track_id)

            results.append(TrackedObject(
                track_id   = track_id,
                class_id   = cls_id,
                class_name = class_name,
                confidence = conf,
                x1 = x1, y1 = y1, x2 = x2, y2 = y2,
                motion     = motion_data,
                trajectory = trajectory,
            ))
            active_ids.add(track_id)

        self._motion.cleanup_stale(active_ids)
        return results

    def reset(self) -> None:
        """Reset tracker and motion history between independent video jobs."""
        self._motion.reset()
