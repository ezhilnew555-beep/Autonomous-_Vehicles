"""
TrackManager — High-level lifecycle coordinator for persistent multi-object tracking.

Wraps ObjectTracker (ByteTrack) and MotionEstimator to provide:
  • Persistent tracking ID assignment.
  • Bounding box center calculation and short position history.
  • Direction estimation (FORWARD, BACKWARD, LEFT, RIGHT, DIAGONAL, STATIONARY, UNKNOWN).
  • Estimated pixel velocity (px/frame and px/second).
  • Temporal class stabilization across frames.
  • Structured tracking data output matching Phase 3 specifications.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Set, Tuple

from app.detection.detector import Detection
from app.tracking.tracker import ObjectTracker, TrackedObject
from app.tracking.motion import MovementDirection, DIRECTION_SYMBOLS
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrackState:
    """Historical state maintained for a single persistent track ID."""
    track_id: int
    first_frame: int
    last_frame: int
    class_votes: dict[str, int] = field(default_factory=dict)
    confidence_history: list[float] = field(default_factory=list)
    speed_history: list[float] = field(default_factory=list)

    @property
    def dominant_class(self) -> str:
        if not self.class_votes:
            return "unknown"
        return max(self.class_votes.items(), key=lambda x: x[1])[0]

    @property
    def avg_confidence(self) -> float:
        return sum(self.confidence_history) / len(self.confidence_history) if self.confidence_history else 0.0

    @property
    def avg_speed(self) -> float:
        return sum(self.speed_history) / len(self.speed_history) if self.speed_history else 0.0

    @property
    def total_frames(self) -> int:
        return len(self.confidence_history)


class TrackManager:
    """
    Orchestrates detection-to-track association, motion estimation, and state tracking.
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.80,
        frame_rate: int = 30,
        history_length: int = 50,
        stationary_threshold_px: float = 0.75,
    ) -> None:
        self.frame_rate = max(1, int(frame_rate))
        self.history_length = history_length
        self._tracker = ObjectTracker(
            track_activation_threshold = track_activation_threshold,
            lost_track_buffer          = lost_track_buffer,
            minimum_matching_threshold = minimum_matching_threshold,
            frame_rate                 = self.frame_rate,
            history_length             = history_length,
        )
        # track_id -> TrackState
        self._track_states: dict[int, TrackState] = {}
        self._all_unique_ids: set[int] = set()
        self._last_tracked_objects: list[TrackedObject] = []
        self._last_structured_tracks: list[dict] = []

    # ─── Public API ───────────────────────────────────────────────────────────

    def update(self, detections: list[Detection], frame_idx: int = 0) -> list[dict]:
        """
        Ingest current frame detections and produce structured tracking records.

        Returns list of structured track records:
        [
            {
                "id": int,
                "class_name": str,
                "confidence": float,
                "bbox": [x1, y1, x2, y2],
                "center": (cx, cy),
                "direction": str,
                "direction_symbol": str,
                "velocity": float,          # px/frame
                "velocity_sec": float,      # px/second
                "history": list[(cx, cy)]
            }, ...
        ]
        """
        tracked = self._tracker.update(detections)
        structured_list: list[dict] = []

        for obj in tracked:
            tid = obj.track_id
            self._all_unique_ids.add(tid)

            # Update track lifetime state & class votes
            if tid not in self._track_states:
                self._track_states[tid] = TrackState(
                    track_id = tid,
                    first_frame = frame_idx,
                    last_frame = frame_idx,
                )
            state = self._track_states[tid]
            state.last_frame = frame_idx
            state.class_votes[obj.class_name] = state.class_votes.get(obj.class_name, 0) + 1
            state.confidence_history.append(obj.confidence)

            # Use stabilized dominant class
            stable_class = state.dominant_class
            obj.class_name = stable_class

            # Motion parameters
            if obj.motion:
                direction = obj.motion.direction
                symbol = obj.motion.direction_symbol
                velocity_px_f = obj.motion.pixel_speed
                velocity_px_s = obj.motion.pixel_speed_sec
                state.speed_history.append(velocity_px_f)
            else:
                direction = MovementDirection.UNKNOWN.value
                symbol = DIRECTION_SYMBOLS.get(direction, "?")
                velocity_px_f = 0.0
                velocity_px_s = 0.0

            record = {
                "id": tid,
                "class_name": stable_class,
                "confidence": round(obj.confidence, 3),
                "bbox": [round(obj.x1, 1), round(obj.y1, 1), round(obj.x2, 1), round(obj.y2, 1)],
                "center": (round(obj.center[0], 1), round(obj.center[1], 1)),
                "direction": direction,
                "direction_symbol": symbol,
                "velocity": round(velocity_px_f, 2),
                "velocity_sec": round(velocity_px_s, 2),
                "history": [(round(p[0], 1), round(p[1], 1)) for p in obj.trajectory],
            }
            structured_list.append(record)

        self._last_tracked_objects = tracked
        self._last_structured_tracks = structured_list
        return structured_list

    @property
    def tracked_objects(self) -> list[TrackedObject]:
        """Return the raw list of TrackedObject instances from the latest update."""
        return self._last_tracked_objects

    @property
    def structured_tracks(self) -> list[dict]:
        """Return structured tracking records from the latest update."""
        return self._last_structured_tracks

    @property
    def active_tracks_count(self) -> int:
        return len(self._last_tracked_objects)

    @property
    def total_unique_tracks(self) -> int:
        return len(self._all_unique_ids)

    def get_active_tracks_table_data(self) -> list[dict]:
        """
        Return tabular data formatted specifically for the ACTIVE TRACKS dashboard table:
        ID | CLASS | CONFIDENCE | DIRECTION | SPEED (px/f)
        """
        table_rows = []
        for r in self._last_structured_tracks:
            table_rows.append({
                "id": r["id"],
                "class_name": r["class_name"].title(),
                "confidence": f"{round(r['confidence'] * 100)}%",
                "direction": r["direction"].title(),
                "direction_symbol": r["direction_symbol"],
                "speed": f"{r['velocity']:.1f} px/f",
                "speed_num": r["velocity"],
            })
        return table_rows

    def get_all_tracks_summary(self) -> list[dict]:
        """Return summary of all tracks observed throughout the video."""
        summaries = []
        for tid, state in self._track_states.items():
            summaries.append({
                "track_id": tid,
                "class_name": state.dominant_class,
                "avg_confidence": round(state.avg_confidence, 3),
                "frame_count": state.total_frames,
                "avg_pixel_speed": round(state.avg_speed, 2),
                "first_frame": state.first_frame,
                "last_frame": state.last_frame,
            })
        return sorted(summaries, key=lambda s: s["track_id"])

    def reset(self) -> None:
        """Reset internal tracker and lifetime track states."""
        self._tracker.reset()
        self._track_states.clear()
        self._all_unique_ids.clear()
        self._last_tracked_objects.clear()
        self._last_structured_tracks.clear()
