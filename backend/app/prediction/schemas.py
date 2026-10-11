"""
Prediction data structures — shared between the prediction module,
the annotation writer, and future collision-analysis modules (Phase 5+).

All fields use plain Python types so no serialisation adapter is needed
when these structs are passed between pipeline stages.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PredictedPoint:
    """
    One predicted position in the future trajectory.

    Attributes:
        x, y              : Predicted centre pixel coordinates.
        frame_offset      : How many frames into the future (1 = next frame).
        uncertainty_radius: 1-sigma positional uncertainty in pixels.
                            Grows with frame_offset; class-dependent.
    """
    x: float
    y: float
    frame_offset: int
    uncertainty_radius: float


@dataclass
class PredictionResult:
    """
    Complete short-term prediction for one tracked object.

    Designed to be consumed by:
        • AnnotatedVideoWriter  (Phase 3/4 — visualisation)
        • CollisionAnalyser     (Phase 5 — TTC / risk)
        • PathPlanner           (Phase 6 — ego avoidance)

    Attributes:
        track_id         : Persistent ByteTrack object ID.
        class_name       : COCO / custom class label.
        method           : Algorithm used: "kalman" | "linear" | "none".
        current_center   : Centre pixel (cx, cy) at the current frame.
        predicted_points : Ordered list of future positions (frame 1 → N).
        is_reliable      : True when Kalman has enough history to converge.
        prediction_horizon: Number of frames predicted ahead.
    """
    track_id:               int
    class_name:             str
    method:                 str   # "kalman" | "linear" | "none"
    current_center:         tuple[float, float]
    predicted_points:       list[PredictedPoint]
    is_reliable:            bool
    prediction_horizon:     int   # frame count
    current_velocity:       float = 0.0   # px/frame
    current_direction:      str = "UNKNOWN" # FORWARD, BACKWARD, LEFT, RIGHT, etc.
    prediction_horizon_sec: float = 2.0   # seconds
    prediction_confidence:  float = 0.0   # 0.0 - 1.0 (class & history weighted)
    is_available:           bool = True
    unavailability_reason:  Optional[str] = None

    # ── Convenience properties (consumed by collision module later) ────────────

    @property
    def current_position(self) -> tuple[float, float]:
        return self.current_center

    @property
    def predicted_positions(self) -> list[tuple[float, float]]:
        """List of (x, y) coordinates for all future predicted positions."""
        return [(pt.x, pt.y) for pt in self.predicted_points]

    def to_dict(self) -> dict:
        return {
            "track_id":              self.track_id,
            "class_name":            self.class_name,
            "current_position":      (round(self.current_center[0], 1), round(self.current_center[1], 1)),
            "current_velocity":      round(self.current_velocity, 2),
            "current_direction":     self.current_direction,
            "prediction_horizon":    round(self.prediction_horizon_sec, 2),
            "predicted_positions":   [(round(p[0], 1), round(p[1], 1)) for p in self.predicted_positions],
            "prediction_confidence": round(self.prediction_confidence, 2),
            "is_available":          self.is_available,
            "unavailability_reason": self.unavailability_reason,
            "method":                self.method,
        }

    @property
    def has_prediction(self) -> bool:
        return self.is_available and len(self.predicted_points) > 0

    @property
    def final_position(self) -> Optional[tuple[float, float]]:
        """Last predicted position, or None if no prediction."""
        if self.predicted_points:
            p = self.predicted_points[-1]
            return (p.x, p.y)
        return None

    @property
    def predicted_direction_deg(self) -> Optional[float]:
        """
        Overall predicted direction in degrees (0° = rightward, 90° = upward).
        Computed from current_center → final predicted position.
        """
        if not self.predicted_points:
            return None
        cx, cy = self.current_center
        px, py = self.predicted_points[-1].x, self.predicted_points[-1].y
        return math.degrees(math.atan2(-(py - cy), px - cx))

    @property
    def predicted_displacement(self) -> Optional[tuple[float, float]]:
        """Total (dx, dy) displacement from current to final predicted position."""
        if not self.predicted_points:
            return None
        cx, cy = self.current_center
        return (
            self.predicted_points[-1].x - cx,
            self.predicted_points[-1].y - cy,
        )

    def position_at(self, frame_offset: int) -> Optional[tuple[float, float]]:
        """Return the predicted position at a specific frame offset, or None."""
        for pt in self.predicted_points:
            if pt.frame_offset == frame_offset:
                return (pt.x, pt.y)
        return None

    def uncertainty_at(self, frame_offset: int) -> Optional[float]:
        """Return the uncertainty radius at a specific frame offset."""
        for pt in self.predicted_points:
            if pt.frame_offset == frame_offset:
                return pt.uncertainty_radius
        return None
