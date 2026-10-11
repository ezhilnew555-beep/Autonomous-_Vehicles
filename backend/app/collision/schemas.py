"""
Collision risk analysis data structures — Phase 5.

Provides models for ego vehicle state, per-object collision risk evaluation,
and persistent risk events logged during video processing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RiskLevel(str, Enum):
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def severity(self) -> int:
        """Numerical severity rank for sorting and threshold comparison."""
        mapping = {
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4,
        }
        return mapping[self]

    def __ge__(self, other: "RiskLevel") -> bool:
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return self.severity >= other.severity

    def __gt__(self, other: "RiskLevel") -> bool:
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return self.severity > other.severity

    def __le__(self, other: "RiskLevel") -> bool:
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return self.severity <= other.severity

    def __lt__(self, other: "RiskLevel") -> bool:
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return self.severity < other.severity


@dataclass
class EgoVehicleState:
    """
    Representation of the autonomous ego vehicle in camera perspective.

    Anchor:
        origin_x, origin_y: Ground contact point of the ego vehicle front bumper
                            (typically bottom-center of the camera frame: (width / 2, height)).
        corridor_half_width: Half-width of the ego vehicle's travel lane / corridor (pixels).
        forward_speed_px: Forward velocity in pixels/frame (positive upwards in image space).
    """
    origin_x: float
    origin_y: float
    corridor_half_width: float = 80.0
    forward_speed_px: float = 0.0


@dataclass
class CollisionRisk:
    """
    Real-time risk assessment for a single tracked object in a single frame.

    Attributes:
        track_id              : Unique object tracking ID.
        class_name            : Road user class name.
        distance_px           : 2D Euclidean distance from ego anchor to object ground contact.
        lateral_offset_px     : Horizontal distance from ego vehicle centerline (positive = right).
        longitudinal_dist_px  : Vertical distance in front of ego vehicle.
        relative_speed_px     : Magnitude of relative velocity vector.
        closing_speed_px      : Speed at which separation is decreasing (positive = closing in).
        ttc_sec               : Estimated Time To Collision in seconds (None if paths diverge).
        risk_level            : Evaluated risk category (LOW, MEDIUM, HIGH, CRITICAL).
        reason                : Human-interpretable mathematical/behavioral justification.
        conflict_point        : (x, y) coordinates of expected intersection if conflict exists.
        conflict_frame_offset : How many frames in the future conflict is projected.
    """
    track_id:              int
    class_name:            str
    distance_px:           float
    lateral_offset_px:     float
    longitudinal_dist_px:  float
    relative_speed_px:     float
    closing_speed_px:      float
    ttc_sec:               Optional[float]
    risk_level:            RiskLevel
    reason:                str
    conflict_point:        Optional[tuple[float, float]] = None
    conflict_frame_offset: Optional[int]                 = None


@dataclass
class RiskEvent:
    """
    Persistent log entry for significant collision-risk occurrences (MEDIUM, HIGH, CRITICAL).
    Recorded for dashboard reporting and safety auditing.
    """
    timestamp_sec: float
    frame_idx:     int
    object_id:     int
    object_class:  str
    distance_px:   float
    ttc_sec:       Optional[float]
    risk_level:    RiskLevel
    reason:        str
