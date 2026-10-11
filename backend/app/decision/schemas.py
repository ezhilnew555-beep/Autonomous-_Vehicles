"""
Decision schemas — Phase 6: Autonomous Decision-Making Logic.

Defines vehicle actions, avoidance directions, road context, and decision records
consumed by the video writer, dashboard, and future adaptive path planner (Phase 7+).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any

from app.collision.schemas import RiskLevel


class VehicleAction(str, Enum):
    CONTINUE       = "CONTINUE"
    SLOW_DOWN      = "SLOW_DOWN"
    STOP           = "STOP"
    AVOID          = "AVOID"
    REPLAN         = "REPLAN"
    EMERGENCY_STOP = "EMERGENCY_STOP"

    @property
    def priority(self) -> int:
        """Numerical urgency rank (higher = more urgent)."""
        mapping = {
            VehicleAction.CONTINUE:       1,
            VehicleAction.SLOW_DOWN:      2,
            VehicleAction.STOP:           3,
            VehicleAction.REPLAN:         4,
            VehicleAction.AVOID:          5,
            VehicleAction.EMERGENCY_STOP: 6,
        }
        return mapping[self]


class AvoidanceDirection(str, Enum):
    LEFT  = "LEFT"
    RIGHT = "RIGHT"
    NONE  = "NONE"


@dataclass
class RoadContext:
    """
    Evaluated spatial clearance and corridor state for decision making.

    Attributes:
        corridor_blocked      : True if a stationary or slow obstacle obstructs the vehicle lane.
        blocking_object_id    : Object ID causing corridor obstruction.
        blocking_object_class : Class of the blocking object.
        left_clearance_px     : Open lateral clearance on the left of ego corridor (pixels).
        right_clearance_px    : Open lateral clearance on the right of ego corridor (pixels).
        can_avoid_left        : True if safe leftward evasive corridor exists without obstacles.
        can_avoid_right       : True if safe rightward evasive corridor exists without obstacles.
    """
    corridor_blocked:      bool
    blocking_object_id:    Optional[int] = None
    blocking_object_class: Optional[str] = None
    left_clearance_px:     float = 0.0
    right_clearance_px:    float = 0.0
    can_avoid_left:        bool  = False
    can_avoid_right:       bool  = False
    environment:           Optional[Any] = None


@dataclass
class DecisionOutput:
    """
    Autonomous driving operational command generated for a single frame.

    Attributes:
        frame_idx           : Current frame index.
        timestamp_sec       : Current video timestamp in seconds.
        action              : VehicleAction command.
        target_object_id    : Primary obstacle ID influencing this decision.
        target_class        : Class of primary obstacle.
        risk_level          : Evaluated risk level of target object.
        ttc_sec             : Time-To-Collision in seconds (if applicable).
        avoidance_direction : Lateral steering directive (LEFT, RIGHT, NONE).
        target_speed_factor : Throttle/brake scaling factor (1.0 = normal, 0.5 = slow, 0.0 = stop).
        reason              : Defensible human-interpretable rationale for the decision.
        road_context        : Spatial clearance context at decision time.
    """
    frame_idx:           int
    timestamp_sec:       float
    action:              VehicleAction
    target_object_id:    Optional[int]
    target_class:        Optional[str]
    risk_level:          Optional[RiskLevel]
    ttc_sec:             Optional[float]
    avoidance_direction: AvoidanceDirection
    target_speed_factor: float
    reason:              str
    road_context:        Optional[RoadContext] = None
