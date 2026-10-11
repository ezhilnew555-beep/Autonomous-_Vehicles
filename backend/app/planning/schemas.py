"""
Planning schemas — Phase 7: Adaptive Collision-Free Path Planning.

Defines trajectory points, candidate paths, planning results, and replanning events
consumed by the video writer, dashboard, and API endpoints.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class PlanningStatus(str, Enum):
    CONTINUE_ORIGINAL     = "CONTINUE_ORIGINAL"
    REPLANNED_AVOID_LEFT  = "REPLANNED_AVOID_LEFT"
    REPLANNED_AVOID_RIGHT = "REPLANNED_AVOID_RIGHT"
    NO_SAFE_PATH          = "NO_SAFE_PATH"


@dataclass
class TrajectoryPoint:
    """
    A single waypoint along a planned trajectory.

    Attributes:
        x, y       : 2D pixel coordinates on camera road plane.
        curvature  : Path curvature kappa = 1 / R at this waypoint.
        s_progress : Cumulative arc length from start (pixels).
    """
    x: float
    y: float
    curvature: float = 0.0
    s_progress: float = 0.0


@dataclass
class CandidatePath:
    """
    A candidate evasive or cruising trajectory evaluated by the planner.

    Attributes:
        path_id                : Unique candidate index.
        label                  : Human-readable identifier (e.g. "centerline", "avoid_left_moderate").
        lateral_offset         : Lateral displacement at the planning horizon (pixels from center).
        points                 : Discretized sequence of waypoints.
        is_collision_free      : True if no waypoint breaches an inflated obstacle envelope.
        min_obstacle_distance  : Smallest distance to any obstacle along this path (pixels).
        safety_cost            : Cost term penalizing proximity to obstacles.
        smoothness_cost        : Cost term penalizing squared curvature / sharp turns.
        deviation_cost         : Cost term penalizing lateral offset from nominal center.
        total_cost             : Weighted sum of all cost terms.
    """
    path_id:               int
    label:                 str
    lateral_offset:        float
    points:                List[TrajectoryPoint] = field(default_factory=list)
    is_collision_free:     bool  = True
    min_obstacle_distance: float = 9999.0
    safety_cost:           float = 0.0
    smoothness_cost:       float = 0.0
    deviation_cost:        float = 0.0
    total_cost:            float = 0.0

    @property
    def coordinates(self) -> List[Tuple[float, float]]:
        """Return (x, y) tuples for rendering and serialization."""
        return [(pt.x, pt.y) for pt in self.points]


@dataclass
class PlanningResult:
    """
    Complete output of the AdaptivePathPlanner for a single frame.

    Attributes:
        frame_idx          : Video frame index.
        timestamp_sec      : Video timestamp in seconds.
        status             : Planning status enum.
        needs_replanning   : True if nominal path was replaced with an alternative.
        replanning_reason  : Documented rationale for replanning or continuing.
        original_path      : Nominal centerline trajectory waypoints [(x, y), ...].
        safe_path          : Selected optimal collision-free trajectory waypoints.
        all_candidates     : Bundle of all candidate paths evaluated.
        selected_path      : The optimal CandidatePath chosen.
        min_clearance_px   : Minimum obstacle clearance along selected path (pixels).
        planning_time_ms   : Computational latency in milliseconds.
    """
    frame_idx:         int
    timestamp_sec:     float
    status:            PlanningStatus
    needs_replanning:  bool
    replanning_reason: str
    original_path:     List[Tuple[float, float]]
    safe_path:         List[Tuple[float, float]]
    all_candidates:    List[CandidatePath] = field(default_factory=list)
    selected_path:     Optional[CandidatePath] = None
    min_clearance_px:  float = 0.0
    planning_time_ms:  float = 0.0


@dataclass
class ReplanningEvent:
    """
    Persistent log record when a path replanning maneuver is triggered.
    """
    timestamp_sec:    float
    frame_idx:        int
    trigger_reason:   str
    previous_status:  str
    new_status:       str
    planning_time_ms: float
    obstacle_id:      Optional[int] = None
    obstacle_class:   Optional[str] = None
