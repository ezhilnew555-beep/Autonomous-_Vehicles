"""
Road & Corridor Context Evaluator — Phase 6.

Analyzes the spatial configuration of obstacles around the ego travel corridor
to determine corridor blockage, lateral clearance, and viable avoidance directions.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from app.tracking.tracker import TrackedObject
from app.collision.schemas import CollisionRisk, RiskLevel
from app.decision.schemas import RoadContext, AvoidanceDirection


def evaluate_road_context(
    frame_width: int,
    frame_height: int,
    corridor_half_width: float,
    tracked_objects: List[TrackedObject],
    risks: Dict[int, CollisionRisk],
    environment: Optional[Any] = None,
) -> RoadContext:
    """
    Examines the current road scene to assess corridor obstruction and evasive clearance.

    Args:
        frame_width         : Width of video frame (pixels).
        frame_height        : Height of video frame (pixels).
        corridor_half_width : Ego lane half-width (pixels).
        tracked_objects     : Active tracked obstacles.
        risks               : Risk assessment per object.

    Returns:
        RoadContext dataclass with corridor state and left/right escape viability.
    """
    ego_cx = frame_width / 2.0
    ego_y = float(frame_height)

    # Lateral lane boundaries
    left_corridor_edge = ego_cx - corridor_half_width
    right_corridor_edge = ego_cx + corridor_half_width

    # Minimum clearance required for vehicle evasive steering (pixels)
    min_escape_corridor_width = 85.0

    # Margin from image boundary (road shoulder edge)
    boundary_margin = 35.0

    corridor_blocked = False
    blocking_id = None
    blocking_cls = None

    # Track closest obstacle on left and right within forward range
    closest_left_dist = 9999.0
    closest_right_dist = 9999.0

    for obj in tracked_objects:
        tid = obj.track_id
        cx = (obj.x1 + obj.x2) / 2.0
        y2 = obj.y2
        longitudinal_dist = max(0.0, ego_y - y2)

        # Only consider obstacles within forward reaction zone (< 380px ahead)
        if longitudinal_dist > 380.0:
            continue

        speed = obj.motion.pixel_speed if obj.motion else 0.0

        # Check if directly in travel corridor
        is_in_corridor = (cx >= left_corridor_edge) and (cx <= right_corridor_edge)
        if is_in_corridor and longitudinal_dist < 260.0:
            # If slow-moving or stationary, it constitutes a blockage
            if speed < 3.5 or longitudinal_dist < 140.0:
                corridor_blocked = True
                blocking_id = tid
                blocking_cls = obj.class_name

        # Check left side obstacles
        if cx < left_corridor_edge:
            if longitudinal_dist < closest_left_dist:
                closest_left_dist = longitudinal_dist

        # Check right side obstacles
        if cx > right_corridor_edge:
            if longitudinal_dist < closest_right_dist:
                closest_right_dist = longitudinal_dist

    # Measure available clearance on left and right
    left_space_available = left_corridor_edge - boundary_margin
    right_space_available = (frame_width - boundary_margin) - right_corridor_edge

    # Escape corridor is viable if there is sufficient physical width and no close obstacle
    can_avoid_left = (left_space_available >= min_escape_corridor_width) and (closest_left_dist > 180.0)
    can_avoid_right = (right_space_available >= min_escape_corridor_width) and (closest_right_dist > 180.0)

    # Apply regulatory sign restrictions (Phase 8B)
    if environment:
        for s in getattr(environment, "detected_signs", []):
            if getattr(s, "confidence", 0.0) >= 0.60:
                from app.environment.schemas import RoadSignType
                if s.sign_type == RoadSignType.NO_LEFT_TURN:
                    can_avoid_left = False
                elif s.sign_type == RoadSignType.NO_RIGHT_TURN:
                    can_avoid_right = False

    return RoadContext(
        corridor_blocked      = corridor_blocked,
        blocking_object_id    = blocking_id,
        blocking_object_class = blocking_cls,
        left_clearance_px     = max(0.0, left_space_available),
        right_clearance_px    = max(0.0, right_space_available),
        can_avoid_left        = can_avoid_left,
        can_avoid_right       = can_avoid_right,
        environment           = environment,
    )
