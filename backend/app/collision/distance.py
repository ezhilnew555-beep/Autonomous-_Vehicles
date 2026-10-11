"""
Distance and relative motion geometry for camera-based ego-vehicle perspective.

All distance calculations operate on ground contact points:
- For objects: bottom center of the bounding box (x_center, y2) where tires/feet touch the road.
- For ego vehicle: bottom center of camera frame (width / 2, height).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

from app.tracking.tracker import TrackedObject
from app.collision.schemas import EgoVehicleState


def get_ground_contact_point(obj: TrackedObject) -> Tuple[float, float]:
    """
    Return ground contact point (cx, y2) for the tracked object.
    In camera perspective, the bottom-center of the 2D bounding box represents
    the contact patch on the ground plane.
    """
    cx = (obj.x1 + obj.x2) / 2.0
    return (cx, obj.y2)


def estimate_separation(
    obj: TrackedObject,
    ego_state: EgoVehicleState,
) -> Tuple[float, float, float]:
    """
    Estimate spatial separation between ego vehicle and an obstacle.

    Returns:
        distance_px          : Euclidean 2D distance to object ground contact (pixels).
        lateral_offset_px    : Horizontal offset from ego centerline (positive = right, negative = left).
        longitudinal_dist_px : Forward distance along travel direction (pixels in front).
    """
    gx, gy = get_ground_contact_point(obj)

    # Lateral offset from ego vehicle center line
    lateral_offset_px = gx - ego_state.origin_x

    # Longitudinal distance (in front of ego vehicle: ego_y - gy)
    longitudinal_dist_px = max(0.0, ego_state.origin_y - gy)

    # Euclidean distance
    distance_px = math.hypot(lateral_offset_px, longitudinal_dist_px)

    return distance_px, lateral_offset_px, longitudinal_dist_px


def compute_relative_motion(
    obj: TrackedObject,
    ego_state: EgoVehicleState,
    prev_distance_px: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Compute relative speed and closing velocity between ego vehicle and the object.

    Args:
        obj              : TrackedObject with optional motion estimates.
        ego_state        : State of the ego vehicle.
        prev_distance_px : Distance measured in the previous frame (if available).

    Returns:
        relative_speed_px : Total relative speed magnitude (pixels/frame).
        closing_speed_px  : Rate at which distance is shrinking along line of sight (px/frame).
                            Positive means closing in; negative means moving away.
    """
    if obj.motion is None:
        return 0.0, 0.0

    # Object velocity in image coordinates: (dx, dy)
    obj_vx = obj.motion.dx
    obj_vy = obj.motion.dy

    # Ego vehicle moves forward (upwards in image coordinates: -forward_speed_px)
    ego_vx = 0.0
    ego_vy = -ego_state.forward_speed_px

    # Relative velocity of object relative to ego: V_rel = V_obj - V_ego
    rel_vx = obj_vx - ego_vx
    rel_vy = obj_vy - ego_vy
    relative_speed_px = math.hypot(rel_vx, rel_vy)

    # Calculate closing speed
    gx, gy = get_ground_contact_point(obj)
    rx = gx - ego_state.origin_x
    ry = gy - ego_state.origin_y  # negative when object is in front
    range_dist = math.hypot(rx, ry)

    if range_dist > 1e-3:
        # Unit vector pointing from ego to object: u = R / |R|
        ux = rx / range_dist
        uy = ry / range_dist
        # Range-rate is d|R|/dt = V_rel · u
        # Closing speed is -d|R|/dt = - (V_rel · u)
        dot_product = (rel_vx * ux) + (rel_vy * uy)
        closing_speed_px = -dot_product
    elif prev_distance_px is not None:
        # Direct temporal difference fallback
        closing_speed_px = prev_distance_px - range_dist
    else:
        closing_speed_px = 0.0

    return relative_speed_px, closing_speed_px
