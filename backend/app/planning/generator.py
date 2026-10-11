"""
Trajectory Candidate Generator — Phase 7.

Generates smooth, kinematically feasible local candidate trajectories using
parameterized quintic polynomial splines across a multi-lane lateral search space.
"""
from __future__ import annotations

import math
from typing import List, Tuple

from app.planning.schemas import CandidatePath, TrajectoryPoint


class TrajectoryGenerator:
    """
    Generates a bundle of smooth candidate trajectories from vehicle origin
    to the lookahead planning horizon.
    """

    # Candidate lateral offsets at planning horizon (pixels relative to centerline)
    DEFAULT_OFFSETS: List[Tuple[float, str]] = [
        (0.0,    "centerline"),
        (-40.0,  "avoid_left_gentle"),
        (-80.0,  "avoid_left_mild"),
        (-130.0, "avoid_left_moderate"),
        (-180.0, "avoid_left_evasive"),
        (-230.0, "avoid_left_wide"),
        (40.0,   "avoid_right_gentle"),
        (80.0,   "avoid_right_mild"),
        (130.0,  "avoid_right_moderate"),
        (180.0,  "avoid_right_evasive"),
        (230.0,  "avoid_right_wide"),
    ]

    def __init__(
        self,
        num_waypoints: int = 25,
        lookahead_distance: float = 340.0,
        road_margin_px: float = 35.0,
    ) -> None:
        self.num_waypoints = num_waypoints
        self.lookahead_distance = lookahead_distance
        self.road_margin_px = road_margin_px

    def generate_candidate_bundle(
        self,
        origin_x: float,
        origin_y: float,
        frame_width: int,
        frame_height: int,
        custom_offsets: List[Tuple[float, str]] | None = None,
    ) -> List[CandidatePath]:
        """
        Generate candidate trajectories across lateral search space.

        Args:
            origin_x     : Starting x-coordinate (bottom center of ego bumper).
            origin_y     : Starting y-coordinate (bottom of camera frame).
            frame_width  : Frame width for road boundary clamping.
            frame_height : Frame height.
            custom_offsets: Optional custom (offset_px, label) list.

        Returns:
            List of CandidatePath instances.
        """
        offsets = custom_offsets or self.DEFAULT_OFFSETS
        horizon_y = max(20.0, origin_y - self.lookahead_distance)
        delta_y = origin_y - horizon_y

        candidates: List[CandidatePath] = []

        for idx, (dx, label) in enumerate(offsets):
            target_x = origin_x + dx

            # Boundary safety check: do not generate paths off road shoulder
            if target_x < self.road_margin_px or target_x > (frame_width - self.road_margin_px):
                continue

            # Generate quintic polynomial waypoint sequence
            waypoints: List[TrajectoryPoint] = []
            s_acc = 0.0
            prev_x, prev_y = origin_x, origin_y

            for step in range(self.num_waypoints):
                u = step / (self.num_waypoints - 1)  # 0.0 to 1.0

                # Quintic smoothstep S(u) = 10u^3 - 15u^4 + 6u^5
                # Guarantees S'(0)=S''(0)=0 and S'(1)=S''(1)=0 (smooth zero-jerk transition)
                s_u = (10.0 * (u ** 3)) - (15.0 * (u ** 4)) + (6.0 * (u ** 5))

                px = origin_x + (dx * s_u)
                py = origin_y - (delta_y * u)

                # Derivatives for exact curvature calculation
                # S'(u) = 30u^2 - 60u^3 + 30u^4
                # S''(u) = 60u - 180u^2 + 120u^3
                du_dy = -1.0 / max(1.0, delta_y)
                ds_du = (30.0 * (u ** 2)) - (60.0 * (u ** 3)) + (30.0 * (u ** 4))
                d2s_du2 = (60.0 * u) - (180.0 * (u ** 2)) + (120.0 * (u ** 3))

                dx_du = dx * ds_du
                dy_du = -delta_y
                d2x_du2 = dx * d2s_du2
                d2y_du2 = 0.0

                denom = (dx_du ** 2 + dy_du ** 2) ** 1.5
                if denom > 1e-4:
                    curvature = abs(dx_du * d2y_du2 - dy_du * d2x_du2) / denom
                else:
                    curvature = 0.0

                dist = math.hypot(px - prev_x, py - prev_y)
                s_acc += dist
                prev_x, prev_y = px, py

                waypoints.append(TrajectoryPoint(
                    x          = round(px, 1),
                    y          = round(py, 1),
                    curvature  = curvature,
                    s_progress = round(s_acc, 1),
                ))

            candidates.append(CandidatePath(
                path_id        = idx,
                label          = label,
                lateral_offset = dx,
                points         = waypoints,
            ))

        return candidates
