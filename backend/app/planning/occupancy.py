"""
Local Obstacle Map & Occupancy Evaluation — Phase 7.

Constructs an inflated 2D safety field from tracked objects and short-term
motion predictions to enable fast collision-checking and clearance queries.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from app.tracking.tracker import TrackedObject
from app.prediction.schemas import PredictionResult


_CLASS_INFLATION_MARGINS: Dict[str, float] = {
    "person":     35.0,  # Vulnerable, sudden lateral steps
    "cow":        45.0,  # Large, unpredictable road crosser
    "horse":      45.0,
    "dog":        30.0,
    "cat":        20.0,
    "sheep":      30.0,
    "bicycle":    30.0,  # Two-wheelers with wobble
    "motorcycle": 30.0,
    "car":        25.0,
    "bus":        35.0,
    "truck":      35.0,
    "_default":   25.0,
}


@dataclass
class ObstacleZone:
    """An inflated safety zone for collision checking."""
    track_id:     int
    class_name:   str
    center_x:     float
    center_y:     float
    radius:       float
    is_predicted: bool = False


class LocalObstacleMap:
    """
    Builds a local obstacle field representing all active obstacles
    and their predicted future positions.
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        tracked_objects: List[TrackedObject],
        predictions: Optional[Dict[int, PredictionResult]] = None,
        lookahead_frames: int = 20,
    ) -> None:
        self.width = frame_width
        self.height = frame_height
        self.zones: List[ObstacleZone] = []

        safe_preds = predictions or {}

        for obj in tracked_objects:
            cls_name = obj.class_name.lower()
            margin = _CLASS_INFLATION_MARGINS.get(cls_name, _CLASS_INFLATION_MARGINS["_default"])

            cx = (obj.x1 + obj.x2) / 2.0
            cy = (obj.y1 + obj.y2) / 2.0  # object center
            half_w = (obj.x2 - obj.x1) / 2.0
            base_r = max(18.0, half_w) + margin

            # 1. Current position zone
            self.zones.append(ObstacleZone(
                track_id     = obj.track_id,
                class_name   = obj.class_name,
                center_x     = cx,
                center_y     = cy,
                radius       = base_r,
                is_predicted = False,
            ))

            # 2. Predicted future positions (short-term envelope)
            pred = safe_preds.get(obj.track_id)
            if pred and pred.predicted_points:
                # Sample 2-3 key future points within lookahead horizon
                step = max(1, len(pred.predicted_points) // 3)
                for pt in pred.predicted_points[:lookahead_frames:step]:
                    future_r = base_r + (pt.uncertainty_radius * 0.35)
                    self.zones.append(ObstacleZone(
                        track_id     = obj.track_id,
                        class_name   = obj.class_name,
                        center_x     = pt.x,
                        center_y     = pt.y,
                        radius       = future_r,
                        is_predicted = True,
                    ))

    # ── Clearance & Collision Queries ─────────────────────────────────────────

    def get_clearance(self, x: float, y: float) -> float:
        """
        Return the Euclidean distance from (x, y) to the nearest obstacle zone boundary.
        Positive = outside obstacle safety zone; Negative = inside safety zone.
        """
        if not self.zones:
            return 9999.0

        min_clearance = 9999.0
        for z in self.zones:
            dist = math.hypot(x - z.center_x, y - z.center_y) - z.radius
            if dist < min_clearance:
                min_clearance = dist

        return min_clearance

    def is_point_in_collision(self, x: float, y: float) -> bool:
        """Check if point penetrates any obstacle safety boundary."""
        return self.get_clearance(x, y) <= 0.0

    def evaluate_path(
        self,
        points: List[Tuple[float, float]],
    ) -> Tuple[bool, float]:
        """
        Evaluate a sequence of path points against all obstacle zones.

        Returns:
            is_collision_free : True if no point breaches any obstacle safety boundary.
            min_clearance     : Smallest distance to any obstacle boundary along the path.
        """
        if not self.zones:
            return True, 9999.0

        min_clearance = 9999.0
        for px, py in points:
            clearance = self.get_clearance(px, py)
            if clearance < min_clearance:
                min_clearance = clearance
            if clearance <= 0.0:
                return False, min_clearance

        return True, min_clearance
