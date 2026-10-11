"""
MotionEstimator — tracks per-object position history and computes velocity/direction.

COORDINATE ASSUMPTIONS:
-----------------------
For a forward-facing vehicle-mounted camera (dashcam):
- Origin (0, 0) is at the top-left of the image frame.
- +X points rightward across the image.
- +Y points downward across the image.

Direction Interpretation in 2D Camera Image Space:
- FORWARD: dy < -threshold (object center moves upward towards horizon/vanishing point, i.e. away from camera vehicle).
- BACKWARD: dy > threshold (object center moves downward toward lower frame boundary, i.e. approaching camera vehicle).
- LEFT: dx < -threshold and |dx| >= 1.4 * |dy| (predominantly lateral leftward motion across field of view).
- RIGHT: dx > threshold and |dx| >= 1.4 * |dy| (predominantly lateral rightward motion across field of view).
- DIAGONAL: Significant combined lateral and longitudinal displacement (e.g. crossing diagonally).
- STATIONARY: Total Euclidean displacement < threshold (default 0.75 pixels/frame).
- UNKNOWN: Insufficient history (fewer than 2 frames).

SPEED / VELOCITY NOTE:
----------------------
All speeds are reported as:
  • pixel_speed: pixels / frame (px/f)
  • pixel_speed_sec: pixels / second (px/s)
Real-world speed estimation (km/h) is NOT claimed without camera intrinsic/extrinsic
calibration, road plane homography, or LiDAR ground truth.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)


class MovementDirection(str, Enum):
    FORWARD    = "FORWARD"
    BACKWARD   = "BACKWARD"
    LEFT       = "LEFT"
    RIGHT      = "RIGHT"
    DIAGONAL   = "DIAGONAL"
    STATIONARY = "STATIONARY"
    UNKNOWN    = "UNKNOWN"


DIRECTION_SYMBOLS: dict[str, str] = {
    MovementDirection.FORWARD.value:    "↑",
    MovementDirection.BACKWARD.value:   "↓",
    MovementDirection.LEFT.value:       "←",
    MovementDirection.RIGHT.value:      "→",
    MovementDirection.DIAGONAL.value:   "↗",
    MovementDirection.STATIONARY.value: "■",
    MovementDirection.UNKNOWN.value:    "?",
}

# 8-point compass labels for compatibility: 0° = East (right), 90° = North (up)
_COMPASS: list[str] = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]


def _angle_to_compass(angle_deg: float) -> str:
    """Map a [-180, 180] degree angle to an 8-point compass label."""
    idx = round(angle_deg / 45.0) % 8
    return _COMPASS[idx]


# ─── Data structure ───────────────────────────────────────────────────────────

@dataclass
class MotionData:
    """Motion state derived from tracked position history."""
    dx: float                              # pixel displacement per frame, positive = rightward
    dy: float                              # pixel displacement per frame, positive = downward
    pixel_speed: float                     # Euclidean pixels/frame (NOT km/h)
    pixel_speed_sec: float                 # Estimated pixels/second (px/f * fps)
    direction_angle: float                 # degrees; 0° = right, 90° = up (image-flipped)
    direction: str                         # FORWARD | BACKWARD | LEFT | RIGHT | DIAGONAL | STATIONARY | UNKNOWN
    direction_symbol: str                  # ↑ | ↓ | ← | → | ↗ | ■ | ?
    direction_label: str                   # compass label: "N", "NE", "S", etc. (compatibility)
    trajectory: list[tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dx":                  round(self.dx, 2),
            "dy":                  round(self.dy, 2),
            "pixel_speed":         round(self.pixel_speed, 2),
            "pixel_speed_sec":     round(self.pixel_speed_sec, 2),
            "direction":           self.direction,
            "direction_symbol":    self.direction_symbol,
            "direction_angle":     round(self.direction_angle, 1),
            "direction_label":     self.direction_label,
        }


# ─── Estimator ────────────────────────────────────────────────────────────────

class MotionEstimator:
    """
    Maintains a rolling window of centre-point history per track ID
    and estimates direction, displacement, and pixel velocity.
    """

    def __init__(
        self,
        history_length: int = 50,
        stationary_threshold_px: float = 0.75,
        fps: float = 30.0,
    ) -> None:
        self.history_length = history_length
        self.stationary_threshold_px = stationary_threshold_px
        self.fps = max(1.0, float(fps))
        # track_id → deque of (cx, cy) float tuples
        self._history: dict[int, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=history_length)
        )

    # ─── Public API ───────────────────────────────────────────────────────────

    def update(self, track_id: int, center: tuple[float, float]) -> None:
        """Append the current centre position to this object's history."""
        self._history[track_id].append(center)

    def get_motion(self, track_id: int) -> Optional[MotionData]:
        """
        Return MotionData for the given track.
        Returns None if the object has fewer than 2 recorded positions.
        """
        positions = list(self._history[track_id])
        if len(positions) < 2:
            return None

        # Calculate displacement over last step, smoothed with short moving window
        prev_x, prev_y = positions[-2]
        curr_x, curr_y = positions[-1]

        dx = curr_x - prev_x
        dy = curr_y - prev_y  # image-space: positive = downward

        # If we have 3+ positions, compute slightly smoothed displacement to reduce detection jitter
        if len(positions) >= 4:
            # Average of last 3 deltas
            deltas_x = [positions[i][0] - positions[i - 1][0] for i in range(-3, 0)]
            deltas_y = [positions[i][1] - positions[i - 1][1] for i in range(-3, 0)]
            smooth_dx = sum(deltas_x) / len(deltas_x)
            smooth_dy = sum(deltas_y) / len(deltas_y)
        else:
            smooth_dx = dx
            smooth_dy = dy

        speed = math.hypot(smooth_dx, smooth_dy)
        speed_sec = speed * self.fps

        # Flip dy for angle calculation so that 90° = upward in standard math coordinates
        angle_rad = math.atan2(-smooth_dy, smooth_dx)
        angle_deg = math.degrees(angle_rad)
        compass   = _angle_to_compass(angle_deg)

        # ── Classify Direction according to camera coordinate assumptions ──────
        if speed < self.stationary_threshold_px:
            direction = MovementDirection.STATIONARY.value
        else:
            abs_x = abs(smooth_dx)
            abs_y = abs(smooth_dy)

            if abs_x >= 1.4 * abs_y:
                direction = MovementDirection.RIGHT.value if smooth_dx > 0 else MovementDirection.LEFT.value
            elif abs_y >= 1.4 * abs_x:
                # -dy means moving upward in image -> forward
                direction = MovementDirection.FORWARD.value if smooth_dy < 0 else MovementDirection.BACKWARD.value
            else:
                direction = MovementDirection.DIAGONAL.value

        symbol = DIRECTION_SYMBOLS.get(direction, "?")

        return MotionData(
            dx              = dx,
            dy              = dy,
            pixel_speed     = speed,
            pixel_speed_sec = speed_sec,
            direction_angle = angle_deg,
            direction       = direction,
            direction_symbol= symbol,
            direction_label = compass,
            trajectory      = positions,
        )

    def get_trajectory(self, track_id: int) -> list[tuple[float, float]]:
        """Return a copy of the position history for the given track."""
        return list(self._history.get(track_id, []))

    def cleanup_stale(self, active_ids: set[int]) -> None:
        """Remove history for track IDs that are no longer active."""
        stale = [k for k in list(self._history) if k not in active_ids]
        for k in stale:
            del self._history[k]
        if stale:
            logger.debug(f"Pruned {len(stale)} stale track(s) from motion history.")

    def reset(self) -> None:
        """Clear all position history (call between video jobs)."""
        self._history.clear()
