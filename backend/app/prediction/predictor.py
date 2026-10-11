"""
PredictionEngine — orchestrates short-term trajectory prediction for all tracked objects.

Phase 4 Strategy
────────────────
Two complementary methods are used; the engine selects automatically per-track:

  1. Kalman Filter (primary, "kalman")
     • Constant-velocity model: state = [x, y, vx, vy].
     • One filter per track ID, updated every frame.
     • Used when the track has ≥ MIN_HISTORY_KALMAN positions.
     • Per-class process-noise parameters encode how predictable each class is:
         car/bus/truck  → low noise  → tight, confident trajectory
         person/animal  → high noise → wide, diffuse uncertainty cone

  2. Linear Velocity Extrapolation (fallback, "linear")
     • Simple: future = current + t * avg_velocity.
     • Used for new tracks that don't yet have enough Kalman history.
     • Uncertainty grows linearly at a class-dependent rate.

Both methods produce a list of PredictedPoint objects that downstream modules
(collision analysis Phase 5, path planning Phase 6) can consume directly.

Note on speed
─────────────
Pixel speed (px/frame) is NOT km/h. Real-world calibration is Phase 5+.
"""
from __future__ import annotations

import math
from typing import Optional

from app.prediction.config import PredictionConfig
from app.prediction.kalman import KalmanFilter2D
from app.prediction.schemas import PredictedPoint, PredictionResult
from app.tracking.tracker import TrackedObject
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ─── Per-class Kalman process-noise parameters ────────────────────────────────
# (sigma_process_pos, sigma_process_vel)
# Lower values → smoother, more confident, less flexible prediction.
# Higher values → wider uncertainty cone, more tolerant of sudden turns.

_KALMAN_PARAMS: dict[str, tuple[float, float]] = {
    # Large, heavy, lane-constrained vehicles
    "bus":        (0.8,  1.0),
    "truck":      (0.8,  1.0),
    # Passenger vehicles (fast, lane-following)
    "car":        (1.0,  1.5),
    # Two-wheelers (faster turns, lane-splitting)
    "motorcycle": (2.0,  3.0),
    "bicycle":    (2.5,  3.5),
    # Pedestrians (arbitrary direction changes)
    "person":     (3.5,  5.0),
    # Animals (very unpredictable)
    "cow":        (5.0,  7.0),
    "horse":      (5.0,  7.0),
    "sheep":      (4.5,  6.5),
    "dog":        (5.5,  8.0),
    "cat":        (5.5,  8.0),
    "bird":       (6.0,  9.0),
    "_default":   (3.0,  4.5),
}

# ─── Per-class linear-fallback uncertainty growth ─────────────────────────────
# (base_uncertainty_px, growth_per_frame_px)
# Applied when Kalman history is too short.

_LINEAR_UNCERTAINTY: dict[str, tuple[float, float]] = {
    "bus":        (6.0,  1.5),
    "truck":      (6.0,  1.5),
    "car":        (5.0,  2.0),
    "motorcycle": (8.0,  3.0),
    "bicycle":    (9.0,  3.5),
    "person":     (14.0, 5.0),
    "cow":        (18.0, 6.0),
    "horse":      (18.0, 6.0),
    "sheep":      (16.0, 5.5),
    "dog":        (20.0, 7.0),
    "cat":        (20.0, 7.0),
    "bird":       (22.0, 8.0),
    "_default":   (12.0, 4.5),
}

# Minimum number of tracked positions before we trust the Kalman estimate.
_MIN_HISTORY_KALMAN: int = 6

# Measurement noise for all classes (centroid uncertainty from bounding-box fit).
_SIGMA_MEASURE: float = 5.0


# ─── Prediction engine ────────────────────────────────────────────────────────

class PredictionEngine:
    """
    Maintains one Kalman filter per active track and generates short-term
    predicted trajectories each frame.
    """

    def __init__(
        self,
        config: Optional[PredictionConfig] = None,
        horizon: Optional[int] = None,
        min_history: Optional[int] = None,
        fps: float = 30.0,
    ) -> None:
        self.config = config or PredictionConfig()
        self.fps = max(1.0, float(fps))
        self.horizon_sec = self.config.PREDICTION_HORIZON_SEC
        self.interval_sec = self.config.PREDICTION_INTERVAL_SEC

        # Frame horizon: default to config horizon_sec * fps if horizon argument is not explicitly given
        if horizon is not None:
            self.horizon = horizon
        else:
            self.horizon = int(round(self.horizon_sec * self.fps))

        self.interval_frames = max(1, int(round(self.interval_sec * self.fps)))
        self.min_history = min_history if min_history is not None else self.config.MIN_HISTORY_FRAMES

        # Per-track Kalman filters — created on first sighting, pruned when lost
        self._filters: dict[int, KalmanFilter2D] = {}
        # Cache class name per track for filter re-creation after re-id
        self._class_cache: dict[int, str] = {}

        logger.info(
            f"PredictionEngine ready: horizon={self.horizon_sec}s ({self.horizon}f) "
            f"interval={self.interval_sec}s ({self.interval_frames}f) min_history={self.min_history}f"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def update_and_predict(
        self,
        tracked_objects: list[TrackedObject],
    ) -> dict[int, PredictionResult]:
        """
        Update all Kalman filters with current-frame observations, then
        generate a PredictionResult for every active track.
        """
        results:    dict[int, PredictionResult] = {}
        active_ids: set[int]                    = set()

        for obj in tracked_objects:
            tid = obj.track_id
            active_ids.add(tid)
            self._class_cache[tid] = obj.class_name

            cur_vel = obj.motion.pixel_speed if obj.motion else 0.0
            cur_dir = obj.motion.direction if obj.motion else "UNKNOWN"

            # ── 1. Stability Checks ───────────────────────────────────────────
            is_available = True
            unavail_reason = None

            if len(obj.trajectory) < self.min_history:
                is_available = False
                unavail_reason = f"Insufficient history ({len(obj.trajectory)}/{self.min_history} frames)"
            elif obj.confidence < self.config.MIN_CONFIDENCE_THRESHOLD:
                is_available = False
                unavail_reason = f"Low confidence ({obj.confidence:.2f} < {self.config.MIN_CONFIDENCE_THRESHOLD:.2f})"
            elif obj.motion is None:
                is_available = False
                unavail_reason = "Velocity unavailable"

            # ── 2. Get or create Kalman filter ────────────────────────────────
            if tid not in self._filters:
                self._filters[tid] = self._make_filter(obj.class_name)
                # Warm-start with first velocity estimate if trajectory is long enough
                if len(obj.trajectory) >= 2:
                    t = obj.trajectory
                    vx = t[-1][0] - t[-2][0]
                    vy = t[-1][1] - t[-2][1]
                    self._filters[tid].initialise(obj.center, (vx, vy))

            # ── Update Kalman with current observation ────────────────────────
            kf = self._filters[tid]
            kf.update(obj.center)

            if not is_available:
                # Return structured unavailable prediction
                results[tid] = PredictionResult(
                    track_id               = tid,
                    class_name             = obj.class_name,
                    method                 = "none",
                    current_center         = obj.center,
                    predicted_points       = [],
                    is_reliable            = False,
                    prediction_horizon     = self.horizon,
                    current_velocity       = cur_vel,
                    current_direction      = cur_dir,
                    prediction_horizon_sec = self.horizon_sec,
                    prediction_confidence  = 0.0,
                    is_available           = False,
                    unavailability_reason  = unavail_reason,
                )
                continue

            # ── 3. Choose method and generate predictions ─────────────────────
            enough_history = len(obj.trajectory) >= _MIN_HISTORY_KALMAN

            if enough_history:
                raw_positions, raw_uncertainties = kf.predict_ahead(self.horizon)
                method = "kalman"
            else:
                raw_positions, raw_uncertainties = self._linear_predict(obj)
                method = "linear"

            # ── 4. Sample waypoints at configured intervals ───────────────────
            step = self.interval_frames
            sampled_points: list[PredictedPoint] = []
            for frame_offset in range(step, self.horizon + 1, step):
                idx = frame_offset - 1
                if idx < len(raw_positions):
                    px, py = raw_positions[idx]
                    u = raw_uncertainties[idx] if idx < len(raw_uncertainties) else 5.0
                    bounded_u = max(4.0, min(16.0, 4.0 + float(u) * 0.025))
                    sampled_points.append(PredictedPoint(
                        x                  = px,
                        y                  = py,
                        frame_offset       = frame_offset,
                        uncertainty_radius = bounded_u,
                    ))

            # If horizon not evenly divisible by step, append the final horizon point
            if self.horizon % step != 0 and len(raw_positions) >= self.horizon:
                px, py = raw_positions[self.horizon - 1]
                u = raw_uncertainties[self.horizon - 1] if len(raw_uncertainties) >= self.horizon else 5.0
                bounded_u = max(4.0, min(16.0, 4.0 + float(u) * 0.025))
                sampled_points.append(PredictedPoint(
                    x                  = px,
                    y                  = py,
                    frame_offset       = self.horizon,
                    uncertainty_radius = bounded_u,
                ))

            # ── 5. Compute class-appropriate prediction confidence ────────────
            class_predictability = self.config.CLASS_PREDICTABILITY.get(
                obj.class_name.lower(), self.config.CLASS_PREDICTABILITY["_default"]
            )
            history_factor = min(1.0, len(obj.trajectory) / 15.0)
            confidence_score = round(
                class_predictability * (0.6 + 0.4 * history_factor) * min(1.0, obj.confidence / 0.8),
                2
            )
            confidence_score = min(1.0, max(0.1, confidence_score))

            results[tid] = PredictionResult(
                track_id               = tid,
                class_name             = obj.class_name,
                method                 = method,
                current_center         = obj.center,
                predicted_points       = sampled_points,
                is_reliable            = len(sampled_points) > 0,
                prediction_horizon     = self.horizon,
                current_velocity       = cur_vel,
                current_direction      = cur_dir,
                prediction_horizon_sec = self.horizon_sec,
                prediction_confidence  = confidence_score,
                is_available           = True,
                unavailability_reason  = None,
            )

        # ── Prune stale Kalman filters ────────────────────────────────────────
        stale = [k for k in self._filters if k not in active_ids]
        for k in stale:
            del self._filters[k]
        if stale:
            logger.debug(f"Pruned {len(stale)} stale Kalman filter(s).")

        return results

    def reset(self) -> None:
        """Clear all filters. Call between independent video jobs."""
        self._filters.clear()
        self._class_cache.clear()

    # ── Private helpers ────────────────────────────────────────────────────────

    def _make_filter(self, class_name: str) -> KalmanFilter2D:
        """Construct a Kalman filter with class-appropriate noise parameters."""
        sp, sv = _KALMAN_PARAMS.get(class_name.lower(), _KALMAN_PARAMS["_default"])
        return KalmanFilter2D(
            sigma_process_pos = sp,
            sigma_process_vel = sv,
            sigma_measure     = _SIGMA_MEASURE,
        )

    def _linear_predict(
        self,
        obj: TrackedObject,
    ) -> tuple[list[tuple[float, float]], list[float]]:
        """
        Fallback: constant-velocity linear extrapolation for tracks with
        insufficient Kalman history.

        Velocity is taken from the MotionEstimator (averaged over recent frames).
        Uncertainty grows linearly at a class-dependent rate.
        """
        if obj.motion is None or obj.motion.pixel_speed < 0.1:
            # Object is stationary or no motion data — predict current position
            cx, cy = obj.center
            base_u, growth = _LINEAR_UNCERTAINTY.get(
                obj.class_name.lower(), _LINEAR_UNCERTAINTY["_default"]
            )
            positions     = [(cx, cy)] * self.horizon
            uncertainties = [base_u + growth * (t + 1) for t in range(self.horizon)]
            return positions, uncertainties

        cx, cy = obj.center
        vx, vy = obj.motion.dx, obj.motion.dy
        base_u, growth = _LINEAR_UNCERTAINTY.get(
            obj.class_name.lower(), _LINEAR_UNCERTAINTY["_default"]
        )

        positions:     list[tuple[float, float]] = []
        uncertainties: list[float]               = []
        for t in range(1, self.horizon + 1):
            positions.append((cx + vx * t, cy + vy * t))
            uncertainties.append(base_u + growth * t)

        return positions, uncertainties
