"""
CollisionRiskAnalyzer — Reusable multi-object collision risk evaluation module (Phase 5).

Evaluates conflicts between the ego vehicle's travel corridor and each tracked object's
current position and predicted trajectory (from Phase 4). Computes:
  • Ground contact distance and relative motion
  • Spatio-temporal trajectory intersection with ego envelope
  • Mathematically justified Time-To-Collision (TTC)
  • Four-level risk classification: LOW, MEDIUM, HIGH, CRITICAL
  • Auditable, human-interpretable reasoning strings
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from app.tracking.tracker import TrackedObject
from app.prediction.schemas import PredictionResult, PredictedPoint
from app.collision.schemas import RiskLevel, CollisionRisk, RiskEvent, EgoVehicleState
from app.collision.distance import (
    get_ground_contact_point,
    estimate_separation,
    compute_relative_motion,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Approximate physical half-widths / clearance per class (pixels)
_CLASS_HALF_WIDTHS: Dict[str, float] = {
    "person":     18.0,
    "bicycle":    18.0,
    "motorcycle": 20.0,
    "cow":        35.0,
    "horse":      30.0,
    "dog":        16.0,
    "cat":        12.0,
    "sheep":      20.0,
    "car":        35.0,
    "bus":        45.0,
    "truck":      45.0,
    "_default":   25.0,
}


class CollisionRiskAnalyzer:
    """
    Reusable collision risk analyzer for autonomous driving scenarios.

    Args:
        ttc_critical_thresh : TTC threshold for CRITICAL level (seconds, default 1.0s).
        ttc_high_thresh     : TTC threshold for HIGH level (seconds, default 2.5s).
        ttc_medium_thresh   : TTC threshold for MEDIUM level (seconds, default 4.5s).
        corridor_half_width : Ego lane corridor half-width (pixels, default 80.0).
        critical_distance   : Distance buffer below which collision is immediate (pixels, default 70.0).
    """

    def __init__(
        self,
        ttc_critical_thresh: float = 1.0,
        ttc_high_thresh:     float = 2.5,
        ttc_medium_thresh:   float = 4.5,
        corridor_half_width: float = 80.0,
        critical_distance:   float = 70.0,
    ) -> None:
        self.ttc_critical_thresh = ttc_critical_thresh
        self.ttc_high_thresh     = ttc_high_thresh
        self.ttc_medium_thresh   = ttc_medium_thresh
        self.corridor_half_width = corridor_half_width
        self.critical_distance   = critical_distance

        # Track previous distances for smoothing: track_id -> dist_px
        self._prev_distances: Dict[int, float] = {}

        # Event logging and de-duplication: track_id -> last_logged_frame
        self._last_logged_frame: Dict[int, int] = {}
        self._recorded_events: List[RiskEvent] = []

        logger.info(
            f"CollisionRiskAnalyzer initialised: "
            f"TTC thresholds [crit={ttc_critical_thresh}s, high={ttc_high_thresh}s, med={ttc_medium_thresh}s], "
            f"corridor_half_width={corridor_half_width}px"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze_frame(
        self,
        frame_idx: int,
        fps: float,
        frame_width: int,
        frame_height: int,
        tracked_objects: List[TrackedObject],
        predictions: Optional[Dict[int, PredictionResult]] = None,
        ego_state: Optional[EgoVehicleState] = None,
    ) -> Dict[int, CollisionRisk]:
        """
        Analyze collision risks for all tracked objects in the current frame.
        """
        if ego_state is None:
            ego_state = EgoVehicleState(
                origin_x            = frame_width / 2.0,
                origin_y            = float(frame_height),
                corridor_half_width = self.corridor_half_width,
                forward_speed_px    = 0.0,
            )

        safe_fps = max(1.0, fps)
        current_time_sec = frame_idx / safe_fps
        active_ids = set()
        risks: Dict[int, CollisionRisk] = {}

        for obj in tracked_objects:
            tid = obj.track_id
            active_ids.add(tid)
            cls_name = obj.class_name.lower()

            # 1. Spatial separation
            dist_px, lat_offset_px, long_dist_px = estimate_separation(obj, ego_state)

            # 2. Relative motion
            prev_dist = self._prev_distances.get(tid)
            rel_speed_px, closing_speed_px = compute_relative_motion(
                obj, ego_state, prev_distance_px=prev_dist
            )
            self._prev_distances[tid] = dist_px

            # 3. Path Intersection & TTC calculation
            pred = predictions.get(tid) if predictions else None
            ttc_sec, conflict_pt, conflict_offset, conflict_type = self._compute_ttc_and_intersection(
                obj              = obj,
                ego_state        = ego_state,
                dist_px          = dist_px,
                lat_offset_px    = lat_offset_px,
                long_dist_px     = long_dist_px,
                closing_speed_px = closing_speed_px,
                prediction       = pred,
                fps              = safe_fps,
            )

            # 4. Risk Level determination with documented reasoning
            risk_level, reason = self._evaluate_risk_level(
                obj              = obj,
                cls_name         = cls_name,
                dist_px          = dist_px,
                lat_offset_px    = lat_offset_px,
                closing_speed_px = closing_speed_px,
                ttc_sec          = ttc_sec,
                conflict_offset  = conflict_offset,
                conflict_type    = conflict_type,
                ego_corridor_hw  = ego_state.corridor_half_width,
            )

            risk_eval = CollisionRisk(
                track_id              = tid,
                class_name            = obj.class_name,
                distance_px           = round(dist_px, 1),
                lateral_offset_px     = round(lat_offset_px, 1),
                longitudinal_dist_px  = round(long_dist_px, 1),
                relative_speed_px     = round(rel_speed_px, 2),
                closing_speed_px      = round(closing_speed_px, 2),
                ttc_sec               = round(ttc_sec, 2) if ttc_sec is not None else None,
                risk_level            = risk_level,
                reason                = reason,
                conflict_point        = conflict_pt,
                conflict_frame_offset = conflict_offset,
            )
            risks[tid] = risk_eval

            # 5. Log significant risk events (MEDIUM, HIGH, CRITICAL)
            if risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL):
                last_frame = self._last_logged_frame.get(tid, -999)
                if frame_idx - last_frame >= 15 or risk_level == RiskLevel.CRITICAL:
                    self._recorded_events.append(RiskEvent(
                        timestamp_sec = round(current_time_sec, 2),
                        frame_idx     = frame_idx,
                        object_id     = tid,
                        object_class  = obj.class_name,
                        distance_px   = round(dist_px, 1),
                        ttc_sec       = round(ttc_sec, 2) if ttc_sec is not None else None,
                        risk_level    = risk_level,
                        reason        = reason,
                    ))
                    self._last_logged_frame[tid] = frame_idx

        # Clean up departed tracks
        stale_ids = [k for k in self._prev_distances if k not in active_ids]
        for k in stale_ids:
            del self._prev_distances[k]
            self._last_logged_frame.pop(k, None)

        return risks

    def get_recorded_events(self) -> List[RiskEvent]:
        """Return all logged risk events recorded across processed frames."""
        return list(self._recorded_events)

    def reset(self) -> None:
        """Reset internal state between jobs."""
        self._prev_distances.clear()
        self._last_logged_frame.clear()
        self._recorded_events.clear()

    # ── Private Mathematical Evaluation ───────────────────────────────────────

    def _compute_ttc_and_intersection(
        self,
        obj: TrackedObject,
        ego_state: EgoVehicleState,
        dist_px: float,
        lat_offset_px: float,
        long_dist_px: float,
        closing_speed_px: float,
        prediction: Optional[PredictionResult],
        fps: float,
    ) -> Tuple[Optional[float], Optional[Tuple[float, float]], Optional[int], str]:
        """
        Calculates Time To Collision (TTC) using spatio-temporal trajectory overlap
        and range closing rate.
        """
        cls_hw = _CLASS_HALF_WIDTHS.get(obj.class_name.lower(), _CLASS_HALF_WIDTHS["_default"])
        lat_tolerance = ego_state.corridor_half_width + cls_hw
        long_tolerance = 45.0  # physical vehicle bumper longitudinal buffer

        ttc_candidates: List[Tuple[float, Optional[Tuple[float, float]], Optional[int], str]] = []

        # ── Method A: Spatio-temporal future trajectory intersection ──────────
        if prediction and prediction.predicted_points:
            for pt in prediction.predicted_points:
                k = pt.frame_offset
                # Position of ego vehicle front at frame k
                ego_y_k = ego_state.origin_y - (ego_state.forward_speed_px * k)

                dx_k = abs(pt.x - ego_state.origin_x)
                dy_k = abs(pt.y - ego_y_k)

                # Collision occurs when object reaches the vehicle envelope at future step k:
                # 1. Lateral corridor overlap
                # 2. Longitudinal envelope contact
                is_contact = (dx_k <= lat_tolerance) and (dy_k <= long_tolerance)

                # Also detect crossing path: object moving laterally across the lane in front
                # within close range (<= 160px ahead) where ego headway is breached
                is_crossing_corridor = (dx_k <= lat_tolerance) and (pt.y <= ego_state.origin_y and pt.y >= ego_state.origin_y - 180.0)

                if is_contact or (is_crossing_corridor and abs(lat_offset_px) > lat_tolerance):
                    ttc_path = k / fps
                    conflict_pt = (round(pt.x, 1), round(pt.y, 1))
                    ttc_candidates.append((ttc_path, conflict_pt, k, "path_intersection"))
                    break

        # ── Method B: Range closing rate TTC along line of sight ───────────────
        if closing_speed_px > 0.4:
            closing_speed_sec = closing_speed_px * fps
            if closing_speed_sec > 1.0:
                ttc_range = dist_px / closing_speed_sec
                # Only valid if object is in or moving into travel path
                if abs(lat_offset_px) <= (lat_tolerance * 1.6):
                    ttc_candidates.append((ttc_range, None, None, "closing_speed"))

        if not ttc_candidates:
            return None, None, None, "none"

        best_ttc, best_pt, best_offset, best_type = min(ttc_candidates, key=lambda c: c[0])
        return best_ttc, best_pt, best_offset, best_type

    def _evaluate_risk_level(
        self,
        obj: TrackedObject,
        cls_name: str,
        dist_px: float,
        lat_offset_px: float,
        closing_speed_px: float,
        ttc_sec: Optional[float],
        conflict_offset: Optional[int],
        conflict_type: str,
        ego_corridor_hw: float,
    ) -> Tuple[RiskLevel, str]:
        """
        Classifies risk into LOW, MEDIUM, HIGH, CRITICAL based on deterministic criteria.
        """
        cls_label = obj.class_name.title()
        cls_hw = _CLASS_HALF_WIDTHS.get(cls_name, _CLASS_HALF_WIDTHS["_default"])
        is_in_corridor = abs(lat_offset_px) <= (ego_corridor_hw + cls_hw)

        # ── 1. CRITICAL Risk ──────────────────────────────────────────────────
        # Short TTC (< 1.0s) with active conflict OR immediate proximity inside corridor
        if ttc_sec is not None and ttc_sec < self.ttc_critical_thresh and (is_in_corridor or conflict_type == "path_intersection"):
            return (
                RiskLevel.CRITICAL,
                f"Immediate collision imminent: predicted trajectory conflict in {ttc_sec:.1f}s. Emergency braking required."
            )

        if dist_px <= self.critical_distance and is_in_corridor and closing_speed_px >= -0.2:
            return (
                RiskLevel.CRITICAL,
                f"Immediate obstacle inside vehicle safety envelope (distance {dist_px:.0f}px). Stop immediately."
            )

        # ── 2. HIGH Risk ──────────────────────────────────────────────────────
        # Short TTC (< 2.5s) with path intersection or high closing speed into corridor
        if ttc_sec is not None and ttc_sec < self.ttc_high_thresh:
            if conflict_type == "path_intersection":
                return (
                    RiskLevel.HIGH,
                    f"Predicted {cls_label} trajectory intersects vehicle corridor in {ttc_sec:.1f}s (headway conflict)."
                )
            return (
                RiskLevel.HIGH,
                f"Rapid closing speed towards {cls_label} with estimated TTC of {ttc_sec:.1f}s."
            )

        if dist_px <= (self.critical_distance * 1.7) and is_in_corridor and closing_speed_px > 0.4:
            return (
                RiskLevel.HIGH,
                f"Obstacle in travel corridor with closing velocity (distance {dist_px:.0f}px)."
            )

        # ── 3. MEDIUM Risk ────────────────────────────────────────────────────
        # Moderate TTC (< 4.5s) OR lateral encroachment by vulnerable road users
        if ttc_sec is not None and ttc_sec < self.ttc_medium_thresh:
            return (
                RiskLevel.MEDIUM,
                f"Potential conflict: predicted trajectory approaches vehicle corridor (TTC ~ {ttc_sec:.1f}s)."
            )

        # Vulnerable road users (pedestrians, animals, two-wheelers) near the lane edge
        is_vulnerable = cls_name in {"person", "cow", "dog", "horse", "bicycle", "motorcycle"}
        is_near_corridor = abs(lat_offset_px) <= (ego_corridor_hw + cls_hw * 3.5)

        if is_vulnerable and is_near_corridor and dist_px < 320.0:
            return (
                RiskLevel.MEDIUM,
                f"Lateral caution: {cls_label} adjacent to travel corridor (high unpredictability buffer)."
            )

        # ── 4. LOW Risk ───────────────────────────────────────────────────────
        if closing_speed_px <= 0.0:
            return (
                RiskLevel.LOW,
                f"Safe separation: {cls_label} moving away or stationary outside immediate corridor."
            )

        return (
            RiskLevel.LOW,
            f"Safe separation maintained ({dist_px:.0f}px distance, no imminent path conflict)."
        )
