"""
AnnotatedVideoWriter — writes annotated frames to an output .mp4 video.

Comprehensive visualisations for Phases 3 through 7:
  • Phase 3: Trajectory polyline, bounding box, ID, class, confidence, speed, direction arrow.
  • Phase 4: Predicted trajectory polylines, semi-transparent uncertainty circles, horizon label.
  • Phase 5: Ego vehicle travel corridor lines, risk-level color coding, TTC badges, conflict markers, top warning banner.
  • Phase 6: Autonomous Decision HUD (live action badge, speed factor, avoidance arrow, reasoning).
  • Phase 7: Adaptive Path Planning:
      - Candidate trajectory search bundle (faint guide lines)
      - ORIGINAL PATH (dashed cyan line)
      - NEW SAFE PATH (solid emerald green curve with waypoints)
      - Planning Status & Latency HUD (planning time in ms, clearance distance)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np

from app.tracking.tracker import TrackedObject
from app.detection.road_classes import get_color, get_display_name
from app.collision.schemas import RiskLevel, CollisionRisk
from app.decision.schemas import VehicleAction, AvoidanceDirection, DecisionOutput
from app.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from app.prediction.schemas import PredictionResult
    _PREDICTION_AVAILABLE = True
except ImportError:
    _PREDICTION_AVAILABLE = False
    PredictionResult = None  # type: ignore

try:
    from app.planning.schemas import PlanningResult, PlanningStatus, CandidatePath
    _PLANNING_AVAILABLE = True
except ImportError:
    _PLANNING_AVAILABLE = False
    PlanningResult = None  # type: ignore

try:
    from app.environment.schemas import (
        EnvironmentState,
        TrafficLightColor,
        TrafficSignal,
        RoadSign,
        RoadSignType,
        RoadGeometryType,
        MapRoadConstraint,
    )
    _ENVIRONMENT_AVAILABLE = True
except ImportError:
    _ENVIRONMENT_AVAILABLE = False
    EnvironmentState = None  # type: ignore


# ─── Risk Colors (BGR format for OpenCV) ──────────────────────────────────────
_RISK_COLORS: Dict[RiskLevel, tuple[int, int, int]] = {
    RiskLevel.CRITICAL: (0, 0, 245),    # Bright Red
    RiskLevel.HIGH:     (0, 130, 255),  # Vivid Orange
    RiskLevel.MEDIUM:   (0, 215, 255),  # Amber Yellow
    RiskLevel.LOW:      (70, 200, 70),  # Safe Green
}

# ─── Action Colors (BGR format for OpenCV) ────────────────────────────────────
_ACTION_COLORS: Dict[VehicleAction, tuple[int, int, int]] = {
    VehicleAction.CONTINUE:       (50, 185, 60),   # Green
    VehicleAction.SLOW_DOWN:      (0, 210, 245),   # Amber
    VehicleAction.STOP:           (0, 90, 230),    # Red-Orange
    VehicleAction.AVOID:          (230, 180, 20),  # Cyan / Electric Blue
    VehicleAction.REPLAN:         (200, 60, 200),  # Magenta
    VehicleAction.EMERGENCY_STOP: (0, 0, 255),     # Bright Red
}

# ─── Road Sign Colors (BGR format for OpenCV) ─────────────────────────────────
_SIGN_COLORS = {
    RoadSignType.STOP:                (0, 0, 240),    # Red
    RoadSignType.NO_ENTRY:            (0, 0, 230),    # Red
    RoadSignType.SPEED_LIMIT:         (20, 20, 220),  # Red border
    RoadSignType.NO_LEFT_TURN:        (0, 50, 230),   # Red border
    RoadSignType.NO_RIGHT_TURN:       (0, 50, 230),   # Red border
    RoadSignType.YIELD:               (0, 160, 245),  # Yellow/Orange inverted triangle
    RoadSignType.PEDESTRIAN_CROSSING: (230, 140, 20), # Blue/Cyan
    RoadSignType.WARNING:             (0, 210, 245),  # Yellow
    RoadSignType.UNKNOWN:             (180, 180, 180),# Grey
}


def _lighten(color: tuple[int, int, int], factor: float = 0.55) -> tuple[int, int, int]:
    """Return a lighter version of the colour for prediction lines."""
    return tuple(min(255, int(c + (255 - c) * factor)) for c in color)  # type: ignore


class AnnotatedVideoWriter:
    """
    Wraps cv2.VideoWriter and provides full frame annotation for Phases 3 through 7.
    """

    _FONT       = cv2.FONT_HERSHEY_SIMPLEX
    _FONT_SCALE = 0.52
    _THICKNESS  = 1
    _BOX_THICK  = 2

    def __init__(
        self,
        output_path: str | Path,
        fps: float,
        width: int,
        height: int,
    ) -> None:
        self.path   = Path(output_path)
        self._fps   = max(1.0, float(fps))
        self._width  = int(width)
        self._height = int(height)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self.path), fourcc, self._fps, (self._width, self._height))

        if not self._writer.isOpened():
            raise IOError(f"cv2.VideoWriter could not open output path: {self.path}")

        logger.info(
            f"AnnotatedVideoWriter ready → '{self.path}'  "
            f"({self._width}×{self._height} @ {self._fps:.1f} fps)"
        )

    # ─── Public API ───────────────────────────────────────────────────────────

    def write_detection_frame(
        self,
        frame: np.ndarray,
        detections: list["Detection"],
        frame_idx: int = 0,
        fps: float = 30.0,
    ) -> np.ndarray:
        """
        Phase 2 pure detection annotation:
        Draw bounding box around every detected object with "Class Name + Confidence".
        No tracking IDs, no persistent IDs, no motion prediction, no collision/planning overlays.
        """
        annotated = frame.copy()
        for d in detections:
            color = get_color(d.class_name)
            x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label: "Car 0.94", "Person 0.89", "Bike 0.91", etc.
            display_name = get_display_name(d.class_name)
            label = f"{display_name} {d.confidence:.2f}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.55
            thick = 1
            (w_txt, h_txt), baseline = cv2.getTextSize(label, font, scale, thick)

            # Label background banner
            ty1 = max(0, y1 - h_txt - 8)
            ty2 = y1
            cv2.rectangle(annotated, (x1, ty1), (x1 + w_txt + 8, ty2), color, -1)
            # Text in white
            cv2.putText(annotated, label, (x1 + 4, ty2 - 4), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

        # Top-left info ribbon
        cv2.rectangle(annotated, (12, 12), (320, 42), (20, 20, 20), -1)
        cv2.rectangle(annotated, (12, 12), (320, 42), (70, 70, 70), 1)
        cv2.putText(
            annotated,
            f"Phase 2 Detection | Objects: {len(detections)}",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 200),
            1,
            cv2.LINE_AA,
        )

        self._writer.write(annotated)
        return annotated

    def write_tracking_frame(
        self,
        frame: np.ndarray,
        tracked_objects: list[TrackedObject],
        frame_idx: int = 0,
        fps: float = 30.0,
        total_unique: int = 0,
    ) -> np.ndarray:
        """
        Phase 3 pure object tracking & movement estimation annotation:
        - Current bounding box in class color with corner brackets.
        - Multi-line tracking label:
            Line 1: Class #ID (e.g. Car #7)
            Line 2: Confidence (e.g. 0.94)
            Line 3: Direction indicator arrow + name (e.g. -> RIGHT)
        - Historical trajectory points drawn as dots and fading line behind each object.
        - Direction indicator arrow originating from center.
        - Estimated Pixel Speed label (px/f).
        - Top HUD ribbon for Phase 3.
        - Strictly NO prediction curves, NO TTC/collision warnings, NO decision HUD, NO path planning lines.
        """
        annotated = frame.copy()

        for obj in tracked_objects:
            color = get_color(obj.class_name)
            x1, y1, x2, y2 = int(obj.x1), int(obj.y1), int(obj.x2), int(obj.y2)
            cx, cy = int(obj.center[0]), int(obj.center[1])

            # 1. Historical trajectory trail & dots behind object
            if obj.trajectory and len(obj.trajectory) > 1:
                pts = [(int(p[0]), int(p[1])) for p in obj.trajectory]
                n = len(pts)
                for j in range(1, n):
                    alpha = (j / n) ** 1.5
                    trail_c = tuple(max(0, min(255, int(v * (0.25 + 0.75 * alpha)))) for v in color)
                    # Trajectory line segment
                    cv2.line(annotated, pts[j - 1], pts[j], trail_c, 2, cv2.LINE_AA)
                    # Trajectory history dots (circles)
                    if j % 3 == 0 or j == n - 1:
                        dot_radius = max(2, int(1 + 3 * alpha))
                        cv2.circle(annotated, pts[j], dot_radius, trail_c, -1, cv2.LINE_AA)

            # 2. Bounding box with corner accents
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            corner_len = min(16, max(6, (x2 - x1) // 4), max(6, (y2 - y1) // 4))
            # Top-left corner
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), 3, cv2.LINE_AA)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), 3, cv2.LINE_AA)
            # Bottom-right corner
            cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), 3, cv2.LINE_AA)
            cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), (255, 255, 255), 3, cv2.LINE_AA)

            # 3. Direction indicator arrow
            if obj.motion and obj.motion.pixel_speed >= 0.75:
                rad = math.radians(obj.motion.direction_angle)
                arrow_len = min(40, max(20, int(obj.motion.pixel_speed * 4)))
                ax2 = int(cx + arrow_len * math.cos(rad))
                ay2 = int(cy - arrow_len * math.sin(rad))
                cv2.arrowedLine(annotated, (cx, cy), (ax2, ay2), (0, 255, 255), 2, tipLength=0.35, line_type=cv2.LINE_AA)

            # 4. Multi-line tracking label badge above box:
            #    Car #7
            #    0.94
            #    ->
            display_name = get_display_name(obj.class_name)
            line1 = f"{display_name} #{obj.track_id}"
            line2 = f"{obj.confidence:.2f}"
            dir_str = obj.motion.direction if obj.motion else "UNKNOWN"
            dir_sym = obj.motion.direction_symbol if obj.motion else "?"
            speed_val = obj.motion.pixel_speed if obj.motion else 0.0
            line3 = f"{dir_sym} {dir_str}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            f_scale = 0.45
            thick = 1
            sz1, _ = cv2.getTextSize(line1, font, f_scale, thick)
            sz2, _ = cv2.getTextSize(line2, font, f_scale, thick)
            sz3, _ = cv2.getTextSize(line3, font, f_scale, thick)
            max_w = max(sz1[0], sz2[0], sz3[0]) + 12
            total_h = sz1[1] + sz2[1] + sz3[1] + 16

            badge_y2 = max(total_h, y1)
            badge_y1 = badge_y2 - total_h
            badge_x1 = max(0, x1)
            badge_x2 = min(self._width, badge_x1 + max_w)

            # Semi-transparent dark background badge
            overlay = annotated.copy()
            cv2.rectangle(overlay, (badge_x1, badge_y1), (badge_x2, badge_y2), (18, 22, 28), -1)
            cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)
            cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), color, 1)

            # Draw lines inside badge
            cur_y = badge_y1 + sz1[1] + 4
            cv2.putText(annotated, line1, (badge_x1 + 6, cur_y), font, f_scale, color, thick, cv2.LINE_AA)
            cur_y += sz2[1] + 4
            cv2.putText(annotated, line2, (badge_x1 + 6, cur_y), font, f_scale, (220, 220, 220), thick, cv2.LINE_AA)
            cur_y += sz3[1] + 4
            cv2.putText(annotated, line3, (badge_x1 + 6, cur_y), font, f_scale, (0, 240, 255), thick, cv2.LINE_AA)

            # 5. Bottom label: Estimated Pixel Speed
            speed_text = f"Estimated Pixel Speed: {speed_val:.1f} px/f"
            cv2.putText(
                annotated,
                speed_text,
                (x1 + 2, min(self._height - 6, y2 + 16)),
                font,
                0.38,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        # 6. Top HUD Ribbon
        act_cnt = len(tracked_objects)
        tot_cnt = max(act_cnt, total_unique)
        hud_text = f"Phase 3: Object Tracking | Active Tracks: {act_cnt} | Total Tracks: {tot_cnt} | Frame #{frame_idx}"
        cv2.rectangle(annotated, (12, 12), (480, 42), (18, 22, 30), -1)
        cv2.rectangle(annotated, (12, 12), (480, 42), (70, 80, 100), 1)
        cv2.putText(annotated, hud_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 200), 1, cv2.LINE_AA)

        self._writer.write(annotated)
        return annotated

    def write_prediction_frame(
        self,
        frame: np.ndarray,
        tracked_objects: list[TrackedObject],
        predictions: dict[int, "PredictionResult"],
        frame_idx: int = 0,
        fps: float = 30.0,
        horizon_sec: float = 2.0,
    ) -> np.ndarray:
        """
        Phase 4 Short-Term Motion Prediction annotation:
        - Actual historical trajectory behind object (dots and trail).
        - Current object bounding box and tracking label.
        - Predicted future trajectory ahead of object with uncertainty circles/waypoints.
        - Waypoint indicators (+0.2s ... +2.0s) with class-appropriate uncertainty radii.
        - Top HUD ribbon for Phase 4.
        - Strictly NO collision risk, NO TTC warnings, NO decision HUD, NO path planning lines.
        """
        annotated = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        # ── 1. Draw Predicted Trajectories and Uncertainty Cones first (background layer) ──
        # Draw semi-transparent uncertainty halos on a single overlay layer to prevent washing out the frame
        overlay = annotated.copy()
        drew_any_halo = False

        for obj in tracked_objects:
            tid = obj.track_id
            pred = predictions.get(tid)
            if not pred or not pred.has_prediction:
                continue

            is_unpredictable = obj.class_name.lower() in ("person", "cow", "dog", "horse", "sheep", "animal")
            pred_color = (180, 100, 255) if is_unpredictable else (255, 200, 50)  # Violet for unpredictable, Cyan for vehicle

            for p in pred.predicted_points:
                px, py = int(p.x), int(p.y)
                # Keep visual uncertainty halo bounded (4 to 14 px) local to waypoint
                u_rad = max(4, min(14, int(round(p.uncertainty_radius))))
                cv2.circle(overlay, (px, py), u_rad, pred_color, -1, cv2.LINE_AA)
                drew_any_halo = True

        if drew_any_halo:
            cv2.addWeighted(overlay, 0.22, annotated, 0.78, 0, annotated)

        # Draw trajectory lines, waypoint outlines and dots
        for obj in tracked_objects:
            tid = obj.track_id
            pred = predictions.get(tid)
            if not pred or not pred.has_prediction:
                continue

            is_unpredictable = obj.class_name.lower() in ("person", "cow", "dog", "horse", "sheep", "animal")
            pred_color = (180, 100, 255) if is_unpredictable else (255, 200, 50)
            cx, cy = int(obj.center[0]), int(obj.center[1])

            pts = [(cx, cy)] + [(int(p.x), int(p.y)) for p in pred.predicted_points]

            # Future trajectory dashed/segmented polyline
            for j in range(1, len(pts)):
                cv2.line(annotated, pts[j - 1], pts[j], pred_color, 2, cv2.LINE_AA)

            # Future waypoint dots and outer rings
            for p in pred.predicted_points:
                px, py = int(p.x), int(p.y)
                u_rad = max(4, min(14, int(round(p.uncertainty_radius))))
                cv2.circle(annotated, (px, py), u_rad, pred_color, 1, cv2.LINE_AA)

                # Waypoint center dot (white with colored border)
                cv2.circle(annotated, (px, py), 2, (255, 255, 255), -1, cv2.LINE_AA)

            # End of trajectory label (+2.0s Conf: 88%)
            if pred.predicted_points:
                last_pt = pred.predicted_points[-1]
                end_x, end_y = int(last_pt.x), int(last_pt.y)
                horizon_txt = f"+{pred.prediction_horizon_sec:.1f}s ({int(pred.prediction_confidence * 100)}%)"
                cv2.putText(
                    annotated,
                    horizon_txt,
                    (end_x + 6, end_y + 4),
                    font,
                    0.36,
                    pred_color,
                    1,
                    cv2.LINE_AA,
                )

        # ── 2. Draw Objects, Historical Trajectory & Labels ────────────────────
        for obj in tracked_objects:
            color = get_color(obj.class_name)
            x1, y1, x2, y2 = int(obj.x1), int(obj.y1), int(obj.x2), int(obj.y2)
            tid = obj.track_id
            pred = predictions.get(tid)

            # A. Historical trajectory trail & dots behind object
            if obj.trajectory and len(obj.trajectory) > 1:
                hist_pts = [(int(p[0]), int(p[1])) for p in obj.trajectory]
                n = len(hist_pts)
                for j in range(1, n):
                    alpha = (j / n) ** 1.5
                    hist_c = tuple(max(0, min(255, int(v * (0.2 + 0.6 * alpha)))) for v in color)
                    cv2.line(annotated, hist_pts[j - 1], hist_pts[j], hist_c, 2, cv2.LINE_AA)
                    if j % 3 == 0 or j == n - 1:
                        cv2.circle(annotated, hist_pts[j], 2, hist_c, -1, cv2.LINE_AA)

            # B. Current bounding box with corner accents
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            corner_len = min(14, max(5, (x2 - x1) // 4))
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), 2, cv2.LINE_AA)

            # C. Multi-line badge
            display_name = get_display_name(obj.class_name)
            line1 = f"{display_name} #{obj.track_id}"
            speed_val = obj.motion.pixel_speed if obj.motion else 0.0
            dir_str = obj.motion.direction if obj.motion else "UNKNOWN"
            dir_sym = obj.motion.direction_symbol if obj.motion else "?"
            line2 = f"{dir_sym} {dir_str} | {speed_val:.1f} px/f"

            if pred and pred.is_available:
                method_name = "Kalman" if pred.method == "kalman" else "Linear"
                line3 = f"Pred: {method_name} ({int(pred.prediction_confidence*100)}% conf)"
            else:
                reason = pred.unavailability_reason if pred else "Unavailable"
                line3 = f"Pred: {reason}"

            f_scale = 0.40
            thick = 1
            sz1, _ = cv2.getTextSize(line1, font, f_scale, thick)
            sz2, _ = cv2.getTextSize(line2, font, f_scale, thick)
            sz3, _ = cv2.getTextSize(line3, font, f_scale, thick)
            max_w = max(sz1[0], sz2[0], sz3[0]) + 10
            total_h = sz1[1] + sz2[1] + sz3[1] + 16

            badge_y2 = max(total_h, y1)
            badge_y1 = badge_y2 - total_h
            badge_x1 = max(0, x1)
            badge_x2 = min(self._width, badge_x1 + max_w)

            # Dark translucent badge
            overlay = annotated.copy()
            cv2.rectangle(overlay, (badge_x1, badge_y1), (badge_x2, badge_y2), (18, 22, 28), -1)
            cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)
            cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), color, 1)

            cur_y = badge_y1 + sz1[1] + 4
            cv2.putText(annotated, line1, (badge_x1 + 5, cur_y), font, f_scale, color, thick, cv2.LINE_AA)
            cur_y += sz2[1] + 4
            cv2.putText(annotated, line2, (badge_x1 + 5, cur_y), font, f_scale, (220, 220, 220), thick, cv2.LINE_AA)
            cur_y += sz3[1] + 4
            pred_txt_color = (0, 240, 255) if (pred and pred.is_available) else (150, 150, 150)
            cv2.putText(annotated, line3, (badge_x1 + 5, cur_y), font, f_scale, pred_txt_color, thick, cv2.LINE_AA)

        # ── 3. Top HUD Ribbon ─────────────────────────────────────────────────
        active_preds = sum(1 for p in predictions.values() if p.has_prediction)
        hud_text = (
            f"Phase 4: Motion Prediction | Horizon: {horizon_sec:.1f}s | "
            f"Active Preds: {active_preds}/{len(tracked_objects)} | Frame #{frame_idx}"
        )
        cv2.rectangle(annotated, (12, 12), (520, 42), (18, 22, 30), -1)
        cv2.rectangle(annotated, (12, 12), (520, 42), (70, 80, 100), 1)
        cv2.putText(annotated, hud_text, (20, 32), font, 0.46, (255, 200, 50), 1, cv2.LINE_AA)

        self._writer.write(annotated)
        return annotated

    def write_frame(
        self,
        frame: np.ndarray,
        tracked_objects: list[TrackedObject],
        predictions: Optional[dict[int, "PredictionResult"]] = None,
        risks: Optional[dict[int, CollisionRisk]] = None,
        decision: Optional[DecisionOutput] = None,
        planning: Optional[PlanningResult] = None,
        environment: Optional[EnvironmentState] = None,
    ) -> np.ndarray:
        """
        Annotate frame in-place (on a copy) and write to output file.
        """
        annotated = frame.copy()

        # 1. Ego vehicle travel corridor overlay (Phase 5)
        self._draw_ego_corridor(annotated)

        # 2. Phase 7: Candidate trajectory search bundle (faint background guide lines)
        if planning and planning.all_candidates:
            self._draw_candidate_trajectories(annotated, planning.all_candidates)

        # 3. Phase 7: ORIGINAL PATH vs NEW SAFE PATH
        if planning:
            self._draw_path_layer(annotated, planning)

        # 4. Prediction trajectories & uncertainty cones (Phase 4)
        if predictions:
            self._draw_prediction_layer(annotated, tracked_objects, predictions)

        # 5. Projected collision conflict points (Phase 5)
        if risks:
            self._draw_collision_points(annotated, risks)

        # 6. Tracked objects with risk color coding (Phase 3 + 5)
        self._annotate_tracked(annotated, tracked_objects, risks)

        # 7. Top Collision Warning Banner (Phase 5)
        if risks:
            self._draw_warning_banner(annotated, risks)

        # 8. Autonomous Decision HUD (Phase 6)
        if decision:
            self._draw_decision_hud(annotated, decision)

        # 9. Path Planning HUD (Phase 7)
        if planning:
            self._draw_planning_hud(annotated, planning)

        # 10. Phase 8: Road-Context Awareness (Traffic Signals, Signs, Map HUD)
        if environment:
            if environment.active_signal:
                self._draw_traffic_signal(annotated, environment.active_signal)
            if environment.detected_signs:
                self._draw_road_signs(annotated, environment.detected_signs)
            if environment.map_constraints:
                self._draw_map_hud(annotated, environment.map_constraints)

        self._writer.write(annotated)
        return annotated

    # ─── Phase 7: Adaptive Path Planning Visualizations ───────────────────────

    def _draw_candidate_trajectories(
        self,
        frame: np.ndarray,
        candidates: List[CandidatePath],
    ) -> None:
        """Draw candidate trajectory search curves with subtle translucent styling."""
        for c in candidates:
            if c.lateral_offset == 0.0:
                continue  # Drawn in path layer
            coords = [(int(pt.x), int(pt.y)) for pt in c.points]
            if len(coords) < 2:
                continue

            color = (130, 130, 160) if c.is_collision_free else (80, 80, 140)
            for j in range(1, len(coords)):
                cv2.line(frame, coords[j - 1], coords[j], color, 1, cv2.LINE_AA)

    def _draw_path_layer(
        self,
        frame: np.ndarray,
        planning: PlanningResult,
    ) -> None:
        """
        Draw ORIGINAL PATH (dashed cyan) and NEW SAFE PATH (emerald green).
        """
        orig_coords = [(int(p[0]), int(p[1])) for p in planning.original_path]
        safe_coords = [(int(p[0]), int(p[1])) for p in planning.safe_path]

        # ── Draw ORIGINAL PATH ────────────────────────────────────────────────
        if orig_coords and len(orig_coords) >= 2:
            # If replanned, show original in dashed cyan/amber to indicate it was replaced
            orig_color = (235, 190, 0) if planning.needs_replanning else (50, 205, 50)
            self._draw_dashed_polyline(frame, orig_coords, orig_color, dash=10, gap=7)

            if planning.needs_replanning:
                # Text label on original path
                end_x, end_y = orig_coords[-1]
                cv2.putText(
                    frame, "ORIGINAL PATH",
                    (end_x - 45, end_y - 8),
                    self._FONT, 0.36, orig_color, 1, cv2.LINE_AA
                )

        # ── Draw NEW SAFE PATH ────────────────────────────────────────────────
        if safe_coords and len(safe_coords) >= 2:
            safe_color = (50, 235, 75)  # Bright Emerald Green
            path_thick = 3

            for j in range(1, len(safe_coords)):
                cv2.line(frame, safe_coords[j - 1], safe_coords[j], safe_color, path_thick, cv2.LINE_AA)

            # Draw waypoint nodes along path
            for j in range(0, len(safe_coords), max(1, len(safe_coords) // 6)):
                wx, wy = safe_coords[j]
                cv2.circle(frame, (wx, wy), 4, safe_color, -1, cv2.LINE_AA)
                cv2.circle(frame, (wx, wy), 6, (255, 255, 255), 1, cv2.LINE_AA)

            # Destination horizon arrow & label
            end_x, end_y = safe_coords[-1]
            cv2.circle(frame, (end_x, end_y), 6, safe_color, -1, cv2.LINE_AA)

            path_label = "NEW SAFE PATH" if planning.needs_replanning else "PLANNED PATH"
            cv2.putText(
                frame, path_label,
                (end_x + 8, end_y - 6),
                self._FONT, 0.42, safe_color, 1, cv2.LINE_AA
            )

    def _draw_planning_hud(
        self,
        frame: np.ndarray,
        planning: PlanningResult,
    ) -> None:
        """Draw path planning metrics HUD in lower-right corner."""
        hud_w = 260
        hud_h = 58
        hx = self._width - hud_w - 16
        hy = self._height - hud_h - 14

        # Translucent dark background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (hx, hy), (hx + hud_w, hy + hud_h), (16, 22, 34), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        status_color = (50, 235, 75) if not planning.needs_replanning else (0, 190, 245)
        if planning.status == PlanningStatus.NO_SAFE_PATH:
            status_color = (0, 0, 255)

        cv2.rectangle(frame, (hx + hud_w - 6, hy), (hx + hud_w, hy + hud_h), status_color, -1)

        # Header: Path Planning State
        status_text = "PATH: SAFE (NOMINAL)" if not planning.needs_replanning else f"PATH: {planning.status.value.replace('_', ' ')}"
        cv2.putText(
            frame, status_text,
            (hx + 12, hy + 22),
            self._FONT, 0.42, status_color, 1, cv2.LINE_AA
        )

        # Subtitle: Planning time and minimum clearance
        metrics_text = f"Plan: {planning.planning_time_ms:.1f}ms | Clearance: {planning.min_clearance_px:.0f}px"
        cv2.putText(
            frame, metrics_text,
            (hx + 12, hy + 45),
            self._FONT, 0.38, (180, 195, 210), 1, cv2.LINE_AA
        )

    # ─── Phase 8: Road-Context Awareness Visualizations ───────────────────────

    def _draw_traffic_signal(self, frame: np.ndarray, sig: TrafficSignal) -> None:
        """Draw traffic light bounding box and miniature 3-lamp signal widget."""
        # 1. Bounding box around light in scene
        bx1, by1, bx2, by2 = [int(v) for v in sig.box]
        color_bgr = (0, 0, 245) if sig.color == TrafficLightColor.RED else (0, 220, 255) if sig.color == TrafficLightColor.YELLOW else (50, 210, 50) if sig.color == TrafficLightColor.GREEN else (160, 160, 160)

        if (bx2 - bx1) > 5 and (by2 - by1) > 10:
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color_bgr, 2)
            lbl = f"[{sig.color.value}] {sig.confidence*100:.0f}%"
            cv2.putText(frame, lbl, (bx1, max(12, by1 - 6)), self._FONT, 0.40, color_bgr, 1, cv2.LINE_AA)

        # 2. Mini 3-lamp HUD in upper right corner
        hud_x = self._width - 55
        hud_y = 65
        hud_w = 38
        hud_h = 92

        overlay = frame.copy()
        cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (20, 22, 28), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (70, 75, 85), 1)

        cx = hud_x + hud_w // 2
        r = 10
        # Red lamp (top)
        red_c = (0, 0, 255) if sig.color == TrafficLightColor.RED else (0, 0, 60)
        cv2.circle(frame, (cx, hud_y + 18), r, red_c, -1)
        cv2.circle(frame, (cx, hud_y + 18), r, (100, 100, 110), 1)

        # Yellow lamp (middle)
        yel_c = (0, 225, 255) if sig.color == TrafficLightColor.YELLOW else (0, 70, 80)
        cv2.circle(frame, (cx, hud_y + 44), r, yel_c, -1)
        cv2.circle(frame, (cx, hud_y + 44), r, (100, 100, 110), 1)

        # Green lamp (bottom)
        grn_c = (50, 240, 60) if sig.color == TrafficLightColor.GREEN else (20, 65, 20)
        cv2.circle(frame, (cx, hud_y + 70), r, grn_c, -1)
        cv2.circle(frame, (cx, hud_y + 70), r, (100, 100, 110), 1)

        # Glow halo on active lamp
        if sig.color == TrafficLightColor.RED:
            cv2.circle(frame, (cx, hud_y + 18), r + 3, (0, 0, 255), 1)
        elif sig.color == TrafficLightColor.YELLOW:
            cv2.circle(frame, (cx, hud_y + 44), r + 3, (0, 225, 255), 1)
        elif sig.color == TrafficLightColor.GREEN:
            cv2.circle(frame, (cx, hud_y + 70), r + 3, (50, 240, 60), 1)

    def _draw_road_signs(self, frame: np.ndarray, signs: List[RoadSign]) -> None:
        """Draw detected road signs with bounding boxes and action directives."""
        for s in signs:
            bx1, by1, bx2, by2 = [int(v) for v in s.box]
            color = _SIGN_COLORS.get(s.sign_type, (0, 180, 240))

            # Bounding box
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)

            # Label above box
            tag = f"[{s.sign_type.value}] {s.confidence*100:.0f}%"
            (tw, th), _ = cv2.getTextSize(tag, self._FONT, 0.40, 1)
            cv2.rectangle(frame, (bx1, max(0, by1 - th - 6)), (bx1 + tw + 6, by1), (18, 22, 30), -1)
            cv2.putText(frame, tag, (bx1 + 3, max(12, by1 - 3)), self._FONT, 0.40, color, 1, cv2.LINE_AA)

            # Directive below box
            if s.action_directive:
                dir_tag = s.action_directive.replace("_", " ")
                cv2.putText(frame, dir_tag, (bx1, min(self._height - 4, by2 + 14)), self._FONT, 0.35, (220, 220, 220), 1, cv2.LINE_AA)

    def _draw_map_hud(self, frame: np.ndarray, map_c: MapRoadConstraint) -> None:
        """Draw top road layout and map navigation ribbon."""
        ribbon_w = min(460, self._width - 120)
        ribbon_h = 24
        rx = (self._width - ribbon_w) // 2
        ry = 8

        overlay = frame.copy()
        cv2.rectangle(overlay, (rx, ry), (rx + ribbon_w, ry + ribbon_h), (16, 20, 30), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (rx, ry), (rx + ribbon_w, ry + ribbon_h), (60, 70, 85), 1)

        geom_text = map_c.geometry_type.value.replace("_", " ")
        info_text = f"ROAD: {geom_text} | NAV: {map_c.destination_direction} | LIMIT: {map_c.speed_limit_kmh:.0f} km/h"
        cv2.putText(frame, info_text, (rx + 10, ry + 16), self._FONT, 0.38, (210, 225, 240), 1, cv2.LINE_AA)

    # ─── Phase 6: Autonomous Decision HUD ─────────────────────────────────────

    def _draw_decision_hud(
        self,
        frame: np.ndarray,
        decision: DecisionOutput,
    ) -> None:
        """Draw digital driving instrument cluster HUD in lower-left corner."""
        action = decision.action
        color = _ACTION_COLORS.get(action, (200, 200, 200))

        hud_w = min(380, self._width // 2 - 24)
        hud_h = 58
        hx = 16
        hy = self._height - hud_h - 14

        overlay = frame.copy()
        cv2.rectangle(overlay, (hx, hy), (hx + hud_w, hy + hud_h), (16, 22, 34), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        cv2.rectangle(frame, (hx, hy), (hx + 6, hy + hud_h), color, -1)

        action_text = action.value
        if decision.avoidance_direction == AvoidanceDirection.LEFT:
            action_text += " <- LEFT"
        elif decision.avoidance_direction == AvoidanceDirection.RIGHT:
            action_text += " RIGHT ->"

        cv2.putText(
            frame, f"ACTION: {action_text}",
            (hx + 14, hy + 22),
            self._FONT, 0.48, color, 2, cv2.LINE_AA
        )

        spd_pct = int(decision.target_speed_factor * 100)
        cv2.putText(
            frame, f"THROTTLE: {spd_pct}%",
            (hx + hud_w - 105, hy + 22),
            self._FONT, 0.40, (220, 220, 220), 1, cv2.LINE_AA
        )

        short_reason = decision.reason[:48]
        cv2.putText(
            frame, short_reason,
            (hx + 14, hy + 45),
            self._FONT, 0.36, (170, 185, 200), 1, cv2.LINE_AA
        )

    # ─── Phase 5: Ego Corridor & Conflict Visualizations ──────────────────────

    def _draw_ego_corridor(self, frame: np.ndarray) -> None:
        cx = self._width // 2
        hw = 85
        top_y = max(0, self._height - 380)

        left_pts = [(cx - hw, self._height), (int(cx - hw * 0.65), top_y)]
        right_pts = [(cx + hw, self._height), (int(cx + hw * 0.65), top_y)]

        lane_color = (60, 180, 75)
        self._draw_dashed_polyline(frame, left_pts, lane_color, dash=12, gap=8)
        self._draw_dashed_polyline(frame, right_pts, lane_color, dash=12, gap=8)

    def _draw_collision_points(
        self,
        frame: np.ndarray,
        risks: dict[int, CollisionRisk],
    ) -> None:
        for tid, risk in risks.items():
            if risk.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL) and risk.conflict_point:
                cpx, cpy = int(risk.conflict_point[0]), int(risk.conflict_point[1])
                color = _RISK_COLORS[risk.risk_level]

                cv2.circle(frame, (cpx, cpy), 8, color, 2, cv2.LINE_AA)
                cv2.line(frame, (cpx - 10, cpy), (cpx + 10, cpy), color, 2, cv2.LINE_AA)
                cv2.line(frame, (cpx, cpy - 10), (cpx, cpy + 10), color, 2, cv2.LINE_AA)

                ttc_str = f"TTC {risk.ttc_sec:.1f}s" if risk.ttc_sec is not None else "CONFLICT"
                cv2.putText(
                    frame, ttc_str,
                    (cpx + 12, cpy + 4),
                    self._FONT, 0.44, color, 1, cv2.LINE_AA
                )

    def _draw_warning_banner(
        self,
        frame: np.ndarray,
        risks: dict[int, CollisionRisk],
    ) -> bool:
        critical_items = [r for r in risks.values() if r.risk_level == RiskLevel.CRITICAL]
        high_items = [r for r in risks.values() if r.risk_level == RiskLevel.HIGH]

        if not critical_items and not high_items:
            return False

        is_critical = len(critical_items) > 0
        top_risk = critical_items[0] if is_critical else high_items[0]
        banner_color = (0, 0, 190) if is_critical else (0, 110, 220)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self._width, 34), banner_color, -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        prefix = "CRITICAL COLLISION ALERT" if is_critical else "COLLISION WARNING"
        ttc_info = f" | TTC: {top_risk.ttc_sec:.1f}s" if top_risk.ttc_sec is not None else ""
        msg = f"! {prefix} ! Object #{top_risk.track_id} ({top_risk.class_name.title()}){ttc_info} - {top_risk.reason[:60]}"

        cv2.putText(
            frame, msg,
            (16, 23),
            self._FONT, 0.54, (255, 255, 255), 2, cv2.LINE_AA
        )
        return True

    # ─── Phase 4: Prediction Drawing ──────────────────────────────────────────

    def _draw_prediction_layer(
        self,
        frame: np.ndarray,
        objects: list[TrackedObject],
        predictions: dict[int, "PredictionResult"],
    ) -> None:
        fps = self._fps
        circle_offsets = sorted(set([
            max(1, round(fps * 0.25)),
            max(1, round(fps * 0.50)),
            max(1, round(fps * 1.00)),
        ]))

        overlay = frame.copy()
        drew_any = False

        for obj in objects:
            pred = predictions.get(obj.track_id)
            if pred is None or not pred.predicted_points:
                continue
            light = _lighten(get_color(obj.class_name), 0.4)

            for offset in circle_offsets:
                if offset > len(pred.predicted_points):
                    continue
                pt = pred.predicted_points[offset - 1]
                cx, cy = int(pt.x), int(pt.y)
                r      = max(4, min(14, int(pt.uncertainty_radius)))
                cv2.circle(overlay, (cx, cy), r, light, 1, cv2.LINE_AA)
                cv2.circle(overlay, (cx, cy), max(2, r - 3), light, -1)
                drew_any = True

        if drew_any:
            cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

        for obj in objects:
            pred = predictions.get(obj.track_id)
            if pred is None or not pred.predicted_points:
                continue
            color = get_color(obj.class_name)
            light = _lighten(color, 0.5)

            cx0, cy0 = int(pred.current_center[0]), int(pred.current_center[1])
            pts = [(cx0, cy0)] + [(int(p.x), int(p.y)) for p in pred.predicted_points]

            self._draw_dashed_polyline(frame, pts, light, dash=9, gap=5)

            if len(pts) >= 2:
                end_x, end_y = pts[-1]
                cv2.circle(frame, (end_x, end_y), 4, light, -1, cv2.LINE_AA)
                horizon_secs = len(pred.predicted_points) / fps
                cv2.putText(
                    frame, f"~{horizon_secs:.1f}s",
                    (end_x + 5, end_y - 5),
                    self._FONT, 0.38, light, 1, cv2.LINE_AA,
                )

    # ─── Phase 3 + 5: Object Annotations ──────────────────────────────────────

    def _annotate_tracked(
        self,
        frame: np.ndarray,
        objects: list[TrackedObject],
        risks: Optional[dict[int, CollisionRisk]] = None,
    ) -> None:
        for obj in objects:
            tid = obj.track_id
            risk = risks.get(tid) if risks else None

            if risk and risk.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL):
                box_color = _RISK_COLORS[risk.risk_level]
                box_thick = 3 if risk.risk_level == RiskLevel.CRITICAL else 2
            else:
                box_color = get_color(obj.class_name)
                box_thick = self._BOX_THICK

            x1, y1, x2, y2 = int(obj.x1), int(obj.y1), int(obj.x2), int(obj.y2)

            self._draw_trajectory(frame, obj.trajectory, box_color)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thick)

            display_name = get_display_name(obj.class_name)
            risk_tag = ""
            if risk and risk.risk_level != RiskLevel.LOW:
                ttc_tag = f" {risk.ttc_sec:.1f}s" if risk.ttc_sec is not None else ""
                risk_tag = f" | {risk.risk_level.value}{ttc_tag}"

            main_label = f"#{obj.track_id} {display_name} {obj.confidence:.0%}{risk_tag}"
            self._draw_label(frame, main_label, x1, y1, box_color)

            motion_parts = []
            if obj.motion:
                motion_parts.append(f"{obj.motion.direction_label} {obj.motion.pixel_speed:.1f}px/f")
            if risk:
                motion_parts.append(f"d:{risk.distance_px:.0f}px")

            if motion_parts:
                info_text = " | ".join(motion_parts)
                cv2.putText(
                    frame, info_text,
                    (x1 + 2, y2 + 16),
                    self._FONT, 0.42, box_color, 1, cv2.LINE_AA,
                )

            if obj.motion and obj.motion.pixel_speed > 0.8:
                self._draw_arrow(frame, obj.center, obj.motion.direction_angle, box_color)

    # ─── Geometric & Text Drawing Primitives ─────────────────────────────────

    @staticmethod
    def _draw_trajectory(
        frame: np.ndarray,
        trajectory: list[tuple[float, float]],
        color: tuple[int, int, int],
    ) -> None:
        if len(trajectory) < 2:
            return
        pts = [(int(p[0]), int(p[1])) for p in trajectory]
        n   = len(pts)
        for j in range(1, n):
            alpha = j / n
            c = tuple(max(0, min(255, int(v * (0.3 + 0.7 * alpha)))) for v in color)
            cv2.line(frame, pts[j - 1], pts[j], c, thickness=2, lineType=cv2.LINE_AA)

    def _draw_label(
        self,
        frame: np.ndarray,
        text: str,
        x1: int,
        y1: int,
        color: tuple[int, int, int],
    ) -> None:
        (tw, th), baseline = cv2.getTextSize(
            text, self._FONT, self._FONT_SCALE, self._THICKNESS
        )
        label_y = max(y1 - 5, th + baseline + 2)
        cv2.rectangle(
            frame,
            (x1, label_y - th - baseline - 2),
            (x1 + tw + 6, label_y + 2),
            color, -1,
        )
        cv2.putText(
            frame, text,
            (x1 + 3, label_y - 1),
            self._FONT, self._FONT_SCALE,
            (0, 0, 0), self._THICKNESS, cv2.LINE_AA,
        )

    @staticmethod
    def _draw_arrow(
        frame: np.ndarray,
        center: tuple[float, float],
        direction_angle_deg: float,
        color: tuple[int, int, int],
        length: int = 32,
    ) -> None:
        cx, cy    = int(center[0]), int(center[1])
        angle_rad = math.radians(direction_angle_deg)
        ex = int(cx + length * math.cos(angle_rad))
        ey = int(cy - length * math.sin(angle_rad))
        cv2.arrowedLine(
            frame, (cx, cy), (ex, ey),
            color, thickness=2, tipLength=0.35, line_type=cv2.LINE_AA,
        )

    @staticmethod
    def _draw_dashed_polyline(
        frame: np.ndarray,
        pts: list[tuple[int, int]],
        color: tuple[int, int, int],
        dash: int = 8,
        gap: int  = 5,
    ) -> None:
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            dx, dy  = x2 - x1, y2 - y1
            dist    = math.hypot(dx, dy)
            if dist < 1:
                continue
            ux, uy  = dx / dist, dy / dist
            pos     = 0.0
            drawing = True
            while pos < dist:
                seg_len = dash if drawing else gap
                end_pos = min(pos + seg_len, dist)
                if drawing:
                    sx = int(x1 + ux * pos)
                    sy = int(y1 + uy * pos)
                    ex = int(x1 + ux * end_pos)
                    ey = int(y1 + uy * end_pos)
                    cv2.line(frame, (sx, sy), (ex, ey), color, 1, cv2.LINE_AA)
                pos     += seg_len
                drawing  = not drawing

    def _convert_to_h264(self) -> None:
        """
        Transcode output video from OpenCV mp4v (MPEG-4 Part 2) to H.264 (AVC1)
        with yuv420p pixel format and faststart moov metadata so HTML5 video tags
        can play it natively in all web browsers (Chrome, Edge, Firefox, Safari).
        """
        if not self.path.exists() or self.path.stat().st_size == 0:
            return

        temp_path = self.path.with_name(self.path.stem + "_h264_tmp.mp4")
        try:
            import av
            from fractions import Fraction

            container = av.open(str(self.path))
            if not container.streams.video:
                container.close()
                return

            video_stream = container.streams.video[0]
            fps = video_stream.average_rate or video_stream.real_rate or Fraction(int(self._fps), 1)

            out_container = av.open(str(temp_path), mode="w", format="mp4", options={"movflags": "faststart"})
            out_stream = out_container.add_stream("h264", rate=fps)
            out_stream.width = video_stream.width
            out_stream.height = video_stream.height
            out_stream.pix_fmt = "yuv420p"
            out_stream.options = {"preset": "ultrafast", "crf": "23"}

            for frame in container.decode(video=0):
                out_frame = frame.reformat(format="yuv420p")
                for packet in out_stream.encode(out_frame):
                    out_container.mux(packet)

            for packet in out_stream.encode():
                out_container.mux(packet)

            out_container.close()
            container.close()

            temp_path.replace(self.path)
            logger.info(f"Successfully converted '{self.path.name}' to web-playable H.264 MP4.")
        except Exception as exc:
            logger.warning(f"H.264 web transcode for '{self.path.name}' encountered an error: {exc}")
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass

    # ─── Context manager ──────────────────────────────────────────────────────

    def close(self) -> None:
        self._writer.release()
        self._convert_to_h264()

    def __enter__(self) -> "AnnotatedVideoWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()

