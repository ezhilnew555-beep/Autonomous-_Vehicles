"""
Traffic Signal Detection & State Recognition — Phase 8.

Identifies traffic light boxes and determines active color state:
  • RED     — Demands stopping before stop line
  • YELLOW  — Caution / decelerate to prepare for stop
  • GREEN   — May proceed if path is collision-free
  • UNKNOWN — Low confidence / unverified; monitored without panicking

Features:
- Multi-band HSV color segmentation.
- Spatial lamp division (Red top, Yellow middle, Green bottom).
- Confidence estimation & temporal smoothing over a rolling frame window.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.detection.detector import Detection
from app.environment.schemas import TrafficLightColor, TrafficSignal
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── HSV Threshold Ranges for Traffic Lights ──────────────────────────────────
# Red wraps around 0/180 in HSV
_HSV_RED_LOW1   = np.array([0,   90, 100], dtype=np.uint8)
_HSV_RED_HIGH1  = np.array([10, 255, 255], dtype=np.uint8)
_HSV_RED_LOW2   = np.array([165, 90, 100], dtype=np.uint8)
_HSV_RED_HIGH2  = np.array([180, 255, 255], dtype=np.uint8)

_HSV_YELLOW_LOW  = np.array([15, 100, 120], dtype=np.uint8)
_HSV_YELLOW_HIGH = np.array([35, 255, 255], dtype=np.uint8)

_HSV_GREEN_LOW   = np.array([40,  80, 100], dtype=np.uint8)
_HSV_GREEN_HIGH  = np.array([90, 255, 255], dtype=np.uint8)

# Minimum reliability confidence threshold
_MIN_RELIABILITY_CONF = 0.55


class TrafficSignalDetector:
    """
    Recognizes traffic light signal colors and maintains temporal confidence.
    """

    def __init__(self, history_len: int = 5) -> None:
        self.history_len = history_len
        self._color_history: deque[TrafficLightColor] = deque(maxlen=history_len)
        self._conf_history: deque[float] = deque(maxlen=history_len)

    def detect_signal(
        self,
        frame: np.ndarray,
        yolo_detections: Optional[List[Detection]] = None,
    ) -> Optional[TrafficSignal]:
        """
        Detect traffic signal in the frame and recognize active light color.

        Args:
            frame           : BGR numpy image frame.
            yolo_detections : Optional detections from YOLO (e.g. class 'traffic light').

        Returns:
            TrafficSignal or None if no traffic light detected.
        """
        h_frame, w_frame = frame.shape[:2]

        # 1. Search for candidate traffic light bounding boxes
        candidate_boxes: List[Tuple[float, float, float, float, float]] = []

        if yolo_detections:
            for d in yolo_detections:
                cls_lower = d.class_name.lower()
                if "traffic light" in cls_lower or "traffic_light" in cls_lower or d.class_id == 9:
                    candidate_boxes.append((d.x1, d.y1, d.x2, d.y2, d.confidence))

        # Fallback heuristic search in upper half of frame if no YOLO box found
        if not candidate_boxes:
            heuristic_box = self._heuristic_signal_search(frame)
            if heuristic_box:
                candidate_boxes.append(heuristic_box)

        if not candidate_boxes:
            # Decay history
            if self._color_history:
                self._color_history.popleft()
            return None

        # Pick largest/highest confidence box
        best_box = max(candidate_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]) * b[4])
        x1, y1, x2, y2, det_conf = best_box

        # Clamp coordinates to frame
        ix1 = max(0, int(round(x1)))
        iy1 = max(0, int(round(y1)))
        ix2 = min(w_frame, int(round(x2)))
        iy2 = min(h_frame, int(round(y2)))

        if (ix2 - ix1) < 6 or (iy2 - iy1) < 12:
            return None

        roi = frame[iy1:iy2, ix1:ix2]
        raw_color, color_conf = self._classify_roi_color(roi)

        # Combine detection and color classification confidence
        overall_conf = round(float(det_conf * color_conf), 2)

        # 2. Temporal Smoothing / Confirmation
        self._color_history.append(raw_color)
        self._conf_history.append(overall_conf)

        # Determine majority color over recent frames
        counts: Dict[TrafficLightColor, int] = {}
        for c in self._color_history:
            counts[c] = counts.get(c, 0) + 1
        smoothed_color = max(counts.keys(), key=lambda k: counts[k])

        avg_conf = round(sum(self._conf_history) / len(self._conf_history), 2)
        is_reliable = avg_conf >= _MIN_RELIABILITY_CONF and smoothed_color != TrafficLightColor.UNKNOWN

        # Distance from ego bumper
        distance_px = round(float(h_frame - iy2), 1)

        reason = self._build_signal_reason(smoothed_color, avg_conf, is_reliable)

        return TrafficSignal(
            color        = smoothed_color,
            confidence   = avg_conf,
            is_reliable  = is_reliable,
            box          = (float(ix1), float(iy1), float(ix2), float(iy2)),
            distance_px  = distance_px,
            reason       = reason,
        )

    # ── Private Color Classification ─────────────────────────────────────────

    def _classify_roi_color(self, roi: np.ndarray) -> Tuple[TrafficLightColor, float]:
        """Classify active lamp color from cropped traffic light ROI."""
        h, w = roi.shape[:2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Masks
        mask_r1 = cv2.inRange(hsv, _HSV_RED_LOW1, _HSV_RED_HIGH1)
        mask_r2 = cv2.inRange(hsv, _HSV_RED_LOW2, _HSV_RED_HIGH2)
        mask_red = cv2.bitwise_or(mask_r1, mask_r2)

        mask_yellow = cv2.inRange(hsv, _HSV_YELLOW_LOW, _HSV_YELLOW_HIGH)
        mask_green  = cv2.inRange(hsv, _HSV_GREEN_LOW, _HSV_GREEN_HIGH)

        # Spatial weighting: Red top, Yellow middle, Green bottom
        top_half    = slice(0, int(h * 0.45))
        mid_half    = slice(int(h * 0.25), int(h * 0.75))
        bot_half    = slice(int(h * 0.55), h)

        red_pts    = int(np.count_nonzero(mask_red[top_half, :]) * 1.3 + np.count_nonzero(mask_red))
        yellow_pts = int(np.count_nonzero(mask_yellow[mid_half, :]) * 1.3 + np.count_nonzero(mask_yellow))
        green_pts  = int(np.count_nonzero(mask_green[bot_half, :]) * 1.3 + np.count_nonzero(mask_green))

        total_pixels = max(1, h * w)
        max_pts = max(red_pts, yellow_pts, green_pts)

        if max_pts < 8:
            return TrafficLightColor.UNKNOWN, 0.30

        conf = min(0.98, max(0.40, max_pts / (total_pixels * 0.20)))

        if max_pts == red_pts:
            return TrafficLightColor.RED, round(conf, 2)
        elif max_pts == yellow_pts:
            return TrafficLightColor.YELLOW, round(conf, 2)
        else:
            return TrafficLightColor.GREEN, round(conf, 2)

    def _heuristic_signal_search(self, frame: np.ndarray) -> Optional[Tuple[float, float, float, float, float]]:
        """Fallback search in upper 60% of frame for high-saturation signal lamps."""
        h, w = frame.shape[:2]
        upper_roi = frame[0:int(h * 0.60), :]

        hsv = cv2.cvtColor(upper_roi, cv2.COLOR_BGR2HSV)
        mask_r = cv2.bitwise_or(
            cv2.inRange(hsv, _HSV_RED_LOW1, _HSV_RED_HIGH1),
            cv2.inRange(hsv, _HSV_RED_LOW2, _HSV_RED_HIGH2),
        )
        mask_g = cv2.inRange(hsv, _HSV_GREEN_LOW, _HSV_GREEN_HIGH)
        mask_y = cv2.inRange(hsv, _HSV_YELLOW_LOW, _HSV_YELLOW_HIGH)

        combined = cv2.bitwise_or(mask_r, cv2.bitwise_or(mask_g, mask_y))
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 30 <= area <= 2500:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                aspect = bh / max(1, bw)
                # Traffic lights are vertical rectangles (aspect ratio ~1.4 to 4.0)
                # Or circular lamp bulbs (aspect ~0.8 to 1.3)
                if 0.7 <= aspect <= 4.2:
                    pad_w = int(bw * 0.4)
                    pad_h = int(bh * 0.6)
                    x1 = max(0, bx - pad_w)
                    y1 = max(0, by - pad_h)
                    x2 = min(w, bx + bw + pad_w)
                    y2 = min(h, by + bh + pad_h)
                    candidates.append((float(x1), float(y1), float(x2), float(y2), 0.72))

        if candidates:
            return max(candidates, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        return None

    def _build_signal_reason(self, color: TrafficLightColor, conf: float, is_reliable: bool) -> str:
        if not is_reliable:
            return f"Traffic light visible but state uncertain ({color.value}, conf {conf*100:.0f}%). Monitoring."
        if color == TrafficLightColor.RED:
            return f"RED traffic signal recognized ({conf*100:.0f}% conf). Compulsory halt required before stop line."
        elif color == TrafficLightColor.YELLOW:
            return f"YELLOW traffic signal recognized ({conf*100:.0f}% conf). Prepare to stop; decelerating."
        elif color == TrafficLightColor.GREEN:
            return f"GREEN traffic signal recognized ({conf*100:.0f}% conf). Right of way clear; proceed with caution."
        return "Traffic light state UNKNOWN."

    def reset(self) -> None:
        self._color_history.clear()
        self._conf_history.clear()
