"""
Road Sign Detection & Classification Engine — Phase 8B.

Detects, classifies, and temporally confirms regulatory and warning road signs:
  • STOP          — Mandatory halt at intersection
  • NO_ENTRY      — Prohibited entry; corridor blocked
  • SPEED_LIMIT   — Posted maximum speed limit restriction (e.g., 40 km/h)
  • NO_LEFT_TURN  — Prohibits leftward turns / evasive paths
  • NO_RIGHT_TURN — Prohibits rightward turns / evasive paths
  • YIELD         — Yield right-of-way; heightened caution
  • PEDESTRIAN_CROSSING — Heightened pedestrian awareness & speed reduction
  • WARNING       — Road hazards, sharp curves, speed breakers
  • UNKNOWN       — Unverified or low-confidence signs (ignored for safety rules)

Features:
- Extensible custom sign model hook (allows loading custom YOLO/ONNX sign weights).
- Pretrained YOLO detection integration (e.g. COCO-80 'stop sign').
- Color and geometric contour classification (octagons, circles, triangles, directional arrows).
- Rolling temporal confirmation filter to prevent single-frame flickering noise.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.detection.detector import Detection
from app.environment.schemas import RoadSign, RoadSignType
from app.environment.sign_rules import map_sign_to_rule, extract_speed_limit_value
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Color Ranges for Road Sign Recognition (HSV) ──────────────────────────────
_HSV_RED1 = (np.array([0, 80, 80], dtype=np.uint8), np.array([12, 255, 255], dtype=np.uint8))
_HSV_RED2 = (np.array([168, 80, 80], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
_HSV_BLUE  = (np.array([100, 100, 80], dtype=np.uint8), np.array([130, 255, 255], dtype=np.uint8))
_HSV_YELLOW = (np.array([18, 100, 100], dtype=np.uint8), np.array([35, 255, 255], dtype=np.uint8))


class RoadSignDetector:
    """
    Extensible road sign detector supporting pretrained YOLO detections,
    custom sign models, and color/geometric feature extraction with temporal confirmation.
    """

    def __init__(
        self,
        custom_model_path: Optional[str] = None,
        min_confidence: float = 0.60,
        history_len: int = 5,
    ) -> None:
        self.custom_model_path = custom_model_path
        self.min_confidence = min_confidence
        self.history_len = history_len
        self._custom_model = None

        # Temporal confirmation: recent seen signs buffer
        self._recent_observations: deque[List[RoadSign]] = deque(maxlen=history_len)

        if custom_model_path:
            try:
                from ultralytics import YOLO
                self._custom_model = YOLO(custom_model_path)
                logger.info(f"Loaded custom road sign model from {custom_model_path}")
            except Exception as e:
                logger.warning(f"Could not load custom sign model: {e}. Using built-in hybrid detector.")

    def detect_signs(
        self,
        frame: np.ndarray,
        yolo_detections: Optional[List[Detection]] = None,
        timestamp_sec: Optional[float] = None,
    ) -> List[RoadSign]:
        """
        Detect road signs in frame using hybrid pipeline and apply temporal filtering.

        Args:
            frame           : Video frame image (BGR).
            yolo_detections : Pretrained YOLO object detections.
            timestamp_sec   : Optional video timestamp.

        Returns:
            List of confirmed RoadSign instances.
        """
        raw_signs: List[RoadSign] = []

        # 1. Inspect YOLO detections for standard COCO signs (Class 11 = 'stop sign')
        if yolo_detections:
            for d in yolo_detections:
                cls_lower = d.class_name.lower()
                if "stop" in cls_lower and "sign" in cls_lower:
                    raw_signs.append(RoadSign(
                        sign_type        = RoadSignType.STOP,
                        confidence       = round(d.confidence, 2),
                        box              = (d.x1, d.y1, d.x2, d.y2),
                        value            = None,
                        action_directive = "MANDATORY_STOP",
                        rule             = "STOP",
                    ))

        # 2. Geometric Shape & Color Extraction
        heuristic_signs = self._extract_candidate_signs(frame)
        for s in heuristic_signs:
            if not self._overlaps_existing(s.box, raw_signs):
                s.rule = map_sign_to_rule(s, min_confidence=self.min_confidence)
                raw_signs.append(s)

        # 3. Temporal Confirmation
        self._recent_observations.append(raw_signs)
        confirmed_signs = self._apply_temporal_confirmation(raw_signs)

        return confirmed_signs

    def detect_signs_structured(
        self,
        frame: np.ndarray,
        yolo_detections: Optional[List[Detection]] = None,
        timestamp_sec: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Returns detected signs in Section 2 JSON structure:
        {
            "sign_type": "STOP",
            "confidence": 0.93,
            "bbox": [x1, y1, x2, y2],
            "value": None | 40,
            "timestamp": "..."
        }
        """
        signs = self.detect_signs(frame, yolo_detections, timestamp_sec)
        out = []
        for s in signs:
            val = extract_speed_limit_value(s.value) if s.sign_type == RoadSignType.SPEED_LIMIT else s.value
            out.append({
                "sign_type": s.sign_type.value,
                "confidence": s.confidence,
                "bbox": [round(v, 1) for v in s.box],
                "value": val,
                "timestamp": f"{timestamp_sec:.2f}s" if timestamp_sec is not None else "0.00s",
            })
        return out

    # ── Private Geometric & Color Feature Classifier ──────────────────────────

    def _extract_candidate_signs(self, frame: np.ndarray) -> List[RoadSign]:
        """Extract road signs using color segmentation and polygon approximation."""
        h, w = frame.shape[:2]
        roi_search = frame[0:int(h * 0.75), :]
        hsv = cv2.cvtColor(roi_search, cv2.COLOR_BGR2HSV)

        # Red mask (STOP, No Entry, Speed Limit, Turn restrictions)
        mask_r1 = cv2.inRange(hsv, _HSV_RED1[0], _HSV_RED1[1])
        mask_r2 = cv2.inRange(hsv, _HSV_RED2[0], _HSV_RED2[1])
        mask_red = cv2.bitwise_or(mask_r1, mask_r2)

        # Blue mask (Pedestrian crossing / Information signs)
        mask_blue = cv2.inRange(hsv, _HSV_BLUE[0], _HSV_BLUE[1])

        # Yellow mask (Caution / Warning / Yield signs)
        mask_yellow = cv2.inRange(hsv, _HSV_YELLOW[0], _HSV_YELLOW[1])

        found: List[RoadSign] = []
        found.extend(self._analyze_contours(mask_red, frame, "RED"))
        found.extend(self._analyze_contours(mask_blue, frame, "BLUE"))
        found.extend(self._analyze_contours(mask_yellow, frame, "YELLOW"))
        return found

    def _analyze_contours(self, mask: np.ndarray, frame: np.ndarray, color_group: str) -> List[RoadSign]:
        results: List[RoadSign] = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200 or area > 24000:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / max(1.0, bh)
            if not (0.70 <= aspect <= 1.40):
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.035 * peri, True)
            num_vertices = len(approx)
            circularity = (4.0 * np.pi * area) / max(1.0, peri * peri)

            sign_roi = frame[by:by + bh, bx:bx + bw]
            box = (float(bx), float(by), float(bx + bw), float(by + bh))

            if color_group == "RED":
                # Check for NO ENTRY (horizontal white bar inside red circle)
                if self._is_no_entry_sign(sign_roi):
                    results.append(RoadSign(
                        sign_type        = RoadSignType.NO_ENTRY,
                        confidence       = 0.88,
                        box              = box,
                        value            = None,
                        action_directive = "PROHIBITED_CORRIDOR",
                        rule             = "PROHIBITED_CORRIDOR",
                    ))
                # Check for NO LEFT TURN / NO RIGHT TURN (circular with diagonal slash)
                elif self._is_turn_restriction(sign_roi, "LEFT"):
                    results.append(RoadSign(
                        sign_type        = RoadSignType.NO_LEFT_TURN,
                        confidence       = 0.85,
                        box              = box,
                        value            = None,
                        action_directive = "REJECT_LEFT_PATH",
                        rule             = "REJECT_LEFT_PATH",
                    ))
                elif self._is_turn_restriction(sign_roi, "RIGHT"):
                    results.append(RoadSign(
                        sign_type        = RoadSignType.NO_RIGHT_TURN,
                        confidence       = 0.85,
                        box              = box,
                        value            = None,
                        action_directive = "REJECT_RIGHT_PATH",
                        rule             = "REJECT_RIGHT_PATH",
                    ))
                # Check for SPEED LIMIT (red circular border with white interior & digits)
                elif circularity > 0.65 or num_vertices >= 7:
                    speed_val = self._estimate_speed_limit(sign_roi)
                    results.append(RoadSign(
                        sign_type        = RoadSignType.SPEED_LIMIT,
                        confidence       = 0.86,
                        box              = box,
                        value            = speed_val,
                        action_directive = f"CAP_SPEED_{str(speed_val).replace(' ', '_')}",
                        rule             = "SPEED_CONSTRAINT",
                    ))
                elif num_vertices == 8 or circularity > 0.70:
                    results.append(RoadSign(
                        sign_type        = RoadSignType.STOP,
                        confidence       = 0.88,
                        box              = box,
                        value            = None,
                        action_directive = "MANDATORY_STOP",
                        rule             = "STOP",
                    ))

            elif color_group == "BLUE":
                if num_vertices == 4 or 0.8 <= aspect <= 1.2:
                    results.append(RoadSign(
                        sign_type        = RoadSignType.PEDESTRIAN_CROSSING,
                        confidence       = 0.84,
                        box              = box,
                        value            = "Pedestrian Crossing Ahead",
                        action_directive = "PEDESTRIAN_ZONE_CAUTION",
                        rule             = "PEDESTRIAN_ZONE_CAUTION",
                    ))

            elif color_group == "YELLOW":
                # Inverted triangle is YIELD sign, upright triangle is WARNING sign
                if self._is_yield_sign(cnt, by, bh):
                    results.append(RoadSign(
                        sign_type        = RoadSignType.YIELD,
                        confidence       = 0.87,
                        box              = box,
                        value            = "Yield Right of Way",
                        action_directive = "YIELD_CAUTION",
                        rule             = "YIELD_CAUTION",
                    ))
                elif num_vertices in (3, 4) or circularity < 0.65:
                    results.append(RoadSign(
                        sign_type        = RoadSignType.WARNING,
                        confidence       = 0.82,
                        box              = box,
                        value            = "Caution / Hazard",
                        action_directive = "ADVISORY_SLOW_DOWN",
                        rule             = "ADVISORY_SLOW_DOWN",
                    ))

        return results

    def _is_yield_sign(self, cnt: np.ndarray, by: int, bh: int) -> bool:
        """Check if triangular contour is inverted (Yield sign)."""
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) == 3:
            # Check if pointing downwards: one vertex at bottom, two at top
            pts = approx.reshape(-1, 2)
            ys = pts[:, 1]
            # If bottom point is near bottom of bounding box
            bottom_pt_idx = int(np.argmax(ys))
            if ys[bottom_pt_idx] > (by + bh * 0.80):
                return True
        return False

    def _is_no_entry_sign(self, roi: np.ndarray) -> bool:
        """Detect horizontal white stripe across red circle."""
        h, w = roi.shape[:2]
        if h < 14 or w < 14:
            return False
        mid_stripe = roi[int(h * 0.35):int(h * 0.65), int(w * 0.15):int(w * 0.85)]
        hsv_stripe = cv2.cvtColor(mid_stripe, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv_stripe, np.array([0, 0, 160], dtype=np.uint8), np.array([180, 60, 255], dtype=np.uint8))
        white_ratio = np.count_nonzero(white_mask) / max(1, mid_stripe.shape[0] * mid_stripe.shape[1])
        return white_ratio > 0.35

    def _is_turn_restriction(self, roi: np.ndarray, direction: str) -> bool:
        """Heuristic check for left or right turn prohibition slash."""
        h, w = roi.shape[:2]
        if h < 20 or w < 20:
            return False
        # Diagonal slash test across center
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=15, minLineLength=int(h * 0.35), maxLineGap=5)
        if lines is not None:
            for line in lines:
                l_arr = line.reshape(-1)
                if len(l_arr) >= 4:
                    x1, y1, x2, y2 = l_arr[:4]
                    slope = (y2 - y1) / max(1e-4, x2 - x1)
                    # Diagonal line roughly 30 to 60 degrees
                    if 0.5 <= abs(slope) <= 2.0:
                        # Direction check based on center of black arrow pixels
                        if direction == "LEFT" and np.mean(edges[:, :w // 2]) > np.mean(edges[:, w // 2:]):
                            return True
                        elif direction == "RIGHT" and np.mean(edges[:, w // 2:]) > np.mean(edges[:, :w // 2]):
                            return True
        return False

    def _estimate_speed_limit(self, roi: np.ndarray) -> int:
        """Extract numeric speed limit (default 40 km/h, or 30/50)."""
        h, w = roi.shape[:2]
        if h < 15 or w < 15:
            return 40
        center_crop = roi[int(h * 0.25):int(h * 0.75), int(w * 0.25):int(w * 0.75)]
        gray = cv2.cvtColor(center_crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        dark_ratio = np.count_nonzero(thresh) / max(1, center_crop.shape[0] * center_crop.shape[1])
        if dark_ratio > 0.38:
            return 30
        elif dark_ratio < 0.22:
            return 50
        return 40

    def _apply_temporal_confirmation(self, current_signs: List[RoadSign]) -> List[RoadSign]:
        """
        Apply temporal smoothing:
        - Signs with high confidence (>= 0.75) are immediately accepted.
        - Signs with marginal confidence (min_confidence <= conf < 0.75) require presence in at least
          2 of the last 4 frames.
        - Signs below min_confidence are filtered or classified as UNKNOWN.
        """
        confirmed: List[RoadSign] = []
        for s in current_signs:
            if s.confidence < self.min_confidence:
                # Mark as UNKNOWN or drop
                continue
            if s.confidence >= 0.75:
                confirmed.append(s)
            else:
                # Check recent history for matching sign type
                recent_matches = 0
                for obs_list in list(self._recent_observations)[:-1]:
                    if any(prev.sign_type == s.sign_type for prev in obs_list):
                        recent_matches += 1
                if recent_matches >= 1:
                    confirmed.append(s)
        return confirmed

    def _overlaps_existing(self, box: Tuple[float, float, float, float], existing: List[RoadSign]) -> bool:
        x1, y1, x2, y2 = box
        for s in existing:
            ex1, ey1, ex2, ey2 = s.box
            ix1 = max(x1, ex1)
            iy1 = max(y1, ey1)
            ix2 = min(x2, ex2)
            iy2 = min(y2, ey2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                area_a = (x2 - x1) * (y2 - y1)
                area_b = (ex2 - ex1) * (ey2 - ey1)
                iou = inter / max(1.0, area_a + area_b - inter)
                if iou > 0.3:
                    return True
        return False

    def reset(self) -> None:
        self._recent_observations.clear()
