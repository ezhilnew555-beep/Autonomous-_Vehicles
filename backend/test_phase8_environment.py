"""
test_phase8_environment.py - Unit & Scenario Test Suite for Phase 8: Road-Context Awareness.

Tests required by specification:
1. Red Signal: RED traffic light recognized -> Vehicle commanded to STOP before stop line.
2. Pedestrian Crossing Sign: Sign detected -> SLOW_DOWN and heightened pedestrian caution.
3. No-Entry Situation: Prohibited corridor -> REPLAN or STOP.
4. Intersection: Approaching T-junction/4-way -> Controlled approach with deceleration.
5. Unmarked Road: Dynamic road corridor without lane markings -> Conservative cruising.
6. Safety Precedence Axiom: GREEN signal active BUT pedestrian enters corridor -> Obstacle hazard strictly overrides green signal!
7. HSV Traffic Light Detector Unit: Image crop classification of RED, YELLOW, GREEN lamps.
"""
import sys
import time
from pathlib import Path

# Add backend to path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import cv2
import numpy as np

from app.tracking.tracker import TrackedObject
from app.collision.schemas import RiskLevel, CollisionRisk
from app.decision.schemas import VehicleAction, AvoidanceDirection
from app.decision.engine import DecisionEngine
from app.planning.planner import AdaptivePathPlanner
from app.planning.schemas import PlanningStatus
from app.environment.schemas import (
    TrafficLightColor,
    TrafficSignal,
    RoadSign,
    RoadSignType,
    RoadGeometryType,
    MapRoadConstraint,
    EnvironmentState,
)
from app.environment.signals import TrafficSignalDetector
from app.environment.signs import RoadSignDetector
from app.environment.map_provider import ModularMapProvider


def make_mock_track(track_id: int, cls_name: str, x1: float, y1: float, x2: float, y2: float) -> TrackedObject:
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return TrackedObject(
        track_id   = track_id,
        class_id   = 0,
        class_name = cls_name,
        confidence = 0.90,
        x1         = x1,
        y1         = y1,
        x2         = x2,
        y2         = y2,
        motion     = None,
        trajectory = [(cx, cy)],
    )


def test_red_signal():
    """Scenario 1: RED traffic light detected -> STOP before stop line"""
    engine = DecisionEngine()
    planner = AdaptivePathPlanner()

    sig = TrafficSignal(
        color        = TrafficLightColor.RED,
        confidence   = 0.92,
        is_reliable  = True,
        box          = (600, 100, 640, 190),
        distance_px  = 400.0,
        reason       = "RED traffic signal recognized (92% conf).",
    )
    env = EnvironmentState(
        frame_idx     = 10,
        timestamp_sec = 0.33,
        active_signal = sig,
    )

    dec = engine.evaluate_frame(
        frame_idx       = 10,
        fps             = 30.0,
        frame_width     = 1280,
        frame_height    = 720,
        tracked_objects = [],
        predictions     = {},
        risks           = {},
        environment     = env,
    )

    assert dec.action == VehicleAction.STOP, f"Expected STOP on RED light, got {dec.action}"
    assert dec.target_speed_factor == 0.0, f"Expected throttle 0.0, got {dec.target_speed_factor}"
    assert "RED traffic signal" in dec.reason

    # Test path planner incorporates stopping constraint
    plan = planner.plan_frame(
        frame_idx       = 10,
        fps             = 30.0,
        frame_width     = 1280,
        frame_height    = 720,
        tracked_objects = [],
        predictions     = {},
        decision        = dec,
        environment     = env,
    )
    assert len(plan.safe_path) > 0, "Expected stopping trajectory"
    # Stopping path should be truncated before full horizon
    assert len(plan.safe_path) <= len(plan.original_path) // 2 + 2

    print(f"  [PASS] Scenario 1: Red Signal correctly commanded {dec.action.value} (throttle: {dec.target_speed_factor*100:.0f}%).")


def test_pedestrian_crossing_sign():
    """Scenario 2: Pedestrian crossing sign recognized -> SLOW_DOWN caution"""
    engine = DecisionEngine()

    sign = RoadSign(
        sign_type        = RoadSignType.PEDESTRIAN_CROSSING,
        confidence       = 0.88,
        box              = (1000, 180, 1080, 260),
        value            = "Pedestrian Crossing Ahead",
        action_directive = "PEDESTRIAN_ZONE_CAUTION",
    )
    env = EnvironmentState(
        frame_idx      = 20,
        timestamp_sec  = 0.66,
        detected_signs = [sign],
    )

    dec = engine.evaluate_frame(
        frame_idx       = 20,
        fps             = 30.0,
        frame_width     = 1280,
        frame_height    = 720,
        tracked_objects = [],
        predictions     = {},
        risks           = {},
        environment     = env,
    )

    assert dec.action == VehicleAction.SLOW_DOWN, f"Expected SLOW_DOWN, got {dec.action}"
    assert dec.target_speed_factor <= 0.60, f"Expected speed cap <= 0.60, got {dec.target_speed_factor}"
    assert "Pedestrian crossing" in dec.reason

    print(f"  [PASS] Scenario 2: Pedestrian Crossing Sign commanded {dec.action.value} (throttle {dec.target_speed_factor*100:.0f}%).")


def test_no_entry_situation():
    """Scenario 3: No-Entry sign detected -> Prohibits corridor, triggers REPLAN or STOP"""
    engine = DecisionEngine()
    planner = AdaptivePathPlanner()

    sign = RoadSign(
        sign_type        = RoadSignType.NO_ENTRY,
        confidence       = 0.90,
        box              = (620, 200, 680, 260),
        value            = "No Entry",
        action_directive = "PROHIBITED_CORRIDOR",
    )
    env = EnvironmentState(
        frame_idx      = 30,
        timestamp_sec  = 1.0,
        detected_signs = [sign],
    )

    dec = engine.evaluate_frame(
        frame_idx       = 30,
        fps             = 30.0,
        frame_width     = 1280,
        frame_height    = 720,
        tracked_objects = [],
        predictions     = {},
        risks           = {},
        environment     = env,
    )

    assert dec.action in (VehicleAction.REPLAN, VehicleAction.STOP), f"Expected REPLAN or STOP on No Entry, got {dec.action}"
    assert "No-Entry" in dec.reason

    # Planner should recognise corridor is prohibited and generate avoidance path
    plan = planner.plan_frame(
        frame_idx       = 30,
        fps             = 30.0,
        frame_width     = 1280,
        frame_height    = 720,
        tracked_objects = [],
        predictions     = {},
        decision        = dec,
        environment     = env,
    )
    assert plan.status in (PlanningStatus.REPLANNED_AVOID_LEFT, PlanningStatus.REPLANNED_AVOID_RIGHT, PlanningStatus.NO_SAFE_PATH)

    print(f"  [PASS] Scenario 3: No-Entry Situation correctly commanded {dec.action.value} and path {plan.status.value}.")


def test_intersection_approach():
    """Scenario 4: Approaching T-junction intersection on map -> Controlled SLOW_DOWN"""
    engine = DecisionEngine()
    map_p = ModularMapProvider(default_profile="intersection_t")
    constraints = map_p.get_constraints(40, 1.33)

    env = EnvironmentState(
        frame_idx       = 40,
        timestamp_sec   = 1.33,
        map_constraints = constraints,
    )

    dec = engine.evaluate_frame(
        frame_idx       = 40,
        fps             = 30.0,
        frame_width     = 1280,
        frame_height    = 720,
        tracked_objects = [],
        predictions     = {},
        risks           = {},
        environment     = env,
    )

    assert dec.action == VehicleAction.SLOW_DOWN, f"Expected SLOW_DOWN approaching intersection, got {dec.action}"
    assert dec.target_speed_factor <= 0.60
    assert "INTERSECTION_T" in dec.reason

    print(f"  [PASS] Scenario 4: Intersection Approach triggered {dec.action.value} (speed factor {dec.target_speed_factor}).")


def test_unmarked_road():
    """Scenario 5: Unmarked road without lane markings -> Cautious cruising speed cap"""
    engine = DecisionEngine()
    map_p = ModularMapProvider(default_profile="unmarked_road")
    constraints = map_p.get_constraints(50, 1.66)

    env = EnvironmentState(
        frame_idx       = 50,
        timestamp_sec   = 1.66,
        map_constraints = constraints,
    )

    dec = engine.evaluate_frame(
        frame_idx       = 50,
        fps             = 30.0,
        frame_width     = 1280,
        frame_height    = 720,
        tracked_objects = [],
        predictions     = {},
        risks           = {},
        environment     = env,
    )

    assert dec.action == VehicleAction.CONTINUE
    assert dec.target_speed_factor <= 0.70, f"Expected conservative cruising speed on unmarked road, got {dec.target_speed_factor}"
    assert "Unmarked road" in dec.reason

    print(f"  [PASS] Scenario 5: Unmarked Road governed cruising speed factor to {dec.target_speed_factor*100:.0f}%.")


def test_safety_precedence_over_green_signal():
    """Scenario 6: GREEN signal active BUT Pedestrian crossing in path -> Collision risk strictly overrides signal!"""
    engine = DecisionEngine()

    # Active GREEN light
    sig = TrafficSignal(
        color        = TrafficLightColor.GREEN,
        confidence   = 0.95,
        is_reliable  = True,
        box          = (600, 100, 640, 190),
        distance_px  = 400.0,
        reason       = "GREEN traffic signal (95% conf). Right of way clear.",
    )
    env = EnvironmentState(
        frame_idx     = 60,
        timestamp_sec = 2.0,
        active_signal = sig,
    )

    # BUT: Pedestrian crossing in front of vehicle with HIGH risk (TTC = 1.3s)
    ped = make_mock_track(12, "person", 620, 520, 660, 600)
    risk = CollisionRisk(
        track_id              = 12,
        class_name            = "person",
        distance_px           = 140.0,
        lateral_offset_px     = 5.0,
        longitudinal_dist_px  = 120.0,
        relative_speed_px     = 3.0,
        closing_speed_px      = 2.5,
        ttc_sec               = 1.3,
        risk_level            = RiskLevel.HIGH,
        reason                = "Predicted Person #12 trajectory crosses vehicle path in 1.3s.",
        conflict_point        = (640.0, 560.0),
        conflict_frame_offset = 39,
    )

    dec = engine.evaluate_frame(
        frame_idx       = 60,
        fps             = 30.0,
        frame_width     = 1280,
        frame_height    = 720,
        tracked_objects = [ped],
        predictions     = {},
        risks           = {12: risk},
        environment     = env,
    )

    # MUST NOT continue just because signal is green! Must STOP or AVOID!
    assert dec.action in (VehicleAction.STOP, VehicleAction.AVOID, VehicleAction.EMERGENCY_STOP), (
        f"CRITICAL SAFETY VIOLATION: Green light overrode pedestrian hazard! Action was {dec.action}"
    )
    assert dec.target_object_id == 12

    print(f"  [PASS] Scenario 6: Safety Precedence Axiom Verified — Hazard commanded {dec.action.value}, overriding GREEN light.")


def test_hsv_traffic_signal_classification():
    """Scenario 7: Unit verification of HSV lamp color extraction."""
    detector = TrafficSignalDetector(history_len=1)

    # 1. Synthetic RED light crop (black housing, bright red top circle)
    red_crop = np.full((90, 30, 3), [20, 20, 20], dtype=np.uint8)
    cv2.circle(red_crop, (15, 18), 8, (0, 0, 255), -1)  # BGR Red
    color_r, conf_r = detector._classify_roi_color(red_crop)
    assert color_r == TrafficLightColor.RED, f"Expected RED, got {color_r}"
    assert conf_r > 0.60

    # 2. Synthetic GREEN light crop (black housing, bright green bottom circle)
    green_crop = np.full((90, 30, 3), [20, 20, 20], dtype=np.uint8)
    cv2.circle(green_crop, (15, 72), 8, (50, 240, 50), -1)  # BGR Green
    color_g, conf_g = detector._classify_roi_color(green_crop)
    assert color_g == TrafficLightColor.GREEN, f"Expected GREEN, got {color_g}"
    assert conf_g > 0.60

    # 3. Synthetic YELLOW light crop (black housing, bright yellow middle circle)
    yel_crop = np.full((90, 30, 3), [20, 20, 20], dtype=np.uint8)
    cv2.circle(yel_crop, (15, 45), 8, (0, 220, 255), -1)  # BGR Yellow
    color_y, conf_y = detector._classify_roi_color(yel_crop)
    assert color_y == TrafficLightColor.YELLOW, f"Expected YELLOW, got {color_y}"
    assert conf_y > 0.60

    print("  [PASS] Scenario 7: HSV Traffic Signal Detector correctly classified Red, Yellow, and Green lamps.")


def run_all_tests():
    print("===========================================================================")
    print("  Eareyes 2k26 — Phase 8: Road-Context Awareness Test Suite")
    print("===========================================================================")

    test_red_signal()
    test_pedestrian_crossing_sign()
    test_no_entry_situation()
    test_intersection_approach()
    test_unmarked_road()
    test_safety_precedence_over_green_signal()
    test_hsv_traffic_signal_classification()

    print("===========================================================================")
    print("  ALL 7 PHASE 8 ENVIRONMENT SCENARIOS PASSED WITH FULL MATHEMATICAL RIGOR!")
    print("===========================================================================")


if __name__ == "__main__":
    run_all_tests()
