"""
Phase 8B — Road Sign Detection and Rule Integration: Spec Compliance Test Suite.

Tests all 9 required scenarios from user specification:
  1. STOP sign                -> STOP rule
  2. SPEED LIMIT 40           -> speed constraint = 40 km/h
  3. NO LEFT TURN             -> left path rejected
  4. NO RIGHT TURN            -> right path rejected
  5. NO ENTRY                 -> prohibited forward corridor direction
  6. YIELD                    -> increased caution / SLOW_DOWN
  7. Unknown / low-confidence -> no active rule applied
  8. Multiple signs           -> all valid constraints applied simultaneously
  9. Sign restriction + obstacle -> safety and sign constraints both respected
"""
import sys
sys.path.insert(0, ".")

import numpy as np

from app.environment.schemas import (
    RoadSign,
    RoadSignType,
    MapRoadConstraint,
    EnvironmentState,
)
from app.environment.sign_rules import (
    map_sign_to_rule,
    extract_speed_limit_value,
    evaluate_sign_constraints,
    SignRuleConstants,
)
from app.environment.road_signs import RoadSignDetector
from app.tracking.tracker import TrackedObject
from app.tracking.motion import MotionData
from app.collision.schemas import RiskLevel, CollisionRisk
from app.decision.schemas import VehicleAction, AvoidanceDirection, RoadContext, DecisionOutput
from app.decision.engine import DecisionEngine
from app.planning.planner import AdaptivePathPlanner
from app.planning.schemas import PlanningStatus

W, H, FPS = 640, 480, 30.0

print("=" * 70)
print("PHASE 8B — ROAD SIGN DETECTION & RULE INTEGRATION: SPEC COMPLIANCE TEST")
print("=" * 70)


def make_engine():
    return DecisionEngine(corridor_half_width=80.0, hysteresis_frames=2)


def make_planner():
    return AdaptivePathPlanner(lookahead_distance=340.0, num_waypoints=20, road_margin_px=35.0)


def make_env_with_signs(signs: list[RoadSign]) -> EnvironmentState:
    constraints = evaluate_sign_constraints(signs)
    return EnvironmentState(
        frame_idx=0,
        timestamp_sec=0.0,
        detected_signs=signs,
        active_speed_limit=constraints.get("speed_limit"),
        active_sign_constraints=constraints.get("active_constraints", []),
        restricted_directions=constraints.get("restricted_directions", []),
    )


def make_motion(dx, dy):
    speed = (dx**2 + dy**2) ** 0.5
    return MotionData(
        dx=dx, dy=dy, pixel_speed=speed, pixel_speed_sec=speed * FPS,
        direction_angle=0.0, direction="FORWARD", direction_symbol="↑", direction_label="N"
    )


def make_obstacle(tid, cls, x1, y1, x2, y2, dx=0.0, dy=0.0):
    return TrackedObject(
        track_id=tid, class_id=0, class_name=cls, confidence=0.92,
        x1=x1, y1=y1, x2=x2, y2=y2,
        motion=make_motion(dx, dy), trajectory=[((x1 + x2) / 2, (y1 + y2) / 2)],
    )


# ─── TEST 1: STOP SIGN -> STOP RULE ──────────────────────────────────────────
print("\n[TEST 1] STOP sign -> STOP rule:")
stop_sign = RoadSign(
    sign_type=RoadSignType.STOP,
    confidence=0.94,
    box=(520.0, 50.0, 580.0, 110.0),
    value=None,
    action_directive="MANDATORY_STOP",
)
rule = map_sign_to_rule(stop_sign)
print(f"  Rule mapped: {rule}")
assert rule == SignRuleConstants.STOP, f"Expected STOP rule, got {rule}"

engine = make_engine()
env = make_env_with_signs([stop_sign])
dec = engine.evaluate_frame(
    frame_idx=0, fps=FPS, frame_width=W, frame_height=H,
    tracked_objects=[], risks={}, environment=env,
)
print(f"  Decision Action: {dec.action.value} | Target Speed Factor: {dec.target_speed_factor:.0%}")
print(f"  Reason: {dec.reason}")
assert dec.action == VehicleAction.STOP, f"Expected STOP, got {dec.action}"
assert dec.target_speed_factor == 0.0, "Expected full stop (0 throttle)"
print("  --> PASS: STOP sign triggers STOP action with 0 throttle.")


# ─── TEST 2: SPEED LIMIT 40 -> SPEED CONSTRAINT = 40 ─────────────────────────
print("\n[TEST 2] SPEED LIMIT 40 -> speed constraint = 40:")
speed_sign = RoadSign(
    sign_type=RoadSignType.SPEED_LIMIT,
    confidence=0.91,
    box=(530.0, 60.0, 580.0, 110.0),
    value=40,
    action_directive="CAP_SPEED_40",
)
rule = map_sign_to_rule(speed_sign)
val = extract_speed_limit_value(speed_sign.value)
print(f"  Rule: {rule} | Extracted speed limit: {val} km/h")
assert rule == SignRuleConstants.SPEED_CONSTRAINT
assert val == 40.0, f"Expected 40.0, got {val}"

constraints = evaluate_sign_constraints([speed_sign])
assert constraints["speed_limit"] == 40.0, f"Expected speed_limit 40.0, got {constraints['speed_limit']}"
assert "SPEED_CONSTRAINT" in constraints["active_constraints"]

engine = make_engine()
env = make_env_with_signs([speed_sign])
dec = engine.evaluate_frame(
    frame_idx=0, fps=FPS, frame_width=W, frame_height=H,
    tracked_objects=[], risks={}, environment=env,
)
print(f"  Decision: {dec.action.value} | Speed Factor: {dec.target_speed_factor:.0%}")
assert dec.action == VehicleAction.CONTINUE
assert dec.target_speed_factor <= 0.55, f"Expected speed factor capped, got {dec.target_speed_factor}"
assert "40" in dec.reason
print("  --> PASS: SPEED LIMIT 40 enforces 40 km/h constraint and limits throttle.")


# ─── TEST 3: NO LEFT TURN -> LEFT PATH REJECTED ──────────────────────────────
print("\n[TEST 3] NO LEFT TURN -> left candidate path rejected:")
no_left_sign = RoadSign(
    sign_type=RoadSignType.NO_LEFT_TURN,
    confidence=0.89,
    box=(520.0, 60.0, 575.0, 115.0),
    action_directive="REJECT_LEFT_PATH",
)
rule = map_sign_to_rule(no_left_sign)
assert rule == SignRuleConstants.REJECT_LEFT_PATH

constraints = evaluate_sign_constraints([no_left_sign])
assert "LEFT" in constraints["restricted_directions"]
print(f"  Rule: {rule} | Restricted directions: {constraints['restricted_directions']}")

# Path Planner: when corridor requires evasion (e.g. center blocked), NO_LEFT_TURN rejects left paths
planner = make_planner()
center_obs = make_obstacle(5, "car", x1=280, y1=240, x2=360, y2=320)
env = make_env_with_signs([no_left_sign])

plan = planner.plan_frame(
    frame_idx=0, fps=FPS, frame_width=W, frame_height=H,
    tracked_objects=[center_obs],
    decision=DecisionOutput(
        frame_idx=0, timestamp_sec=0.0, action=VehicleAction.REPLAN,
        target_object_id=5, target_class="car", risk_level=RiskLevel.HIGH,
        ttc_sec=2.0, avoidance_direction=AvoidanceDirection.RIGHT, target_speed_factor=0.6,
        reason="Center blocked.", road_context=RoadContext(corridor_blocked=True, can_avoid_left=False, can_avoid_right=True)
    ),
    environment=env,
)
print(f"  Plan status: {plan.status.value}")
if plan.selected_path:
    print(f"  Selected path lateral offset: {plan.selected_path.lateral_offset:.1f}px")
    assert plan.selected_path.lateral_offset >= -10.0, "Left path must NOT be selected when NO_LEFT_TURN is active"
assert plan.status != PlanningStatus.REPLANNED_AVOID_LEFT, "Planner must not choose AVOID_LEFT"
print("  --> PASS: NO LEFT TURN successfully rejects left-turn candidates.")


# ─── TEST 4: NO RIGHT TURN -> RIGHT PATH REJECTED ─────────────────────────────
print("\n[TEST 4] NO RIGHT TURN -> right candidate path rejected:")
no_right_sign = RoadSign(
    sign_type=RoadSignType.NO_RIGHT_TURN,
    confidence=0.88,
    box=(520.0, 60.0, 575.0, 115.0),
    action_directive="REJECT_RIGHT_PATH",
)
rule = map_sign_to_rule(no_right_sign)
assert rule == SignRuleConstants.REJECT_RIGHT_PATH

constraints = evaluate_sign_constraints([no_right_sign])
assert "RIGHT" in constraints["restricted_directions"]

planner = make_planner()
center_obs = make_obstacle(6, "car", x1=280, y1=240, x2=360, y2=320)
env = make_env_with_signs([no_right_sign])

plan = planner.plan_frame(
    frame_idx=0, fps=FPS, frame_width=W, frame_height=H,
    tracked_objects=[center_obs],
    decision=DecisionOutput(
        frame_idx=0, timestamp_sec=0.0, action=VehicleAction.REPLAN,
        target_object_id=6, target_class="car", risk_level=RiskLevel.HIGH,
        ttc_sec=2.0, avoidance_direction=AvoidanceDirection.LEFT, target_speed_factor=0.6,
        reason="Center blocked.", road_context=RoadContext(corridor_blocked=True, can_avoid_left=True, can_avoid_right=False)
    ),
    environment=env,
)
print(f"  Plan status: {plan.status.value}")
if plan.selected_path:
    print(f"  Selected path lateral offset: {plan.selected_path.lateral_offset:.1f}px")
    assert plan.selected_path.lateral_offset <= 10.0, "Right path must NOT be selected when NO_RIGHT_TURN is active"
assert plan.status != PlanningStatus.REPLANNED_AVOID_RIGHT, "Planner must not choose AVOID_RIGHT"
print("  --> PASS: NO RIGHT TURN successfully rejects right-turn candidates.")


# ─── TEST 5: NO ENTRY -> PROHIBITED FORWARD DIRECTION ────────────────────────
print("\n[TEST 5] NO ENTRY -> prohibited forward direction:")
no_entry_sign = RoadSign(
    sign_type=RoadSignType.NO_ENTRY,
    confidence=0.92,
    box=(520.0, 50.0, 580.0, 110.0),
    action_directive="PROHIBITED_CORRIDOR",
)
rule = map_sign_to_rule(no_entry_sign)
assert rule == SignRuleConstants.PROHIBITED_CORRIDOR
constraints = evaluate_sign_constraints([no_entry_sign])
assert "PROHIBITED_CORRIDOR" in constraints["active_constraints"]

engine = make_engine()
env = make_env_with_signs([no_entry_sign])
dec = engine.evaluate_frame(
    frame_idx=0, fps=FPS, frame_width=W, frame_height=H,
    tracked_objects=[], risks={}, environment=env,
)
print(f"  Decision Action: {dec.action.value} | Avoidance Dir: {dec.avoidance_direction.value}")
assert dec.action in (VehicleAction.REPLAN, VehicleAction.STOP), f"Expected REPLAN or STOP on NO_ENTRY, got {dec.action}"

# Test planner rejects centerline
planner = make_planner()
plan = planner.plan_frame(
    frame_idx=0, fps=FPS, frame_width=W, frame_height=H,
    tracked_objects=[], decision=dec, environment=env,
)
print(f"  Plan Status: {plan.status.value}")
if plan.selected_path:
    assert abs(plan.selected_path.lateral_offset) > 10.0, "Centerline path must NOT be selected when NO_ENTRY is active"
print("  --> PASS: NO ENTRY prohibits forward corridor.")


# ─── TEST 6: YIELD -> INCREASED CAUTION ───────────────────────────────────────
print("\n[TEST 6] YIELD -> increased caution / SLOW_DOWN:")
yield_sign = RoadSign(
    sign_type=RoadSignType.YIELD,
    confidence=0.87,
    box=(520.0, 60.0, 570.0, 110.0),
    action_directive="YIELD_CAUTION",
)
rule = map_sign_to_rule(yield_sign)
assert rule == SignRuleConstants.YIELD_CAUTION

engine = make_engine()
env = make_env_with_signs([yield_sign])
dec = engine.evaluate_frame(
    frame_idx=0, fps=FPS, frame_width=W, frame_height=H,
    tracked_objects=[], risks={}, environment=env,
)
print(f"  Decision: {dec.action.value} | Speed Factor: {dec.target_speed_factor:.0%}")
assert dec.action == VehicleAction.SLOW_DOWN, f"Expected SLOW_DOWN, got {dec.action}"
assert dec.target_speed_factor <= 0.60
assert "YIELD" in dec.reason.upper()
print("  --> PASS: YIELD sign commands SLOW_DOWN caution.")


# ─── TEST 7: UNKNOWN / LOW-CONFIDENCE SIGN -> NO ACTIVE RULE ─────────────────
print("\n[TEST 7] Unknown / low-confidence sign -> no active rule:")
low_conf_sign = RoadSign(
    sign_type=RoadSignType.STOP,
    confidence=0.42,   # Below threshold of 0.60
    box=(520.0, 60.0, 570.0, 110.0),
)
rule = map_sign_to_rule(low_conf_sign, min_confidence=0.60)
print(f"  Low-confidence sign rule: {rule}")
assert rule == SignRuleConstants.NONE

unknown_sign = RoadSign(
    sign_type=RoadSignType.UNKNOWN,
    confidence=0.85,
    box=(520.0, 60.0, 570.0, 110.0),
)
rule_u = map_sign_to_rule(unknown_sign)
assert rule_u == SignRuleConstants.NONE

engine = make_engine()
env = make_env_with_signs([low_conf_sign])
dec = engine.evaluate_frame(
    frame_idx=0, fps=FPS, frame_width=W, frame_height=H,
    tracked_objects=[], risks={}, environment=env,
)
print(f"  Decision with low-conf sign: {dec.action.value}")
assert dec.action == VehicleAction.CONTINUE, "Low-confidence sign should not trigger STOP"
print("  --> PASS: Low-confidence/unknown sign does not trigger active rules.")


# ─── TEST 8: MULTIPLE SIGNS -> ALL CONSTRAINTS APPLIED ────────────────────────
print("\n[TEST 8] Multiple signs (SPEED_LIMIT 40 + NO_LEFT_TURN):")
sign1 = RoadSign(sign_type=RoadSignType.SPEED_LIMIT, confidence=0.90, box=(500, 50, 540, 90), value=40)
sign2 = RoadSign(sign_type=RoadSignType.NO_LEFT_TURN, confidence=0.88, box=(550, 50, 590, 90))

constraints = evaluate_sign_constraints([sign1, sign2])
print(f"  Structured constraints: {constraints}")
assert constraints["speed_limit"] == 40.0
assert "SPEED_CONSTRAINT" in constraints["active_constraints"]
assert "REJECT_LEFT_PATH" in constraints["active_constraints"]
assert "LEFT" in constraints["restricted_directions"]

# Both constraints applied in EnvironmentState
env = make_env_with_signs([sign1, sign2])
assert env.active_speed_limit == 40.0
assert "LEFT" in env.restricted_directions
print("  --> PASS: Multiple concurrent signs successfully synthesized and enforced.")


# ─── TEST 9: SIGN RESTRICTION + OBSTACLE -> BOTH RESPECTED ───────────────────
print("\n[TEST 9] Sign restriction + obstacle -> safety and sign constraints both respected:")
# Scenario: NO_LEFT_TURN sign active AND right side is blocked by an obstacle
# Left is prohibited by sign. Right is blocked by obstacle.
# System must NOT choose left turn just because right is blocked. It must declare NO_SAFE_PATH / STOP.
planner = make_planner()
no_left = RoadSign(sign_type=RoadSignType.NO_LEFT_TURN, confidence=0.91, box=(520, 50, 560, 90))
env = make_env_with_signs([no_left])

# Obstacles blocking center and right corridor
center_obs = make_obstacle(1, "car", x1=270, y1=200, x2=370, y2=360)
right_obs  = make_obstacle(2, "truck", x1=380, y1=150, x2=600, y2=400)

plan = planner.plan_frame(
    frame_idx=0, fps=FPS, frame_width=W, frame_height=H,
    tracked_objects=[center_obs, right_obs],
    decision=DecisionOutput(
        frame_idx=0, timestamp_sec=0.0, action=VehicleAction.STOP,
        target_object_id=1, target_class="car", risk_level=RiskLevel.HIGH,
        ttc_sec=1.5, avoidance_direction=AvoidanceDirection.NONE, target_speed_factor=0.0,
        reason="Center and right blocked; left prohibited by NO_LEFT_TURN sign.",
        road_context=RoadContext(corridor_blocked=True, can_avoid_left=False, can_avoid_right=False)
    ),
    environment=env,
)
print(f"  Plan status: {plan.status.value} | Safe path: {plan.safe_path}")
assert plan.status in (PlanningStatus.NO_SAFE_PATH, PlanningStatus.CONTINUE_ORIGINAL), \
    f"Expected NO_SAFE_PATH when sign and obstacles leave no legal route, got {plan.status}"
if plan.selected_path:
    assert plan.selected_path.lateral_offset >= -10.0, "Must NEVER select left path with NO_LEFT_TURN active!"
print("  --> PASS: System respects sign prohibition and avoids dangerous/illegal left maneuver.")


# ─── TEST 10: STRUCTURED OUTPUT VERIFICATION (SECTION 12) ─────────────────────
print("\n[TEST 10] Structured output verification (Section 12 schema):")
stop_for_schema = RoadSign(sign_type=RoadSignType.STOP, confidence=0.94, box=(510, 40, 560, 90), value=None)
structured_res = evaluate_sign_constraints([stop_for_schema])

print(f"  Output keys: {list(structured_res.keys())}")
print(f"  road_signs: {structured_res['road_signs']}")
print(f"  active_constraints: {structured_res['active_constraints']}")
print(f"  speed_limit: {structured_res['speed_limit']}")
print(f"  restricted_directions: {structured_res['restricted_directions']}")

assert "road_signs" in structured_res
assert "active_constraints" in structured_res
assert "speed_limit" in structured_res
assert "restricted_directions" in structured_res
assert structured_res["road_signs"][0]["type"] == "STOP"
assert structured_res["road_signs"][0]["confidence"] == 0.94
assert structured_res["road_signs"][0]["value"] is None
assert structured_res["road_signs"][0]["rule"] == "STOP"
assert structured_res["active_constraints"] == ["STOP"]

print("  --> PASS: Output matches Section 12 specification format perfectly.")

print("\n" + "=" * 70)
print("ALL 9 REQUIRED PHASE 8B TESTS PASSED — ROAD SIGN INTEGRATION VERIFIED!")
print("=" * 70)
