"""
Phase 6 — Spec Compliance Test: All 7 Scenarios from the User Requirement.

Tests exactly the scenarios specified:
  1. Empty/clear road → CONTINUE
  2. Object far away → CONTINUE
  3. Object approaching but enough time → SLOW_DOWN
  4. Object blocking current path → REPLAN / AVOID
  5. Immediate collision risk → EMERGENCY_STOP
  6. Multiple obstacles → most immediate safety constraint chosen
  7. Uncertain prediction → conservative behavior (SLOW_DOWN/STOP)
"""
import sys
sys.path.insert(0, ".")

from app.tracking.tracker import TrackedObject
from app.tracking.motion import MotionData
from app.collision.schemas import RiskLevel, CollisionRisk, EgoVehicleState
from app.collision.analyzer import CollisionRiskAnalyzer
from app.decision.schemas import VehicleAction, AvoidanceDirection
from app.decision.engine import DecisionEngine

print("=" * 70)
print("PHASE 6 SPEC COMPLIANCE — 7 REQUIRED SCENARIOS")
print("=" * 70)

W, H, FPS = 640, 480, 30.0

def make_motion(dx, dy, direction="FORWARD", symbol="↑"):
    speed = (dx**2 + dy**2) ** 0.5
    return MotionData(
        dx=dx, dy=dy,
        pixel_speed=speed,
        pixel_speed_sec=speed * FPS,
        direction_angle=0.0,
        direction=direction,
        direction_symbol=symbol,
        direction_label="N",
    )

def make_engine():
    return DecisionEngine(corridor_half_width=80.0, hysteresis_frames=2)


# ─── SCENARIO 1: EMPTY / CLEAR ROAD → CONTINUE ───────────────────────────────
print("\n[SCENARIO 1] Empty/clear road (no objects, no risks):")
engine = make_engine()
dec = engine.evaluate_frame(
    frame_idx=0, fps=FPS, frame_width=W, frame_height=H,
    tracked_objects=[], risks={},
)
print(f"  Action: {dec.action.value} | Speed Factor: {dec.target_speed_factor:.0%}")
print(f"  Reason: {dec.reason}")
assert dec.action == VehicleAction.CONTINUE, f"Expected CONTINUE, got {dec.action}"
assert dec.target_speed_factor == 1.0
print("  --> PASS: Empty road correctly yields CONTINUE at full speed.")


# ─── SCENARIO 2: OBJECT FAR AWAY → CONTINUE ──────────────────────────────────
print("\n[SCENARIO 2] Car very far away (no corridor threat):")
engine = make_engine()
far_car = TrackedObject(
    track_id=1, class_id=2, class_name="car", confidence=0.92,
    x1=480.0, y1=50.0, x2=560.0, y2=110.0,   # Far right, far up
    motion=make_motion(1.0, -5.0, "FORWARD", "↑"),
    trajectory=[(520.0, 80.0)],
)
far_risk = {
    1: CollisionRisk(
        track_id=1, class_name="car",
        distance_px=420.0, lateral_offset_px=200.0, longitudinal_dist_px=370.0,
        relative_speed_px=5.0, closing_speed_px=-2.0,   # moving away
        ttc_sec=None, risk_level=RiskLevel.LOW,
        reason="Safe separation: Car moving away outside corridor.",
    )
}
dec = engine.evaluate_frame(
    frame_idx=10, fps=FPS, frame_width=W, frame_height=H,
    tracked_objects=[far_car], risks=far_risk,
)
print(f"  Action: {dec.action.value} | TTC: {dec.ttc_sec}")
print(f"  Reason: {dec.reason}")
assert dec.action == VehicleAction.CONTINUE, f"Expected CONTINUE, got {dec.action}"
print("  --> PASS: Distant non-threatening car correctly yields CONTINUE.")


# ─── SCENARIO 3: OBJECT APPROACHING WITH ENOUGH TIME → SLOW_DOWN ─────────────
print("\n[SCENARIO 3] Object approaching but enough time available (TTC ~ 3.5s):")
engine = make_engine()
approach_car = TrackedObject(
    track_id=5, class_id=2, class_name="car", confidence=0.94,
    x1=285.0, y1=230.0, x2=355.0, y2=290.0,
    motion=make_motion(0.0, 3.5, "FORWARD", "↓"),
    trajectory=[(320.0, 260.0)],
)
approach_risk = {
    5: CollisionRisk(
        track_id=5, class_name="car",
        distance_px=190.0, lateral_offset_px=0.0, longitudinal_dist_px=190.0,
        relative_speed_px=3.5, closing_speed_px=3.5,
        ttc_sec=3.8,    # ~3.8s — MEDIUM (2.5–4.5s range)
        risk_level=RiskLevel.MEDIUM,
        reason="Potential conflict: predicted trajectory approaches vehicle corridor (TTC ~ 3.8s).",
    )
}
dec = engine.evaluate_frame(
    frame_idx=20, fps=FPS, frame_width=W, frame_height=H,
    tracked_objects=[approach_car], risks=approach_risk,
)
print(f"  Action: {dec.action.value} | TTC: {dec.ttc_sec} | Speed: {dec.target_speed_factor:.0%}")
print(f"  Reason: {dec.reason}")
assert dec.action == VehicleAction.SLOW_DOWN, f"Expected SLOW_DOWN, got {dec.action}"
assert dec.target_speed_factor < 1.0
print("  --> PASS: Approaching object with adequate TTC yields SLOW_DOWN.")


# ─── SCENARIO 4: OBJECT BLOCKING PATH (LATERAL CLEAR) → REPLAN ───────────────
print("\n[SCENARIO 4] Stationary truck blocking corridor — left lane open (HIGH):")
engine = make_engine()
blocked_truck = TrackedObject(
    track_id=8, class_id=7, class_name="truck", confidence=0.97,
    x1=285.0, y1=290.0, x2=365.0, y2=380.0,  # centered in corridor
    motion=make_motion(0.0, 0.1, "STATIONARY", "■"),
    trajectory=[(325.0, 335.0)],
)
blocked_risk = {
    8: CollisionRisk(
        track_id=8, class_name="truck",
        distance_px=105.0, lateral_offset_px=5.0, longitudinal_dist_px=100.0,
        relative_speed_px=0.1, closing_speed_px=0.1,
        ttc_sec=2.0, risk_level=RiskLevel.HIGH,
        reason="Stationary truck blocks travel corridor (TTC 2.0s).",
    )
}
dec = engine.evaluate_frame(
    frame_idx=30, fps=FPS, frame_width=W, frame_height=H,
    tracked_objects=[blocked_truck], risks=blocked_risk,
)
print(f"  Action: {dec.action.value} | Avoidance: {dec.avoidance_direction.value}")
print(f"  Reason: {dec.reason}")
assert dec.action in (VehicleAction.REPLAN, VehicleAction.AVOID, VehicleAction.STOP), \
    f"Expected REPLAN/AVOID/STOP, got {dec.action}"
print(f"  --> PASS: Blocked path yields {dec.action.value} with {dec.avoidance_direction.value} avoidance.")


# ─── SCENARIO 5: IMMEDIATE COLLISION RISK → EMERGENCY_STOP ───────────────────
print("\n[SCENARIO 5] Immediate collision — car head-on inside safety envelope:")
engine = make_engine()
crash_car = TrackedObject(
    track_id=15, class_id=2, class_name="car", confidence=0.99,
    x1=290.0, y1=420.0, x2=360.0, y2=475.0,  # 5px from ego bumper
    motion=make_motion(0.0, 18.0, "FORWARD", "↓"),
    trajectory=[(325.0, 448.0)],
)
crash_risk = {
    15: CollisionRisk(
        track_id=15, class_name="car",
        distance_px=35.0, lateral_offset_px=0.0, longitudinal_dist_px=5.0,
        relative_speed_px=18.0, closing_speed_px=18.0,
        ttc_sec=0.1, risk_level=RiskLevel.CRITICAL,
        reason="Immediate obstacle inside vehicle safety envelope (distance 35px). Stop immediately.",
    )
}
dec = engine.evaluate_frame(
    frame_idx=40, fps=FPS, frame_width=W, frame_height=H,
    tracked_objects=[crash_car], risks=crash_risk,
)
print(f"  Action: {dec.action.value} | TTC: {dec.ttc_sec}s | Speed: {dec.target_speed_factor:.0%}")
print(f"  Reason: {dec.reason}")
assert dec.action == VehicleAction.EMERGENCY_STOP, f"Expected EMERGENCY_STOP, got {dec.action}"
assert dec.target_speed_factor == 0.0
print("  --> PASS: Immediate collision correctly yields EMERGENCY_STOP.")


# ─── SCENARIO 6: MULTIPLE OBSTACLES → MOST IMMEDIATE THREAT CHOSEN ───────────
print("\n[SCENARIO 6] Multiple mixed threats — engine selects most critical:")
engine = make_engine()

# Car #7: LOW (far away, moving away)
obj_low = TrackedObject(
    track_id=7, class_id=2, class_name="car", confidence=0.93,
    x1=480.0, y1=80.0, x2=550.0, y2=130.0,
    motion=make_motion(2.0, -6.0, "FORWARD", "↑"),
    trajectory=[(515.0, 105.0)],
)
# Bike #12: MEDIUM (adjacent, approaching slowly)
obj_med = TrackedObject(
    track_id=12, class_id=1, class_name="bicycle", confidence=0.85,
    x1=175.0, y1=310.0, x2=205.0, y2=360.0,
    motion=make_motion(0.8, 0.3, "RIGHT", "→"),
    trajectory=[(190.0, 335.0)],
)
# Cow #21: HIGH (crossing path, TTC 1.5s)
obj_high = TrackedObject(
    track_id=21, class_id=19, class_name="cow", confidence=0.91,
    x1=100.0, y1=320.0, x2=170.0, y2=390.0,
    motion=make_motion(3.0, 0.2, "RIGHT", "→"),
    trajectory=[(135.0, 355.0)],
)
# Person #25: LOW (far left, stationary)
obj_low2 = TrackedObject(
    track_id=25, class_id=0, class_name="person", confidence=0.88,
    x1=30.0, y1=200.0, x2=60.0, y2=260.0,
    motion=make_motion(0.1, 0.0, "STATIONARY", "■"),
    trajectory=[(45.0, 230.0)],
)

multi_risks = {
    7:  CollisionRisk(track_id=7,  class_name="car",     distance_px=390.0, lateral_offset_px=195.0, longitudinal_dist_px=350.0, relative_speed_px=6.3, closing_speed_px=-3.0, ttc_sec=None,  risk_level=RiskLevel.LOW,    reason="Moving away."),
    12: CollisionRisk(track_id=12, class_name="bicycle", distance_px=155.0, lateral_offset_px=-130.0,longitudinal_dist_px=120.0, relative_speed_px=0.9, closing_speed_px=0.6,  ttc_sec=3.2,   risk_level=RiskLevel.MEDIUM, reason="Potential conflict TTC 3.2s."),
    21: CollisionRisk(track_id=21, class_name="cow",     distance_px=210.0, lateral_offset_px=-185.0,longitudinal_dist_px=90.0,  relative_speed_px=3.0, closing_speed_px=2.8,  ttc_sec=1.5,   risk_level=RiskLevel.HIGH,   reason="Cow crosses corridor in 1.5s.", conflict_point=(320.0, 340.0), conflict_frame_offset=45),
    25: CollisionRisk(track_id=25, class_name="person",  distance_px=340.0, lateral_offset_px=-275.0,longitudinal_dist_px=220.0, relative_speed_px=0.1, closing_speed_px=-0.1, ttc_sec=None,  risk_level=RiskLevel.LOW,    reason="Stationary pedestrian far left."),
}

dec = engine.evaluate_frame(
    frame_idx=50, fps=FPS, frame_width=W, frame_height=H,
    tracked_objects=[obj_low, obj_med, obj_high, obj_low2],
    risks=multi_risks,
)
print(f"  Action: {dec.action.value} | Primary Threat: #{dec.target_object_id} {dec.target_class}")
print(f"  Risk Level: {dec.risk_level.value if dec.risk_level else '—'} | TTC: {dec.ttc_sec}")
print(f"  Reason: {dec.reason}")

# Must select the cow (#21) as highest priority threat
assert dec.target_object_id == 21, f"Expected cow #21 as primary threat, got #{dec.target_object_id}"
assert dec.action not in (VehicleAction.CONTINUE,), f"Should not CONTINUE with HIGH risk cow in path"
print("  --> PASS: Engine correctly identifies Cow #21 (HIGH/TTC=1.5s) as most critical threat.")


# ─── SCENARIO 7: UNCERTAIN / LOW-CONFIDENCE PREDICTION → CONSERVATIVE ────────
print("\n[SCENARIO 7] Uncertain detection — conservative SLOW_DOWN/STOP behavior:")
engine = make_engine()

# Low-confidence animal detection near corridor
uncertain_obj = TrackedObject(
    track_id=33, class_id=19, class_name="cow", confidence=0.41,  # low confidence
    x1=230.0, y1=280.0, x2=300.0, y2=360.0,
    motion=make_motion(1.5, 0.5, "DIAGONAL", "↗"),
    trajectory=[(265.0, 320.0)],
)
# Still classified as MEDIUM risk due to proximity (conservative)
uncertain_risk = {
    33: CollisionRisk(
        track_id=33, class_name="cow",
        distance_px=165.0, lateral_offset_px=-55.0, longitudinal_dist_px=120.0,
        relative_speed_px=1.6, closing_speed_px=1.2,
        ttc_sec=3.0,
        risk_level=RiskLevel.MEDIUM,
        reason="Lateral caution: Cow adjacent to travel corridor (high unpredictability buffer).",
    )
}
dec = engine.evaluate_frame(
    frame_idx=60, fps=FPS, frame_width=W, frame_height=H,
    tracked_objects=[uncertain_obj], risks=uncertain_risk,
)
print(f"  Action: {dec.action.value} | Target: #{dec.target_object_id} {dec.target_class} (conf: {uncertain_obj.confidence:.0%})")
print(f"  Speed Factor: {dec.target_speed_factor:.0%} | TTC: {dec.ttc_sec}s")
print(f"  Reason: {dec.reason}")
assert dec.action != VehicleAction.CONTINUE, \
    "Should NOT continue with a low-confidence uncertain animal near corridor"
assert dec.target_speed_factor < 1.0, "Should reduce speed under uncertainty"
print("  --> PASS: Uncertain low-confidence animal detection yields conservative deceleration.")


# ─── VERIFY DECISION HISTORY ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("DECISION HISTORY (last engine instance):")
for d in engine.get_history():
    tid = f"#{d.target_object_id}" if d.target_object_id else "—"
    ttc = f"{d.ttc_sec:.1f}s" if d.ttc_sec else "—"
    print(f"  {d.timestamp_sec:.2f}s → {d.action.value:<14} Threat={tid:<6} TTC={ttc:<6} | {d.reason[:50]}...")

print("\n" + "=" * 70)
print("ALL 7 PHASE 6 SPEC SCENARIOS PASSED — DECISION ENGINE VERIFIED!")
print("=" * 70)
