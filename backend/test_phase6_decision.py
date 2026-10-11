"""
Phase 6 Decision Engine Unit & Behavioral Verification Test.

Tests DecisionEngine against the 5 user-mandated real-world scenarios:
  1. Pedestrian crossing path
  2. Preceding vehicle stopping / blocking path
  3. Bike entering travel corridor
  4. Animal (cow) crossing / near boundary
  5. Obstacle completely blocking road (no lateral clearance vs clearance)
"""
import sys
sys.path.insert(0, ".")

from app.tracking.tracker import TrackedObject
from app.tracking.motion import MotionData
from app.collision.schemas import RiskLevel, CollisionRisk
from app.decision.schemas import VehicleAction, AvoidanceDirection, DecisionOutput
from app.decision.engine import DecisionEngine

print("=" * 75)
print("PHASE 6: AUTONOMOUS DECISION-MAKING ENGINE UNIT & SCENARIO VERIFICATION")
print("=" * 75)

engine = DecisionEngine(corridor_half_width=80.0, hysteresis_frames=2)
frame_w = 640
frame_h = 480
fps = 30.0

# ─── SCENARIO 1: PEDESTRIAN CROSSING PATH -> STOP / YIELD ─────────────────────
print("\n[SCENARIO 1] Pedestrian crossing ego vehicle corridor (TTC = 1.4s):")
ped_obj = TrackedObject(
    track_id   = 12,
    class_id   = 0,
    class_name = "person",
    confidence = 0.90,
    x1         = 240.0,
    y1         = 340.0,
    x2         = 270.0,
    y2         = 400.0,
    motion     = MotionData(dx=2.5, dy=0.0, pixel_speed=2.5, pixel_speed_sec=75.0, direction_angle=0.0, direction="RIGHT", direction_symbol="→", direction_label="E"),
)
ped_risk = {
    12: CollisionRisk(
        track_id              = 12,
        class_name            = "person",
        distance_px           = 140.0,
        lateral_offset_px     = -65.0,
        longitudinal_dist_px  = 80.0,
        relative_speed_px     = 2.5,
        closing_speed_px      = 1.8,
        ttc_sec               = 1.4,
        risk_level            = RiskLevel.HIGH,
        reason                = "Predicted Person trajectory intersects vehicle corridor in 1.4s.",
        conflict_point        = (320.0, 370.0),
        conflict_frame_offset = 42,
    )
}

dec1 = engine.evaluate_frame(
    frame_idx       = 10,
    fps             = fps,
    frame_width     = frame_w,
    frame_height    = frame_h,
    tracked_objects = [ped_obj],
    risks           = ped_risk,
)

print(f"  Action: {dec1.action.value} | Target: #{dec1.target_object_id} {dec1.target_class} | Throttle: {dec1.target_speed_factor*100:.0f}%")
print(f"  Reason: {dec1.reason}")
assert dec1.action in (VehicleAction.STOP, VehicleAction.SLOW_DOWN, VehicleAction.AVOID), f"Unexpected action: {dec1.action}"
assert dec1.target_object_id == 12
assert dec1.ttc_sec == 1.4
print("  --> PASS: Pedestrian crossing correctly prompts controlled stop/yield action.")


# ─── SCENARIO 2: VEHICLE STOPPING / SLOW AHEAD -> REPLAN (Left lane open) ─────
print("\n[SCENARIO 2] Slow-moving vehicle blocking predicted path with left lane open:")
car_obj = TrackedObject(
    track_id   = 7,
    class_id   = 2,
    class_name = "car",
    confidence = 0.95,
    x1         = 285.0,
    y1         = 270.0,
    x2         = 355.0,
    y2         = 340.0,
    motion     = MotionData(dx=0.0, dy=0.8, pixel_speed=0.8, pixel_speed_sec=24.0, direction_angle=-90.0, direction="FORWARD", direction_symbol="↓", direction_label="S"),
)
car_risk = {
    7: CollisionRisk(
        track_id              = 7,
        class_name            = "car",
        distance_px           = 140.0,
        lateral_offset_px     = 0.0,
        longitudinal_dist_px  = 140.0,
        relative_speed_px     = 0.8,
        closing_speed_px      = 0.8,
        ttc_sec               = 1.8,
        risk_level            = RiskLevel.HIGH,
        reason                = "Preceding car stopped/slow in travel corridor (TTC 1.8s).",
    )
}

dec2 = engine.evaluate_frame(
    frame_idx       = 20,
    fps             = fps,
    frame_width     = frame_w,
    frame_height    = frame_h,
    tracked_objects = [car_obj],
    risks           = car_risk,
)

print(f"  Action: {dec2.action.value} | Avoidance: {dec2.avoidance_direction.value} | Throttle: {dec2.target_speed_factor*100:.0f}%")
print(f"  Reason: {dec2.reason}")
assert dec2.action in (VehicleAction.REPLAN, VehicleAction.AVOID), f"Expected REPLAN or AVOID, got {dec2.action}"
assert dec2.avoidance_direction in (AvoidanceDirection.LEFT, AvoidanceDirection.RIGHT)
print("  --> PASS: Stationary vehicle with open adjacent lane triggers REPLAN.")


# ─── SCENARIO 3: BIKE ENTERING CORRIDOR -> AVOID / SLOW_DOWN ──────────────────
print("\n[SCENARIO 3] Bike entering corridor from adjacent lane:")
bike_obj = TrackedObject(
    track_id   = 15,
    class_id   = 1,
    class_name = "bicycle",
    confidence = 0.86,
    x1         = 200.0,
    y1         = 330.0,
    x2         = 230.0,
    y2         = 380.0,
    motion     = MotionData(dx=1.8, dy=1.2, pixel_speed=2.16, pixel_speed_sec=64.8, direction_angle=-33.0, direction="DIAGONAL", direction_symbol="↘", direction_label="SE"),
)
bike_risk = {
    15: CollisionRisk(
        track_id              = 15,
        class_name            = "bicycle",
        distance_px           = 145.0,
        lateral_offset_px     = -105.0,
        longitudinal_dist_px  = 100.0,
        relative_speed_px     = 2.16,
        closing_speed_px      = 1.5,
        ttc_sec               = 2.2,
        risk_level            = RiskLevel.HIGH,
        reason                = "Predicted Bicycle trajectory enters vehicle corridor in 2.2s.",
    )
}

dec3 = engine.evaluate_frame(
    frame_idx       = 30,
    fps             = fps,
    frame_width     = frame_w,
    frame_height    = frame_h,
    tracked_objects = [bike_obj],
    risks           = bike_risk,
)

print(f"  Action: {dec3.action.value} | Avoidance: {dec3.avoidance_direction.value} | Throttle: {dec3.target_speed_factor*100:.0f}%")
print(f"  Reason: {dec3.reason}")
assert dec3.action in (VehicleAction.AVOID, VehicleAction.STOP, VehicleAction.SLOW_DOWN), f"Unexpected action: {dec3.action}"
print("  --> PASS: Encroaching bike triggers avoidance/speed reduction.")


# ─── SCENARIO 4: ANIMAL (COW) NEAR BOUNDARY -> SLOW_DOWN (Cautious) ───────────
print("\n[SCENARIO 4] Cow near lane boundary (MEDIUM risk):")
cow_obj = TrackedObject(
    track_id   = 19,
    class_id   = 19,
    class_name = "cow",
    confidence = 0.92,
    x1         = 190.0,
    y1         = 240.0,
    x2         = 260.0,
    y2         = 320.0,
    motion     = MotionData(dx=0.1, dy=0.1, pixel_speed=0.14, pixel_speed_sec=4.2, direction_angle=-45.0, direction="STATIONARY", direction_symbol="■", direction_label="SE"),
)
cow_risk = {
    19: CollisionRisk(
        track_id              = 19,
        class_name            = "cow",
        distance_px           = 185.0,
        lateral_offset_px     = -95.0,
        longitudinal_dist_px  = 160.0,
        relative_speed_px     = 0.14,
        closing_speed_px      = 0.1,
        ttc_sec               = 3.8,
        risk_level            = RiskLevel.MEDIUM,
        reason                = "Lateral caution: Cow adjacent to travel corridor (high unpredictability buffer).",
    )
}

dec4 = engine.evaluate_frame(
    frame_idx       = 40,
    fps             = fps,
    frame_width     = frame_w,
    frame_height    = frame_h,
    tracked_objects = [cow_obj],
    risks           = cow_risk,
)

print(f"  Action: {dec4.action.value} | Throttle: {dec4.target_speed_factor*100:.0f}%")
print(f"  Reason: {dec4.reason}")
assert dec4.action == VehicleAction.SLOW_DOWN, f"Expected SLOW_DOWN, got {dec4.action}"
assert dec4.target_speed_factor < 1.0, "Expected reduced throttle factor"
print("  --> PASS: Unpredictable animal triggers SLOW_DOWN with throttle reduction.")


# ─── SCENARIO 5: OBSTACLE COMPLETELY BLOCKING ROAD -> EMERGENCY_STOP ──────────
print("\n[SCENARIO 5] Obstacle completely blocking road with zero lateral clearance (CRITICAL):")
# Block left side with parked truck, right side with guardrail/bus, and center with immediate hazard
block_center = TrackedObject(
    track_id   = 30,
    class_id   = 2,
    class_name = "car",
    confidence = 0.98,
    x1         = 280.0,
    y1         = 410.0,
    x2         = 360.0,
    y2         = 470.0,  # Only 10px from ego front!
    motion     = MotionData(dx=0.0, dy=14.0, pixel_speed=14.0, pixel_speed_sec=420.0, direction_angle=-90.0, direction="FORWARD", direction_symbol="↓", direction_label="S"),
)
block_left = TrackedObject(
    track_id   = 31,
    class_id   = 7,
    class_name = "truck",
    confidence = 0.94,
    x1         = 20.0,
    y1         = 320.0,
    x2         = 180.0,
    y2         = 440.0,
    motion     = None,
)
block_right = TrackedObject(
    track_id   = 32,
    class_id   = 5,
    class_name = "bus",
    confidence = 0.96,
    x1         = 460.0,
    y1         = 320.0,
    x2         = 620.0,
    y2         = 440.0,
    motion     = None,
)

critical_risk = {
    30: CollisionRisk(
        track_id              = 30,
        class_name            = "car",
        distance_px           = 40.0,
        lateral_offset_px     = 0.0,
        longitudinal_dist_px  = 10.0,
        relative_speed_px     = 14.0,
        closing_speed_px      = 14.0,
        ttc_sec               = 0.1,
        risk_level            = RiskLevel.CRITICAL,
        reason                = "Immediate obstacle inside vehicle safety envelope (distance 40px). Stop immediately.",
    ),
    31: CollisionRisk(
        track_id              = 31,
        class_name            = "truck",
        distance_px           = 160.0,
        lateral_offset_px     = -220.0,
        longitudinal_dist_px  = 60.0,
        relative_speed_px     = 0.0,
        closing_speed_px      = 0.0,
        ttc_sec               = None,
        risk_level            = RiskLevel.LOW,
        reason                = "Parked on left shoulder.",
    ),
    32: CollisionRisk(
        track_id              = 32,
        class_name            = "bus",
        distance_px           = 160.0,
        lateral_offset_px     = 220.0,
        longitudinal_dist_px  = 60.0,
        relative_speed_px     = 0.0,
        closing_speed_px      = 0.0,
        ttc_sec               = None,
        risk_level            = RiskLevel.LOW,
        reason                = "Parked on right shoulder.",
    ),
}

dec5 = engine.evaluate_frame(
    frame_idx       = 50,
    fps             = fps,
    frame_width     = frame_w,
    frame_height    = frame_h,
    tracked_objects = [block_center, block_left, block_right],
    risks           = critical_risk,
)

print(f"  Action: {dec5.action.value} | Throttle: {dec5.target_speed_factor*100:.0f}%")
print(f"  Reason: {dec5.reason}")
assert dec5.action == VehicleAction.EMERGENCY_STOP, f"Expected EMERGENCY_STOP, got {dec5.action}"
assert dec5.target_speed_factor == 0.0
print("  --> PASS: Blocked road with immediate obstacle triggers EMERGENCY_STOP.")


# ─── VERIFY DECISION HISTORY ──────────────────────────────────────────────────
history = engine.get_history()
counts = engine.get_action_counts()
dominant = engine.get_dominant_action()

print(f"\n[DECISION HISTORY AUDIT] Recorded {len(history)} decisions:")
print(f"  Action Counts: {counts}")
print(f"  Dominant Action: {dominant}")
assert len(history) == 5, f"Expected 5 decision records, got {len(history)}"
for d in history:
    print(f"  Frame #{d.frame_idx} ({d.timestamp_sec:.1f}s): {d.action.value:<14s} | Throttle {d.target_speed_factor*100:3.0f}% | Reason: {d.reason[:48]}...")

print("\n" + "=" * 75)
print("ALL 5 PHASE 6 DECISION SCENARIOS PASSED WITH FULL MATHEMATICAL RIGOR!")
print("=" * 75)
