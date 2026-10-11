"""
Phase 5 Comprehensive Unit & Mathematical Validation Test.

Tests CollisionRiskAnalyzer against cars, pedestrians, bikes, and animals
across CRITICAL, HIGH, MEDIUM, and LOW risk scenarios.
"""
import sys
sys.path.insert(0, ".")

from app.tracking.tracker import TrackedObject
from app.tracking.motion import MotionData
from app.prediction.schemas import PredictionResult, PredictedPoint
from app.collision.schemas import RiskLevel, EgoVehicleState, CollisionRisk
from app.collision.analyzer import CollisionRiskAnalyzer

print("=" * 70)
print("PHASE 5: COLLISION RISK ANALYSIS UNIT & MATHEMATICAL VERIFICATION")
print("=" * 70)

analyzer = CollisionRiskAnalyzer(
    ttc_critical_thresh = 1.0,
    ttc_high_thresh     = 2.5,
    ttc_medium_thresh   = 4.5,
    corridor_half_width = 80.0,
    critical_distance   = 70.0,
)

frame_w = 640
frame_h = 480
fps = 30.0

ego_state = EgoVehicleState(
    origin_x            = 320.0,
    origin_y            = 480.0,
    corridor_half_width = 80.0,
    forward_speed_px    = 0.0,
)


# ─── SCENARIO 1: CAR Head-On Rapid Closing -> CRITICAL (TTC < 1.0s) ───────────
print("\n[TEST 1] Car on head-on collision course inside corridor:")
car_obj = TrackedObject(
    track_id   = 7,
    class_id   = 2,
    class_name = "car",
    confidence = 0.94,
    x1         = 280.0,
    y1         = 360.0,
    x2         = 360.0,
    y2         = 430.0,  # 50px in front of ego bumper (480 - 430)
    motion     = MotionData(
        dx               = 0.0,
        dy               = 12.0,   # Moving downwards towards ego at 12 px/frame
        pixel_speed      = 12.0,
        pixel_speed_sec  = 360.0,  # 12 px/f * 30fps
        direction_angle  = -90.0,
        direction        = "FORWARD",
        direction_symbol = "↓",
        direction_label  = "S",
        trajectory       = [(320.0, 380.0), (320.0, 395.0)],
    ),
    trajectory = [(320.0, 380.0), (320.0, 395.0)],
)

# Predicted points closing in rapidly
car_preds = {
    7: PredictionResult(
        track_id           = 7,
        class_name         = "car",
        method             = "kalman",
        current_center     = (320.0, 395.0),
        predicted_points   = [
            PredictedPoint(x=320.0, y=395.0 + i * 12.0, frame_offset=i, uncertainty_radius=8.0)
            for i in range(1, 15)
        ],
        is_reliable        = True,
        prediction_horizon = 30,
    )
}

risks1 = analyzer.analyze_frame(
    frame_idx       = 10,
    fps             = fps,
    frame_width     = frame_w,
    frame_height    = frame_h,
    tracked_objects = [car_obj],
    predictions     = car_preds,
    ego_state       = ego_state,
)

car_risk = risks1[7]
print(f"  Object: Car #7 | Level: {car_risk.risk_level.value} | TTC: {car_risk.ttc_sec}s | Dist: {car_risk.distance_px}px")
print(f"  Reason: {car_risk.reason}")
assert car_risk.risk_level == RiskLevel.CRITICAL, f"Expected CRITICAL, got {car_risk.risk_level}"
assert car_risk.ttc_sec is not None and car_risk.ttc_sec < 1.0, f"Expected TTC < 1.0s, got {car_risk.ttc_sec}"
print("  --> PASS: Immediate head-on car collision accurately rated CRITICAL.")


# ─── SCENARIO 2: PEDESTRIAN Crossing Corridor -> HIGH (TTC ~ 1.4s) ────────────
print("\n[TEST 2] Pedestrian crossing road corridor at T = 1.4s:")
ped_obj = TrackedObject(
    track_id   = 12,
    class_id   = 0,
    class_name = "person",
    confidence = 0.88,
    x1         = 105.0,
    y1         = 340.0,
    x2         = 135.0,
    y2         = 400.0,
    motion     = MotionData(
        dx               = 2.5,   # Moving rightward across road towards corridor edge
        dy               = 0.2,
        pixel_speed      = 2.51,
        pixel_speed_sec  = 75.3,  # 2.51 * 30fps
        direction_angle  = 0.0,
        direction        = "RIGHT",
        direction_symbol = "→",
        direction_label  = "E",
        trajectory       = [(110.0, 370.0), (120.0, 370.0)],
    ),
    trajectory = [(110.0, 370.0), (120.0, 370.0)],
)

# Enters ego corridor at frame 42 (1.40s at 30fps): 117 + 42 * 2.5 = 222px
ped_pred_pts = [
    PredictedPoint(x=117.0 + i * 2.5, y=370.0 + i * 0.2, frame_offset=i, uncertainty_radius=10.0 + i * 0.2)
    for i in range(1, 60)
]
ped_preds = {
    12: PredictionResult(
        track_id           = 12,
        class_name         = "person",
        method             = "kalman",
        current_center     = (120.0, 370.0),
        predicted_points   = ped_pred_pts,
        is_reliable        = True,
        prediction_horizon = 60,
    )
}

risks2 = analyzer.analyze_frame(
    frame_idx       = 20,
    fps             = fps,
    frame_width     = frame_w,
    frame_height    = frame_h,
    tracked_objects = [ped_obj],
    predictions     = ped_preds,
    ego_state       = ego_state,
)

ped_risk = risks2[12]
print(f"  Object: Person #12 | Level: {ped_risk.risk_level.value} | TTC: {ped_risk.ttc_sec}s | Dist: {ped_risk.distance_px}px")
print(f"  Reason: {ped_risk.reason}")
assert ped_risk.risk_level == RiskLevel.HIGH, f"Expected HIGH, got {ped_risk.risk_level}"
assert ped_risk.ttc_sec is not None and abs(ped_risk.ttc_sec - 1.4) <= 0.15, f"Expected TTC ~ 1.4s, got {ped_risk.ttc_sec}"
assert ped_risk.conflict_point is not None, "Expected conflict point to be set"
print("  --> PASS: Pedestrian crossing trajectory accurately detected with TTC = 1.4s (HIGH).")


# ─── SCENARIO 3: BICYCLE in Adjacent Lane with Moderate Approach -> MEDIUM ────
print("\n[TEST 3] Bicycle in adjacent lane with moderate approach (TTC ~ 3.0s):")
bike_obj = TrackedObject(
    track_id   = 15,
    class_id   = 1,
    class_name = "bicycle",
    confidence = 0.82,
    x1         = 165.0,
    y1         = 340.0,
    x2         = 195.0,
    y2         = 390.0,
    motion     = MotionData(
        dx               = 0.5,
        dy               = 0.2,
        pixel_speed      = 0.54,
        pixel_speed_sec  = 16.2,  # 0.54 * 30fps
        direction_angle  = -20.0,
        direction        = "DIAGONAL",
        direction_symbol = "↗",
        direction_label  = "E",
        trajectory       = [(175.0, 365.0), (180.0, 365.0)],
    ),
    trajectory = [(175.0, 365.0), (180.0, 365.0)],
)

# Reaches corridor boundary at frame ~85 (85 / 30 = 2.83s, between 2.5s and 4.5s -> MEDIUM)
bike_preds = {
    15: PredictionResult(
        track_id           = 15,
        class_name         = "bicycle",
        method             = "linear",
        current_center     = (180.0, 365.0),
        predicted_points   = [
            PredictedPoint(x=180.0 + i * 0.5, y=365.0 + i * 0.2, frame_offset=i, uncertainty_radius=12.0)
            for i in range(1, 100)
        ],
        is_reliable        = False,
        prediction_horizon = 100,
    )
}

risks3 = analyzer.analyze_frame(
    frame_idx       = 30,
    fps             = fps,
    frame_width     = frame_w,
    frame_height    = frame_h,
    tracked_objects = [bike_obj],
    predictions     = bike_preds,
    ego_state       = ego_state,
)

bike_risk = risks3[15]
print(f"  Object: Bike #15 | Level: {bike_risk.risk_level.value} | TTC: {bike_risk.ttc_sec}s | Dist: {bike_risk.distance_px}px")
print(f"  Reason: {bike_risk.reason}")
assert bike_risk.risk_level == RiskLevel.MEDIUM, f"Expected MEDIUM, got {bike_risk.risk_level}"
assert bike_risk.ttc_sec is not None and bike_risk.ttc_sec >= 2.5 and bike_risk.ttc_sec <= 4.5, f"Expected 2.5 <= TTC <= 4.5, got {bike_risk.ttc_sec}"
print("  --> PASS: Approaching bicycle classified as MEDIUM risk.")


# ─── SCENARIO 4: ANIMAL (COW) Near Lane Boundary -> MEDIUM (Cautious) ────────
print("\n[TEST 4] Animal (Cow) stationary near road boundary:")
cow_obj = TrackedObject(
    track_id   = 19,
    class_id   = 19,
    class_name = "cow",
    confidence = 0.91,
    x1         = 200.0,
    y1         = 250.0,
    x2         = 260.0,
    y2         = 330.0,
    motion     = MotionData(
        dx               = 0.05,
        dy               = 0.05,
        pixel_speed      = 0.07,
        pixel_speed_sec  = 2.1,   # 0.07 * 30fps
        direction_angle  = -45.0,
        direction        = "STATIONARY",
        direction_symbol = "■",
        direction_label  = "SE",
        trajectory       = [(230.0, 290.0), (230.0, 290.0)],
    ),
    trajectory = [(230.0, 290.0), (230.0, 290.0)],
)

risks4 = analyzer.analyze_frame(
    frame_idx       = 40,
    fps             = fps,
    frame_width     = frame_w,
    frame_height    = frame_h,
    tracked_objects = [cow_obj],
    predictions     = None,
    ego_state       = ego_state,
)

cow_risk = risks4[19]
print(f"  Object: Cow #19 | Level: {cow_risk.risk_level.value} | TTC: {cow_risk.ttc_sec} | Dist: {cow_risk.distance_px}px")
print(f"  Reason: {cow_risk.reason}")
assert cow_risk.risk_level == RiskLevel.MEDIUM, f"Expected MEDIUM for unpredictable animal near boundary, got {cow_risk.risk_level}"
print("  --> PASS: Cow near road corridor triggers lateral caution MEDIUM risk.")


# ─── SCENARIO 5: CAR Moving Away / Parallel Lane -> LOW ──────────────────────
print("\n[TEST 5] Car in adjacent lane moving away:")
safe_car = TrackedObject(
    track_id   = 25,
    class_id   = 2,
    class_name = "car",
    confidence = 0.95,
    x1         = 480.0,
    y1         = 100.0,
    x2         = 560.0,
    y2         = 160.0,
    motion     = MotionData(
        dx               = 1.0,
        dy               = -8.0,  # Moving forward away from ego
        pixel_speed      = 8.06,
        pixel_speed_sec  = 241.8, # 8.06 * 30fps
        direction_angle  = 97.0,
        direction        = "FORWARD",
        direction_symbol = "↑",
        direction_label  = "N",
        trajectory       = [(520.0, 140.0), (520.0, 130.0)],
    ),
    trajectory = [(520.0, 140.0), (520.0, 130.0)],
)

risks5 = analyzer.analyze_frame(
    frame_idx       = 50,
    fps             = fps,
    frame_width     = frame_w,
    frame_height    = frame_h,
    tracked_objects = [safe_car],
    predictions     = None,
    ego_state       = ego_state,
)

safe_risk = risks5[25]
print(f"  Object: Car #25 | Level: {safe_risk.risk_level.value} | TTC: {safe_risk.ttc_sec} | Dist: {safe_risk.distance_px}px")
print(f"  Reason: {safe_risk.reason}")
assert safe_risk.risk_level == RiskLevel.LOW, f"Expected LOW, got {safe_risk.risk_level}"
print("  --> PASS: Moving-away car correctly rated LOW risk.")


# ─── SCENARIO 6: TTC NOT FALSELY REPORTED when paths diverge ─────────────────
print("\n[TEST 6] Verifying TTC not falsely reported when object paths diverge:")
diverging_car = TrackedObject(
    track_id   = 30,
    class_id   = 2,
    class_name = "car",
    confidence = 0.90,
    x1         = 10.0,
    y1         = 200.0,
    x2         = 70.0,
    y2         = 260.0,
    motion     = MotionData(
        dx               = -5.0,  # Moving hard left, away from ego corridor
        dy               = 0.0,
        pixel_speed      = 5.0,
        pixel_speed_sec  = 150.0,
        direction_angle  = 180.0,
        direction        = "LEFT",
        direction_symbol = "←",
        direction_label  = "W",
        trajectory       = [(40.0, 230.0), (35.0, 230.0)],
    ),
    trajectory = [(40.0, 230.0), (35.0, 230.0)],
)

# Predicted to move further left, out of frame
diverge_preds = {
    30: PredictionResult(
        track_id           = 30,
        class_name         = "car",
        method             = "kalman",
        current_center     = (35.0, 230.0),
        predicted_points   = [
            PredictedPoint(x=35.0 - i * 5.0, y=230.0, frame_offset=i, uncertainty_radius=8.0)
            for i in range(1, 30)
        ],
        is_reliable        = True,
        prediction_horizon = 30,
    )
}

risks6 = analyzer.analyze_frame(
    frame_idx       = 60,
    fps             = fps,
    frame_width     = frame_w,
    frame_height    = frame_h,
    tracked_objects = [diverging_car],
    predictions     = diverge_preds,
    ego_state       = ego_state,
)

div_risk = risks6[30]
print(f"  Object: Car #30 | Level: {div_risk.risk_level.value} | TTC: {div_risk.ttc_sec} | Dist: {div_risk.distance_px}px")
print(f"  Reason: {div_risk.reason}")
assert div_risk.ttc_sec is None or div_risk.risk_level == RiskLevel.LOW, \
    f"Diverging car should have no TTC or LOW risk, got {div_risk.risk_level} / TTC={div_risk.ttc_sec}"
print("  --> PASS: Diverging car correctly shows no false TTC / LOW risk.")


# ─── VERIFY LOGGED RISK EVENTS ────────────────────────────────────────────────
events = analyzer.get_recorded_events()
print(f"\n[TEST 7] Verifying recorded risk events (logged: {len(events)} events):")
assert len(events) >= 3, f"Expected at least 3 risk events recorded, got {len(events)}"

for ev in events:
    print(f"  Time: {ev.timestamp_sec:.1f}s | Frame #{ev.frame_idx} | {ev.object_class.upper()} #{ev.object_id} | Risk: {ev.risk_level.value} | TTC: {ev.ttc_sec}s | Reason: {ev.reason[:55]}...")
    assert ev.object_id in (7, 12, 15, 19)
    assert ev.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL)
    assert len(ev.reason) > 5

print("\n" + "=" * 70)
print("ALL PHASE 5 COLLISION RISK TESTS PASSED SUCCESSFULLY!")
print("=" * 70)
