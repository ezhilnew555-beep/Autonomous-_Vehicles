"""
test_phase7_planner.py - Unit & Scenario Test Suite for Adaptive Path Planning (Phase 7)

Tests:
1. Nominal Path Clear: Empty corridor -> CONTINUE_ORIGINAL
2. Stationary Vehicle: Vehicle directly ahead in corridor -> REPLANNED_AVOID_LEFT or RIGHT
3. Crossing Pedestrian: Pedestrian trajectory crossing vehicle path -> Replan evasion
4. Roadside Animal: Cow near shoulder with inflated safety envelope -> Clearance verified
5. Impassable Road: Full width blocked -> NO_SAFE_PATH
6. Latency Benchmark: 100 planning cycles execute in < 15ms per cycle.
"""

import sys
import time
from pathlib import Path

# Add backend to path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.tracking.tracker import TrackedObject
from app.prediction.schemas import PredictionResult, PredictedPoint
from app.planning.schemas import PlanningStatus
from app.planning.planner import AdaptivePathPlanner


def make_track(track_id: int, cls_name: str, x1: float, y1: float, x2: float, y2: float) -> TrackedObject:
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


def make_prediction(track_id: int, cls_name: str, cx: float, cy: float, points=None, uncertainty=15.0) -> PredictionResult:
    if points is None:
        pts = [PredictedPoint(x=cx, y=cy, frame_offset=i, uncertainty_radius=uncertainty) for i in range(1, 16)]
    else:
        pts = [PredictedPoint(x=px, y=py, frame_offset=i, uncertainty_radius=uncertainty) for i, (px, py) in enumerate(points, 1)]
    return PredictionResult(
        track_id           = track_id,
        class_name         = cls_name,
        method             = "kalman",
        current_center     = (cx, cy),
        predicted_points   = pts,
        is_reliable        = True,
        prediction_horizon = len(pts),
    )


def test_clear_road():
    """Scenario 1: Clear road corridor -> CONTINUE_ORIGINAL"""
    planner = AdaptivePathPlanner(lookahead_distance=340.0)
    
    res = planner.plan_frame(
        frame_idx       = 1,
        fps             = 30.0,
        frame_width     = 1280,
        frame_height    = 720,
        tracked_objects = [],
        predictions     = {},
        decision        = None,
    )
    
    assert res.status == PlanningStatus.CONTINUE_ORIGINAL, f"Expected CONTINUE_ORIGINAL, got {res.status}"
    assert len(res.original_path) > 0
    assert len(res.safe_path) > 0
    assert not res.needs_replanning
    assert len(planner.get_replanning_events()) == 0
    print("  [PASS] Scenario 1: Clear road maintains nominal path.")


def test_stationary_vehicle_blocking():
    """Scenario 2: Car stopped in vehicle corridor -> Replan evasion"""
    planner = AdaptivePathPlanner(lookahead_distance=340.0)
    
    # Car stopped directly ahead on centerline: x=640, y=480
    car = make_track(1, "car", 580, 420, 700, 540)
    pred = make_prediction(1, "car", 640, 480)
    
    res = planner.plan_frame(
        frame_idx       = 10,
        fps             = 30.0,
        frame_width     = 1280,
        frame_height    = 720,
        tracked_objects = [car],
        predictions     = {1: pred},
        decision        = None,
    )
    
    assert res.status in (PlanningStatus.REPLANNED_AVOID_LEFT, PlanningStatus.REPLANNED_AVOID_RIGHT), (
        f"Expected avoidance replan, got {res.status}"
    )
    assert res.needs_replanning is True
    assert res.selected_path is not None
    assert res.selected_path.is_collision_free is True
    assert res.selected_path.lateral_offset != 0.0
    print(f"  [PASS] Scenario 2: Stationary vehicle triggered {res.status.value} (offset {res.selected_path.lateral_offset:.1f}px).")


def test_crossing_pedestrian():
    """Scenario 3: Pedestrian crossing across vehicle path"""
    planner = AdaptivePathPlanner(lookahead_distance=340.0)
    
    # Pedestrian crossing from left to right across centerline
    ped = make_track(2, "person", 490, 450, 530, 530)
    cross_pts = [(510 + i * 15, 490) for i in range(12)]
    pred = make_prediction(2, "person", 510, 490, points=cross_pts, uncertainty=20.0)
    
    res = planner.plan_frame(
        frame_idx       = 25,
        fps             = 30.0,
        frame_width     = 1280,
        frame_height    = 720,
        tracked_objects = [ped],
        predictions     = {2: pred},
        decision        = None,
    )
    
    # Must find a collision-free path or successfully replan
    assert res.status != PlanningStatus.NO_SAFE_PATH
    assert res.selected_path is not None
    assert res.selected_path.is_collision_free is True
    print(f"  [PASS] Scenario 3: Crossing pedestrian safely navigated with status {res.status.value}.")


def test_roadside_animal_safety_margin():
    """Scenario 4: Stray animal near road shoulder"""
    planner = AdaptivePathPlanner(lookahead_distance=340.0)
    
    # Cow on right side: x=740, y=490
    cow = make_track(3, "cow", 690, 440, 790, 540)
    pred = make_prediction(3, "cow", 740, 490, uncertainty=25.0)
    
    res = planner.plan_frame(
        frame_idx       = 40,
        fps             = 30.0,
        frame_width     = 1280,
        frame_height    = 720,
        tracked_objects = [cow],
        predictions     = {3: pred},
        decision        = None,
    )
    
    assert res.selected_path is not None
    assert res.selected_path.is_collision_free is True
    # If replanned, must steer to the left away from the right-side animal
    if res.status != PlanningStatus.CONTINUE_ORIGINAL:
        assert res.selected_path.lateral_offset < 0.0, "Should steer left away from right-side animal"
    print(f"  [PASS] Scenario 4: Roadside animal handled safely with status {res.status.value}.")


def test_completely_blocked_road():
    """Scenario 5: Wall of obstacles blocking entire road -> NO_SAFE_PATH"""
    planner = AdaptivePathPlanner(lookahead_distance=340.0)
    
    # Obstacles spanning the entire road width
    tracks = []
    preds = {}
    for i, x in enumerate([200, 350, 500, 640, 780, 920, 1080]):
        tid = 10 + i
        t = make_track(tid, "truck", x - 90, 420, x + 90, 560)
        p = make_prediction(tid, "truck", x, 490)
        tracks.append(t)
        preds[tid] = p
    
    res = planner.plan_frame(
        frame_idx       = 60,
        fps             = 30.0,
        frame_width     = 1280,
        frame_height    = 720,
        tracked_objects = tracks,
        predictions     = preds,
        decision        = None,
    )
    
    assert res.status == PlanningStatus.NO_SAFE_PATH, f"Expected NO_SAFE_PATH, got {res.status}"
    assert res.safe_path == []
    print("  [PASS] Scenario 5: Completely blocked road correctly yielded NO_SAFE_PATH.")


def test_latency_benchmark():
    """Scenario 6: Real-time computational latency benchmark (< 15ms)"""
    planner = AdaptivePathPlanner(lookahead_distance=340.0)
    
    tracks = [
        make_track(1, "car", 580, 420, 700, 540),
        make_track(2, "person", 490, 450, 530, 530),
        make_track(3, "motorcycle", 710, 450, 750, 520),
        make_track(4, "auto_rickshaw", 430, 420, 490, 510),
    ]
    preds = {t.track_id: make_prediction(t.track_id, t.class_name, (t.x1 + t.x2) / 2.0, (t.y1 + t.y2) / 2.0) for t in tracks}
    
    latencies = []
    for i in range(100):
        t0 = time.perf_counter()
        res = planner.plan_frame(
            frame_idx       = i + 1,
            fps             = 30.0,
            frame_width     = 1280,
            frame_height    = 720,
            tracked_objects = tracks,
            predictions     = preds,
            decision        = None,
        )
        t_elapsed = (time.perf_counter() - t0) * 1000.0
        latencies.append(t_elapsed)
    
    avg_lat = sum(latencies) / len(latencies)
    max_lat = max(latencies)
    print(f"  [PASS] Scenario 6: Latency Benchmark across 100 cycles: Avg = {avg_lat:.2f} ms, Max = {max_lat:.2f} ms (Target < 15 ms).")
    assert avg_lat < 15.0, f"Average planning latency {avg_lat:.2f} ms exceeded 15.0 ms threshold!"


def run_all_tests():
    print("============================================================")
    print("  Eareyes 2k26 - Phase 7: Adaptive Path Planning Test Suite")
    print("============================================================")
    
    test_clear_road()
    test_stationary_vehicle_blocking()
    test_crossing_pedestrian()
    test_roadside_animal_safety_margin()
    test_completely_blocked_road()
    test_latency_benchmark()
    
    print("============================================================")
    print("  ALL 6 SCENARIOS & BENCHMARKS PASSED SUCCESSFULLY!")
    print("============================================================")


if __name__ == "__main__":
    run_all_tests()
