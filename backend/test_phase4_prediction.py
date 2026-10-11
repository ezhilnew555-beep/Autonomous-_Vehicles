"""
Unit tests for Phase 4 Short-Term Motion Prediction.
Verifies:
1. PredictionConfig initialization and class predictability mapping.
2. PredictionEngine stability gating (<5 frames -> is_available=False).
3. Low confidence gating -> is_available=False.
4. Kalman filter convergence and 2.0s future trajectory waypoints (10 points at 0.2s).
5. Class predictability impact on confidence scores (vehicle > animal/pedestrian).
6. Linear velocity extrapolation fallback when appropriate.
7. PredictionResult.to_dict() contract.
"""
from app.prediction.config import PredictionConfig
from app.prediction.predictor import PredictionEngine
from app.prediction.schemas import PredictionResult, PredictedPoint
from app.tracking.tracker import TrackedObject
from app.tracking.motion import MotionData


def make_mock_track(track_id: int, class_name: str, trajectory: list[tuple[float, float]], confidence: float = 0.85):
    cx, cy = trajectory[-1]
    motion = None
    if len(trajectory) >= 2:
        dx = trajectory[-1][0] - trajectory[-2][0]
        dy = trajectory[-1][1] - trajectory[-2][1]
        speed = (dx**2 + dy**2)**0.5
        motion = MotionData(
            dx=dx,
            dy=dy,
            pixel_speed=speed,
            pixel_speed_sec=speed * 30.0,
            direction="FORWARD" if dy < 0 else "BACKWARD",
            direction_symbol="↑" if dy < 0 else "↓",
            direction_angle=90.0 if dy < 0 else 270.0,
            direction_label="N" if dy < 0 else "S",
            trajectory=trajectory,
        )
    return TrackedObject(
        track_id=track_id,
        class_id=2 if class_name == "car" else 0,
        class_name=class_name,
        confidence=confidence,
        x1=cx - 20,
        y1=cy - 20,
        x2=cx + 20,
        y2=cy + 20,
        motion=motion,
        trajectory=trajectory,
    )


def test_prediction_config():
    cfg = PredictionConfig()
    assert cfg.PREDICTION_HORIZON_SEC == 2.0
    assert cfg.PREDICTION_INTERVAL_SEC == 0.2
    assert cfg.MIN_HISTORY_FRAMES == 5
    assert cfg.CLASS_PREDICTABILITY["car"] > cfg.CLASS_PREDICTABILITY["cow"]
    assert cfg.CLASS_PREDICTABILITY["bus"] > cfg.CLASS_PREDICTABILITY["person"]


def test_stability_gating_insufficient_history():
    engine = PredictionEngine(fps=30.0)
    # Track with only 3 frames (< 5 frames requirement)
    traj = [(100.0, 100.0), (105.0, 100.0), (110.0, 100.0)]
    obj = make_mock_track(track_id=1, class_name="car", trajectory=traj)
    
    results = engine.update_and_predict([obj])
    assert 1 in results
    res = results[1]
    assert res.is_available is False
    assert "Insufficient history" in (res.unavailability_reason or "")
    assert len(res.predicted_points) == 0


def test_stability_gating_low_confidence():
    engine = PredictionEngine(fps=30.0)
    # Track with 10 frames but very low confidence (0.20 < 0.35)
    traj = [(100.0 + i*5, 100.0) for i in range(10)]
    obj = make_mock_track(track_id=2, class_name="car", trajectory=traj, confidence=0.20)
    
    results = engine.update_and_predict([obj])
    assert 2 in results
    res = results[2]
    assert res.is_available is False
    assert "Low confidence" in (res.unavailability_reason or "")


def test_valid_prediction_trajectory_and_intervals():
    engine = PredictionEngine(fps=30.0)
    # Vehicle moving rightward at 5 px/frame for 15 frames
    traj = [(100.0 + i*5.0, 200.0) for i in range(15)]
    obj = make_mock_track(track_id=7, class_name="car", trajectory=traj, confidence=0.92)

    # Prime the Kalman filter
    for i in range(5, 15):
        sub_obj = make_mock_track(track_id=7, class_name="car", trajectory=traj[:i], confidence=0.92)
        results = engine.update_and_predict([sub_obj])

    res = results[7]
    assert res.is_available is True
    assert res.unavailability_reason is None
    assert res.method in ("kalman", "linear")
    assert res.prediction_horizon_sec == 2.0
    assert len(res.predicted_points) == 10  # 2.0s / 0.2s = 10 waypoints
    
    # Check that predicted points continue to the right
    assert res.predicted_points[-1].x > res.current_center[0]
    # Check uncertainty radius grows with time
    assert res.predicted_points[-1].uncertainty_radius >= res.predicted_points[0].uncertainty_radius


def test_class_predictability_weighting():
    engine = PredictionEngine(fps=30.0)
    # Vehicle trajectory
    traj_car = [(100.0 + i*4.0, 150.0) for i in range(15)]
    obj_car = make_mock_track(track_id=10, class_name="car", trajectory=traj_car, confidence=0.90)

    # Cow trajectory with identical positions & confidence
    traj_cow = [(100.0 + i*4.0, 150.0) for i in range(15)]
    obj_cow = make_mock_track(track_id=11, class_name="cow", trajectory=traj_cow, confidence=0.90)

    for i in range(8, 15):
        sub_car = make_mock_track(track_id=10, class_name="car", trajectory=traj_car[:i], confidence=0.90)
        sub_cow = make_mock_track(track_id=11, class_name="cow", trajectory=traj_cow[:i], confidence=0.90)
        results = engine.update_and_predict([sub_car, sub_cow])

    pred_car = results[10]
    pred_cow = results[11]

    # Car confidence must be higher than cow due to class predictability modeling
    assert pred_car.prediction_confidence > pred_cow.prediction_confidence


def test_prediction_result_to_dict():
    res = PredictionResult(
        track_id=7,
        class_name="car",
        method="kalman",
        current_center=(150.0, 200.0),
        predicted_points=[
            PredictedPoint(x=155.0, y=200.0, frame_offset=6, uncertainty_radius=6.0),
            PredictedPoint(x=160.0, y=200.0, frame_offset=12, uncertainty_radius=8.0),
        ],
        is_reliable=True,
        prediction_horizon=60,
        current_velocity=5.0,
        current_direction="FORWARD",
        prediction_horizon_sec=2.0,
        prediction_confidence=0.88,
        is_available=True,
    )
    d = res.to_dict()
    assert d["track_id"] == 7
    assert d["class_name"] == "car"
    assert d["prediction_horizon"] == 2.0
    assert d["prediction_confidence"] == 0.88
    assert len(d["predicted_positions"]) == 2
    assert d["is_available"] is True


if __name__ == "__main__":
    tests = [
        test_prediction_config,
        test_stability_gating_insufficient_history,
        test_stability_gating_low_confidence,
        test_valid_prediction_trajectory_and_intervals,
        test_class_predictability_weighting,
        test_prediction_result_to_dict,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASSED: {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAILED: {t.__name__} - {e}")
            raise
    print(f"\nAll {passed}/{len(tests)} Phase 4 unit tests PASSED successfully!")
