"""
Unit tests for Phase 3: Object Tracking & Movement Estimation.
Tests MotionEstimator, TrackManager, and AnnotatedVideoWriter.write_tracking_frame.
"""
import math
import numpy as np

from app.detection.detector import Detection
from app.tracking.motion import (
    MotionEstimator,
    MovementDirection,
    DIRECTION_SYMBOLS,
)
from app.tracking.track_manager import TrackManager
from app.video.writer import AnnotatedVideoWriter


def test_motion_estimator_directions():
    estimator = MotionEstimator(history_length=10, stationary_threshold_px=0.75, fps=30.0)

    # 1. Stationary object
    estimator.update(1, (100.0, 100.0))
    estimator.update(1, (100.2, 100.3))
    m_stat = estimator.get_motion(1)
    assert m_stat is not None
    assert m_stat.direction == MovementDirection.STATIONARY.value
    assert m_stat.direction_symbol == "■"
    assert m_stat.pixel_speed < 0.75

    # 2. Forward motion (moving upwards in image towards horizon, -dy)
    estimator.update(2, (200.0, 300.0))
    estimator.update(2, (200.0, 280.0))
    m_fwd = estimator.get_motion(2)
    assert m_fwd is not None
    assert m_fwd.direction == MovementDirection.FORWARD.value
    assert m_fwd.direction_symbol == "↑"
    assert abs(m_fwd.pixel_speed - 20.0) < 1.0
    assert abs(m_fwd.pixel_speed_sec - 600.0) < 30.0

    # 3. Backward motion (moving downwards in image towards camera, +dy)
    estimator.update(3, (200.0, 100.0))
    estimator.update(3, (200.0, 125.0))
    m_bwd = estimator.get_motion(3)
    assert m_bwd is not None
    assert m_bwd.direction == MovementDirection.BACKWARD.value
    assert m_bwd.direction_symbol == "↓"
    assert abs(m_bwd.pixel_speed - 25.0) < 1.0

    # 4. Rightward motion (+dx)
    estimator.update(4, (100.0, 200.0))
    estimator.update(4, (130.0, 201.0))
    m_right = estimator.get_motion(4)
    assert m_right is not None
    assert m_right.direction == MovementDirection.RIGHT.value
    assert m_right.direction_symbol == "→"

    # 5. Leftward motion (-dx)
    estimator.update(5, (300.0, 200.0))
    estimator.update(5, (270.0, 201.0))
    m_left = estimator.get_motion(5)
    assert m_left is not None
    assert m_left.direction == MovementDirection.LEFT.value
    assert m_left.direction_symbol == "←"

    # 6. Diagonal motion (both dx and dy significant)
    estimator.update(6, (200.0, 200.0))
    estimator.update(6, (230.0, 170.0))
    m_diag = estimator.get_motion(6)
    assert m_diag is not None
    assert m_diag.direction == MovementDirection.DIAGONAL.value

    print("MotionEstimator directions test passed!")


def test_track_manager_persistence():
    tm = TrackManager(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.5,
        frame_rate=30,
        history_length=20,
    )

    # Simulate 5 consecutive frames with 2 objects (Car moving right, Person stationary)
    for frame_idx in range(5):
        dets = [
            Detection(
                class_id=2,
                class_name="car",
                confidence=0.92,
                x1=50.0 + frame_idx * 6.0,
                y1=100.0,
                x2=120.0 + frame_idx * 6.0,
                y2=160.0,
            ),
            Detection(
                class_id=0,
                class_name="person",
                confidence=0.88,
                x1=250.0,
                y1=150.0,
                x2=280.0,
                y2=210.0,
            ),
        ]
        structured = tm.update(dets, frame_idx=frame_idx)
        if frame_idx >= 1:
            assert len(structured) == 2

    # Check persistence: track IDs should stay consistent across frames
    tracked_objs = tm.tracked_objects
    assert len(tracked_objs) == 2
    track_ids = {obj.track_id for obj in tracked_objs}
    assert len(track_ids) == 2

    # Check structured output format
    car_record = next(r for r in tm.structured_tracks if r["class_name"] == "car")
    assert "id" in car_record
    assert car_record["confidence"] == 0.92
    assert "bbox" in car_record
    assert "center" in car_record
    assert "direction" in car_record
    assert "velocity" in car_record
    assert "history" in car_record
    assert len(car_record["history"]) >= 4

    assert tm.active_tracks_count == 2
    assert tm.total_unique_tracks == 2

    summary = tm.get_active_tracks_table_data()
    assert len(summary) == 2
    for row in summary:
        assert "id" in row
        assert "class_name" in row
        assert "confidence" in row
        assert "direction" in row
        assert "speed" in row

    print("TrackManager persistence test passed!")


def test_write_tracking_frame():
    import tempfile
    from pathlib import Path

    tm = TrackManager(frame_rate=30)
    dets = [
        Detection(class_id=2, class_name="car", confidence=0.94, x1=60.0, y1=80.0, x2=140.0, y2=150.0),
        Detection(class_id=19, class_name="cow", confidence=0.89, x1=200.0, y1=100.0, x2=270.0, y2=170.0),
    ]
    tm.update(dets, frame_idx=0)
    # 2nd frame to have velocity
    dets2 = [
        Detection(class_id=2, class_name="car", confidence=0.94, x1=65.0, y1=80.0, x2=145.0, y2=150.0),
        Detection(class_id=19, class_name="cow", confidence=0.89, x1=200.0, y1=100.0, x2=270.0, y2=170.0),
    ]
    tm.update(dets2, frame_idx=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test_track_out.mp4"
        writer = AnnotatedVideoWriter(str(out_path), fps=30.0, width=640, height=480)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        annotated = writer.write_tracking_frame(
            dummy_frame,
            tm.tracked_objects,
            frame_idx=1,
            fps=30.0,
            total_unique=tm.total_unique_tracks,
        )
        assert annotated.shape == (480, 640, 3)
        writer.close()
        assert out_path.exists()
        print("write_tracking_frame test passed!")


if __name__ == "__main__":
    test_motion_estimator_directions()
    test_track_manager_persistence()
    test_write_tracking_frame()
    print("ALL PHASE 3 UNIT TESTS PASSED!")
