"""
Unit and integration tests for Phase 9A — Dashboard API Endpoints.
Verifies all 5 endpoints:
  - /api/v1/dashboard/state
  - /api/v1/dashboard/metrics
  - /api/v1/dashboard/events
  - /api/v1/dashboard/environment
  - /api/v1/dashboard/path
Verifies fallback to UNKNOWN/NOT AVAILABLE/INSUFFICIENT DATA, and state-change deduplication.
"""
import sys
import unittest
from starlette.testclient import TestClient

from app.main import app
from app.services.video_service import register_job, get_job
from app.models.schemas import (
    JobStatus,
    DetectionStats,
    DecisionItem,
    RiskEventItem,
    ReplanningEventItem,
    TrackedObjectSummary,
)


class TestPhase9ADashboardAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_fallback_empty_state(self):
        """When querying a non-existent video ID, endpoints should return clean fallback values."""
        res = self.client.get("/api/v1/dashboard/state?video_id=nonexistent_123")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["current_action"], "UNKNOWN")
        self.assertEqual(data["risk_level"], "UNKNOWN")
        self.assertEqual(data["threat_object"], "NOT AVAILABLE")
        self.assertEqual(data["safety_status"], "INSUFFICIENT DATA")
        self.assertFalse(data["is_available"])

        res_metrics = self.client.get("/api/v1/dashboard/metrics?video_id=nonexistent_123")
        self.assertEqual(res_metrics.status_code, 200)
        metrics_data = res_metrics.json()
        self.assertEqual(metrics_data["detection_fps"], 0.0)
        self.assertEqual(metrics_data["replanning_count"], 0)
        self.assertFalse(metrics_data["is_available"])

        res_path = self.client.get("/api/v1/dashboard/path?video_id=nonexistent_123")
        self.assertEqual(res_path.status_code, 200)
        path_data = res_path.json()
        self.assertEqual(path_data["selected_path"], "NOT AVAILABLE")
        self.assertFalse(path_data["is_available"])

    def test_02_completed_job_endpoints(self):
        """When a completed job exists, endpoints return actual system telemetry."""
        vid = "test_vid_p9a_001"
        job = register_job(
            video_id=vid,
            original_filename="traffic_demo.mp4",
            input_path="traffic_demo.mp4",
            output_path="traffic_demo_annotated.mp4",
            size_bytes=102400,
        )

        decisions = [
            DecisionItem(
                frame_idx=10,
                timestamp_sec=0.4,
                action="CONTINUE",
                avoidance_direction="NONE",
                reason="Clear corridor.",
            ),
            DecisionItem(
                frame_idx=20,
                timestamp_sec=0.8,
                action="CONTINUE",
                avoidance_direction="NONE",
                reason="Clear corridor.",
            ),
            DecisionItem(
                frame_idx=30,
                timestamp_sec=1.2,
                action="REPLAN",
                target_object_id=12,
                target_class="cow",
                risk_level="HIGH",
                ttc_sec=1.8,
                avoidance_direction="LEFT",
                target_speed_factor=0.6,
                reason="Slow-moving cattle predicted to enter the vehicle path.",
            ),
            DecisionItem(
                frame_idx=40,
                timestamp_sec=1.6,
                action="REPLAN",
                target_object_id=12,
                target_class="cow",
                risk_level="HIGH",
                ttc_sec=1.8,
                avoidance_direction="LEFT",
                target_speed_factor=0.6,
                reason="Slow-moving cattle predicted to enter the vehicle path.",
            ),
        ]

        replans = [
            ReplanningEventItem(
                timestamp_sec=1.2,
                frame_idx=30,
                trigger_reason="Obstacle Cow #12",
                previous_status="CONTINUE_ORIGINAL",
                new_status="REPLANNED_AVOID_LEFT",
                planning_time_ms=1.45,
                obstacle_id=12,
                obstacle_class="cow",
            )
        ]

        risks = [
            RiskEventItem(
                timestamp_sec=1.2,
                frame_idx=30,
                object_id=12,
                object_class="cow",
                distance_px=110.0,
                ttc_sec=1.8,
                risk_level="HIGH",
                reason="Approaching ego path",
            )
        ]

        job.stats = DetectionStats(
            total_frames=100,
            processed_frames=100,
            processing_fps=18.5,
            video_fps=25.0,
            duration_seconds=4.0,
            unique_track_ids=3,
            objects_by_class={"cow": 1, "car": 2},
            tracked_objects=[
                TrackedObjectSummary(
                    track_id=12,
                    class_name="cow",
                    avg_confidence=0.92,
                    frame_count=30,
                    avg_pixel_speed=2.5,
                    last_direction="W",
                    has_prediction=True,
                    prediction_method="kalman",
                    predicted_direction="W",
                    max_risk_level="HIGH",
                    min_ttc_sec=1.8,
                    last_risk_reason="Approaching ego path",
                )
            ],
            pipeline_mode="full",
            risk_counts={"LOW": 90, "MEDIUM": 5, "HIGH": 5, "CRITICAL": 0},
            critical_events_count=0,
            high_events_count=1,
            risk_events=risks,
            action_counts={"CONTINUE": 50, "REPLAN": 50},
            dominant_action="REPLAN",
            decision_history=decisions,
            replanning_count=1,
            avg_planning_time_ms=1.45,
            final_path_status="REPLANNED_AVOID_LEFT",
            replanning_events=replans,
            active_traffic_signal={"color": "GREEN", "confidence": 0.95},
            traffic_signals_detected=[{"color": "GREEN", "confidence": 0.95}],
            road_signs_detected=[{"sign_type": "SPEED_LIMIT", "value": 40}],
            active_map_profile={
                "road_id": "VILLAGE_001",
                "road_type": "VILLAGE_ROAD",
                "geometry_type": "STRAIGHT_CORRIDOR",
                "lane_count": 1,
                "speed_limit_kmh": 30,
                "has_intersection": False,
                "destination_direction": "FORWARD",
                "restricted_directions": [],
                "road_notes": "Unmarked village road with loose gravel shoulders.",
            },
        )
        job.status = JobStatus.DONE

        # Verify /state
        res_state = self.client.get(f"/api/v1/dashboard/state?video_id={vid}")
        self.assertEqual(res_state.status_code, 200)
        state = res_state.json()
        self.assertTrue(state["is_available"])
        self.assertEqual(state["current_action"], "REPLAN")
        self.assertEqual(state["risk_level"], "HIGH")
        self.assertEqual(state["ttc_sec"], 1.8)
        self.assertIn("Cow #12", state["threat_object"])
        self.assertEqual(state["current_path"], "LEFT_AVOIDANCE")
        self.assertEqual(state["safety_status"], "CAUTION")

        # Verify /metrics
        res_metrics = self.client.get(f"/api/v1/dashboard/metrics?video_id={vid}")
        self.assertEqual(res_metrics.status_code, 200)
        metrics = res_metrics.json()
        self.assertTrue(metrics["is_available"])
        self.assertEqual(metrics["processing_fps"], 18.5)
        self.assertEqual(metrics["avg_planning_latency_ms"], 1.45)
        self.assertEqual(metrics["replanning_count"], 1)
        self.assertEqual(metrics["collision_events_count"], 0)
        self.assertEqual(metrics["completed_scenarios"], 5)

        # Verify /events (deduplication check)
        res_events = self.client.get(f"/api/v1/dashboard/events?video_id={vid}")
        self.assertEqual(res_events.status_code, 200)
        events_data = res_events.json()
        self.assertTrue(events_data["is_available"])
        # 4 decision frames, but only 2 unique state transitions:
        # Frame 10 (CONTINUE) and Frame 30 (REPLAN)
        self.assertEqual(len(events_data["events"]), 2)
        self.assertEqual(events_data["events"][0]["decision"], "CONTINUE")
        self.assertEqual(events_data["events"][1]["decision"], "REPLAN")
        self.assertEqual(events_data["events"][1]["threat"], "Cow #12")

        # Verify /environment
        res_env = self.client.get(f"/api/v1/dashboard/environment?video_id={vid}")
        self.assertEqual(res_env.status_code, 200)
        env = res_env.json()
        self.assertTrue(env["is_available"])
        self.assertEqual(env["road_type"], "VILLAGE_ROAD")
        self.assertFalse(env["intersection"])
        self.assertEqual(env["speed_limit_kmh"], 30)
        self.assertIn("SPEED_LIMIT_30_KMH", env["planning_constraints"])

        # Verify /path
        res_path = self.client.get(f"/api/v1/dashboard/path?video_id={vid}")
        self.assertEqual(res_path.status_code, 200)
        path = res_path.json()
        self.assertTrue(path["is_available"])
        self.assertEqual(path["path_type"], "LEFT_AVOIDANCE")
        self.assertEqual(len(path["trajectory_2d"]), 25)
        self.assertEqual(len(path["nominal_path_2d"]), 25)
        self.assertEqual(path["drivable_boundaries"], [120, 520])


if __name__ == "__main__":
    unittest.main()
