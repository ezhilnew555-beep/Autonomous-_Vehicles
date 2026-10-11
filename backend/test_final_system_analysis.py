"""
Unit and integration tests for the unified road analysis system (Final System Integration).
Tests ObjectCounter, EventManager, AnalysisSummaryGenerator, AnalysisOrchestrator, and API endpoints.
"""
from __future__ import annotations
import unittest
import numpy as np
import cv2
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.analysis.object_counter import ObjectCounter, CANONICAL_CATEGORIES, canonicalize_class
from app.analysis.event_manager import EventManager
from app.analysis.analysis_summary import AnalysisSummaryGenerator
from app.analysis.analysis_state import (
    RoadStatus,
    ROAD_STATUS_DESCRIPTIONS,
    map_decision_to_plain_action,
    map_planning_status_to_label,
    determine_road_status,
)
from app.analysis.analysis_result import (
    AnalysisResult,
    analysis_store,
    ObjectCountsModel,
)
from app.analysis.analysis_orchestrator import AnalysisOrchestrator
from app.services.video_service import register_job, get_job, JobStatus


class TestObjectCounter(unittest.TestCase):
    """Test ObjectCounter normalization and track ID deduplication."""

    def test_canonical_mapping(self):
        self.assertEqual(canonicalize_class("car"), "Cars")
        self.assertEqual(canonicalize_class("motorcycle"), "Bikes")
        self.assertEqual(canonicalize_class("bicycle"), "Bicycles")
        self.assertEqual(canonicalize_class("person"), "People")
        self.assertEqual(canonicalize_class("cow"), "Animals")
        self.assertEqual(canonicalize_class("dog"), "Animals")
        self.assertEqual(canonicalize_class("bus"), "Buses")
        self.assertEqual(canonicalize_class("truck"), "Trucks")
        self.assertEqual(canonicalize_class("autorickshaw"), "Auto-rickshaws")
        self.assertEqual(canonicalize_class("rickshaw"), "Auto-rickshaws")
        self.assertEqual(canonicalize_class("unknown_thing"), "Other")
        self.assertEqual(canonicalize_class(""), "Unknown")

    def test_track_id_deduplication(self):
        counter = ObjectCounter(min_persistence_frames=2)
        # Simulate track 1 (car) seen over 20 consecutive frames
        for frame in range(20):
            counter.update_frame([{"track_id": 1, "class_name": "car"}])

        counts = counter.get_counts()
        # Current frame should have 1 car
        self.assertEqual(counts["current_frame"]["Cars"], 1)
        # Total unique count across all 20 frames must be 1 car (NOT 20!)
        self.assertEqual(counts["total_unique"]["Cars"], 1)
        self.assertEqual(counts["total_unique_all"], 1)

        # Now simulate track 2 (person) and track 3 (motorcycle) appearing
        for frame in range(5):
            counter.update_frame([
                {"track_id": 1, "class_name": "car"},
                {"track_id": 2, "class_name": "person"},
                {"track_id": 3, "class_name": "motorcycle"},
            ])

        counts2 = counter.get_counts()
        self.assertEqual(counts2["current_frame"]["Cars"], 1)
        self.assertEqual(counts2["current_frame"]["People"], 1)
        self.assertEqual(counts2["current_frame"]["Bikes"], 1)
        self.assertEqual(counts2["total_unique"]["Cars"], 1)
        self.assertEqual(counts2["total_unique"]["People"], 1)
        self.assertEqual(counts2["total_unique"]["Bikes"], 1)
        self.assertEqual(counts2["total_unique_all"], 3)


class TestEventManager(unittest.TestCase):
    """Test timeline event management and plain English descriptions."""

    def test_event_logging_and_debounce(self):
        em = EventManager(debounce_window_sec=1.0)
        e1 = em.log_decision_change(
            frame_index=10,
            timestamp_sec=0.5,
            prev_action="Continue normally",
            new_action="Slow down",
            reason="Approaching pedestrian",
        )
        self.assertIsNotNone(e1)
        self.assertIn("changed from Continue normally to Slow down", e1.description)

        # Same decision transition within debounce window should be ignored
        e2 = em.log_decision_change(
            frame_index=12,
            timestamp_sec=0.6,
            prev_action="Continue normally",
            new_action="Slow down",
            reason="Approaching pedestrian",
        )
        self.assertIsNone(e2)

        # Risk detected
        e_risk = em.log_risk_detected(
            frame_index=20,
            timestamp_sec=1.0,
            track_id=4,
            class_name="person",
            risk_level="HIGH",
            ttc_sec=2.1,
            reason="Trajectory intersection",
        )
        self.assertIsNotNone(e_risk)
        self.assertEqual(e_risk.severity, "WARNING")
        self.assertIn("HIGH collision risk", e_risk.description)

        timeline = em.get_timeline()
        self.assertEqual(len(timeline), 2)


class TestAnalysisSummary(unittest.TestCase):
    """Test dynamic summary generation from structured data."""

    def test_summary_generation(self):
        summary = AnalysisSummaryGenerator.generate(
            duration_sec=10.0,
            total_frames=300,
            unique_counts={"Cars": 2, "People": 1, "Bikes": 1, "Animals": 1},
            total_unique_all=5,
            risk_counter={"LOW": 5, "MEDIUM": 2, "HIGH": 1, "CRITICAL": 0},
            decisions_counter={"CRUISE": 250, "SLOW_DOWN": 40, "AVOID_OBSTACLE": 10},
            replanning_count=2,
            average_smoothness=0.93,
            average_latency_ms=1.45,
            road_status="CAUTION",
            critical_events=[{
                "description": "HIGH collision risk: person #2 with TTC 2.4s",
                "severity": "WARNING",
            }],
            detected_signals=[{"color": "green", "confidence": 0.92}],
            detected_signs=[{"sign_type": "SPEED_LIMIT_40", "action_directive": "Obey 40 km/h limit"}],
        )

        self.assertIn("300 frames", summary["executive_summary"])
        self.assertIn("0 collisions", summary["executive_summary"])
        self.assertIn("Cars: 2", summary["traffic_breakdown"])
        self.assertIn("People: 1", summary["traffic_breakdown"])
        self.assertIn("replanning", summary["navigation_assessment"])
        self.assertIn("ZERO COLLISIONS DETECTED", summary["safety_verdict"])


class TestAnalysisEndpoints(unittest.TestCase):
    """Test REST API endpoints for unified road analysis."""

    def setUp(self):
        self.client = TestClient(app)

    def test_analysis_endpoints_workflow(self):
        # Create a mock synthetic video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f_in, \
             tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f_out:
            in_path = f_in.name
            out_path = f_out.name

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(in_path, fourcc, 25.0, (320, 240))
        for _ in range(15):
            frame = np.full((240, 320, 3), 70, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        video_id = "test_analysis_vid_001"
        register_job(
            video_id=video_id,
            original_filename="test_video.mp4",
            input_path=in_path,
            output_path=out_path,
            size_bytes=1000,
        )

        # Run pipeline synchronously for test
        res = AnalysisOrchestrator.start_analysis(video_id)
        self.assertEqual(res.status, "PROCESSING")
        
        # Execute pipeline
        completed_res = AnalysisOrchestrator.run_pipeline(res.analysis_id)
        self.assertEqual(completed_res.status, "COMPLETED")
        self.assertEqual(completed_res.performance.collision_count, 0)
        self.assertIsNotNone(completed_res.summary.executive_summary)

        # Query GET /api/v1/analysis/{analysis_id}
        resp = self.client.get(f"/api/v1/analysis/{completed_res.analysis_id}")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["analysis_id"], completed_res.analysis_id)
        self.assertIn("object_counts", data)
        self.assertIn("total_unique", data["object_counts"])
        self.assertIn("timeline", data)
        self.assertIn("summary", data)
        self.assertIn("performance", data)
        self.assertIn("video_info", data)
        self.assertIn("safety_analysis", data)
        self.assertIn("road_information", data)
        self.assertEqual(data["performance"]["total_collision_events"], 0)
        self.assertEqual(data["performance"]["collision_free_rate"], "100%")

        # Query GET /api/v1/analysis/status/{analysis_id}
        resp_status = self.client.get(f"/api/v1/analysis/status/{completed_res.analysis_id}")
        self.assertEqual(resp_status.status_code, 200)
        status_data = resp_status.json()
        self.assertEqual(status_data["progress"], 100.0)

        # Query GET /api/v1/analysis/video/{video_id}
        resp_vid = self.client.get(f"/api/v1/analysis/video/{video_id}")
        self.assertEqual(resp_vid.status_code, 200)
        self.assertEqual(resp_vid.json()["analysis_id"], completed_res.analysis_id)

    def test_prediction_result_schema_compatibility(self):
        """Verify PredictionResult attributes match TechnicalDetails requirements."""
        from app.prediction.schemas import PredictionResult, PredictedPoint
        pred = PredictionResult(
            track_id=1,
            class_name="car",
            method="kalman",
            current_center=(100.0, 150.0),
            predicted_points=[PredictedPoint(x=102.0, y=148.0, frame_offset=1, uncertainty_radius=5.0)],
            is_reliable=True,
            prediction_horizon=10,
            prediction_confidence=0.88,
        )
        self.assertTrue(pred.has_prediction)
        self.assertEqual(getattr(pred, "prediction_confidence", 0.0), 0.88)
        self.assertEqual(len(getattr(pred, "predicted_points", [])), 1)


if __name__ == "__main__":
    unittest.main()
