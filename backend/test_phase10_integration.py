"""
Eareyes 2k26 — Phase 10: Final Full-System Integration, Validation, Testing & Demo Readiness.

Automates the verification of the complete 10-stage autonomous driving stack:
1. Standardized Data Contracts (Detection, Tracking, Prediction, Risk, Decision, Planning, Environment)
2. Safety Priority Hierarchy (Immediate Collision Safety > Signal/Speed Restrictions > Path Efficiency)
3. Multi-Object Threat Arbitration (Cars, Trucks, Bikes, Pedestrians, Animals resolved via Risk/TTC)
4. Conservative Uncertainty Handling (Unknown signals/signs, low confidence, corridor constraints)
5. Path Planning Quality & Dynamic Replanning (No safe path -> Stop, Obstacle -> Quintic Evasion)
6. All 5 Required Indian Road Scenarios (Village, Intersection, Merge, Market, Cattle)
7. Measurable System Performance Metrics (Real latency, FPS, smoothness, zero collisions)
8. End-to-End API Pipeline Execution
"""
import math
import os
import sys
import unittest
from typing import Dict, List

# Add parent directory to sys.path
sys.path.insert(0, os.path.dirname(__file__))

from starlette.testclient import TestClient

from app.main import app
from app.contracts import (
    format_detection_contract,
    format_tracking_contract,
    format_prediction_contract,
    format_risk_contract,
    format_decision_contract,
    format_planning_contract,
    format_environment_contract,
)
from app.tracking.tracker import TrackedObject
from app.prediction.schemas import PredictionResult, PredictedPoint
from app.collision.schemas import RiskLevel, CollisionRisk
from app.decision.schemas import VehicleAction, AvoidanceDirection, DecisionOutput
from app.decision.engine import DecisionEngine
from app.planning.schemas import PlanningResult, PlanningStatus
from app.planning.planner import AdaptivePathPlanner
from app.environment.schemas import (
    RoadType,
    TrafficLightColor,
    TrafficSignal,
    RoadSignType,
    RoadSign,
    MapRoadConstraint,
    DrivableRegion,
    EnvironmentState,
)
from app.environment.road_context import RoadContextModel
from app.environment.environment_rules import EnvironmentRuleEngine
from app.scenarios.scenario_config import ScenarioID, SCENARIO_CONFIGS
from app.scenarios.scenario_runner import ScenarioRunner
from app.scenarios.scenario_manager import ScenarioManager


class TestPhase10FullSystemIntegration(unittest.TestCase):
    """Phase 10 Comprehensive Verification and Acceptance Test Suite."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.decision_engine = DecisionEngine(corridor_half_width=80.0, hysteresis_frames=1)
        cls.planner = AdaptivePathPlanner(lookahead_distance=340.0, num_waypoints=25, road_margin_px=35.0)

    # ── Test 1: Standardized Data Contracts ──────────────────────────────────
    def test_01_standardized_data_contracts(self):
        """Validate exact keys and types for all 7 primary data contracts."""
        # 1. Detection
        det = format_detection_contract(class_name="car", confidence=0.925, bbox=[100, 200, 180, 280])
        self.assertIn("object_id", det)
        self.assertEqual(det["class_name"], "car")
        self.assertEqual(det["confidence"], 0.925)
        self.assertEqual(det["bbox"], [100.0, 200.0, 180.0, 280.0])

        # 2. Tracking
        trk = format_tracking_contract(object_id=12, class_name="car", position=[140, 240], velocity=[0.0, -5.2])
        self.assertEqual(trk["object_id"], 12)
        self.assertEqual(trk["position"], [140.0, 240.0])
        self.assertEqual(trk["velocity"], [0.0, -5.2])
        self.assertIn("direction", trk)

        # 3. Prediction
        pred = format_prediction_contract(object_id=12, future_positions=[[140, 230], [140, 220]], prediction_confidence=0.88)
        self.assertEqual(pred["object_id"], 12)
        self.assertEqual(len(pred["future_positions"]), 2)
        self.assertEqual(pred["prediction_confidence"], 0.88)

        # 4. Risk
        risk = format_risk_contract(object_id=12, risk_level="HIGH", ttc=1.8, trajectory_conflict=True)
        self.assertEqual(risk["object_id"], 12)
        self.assertEqual(risk["risk_level"], "HIGH")
        self.assertEqual(risk["ttc"], 1.8)
        self.assertTrue(risk["trajectory_conflict"])

        # 5. Decision
        dec = format_decision_contract(action="REPLAN", reason="Obstacle ahead", threat_object=12, risk_level="HIGH")
        self.assertEqual(dec["action"], "REPLAN")
        self.assertEqual(dec["threat_object"], 12)
        self.assertEqual(dec["risk_level"], "HIGH")

        # 6. Planning
        plan = format_planning_contract(path_type="LEFT_AVOIDANCE", selected_path=[[320, 480], [300, 400]], safe=True, replanned=True)
        self.assertEqual(plan["path_type"], "LEFT_AVOIDANCE")
        self.assertTrue(plan["safe"])
        self.assertTrue(plan["replanned"])

        # 7. Environment
        env = format_environment_contract(road_type="URBAN_ROAD", traffic_signal="GREEN", road_signs=[{"type": "STOP"}])
        self.assertEqual(env["road_type"], "URBAN_ROAD")
        self.assertEqual(env["traffic_signal"], "GREEN")
        self.assertEqual(len(env["road_signs"]), 1)

    # ── Test 2: Safety Priority Hierarchy ────────────────────────────────────
    def test_02_safety_priority_hierarchy_green_light_vs_pedestrian(self):
        """
        Verify that immediate collision safety overrides traffic signals.
        Example: GREEN traffic signal + Pedestrian predicted to collide in corridor.
        Outcome MUST NOT be CONTINUE; it MUST command STOP or controlled avoidance.
        """
        engine = DecisionEngine(corridor_half_width=80.0, hysteresis_frames=1)

        # Crossing pedestrian in forward path with critical TTC = 1.3s
        ped_track = TrackedObject(
            track_id=101, class_id=0, class_name="person", confidence=0.94,
            x1=310.0, y1=320.0, x2=330.0, y2=370.0
        )
        ped_risk = CollisionRisk(
            track_id=101, class_name="person", distance_px=110.0,
            lateral_offset_px=0.0, longitudinal_dist_px=110.0,
            relative_speed_px=-15.0, closing_speed_px=15.0,
            ttc_sec=1.3, risk_level=RiskLevel.HIGH,
            reason="Predicted Person #101 trajectory crosses vehicle corridor in 1.3s"
        )
        pred = PredictionResult(
            track_id=101, class_name="person", method="kalman",
            current_center=(320.0, 345.0),
            predicted_points=[PredictedPoint(320.0, 345.0 + i*5, i, 10.0) for i in range(1, 10)],
            is_reliable=True, prediction_horizon=10
        )

        # Environment has a reliable GREEN signal (which normally allows cruising at 100%)
        green_signal = TrafficSignal(
            color=TrafficLightColor.GREEN, confidence=0.95, is_reliable=True,
            box=(550.0, 50.0, 580.0, 120.0), distance_px=300.0, reason="GREEN signal"
        )
        env_state = EnvironmentState(
            frame_idx=10, timestamp_sec=0.4, active_signal=green_signal,
            detected_signs=[], road_type=RoadType.URBAN_ROAD,
            drivable_region=DrivableRegion(35.0, 605.0, 570.0),
        )

        decision = engine.evaluate_frame(
            frame_idx=10, fps=25.0, frame_width=640, frame_height=480,
            tracked_objects=[ped_track],
            predictions={101: pred},
            risks={101: ped_risk},
            environment=env_state,
        )

        # Verification: System must yield to pedestrian despite the green signal
        self.assertNotEqual(decision.action, VehicleAction.CONTINUE, "GREEN signal must NEVER override pedestrian collision hazard")
        self.assertIn(decision.action, (VehicleAction.STOP, VehicleAction.AVOID, VehicleAction.EMERGENCY_STOP))
        self.assertEqual(decision.target_object_id, 101)
        self.assertIn("Person", decision.reason)

    # ── Test 3: Multi-Object Threat Selection ─────────────────────────────────
    def test_03_multi_object_threat_arbitration(self):
        """
        Verify multi-object threat arbitration among 7 simultaneous entities:
        Car, Bus, Truck, Motorcycle, Bicycle, Pedestrian, Animal.
        The most immediate collision threat must be selected, not just highest detection confidence.
        """
        engine = DecisionEngine(corridor_half_width=80.0, hysteresis_frames=1)

        # 7 simultaneous tracked road users
        tracks = [
            TrackedObject(1, 2, "car", 0.99, 100.0, 200.0, 160.0, 260.0),          # High confidence, far left (no threat)
            TrackedObject(2, 5, "bus", 0.95, 480.0, 180.0, 560.0, 280.0),          # High confidence, far right (no threat)
            TrackedObject(3, 7, "truck", 0.92, 120.0, 150.0, 190.0, 240.0),        # Left lane
            TrackedObject(4, 3, "motorcycle", 0.85, 420.0, 320.0, 450.0, 360.0),   # Right shoulder
            TrackedObject(5, 1, "bicycle", 0.88, 360.0, 340.0, 380.0, 370.0),      # Near corridor edge (MEDIUM risk, TTC 3.2s)
            TrackedObject(6, 0, "person", 0.78, 315.0, 390.0, 335.0, 430.0),       # Directly in corridor close (CRITICAL risk, TTC 0.8s)
            TrackedObject(7, 19, "cow", 0.82, 220.0, 280.0, 270.0, 330.0),         # Left berm (LOW risk)
        ]

        risks = {
            1: CollisionRisk(1, "car", 280.0, -180.0, 280.0, -2.0, 2.0, None, RiskLevel.LOW, "Safe distance"),
            2: CollisionRisk(2, "bus", 300.0, 190.0, 300.0, -1.0, 1.0, None, RiskLevel.LOW, "Safe distance"),
            3: CollisionRisk(3, "truck", 330.0, -160.0, 330.0, 0.0, 0.0, None, RiskLevel.LOW, "Left lane"),
            4: CollisionRisk(4, "motorcycle", 160.0, 110.0, 160.0, -4.0, 4.0, None, RiskLevel.LOW, "Shoulder clear"),
            5: CollisionRisk(5, "bicycle", 140.0, 50.0, 140.0, -10.0, 10.0, 3.2, RiskLevel.MEDIUM, "Approaching corridor"),
            6: CollisionRisk(6, "person", 70.0, 5.0, 70.0, -20.0, 20.0, 0.8, RiskLevel.CRITICAL, "Imminent corridor breach"),
            7: CollisionRisk(7, "cow", 200.0, -90.0, 200.0, -3.0, 3.0, None, RiskLevel.LOW, "On berm"),
        }

        decision = engine.evaluate_frame(
            frame_idx=25, fps=25.0, frame_width=640, frame_height=480,
            tracked_objects=tracks,
            risks=risks,
        )

        # Threat must be Track #6 (Person, Critical risk TTC 0.8s) even though Car #1 had higher confidence (0.99)
        self.assertEqual(decision.target_object_id, 6, "Must select most urgent threat (Track #6), not highest confidence detection")
        self.assertEqual(decision.risk_level, RiskLevel.CRITICAL)
        self.assertIn(decision.action, (VehicleAction.EMERGENCY_STOP, VehicleAction.AVOID))

    # ── Test 4: Conservative Behavior Under Uncertainty ───────────────────────
    def test_04_uncertainty_handling_unknown_signal_and_no_safe_path(self):
        """
        Verify conservative system behavior when sensory information is uncertain:
        1. UNKNOWN traffic signal at an intersection -> approach with caution (do NOT assume green).
        2. Totally blocked corridor with no safe evasion -> NO_SAFE_PATH and safe STOP.
        """
        engine = DecisionEngine(corridor_half_width=80.0, hysteresis_frames=1)

        # 1. Unknown traffic signal approaching an intersection
        unknown_signal = TrafficSignal(
            color=TrafficLightColor.UNKNOWN, confidence=0.35, is_reliable=False,
            box=(550.0, 50.0, 580.0, 120.0), distance_px=220.0, reason="Occluded/ambiguous signal"
        )
        env_intersection = EnvironmentState(
            frame_idx=15, timestamp_sec=0.6, active_signal=unknown_signal,
            has_intersection=True, road_type=RoadType.URBAN_INTERSECTION,
            map_constraints=MapRoadConstraint(road_type=RoadType.URBAN_INTERSECTION, has_intersection=True),
            drivable_region=DrivableRegion(35.0, 605.0, 570.0),
        )

        dec_unknown = engine.evaluate_frame(
            frame_idx=15, fps=25.0, frame_width=640, frame_height=480,
            tracked_objects=[], risks={}, environment=env_intersection
        )
        # Must slow down and approach conservatively, never accelerate or assume free passage
        self.assertIn(dec_unknown.action, (VehicleAction.SLOW_DOWN, VehicleAction.CONTINUE))
        self.assertLessEqual(dec_unknown.target_speed_factor, 0.70)

        # 2. Path planning with completely blocked corridor (large obstacle spanning road)
        planner = AdaptivePathPlanner(lookahead_distance=340.0, num_waypoints=25, road_margin_px=35.0)
        wall_obstacle = TrackedObject(
            track_id=99, class_id=7, class_name="truck", confidence=0.95,
            x1=40.0, y1=240.0, x2=600.0, y2=320.0  # Spanning the entire width
        )
        dec_stop = DecisionOutput(
            frame_idx=16, timestamp_sec=0.64, action=VehicleAction.STOP,
            target_object_id=99, target_class="truck", risk_level=RiskLevel.HIGH,
            ttc_sec=None, avoidance_direction=AvoidanceDirection.NONE,
            target_speed_factor=0.0, reason="Road fully blocked"
        )

        plan_res = planner.plan_frame(
            frame_idx=16, fps=25.0, frame_width=640, frame_height=480,
            tracked_objects=[wall_obstacle],
            decision=dec_stop,
            environment=env_intersection
        )
        # Must NOT select an unsafe path; must report NO_SAFE_PATH
        self.assertEqual(plan_res.status, PlanningStatus.NO_SAFE_PATH)
        self.assertEqual(len(plan_res.safe_path), 0)

    # ── Test 5: All 5 Required Indian Road Scenarios ───────────────────────────
    def test_05_all_five_scenarios_execution(self):
        """
        Run the complete pipeline across all 5 Indian road situations:
        1. VILLAGE_ROAD
        2. URBAN_UNSIGNALLED_INTERSECTION
        3. HIGHWAY_MERGE
        4. MARKET_AREA
        5. CATTLE_CROSSING
        """
        for sid in ScenarioID:
            res = ScenarioRunner.run(sid)
            self.assertTrue(res.completed, f"{sid.value} must complete successfully")
            self.assertTrue(res.collision_free, f"{sid.value} must be collision free")
            self.assertEqual(res.collision_events, 0, f"{sid.value} must record zero collisions")
            self.assertGreaterEqual(res.path_smoothness_score, 0.40)
            self.assertGreater(res.processing_fps, 0.0)
            self.assertGreaterEqual(res.average_replanning_latency_ms, 0.0)
            self.assertTrue(len(res.events) > 0)

        # Verify summary aggregation
        summary = ScenarioManager.get_summary()
        self.assertEqual(summary["total_scenarios"], 5)
        self.assertEqual(summary["completed_scenarios"], 5)
        self.assertEqual(summary["collision_free_scenarios"], 5)
        self.assertEqual(summary["collision_free_rate_pct"], 100.0)
        self.assertEqual(summary["scenario_completion_rate_text"], "5 / 5")
        self.assertEqual(summary["total_collision_events"], 0)

    # ── Test 6: REST API Pipeline Integration ────────────────────────────────
    def test_06_api_endpoints_and_export(self):
        """Verify REST API endpoints under /api/v1/scenarios and /api/v1/dashboard."""
        # 1. Scenarios Catalog
        res_cat = self.client.get("/api/v1/scenarios")
        self.assertEqual(res_cat.status_code, 200)
        data_cat = res_cat.json()
        self.assertEqual(len(data_cat), 5)
        for item in data_cat:
            self.assertIn("scenario_id", item)
            self.assertEqual(item["status"], "TESTED")

        # 2. Scenarios Comparison Table
        res_comp = self.client.get("/api/v1/scenarios/comparison")
        self.assertEqual(res_comp.status_code, 200)
        table = res_comp.json()
        self.assertEqual(len(table), 5)

        # 3. JSON and CSV Export
        res_json = self.client.get("/api/v1/scenarios/export?format=json")
        self.assertEqual(res_json.status_code, 200)
        self.assertIn("comparison_table", res_json.json())
        self.assertIn("summary", res_json.json())

        res_csv = self.client.get("/api/v1/scenarios/export?format=csv")
        self.assertEqual(res_csv.status_code, 200)
        self.assertIn("VILLAGE_ROAD", res_csv.text)
        self.assertIn("CATTLE_CROSSING", res_csv.text)

        # 4. Dashboard Endpoints
        res_state = self.client.get("/api/v1/dashboard/state")
        self.assertEqual(res_state.status_code, 200)
        self.assertIn("current_action", res_state.json())

        res_metrics = self.client.get("/api/v1/dashboard/metrics")
        self.assertEqual(res_metrics.status_code, 200)
        self.assertIn("processing_fps", res_metrics.json())


if __name__ == "__main__":
    unittest.main()
