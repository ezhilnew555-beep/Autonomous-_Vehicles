"""
Unit and integration tests for Phase 9B — Indian Road Scenario Testing & Performance Evaluation.
Verifies:
  - 5 scenario configurations (VILLAGE_ROAD, URBAN_UNSIGNALLED_INTERSECTION, HIGHWAY_MERGE, MARKET_AREA, CATTLE_CROSSING)
  - Full pipeline execution via ScenarioRunner
  - Measurable replanning latency (ms) and transparent path smoothness calculation
  - Independence of completed vs collision_free metrics
  - Comparison table with NOT TESTED handling
  - Summary metrics and JSON/CSV export
  - REST API endpoints under /api/v1/scenarios
"""
import sys
import unittest
from starlette.testclient import TestClient

from app.main import app
from app.scenarios.scenario_config import ScenarioID, SCENARIO_CONFIGS
from app.scenarios.scenario_runner import ScenarioRunner
from app.scenarios.scenario_results import scenario_store
from app.scenarios.scenario_manager import ScenarioManager
from app.scenarios.scenario_validators import calculate_path_smoothness, validate_scenario_execution


class TestPhase9BScenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_scenario_configs(self):
        """Verify all 5 required scenarios are defined with full configuration metadata."""
        expected_ids = [
            ScenarioID.VILLAGE_ROAD,
            ScenarioID.URBAN_UNSIGNALLED_INTERSECTION,
            ScenarioID.HIGHWAY_MERGE,
            ScenarioID.MARKET_AREA,
            ScenarioID.CATTLE_CROSSING,
        ]
        self.assertEqual(len(SCENARIO_CONFIGS), 5)
        for sid in expected_ids:
            self.assertIn(sid, SCENARIO_CONFIGS)
            cfg = SCENARIO_CONFIGS[sid]
            self.assertTrue(len(cfg.scenario_name) > 0)
            self.assertTrue(len(cfg.description) > 0)
            self.assertTrue(len(cfg.expected_objects) > 0)
            self.assertTrue(len(cfg.expected_decision_types) > 0)
            self.assertIn("speed_limit_kmh", cfg.expected_environment)

    def test_02_smoothness_calculation(self):
        """Verify reproducible path smoothness formula."""
        # Straight line: should have high smoothness close to 1.0
        straight = [(160.0, 480.0 - i * 10) for i in range(25)]
        s_straight = calculate_path_smoothness(straight)
        self.assertGreaterEqual(s_straight, 0.85)

        # Curved avoidance path: smoothness should be positive and bounded [0.0, 1.0]
        curved = [(160.0 - math_dx(i), 480.0 - i * 10) for i in range(25)]
        s_curved = calculate_path_smoothness(curved, [0.005] * 25)
        self.assertGreaterEqual(s_curved, 0.40)
        self.assertLessEqual(s_curved, 1.0)

    def test_03_scenario_runner_execution_all_5(self):
        """Run each of the 5 scenarios through the complete pipeline."""
        for sid in ScenarioID:
            res = ScenarioRunner.run(sid)
            self.assertEqual(res.scenario_id, sid.value)
            self.assertTrue(res.completed, f"Scenario {sid.value} must complete without fatal pipeline crash")
            self.assertTrue(res.collision_free, f"Adaptive planner must maintain collision-free travel for {sid.value}")
            self.assertEqual(res.collision_events, 0)
            self.assertGreater(res.processing_fps, 0.0)
            self.assertGreaterEqual(res.path_smoothness_score, 0.40)
            self.assertLessEqual(res.path_smoothness_score, 1.0)
            self.assertGreaterEqual(res.average_replanning_latency_ms, 0.0)
            self.assertTrue(len(res.events) > 0)

            # Validate execution against config criteria
            val = validate_scenario_execution(SCENARIO_CONFIGS[sid], res)
            self.assertTrue(val.is_valid, f"Scenario {sid.value} validation failed: {val.failed_checks}")

    def test_04_comparison_table_and_summary(self):
        """Verify 5-scenario comparison table and summary metrics."""
        table = ScenarioManager.get_comparison_table()
        self.assertEqual(len(table), 5)
        for row in table:
            self.assertIn(row["status"], ("TESTED", "NOT TESTED"))
            self.assertIn("scenario", row)
            self.assertIn("completed", row)
            self.assertIn("collision_free", row)
            self.assertIn("avg_replanning_latency", row)
            self.assertIn("path_smoothness", row)

        summary = ScenarioManager.get_summary()
        self.assertEqual(summary["total_scenarios"], 5)
        self.assertEqual(summary["completed_scenarios"], 5)
        self.assertEqual(summary["collision_free_scenarios"], 5)
        self.assertEqual(summary["collision_free_rate_pct"], 100.0)
        self.assertEqual(summary["scenario_completion_rate_text"], "5 / 5")
        self.assertGreater(summary["average_path_smoothness"], 0.0)

    def test_05_api_endpoints(self):
        """Verify REST API endpoints for scenarios, summary, comparison, and export."""
        # 1. GET /api/v1/scenarios
        res_list = self.client.get("/api/v1/scenarios")
        self.assertEqual(res_list.status_code, 200)
        catalog = res_list.json()
        self.assertEqual(len(catalog), 5)

        # 2. GET /api/v1/scenarios/CATTLE_CROSSING
        res_item = self.client.get("/api/v1/scenarios/CATTLE_CROSSING")
        self.assertEqual(res_item.status_code, 200)
        item = res_item.json()
        self.assertEqual(item["config"]["scenario_id"], "CATTLE_CROSSING")
        self.assertIn("latest_result", item)

        # 3. POST /api/v1/scenarios/VILLAGE_ROAD/run
        res_run = self.client.post("/api/v1/scenarios/VILLAGE_ROAD/run")
        self.assertEqual(res_run.status_code, 200)
        run_data = res_run.json()
        self.assertEqual(run_data["scenario_id"], "VILLAGE_ROAD")
        self.assertTrue(run_data["completed"])

        # 4. GET /api/v1/scenarios/summary
        res_sum = self.client.get("/api/v1/scenarios/summary")
        self.assertEqual(res_sum.status_code, 200)
        self.assertEqual(res_sum.json()["total_scenarios"], 5)

        # 5. GET /api/v1/scenarios/comparison
        res_comp = self.client.get("/api/v1/scenarios/comparison")
        self.assertEqual(res_comp.status_code, 200)
        self.assertEqual(len(res_comp.json()), 5)

        # 6. GET /api/v1/scenarios/export?format=json
        res_exp_json = self.client.get("/api/v1/scenarios/export?format=json")
        self.assertEqual(res_exp_json.status_code, 200)
        self.assertEqual(res_exp_json.headers["content-type"], "application/json")
        self.assertIn("summary", res_exp_json.json())

        # 7. GET /api/v1/scenarios/export?format=csv
        res_exp_csv = self.client.get("/api/v1/scenarios/export?format=csv")
        self.assertEqual(res_exp_csv.status_code, 200)
        self.assertIn("text/csv", res_exp_csv.headers["content-type"])
        self.assertIn("Scenario Name", res_exp_csv.text)


def math_dx(i: int) -> float:
    u = i / 24.0
    return 80.0 * (10.0 * (u ** 3) - 15.0 * (u ** 4) + 6.0 * (u ** 5))


if __name__ == "__main__":
    unittest.main()
