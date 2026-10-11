"""
Scenario Manager — Phase 9B.
Coordinates scenario configuration, benchmark execution, comparison reporting,
metrics aggregation, and result export.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any
from app.scenarios.scenario_config import ScenarioID, ScenarioConfig, SCENARIO_CONFIGS
from app.scenarios.scenario_results import ScenarioMetrics, scenario_store
from app.scenarios.scenario_runner import ScenarioRunner


class ScenarioManager:
    """
    Central manager for Indian road scenario benchmarking & performance evaluation.
    """

    @classmethod
    def list_scenarios(cls) -> List[Dict[str, Any]]:
        """List all 5 required scenarios with current testing status."""
        catalog = []
        for sid, config in SCENARIO_CONFIGS.items():
            sid_str = sid.value
            latest = scenario_store.get_result(sid_str)
            catalog.append({
                "scenario_id": sid_str,
                "scenario_name": config.scenario_name,
                "description": config.description,
                "challenges": config.challenges,
                "expected_objects": config.expected_objects,
                "expected_decision_types": config.expected_decision_types,
                "nominal_speed_kmh": config.nominal_speed_kmh,
                "status": "TESTED" if latest is not None else "NOT TESTED",
                "completed": latest.completed if latest else None,
                "collision_free": latest.collision_free if latest else None,
                "path_smoothness": latest.path_smoothness_score if latest else None,
                "replanning_latency_ms": latest.average_replanning_latency_ms if latest else None,
            })
        return catalog

    @classmethod
    def get_scenario_details(cls, scenario_id: str) -> Optional[Dict[str, Any]]:
        """Get complete configuration and latest evaluation result for a scenario."""
        try:
            sid = ScenarioID(scenario_id)
        except ValueError:
            return None

        config = SCENARIO_CONFIGS.get(sid)
        if not config:
            return None

        latest = scenario_store.get_result(sid.value)
        return {
            "config": config.model_dump(),
            "latest_result": latest.model_dump() if latest else None,
            "status": "TESTED" if latest is not None else "NOT TESTED",
        }

    @classmethod
    def run_scenario(cls, scenario_id: str, video_path: Optional[str] = None) -> ScenarioMetrics:
        """Run single scenario through the complete pipeline."""
        return ScenarioRunner.run(scenario_id, video_path=video_path)

    @classmethod
    def run_all_scenarios(cls) -> Dict[str, Any]:
        """Execute all five scenarios sequentially through the complete pipeline."""
        results = {}
        for sid in SCENARIO_CONFIGS.keys():
            res = ScenarioRunner.run(sid)
            results[sid.value] = res
        return {
            "summary": cls.get_summary(),
            "comparison_table": cls.get_comparison_table(),
            "results_count": len(results),
        }

    @classmethod
    def get_comparison_table(cls) -> List[Dict[str, Any]]:
        """Get 5-scenario comparison table with 'NOT TESTED' markers."""
        return scenario_store.get_comparison_table()

    @classmethod
    def get_summary(cls) -> Dict[str, Any]:
        """Get system-wide summary metrics."""
        return scenario_store.get_summary()

    @classmethod
    def export(cls, format: str = "json") -> Tuple[str, str]:
        """Export evaluation results as JSON or CSV with appropriate media type."""
        fmt = format.lower().strip()
        if fmt == "csv":
            content = scenario_store.export_csv()
            media_type = "text/csv"
        else:
            content = scenario_store.export_json()
            media_type = "application/json"
        return content, media_type
