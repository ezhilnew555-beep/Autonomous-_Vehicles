"""
Scenario Results, Metrics & Storage — Phase 9B.
Handles structured metrics, event logging (with state-change deduplication),
persistent JSON storage, and JSON/CSV export.
"""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from app.scenarios.scenario_config import ScenarioID, SCENARIO_CONFIGS


class ScenarioEvent(BaseModel):
    timestamp: str
    scenario: str
    event: str
    object_id: Optional[int] = None
    object_class: Optional[str] = None
    risk: str = "LOW"
    ttc: Optional[float] = None
    decision: str = "CONTINUE"
    path: str = "KEEP_CURRENT"
    reason: str = ""


class ScenarioMetrics(BaseModel):
    scenario_id: str
    scenario_name: str
    video_name: str
    start_time: str
    end_time: str
    completed: bool = True
    collision_free: bool = True
    collision_events: int = 0
    replans: int = 0
    average_replanning_latency_ms: float = 0.0
    minimum_replanning_latency_ms: float = 0.0
    maximum_replanning_latency_ms: float = 0.0
    path_smoothness_score: float = 1.0
    detection_fps: float = 0.0
    processing_fps: float = 0.0
    average_planning_latency_ms: float = 0.0
    maximum_planning_latency_ms: float = 0.0
    total_detected_objects: int = 0
    total_tracked_objects: int = 0
    total_risk_events: int = 0
    emergency_stops: int = 0
    final_decision: str = "CONTINUE"
    final_path: str = "KEEP_CURRENT"
    failure_reason: Optional[str] = None
    events: List[ScenarioEvent] = Field(default_factory=list)


class ScenarioResultsStore:
    """
    Local persistent store for scenario evaluation benchmark results.
    Avoids external database dependencies by maintaining a clean JSON store.
    """

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        if storage_path is None:
            # backend/app/data/scenario_evaluations.json
            base_dir = Path(__file__).resolve().parent.parent
            self.storage_file = base_dir / "data" / "scenario_evaluations.json"
        else:
            self.storage_file = storage_path
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        self._results: Dict[str, ScenarioMetrics] = {}
        self._load()

    def _load(self) -> None:
        if not self.storage_file.exists():
            return
        try:
            with open(self.storage_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for k, v in data.items():
                    self._results[k] = ScenarioMetrics(**v)
        except Exception:
            self._results = {}

    def _save(self) -> None:
        try:
            with open(self.storage_file, "w", encoding="utf-8") as f:
                serialized = {k: v.model_dump() for k, v in self._results.items()}
                json.dump(serialized, f, indent=2)
        except Exception:
            pass

    def save_result(self, result: ScenarioMetrics) -> None:
        self._results[result.scenario_id] = result
        self._save()

    def get_result(self, scenario_id: str) -> Optional[ScenarioMetrics]:
        return self._results.get(scenario_id)

    def get_all_results(self) -> Dict[str, ScenarioMetrics]:
        return dict(self._results)

    def get_comparison_table(self) -> List[Dict[str, Any]]:
        """
        Comparison table across all 5 required scenarios.
        Untested scenarios are explicitly marked 'NOT TESTED' per Section 18.
        """
        table: List[Dict[str, Any]] = []

        for sid, config in SCENARIO_CONFIGS.items():
            sid_str = sid.value
            res = self._results.get(sid_str)

            if res is not None:
                table.append({
                    "scenario_id": sid_str,
                    "scenario": config.scenario_name,
                    "status": "TESTED",
                    "completed": "YES" if res.completed else "NO",
                    "collision_free": "YES" if res.collision_free else "NO",
                    "replans": res.replans,
                    "avg_replanning_latency": f"{res.average_replanning_latency_ms:.1f} ms",
                    "path_smoothness": f"{res.path_smoothness_score:.2f}",
                    "processing_fps": f"{res.processing_fps:.1f} fps",
                    "collision_events": res.collision_events,
                    "final_decision": res.final_decision,
                    "final_path": res.final_path,
                })
            else:
                table.append({
                    "scenario_id": sid_str,
                    "scenario": config.scenario_name,
                    "status": "NOT TESTED",
                    "completed": "NOT TESTED",
                    "collision_free": "NOT TESTED",
                    "replans": "NOT TESTED",
                    "avg_replanning_latency": "NOT TESTED",
                    "path_smoothness": "NOT TESTED",
                    "processing_fps": "NOT TESTED",
                    "collision_events": "NOT TESTED",
                    "final_decision": "NOT TESTED",
                    "final_path": "NOT TESTED",
                })

        return table

    def get_summary(self) -> Dict[str, Any]:
        """
        Overall evaluation summary per Section 19:
        - SCENARIO COMPLETION RATE (e.g. 5 / 5)
        - COLLISION-FREE RATE (%)
        - AVERAGE REPLANNING LATENCY (ms)
        - AVERAGE PATH SMOOTHNESS
        - TOTAL REPLANS
        - TOTAL COLLISION EVENTS
        - PROCESSING FPS
        """
        total_scenarios = len(SCENARIO_CONFIGS)
        completed_scenarios = sum(1 for r in self._results.values() if r.completed)
        collision_free_scenarios = sum(1 for r in self._results.values() if r.completed and r.collision_free)
        total_collisions = sum(r.collision_events for r in self._results.values())
        total_replans = sum(r.replans for r in self._results.values())

        latencies = [r.average_replanning_latency_ms for r in self._results.values() if r.average_replanning_latency_ms > 0]
        avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0.0

        smoothness_scores = [r.path_smoothness_score for r in self._results.values()]
        avg_smoothness = round(sum(smoothness_scores) / len(smoothness_scores), 2) if smoothness_scores else 0.0

        fps_list = [r.processing_fps for r in self._results.values() if r.processing_fps > 0]
        avg_fps = round(sum(fps_list) / len(fps_list), 1) if fps_list else 0.0

        collision_free_rate = (
            round((collision_free_scenarios / completed_scenarios) * 100.0, 1)
            if completed_scenarios > 0 else 0.0
        )

        return {
            "total_scenarios": total_scenarios,
            "completed_scenarios": completed_scenarios,
            "scenario_completion_rate_text": f"{completed_scenarios} / {total_scenarios}",
            "scenario_completion_rate_pct": round((completed_scenarios / total_scenarios) * 100.0, 1),
            "collision_free_scenarios": collision_free_scenarios,
            "collision_free_rate_pct": collision_free_rate,
            "total_collision_events": total_collisions,
            "total_replans": total_replans,
            "average_replanning_latency_ms": avg_latency,
            "average_path_smoothness": avg_smoothness,
            "average_processing_fps": avg_fps,
            "tested_count": len(self._results),
            "untested_count": max(0, total_scenarios - len(self._results)),
        }

    def export_json(self) -> str:
        """Export all tested scenario evaluation results as formatted JSON."""
        summary = self.get_summary()
        comparison = self.get_comparison_table()
        export_payload = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "comparison_table": comparison,
            "detailed_results": {k: v.model_dump() for k, v in self._results.items()},
        }
        return json.dumps(export_payload, indent=2)

    def export_csv(self) -> str:
        """Export comparison table as standard CSV."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Scenario ID",
            "Scenario Name",
            "Status",
            "Completed",
            "Collision Free",
            "Replans",
            "Avg Replanning Latency",
            "Path Smoothness",
            "Processing FPS",
            "Collision Events",
            "Final Decision",
            "Final Path",
        ])
        for row in self.get_comparison_table():
            writer.writerow([
                row["scenario_id"],
                row["scenario"],
                row["status"],
                row["completed"],
                row["collision_free"],
                row["replans"],
                row["avg_replanning_latency"],
                row["path_smoothness"],
                row["processing_fps"],
                row["collision_events"],
                row["final_decision"],
                row["final_path"],
            ])
        return output.getvalue()


# Default singleton instance
scenario_store = ScenarioResultsStore()
