"""
Evaluation & Scenario Benchmarking Data Schemas — Phase 9.

Defines schemas for:
- 5 Mandatory Indian road test scenarios
- Baseline (Fixed Path) vs. Adaptive Path Planner performance metrics
- Frame-by-frame trajectory telemetry
- Comprehensive system-level evaluation summaries
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ScenarioId(str, Enum):
    UNMARKED_VILLAGE_ROAD          = "UNMARKED_VILLAGE_ROAD"
    URBAN_INTERSECTION_NO_SIGNALS  = "URBAN_INTERSECTION_NO_SIGNALS"
    HIGHWAY_MERGE_SLOW_VEHICLES    = "HIGHWAY_MERGE_SLOW_VEHICLES"
    DENSE_MARKET_MIXED_TRAFFIC     = "DENSE_MARKET_MIXED_TRAFFIC"
    SUDDEN_CATTLE_CROSSING         = "SUDDEN_CATTLE_CROSSING"


class ScenarioInfo(BaseModel):
    id: ScenarioId
    title: str
    subtitle: str
    indian_road_context: str
    challenges: List[str]
    road_geometry: str
    obstacle_types: List[str]
    simulated_duration_sec: float
    nominal_speed_kmh: float


class TrajectoryPoint2D(BaseModel):
    step: int
    x: float
    y: float
    speed_kmh: float
    clearance_m: float
    is_evading: bool = False


class PlannerPerformanceMetrics(BaseModel):
    planner_type: str = Field(..., description="'BASELINE_FIXED_PATH' or 'ADAPTIVE_PATH_PLANNER'")
    collision_count: int
    min_safety_distance_m: float
    avg_safety_distance_m: float
    min_ttc_sec: Optional[float] = None
    avg_ttc_sec: Optional[float] = None
    critical_ttc_events: int
    avg_replanning_latency_ms: float
    max_replanning_latency_ms: float
    total_path_length_m: float
    path_smoothness_score: float = Field(..., description="Curvature integral (lower = smoother, less jerk)")
    scenario_completed: bool
    completion_time_sec: float
    dominant_action: str
    avoidance_maneuvers: int
    final_status: str
    trajectory: List[TrajectoryPoint2D] = []
    telemetry_summary: str


class ScenarioBenchmarkResult(BaseModel):
    scenario: ScenarioInfo
    baseline_metrics: PlannerPerformanceMetrics
    adaptive_metrics: PlannerPerformanceMetrics
    safety_improvement_pct: float
    latency_advantage: str
    engineering_verdict: str


class SystemEvaluationSummary(BaseModel):
    detection_precision: float
    detection_recall: float
    detection_map50: float
    tracking_mota: float
    tracking_id_switches: int
    total_scenarios_tested: int
    adaptive_completion_rate: float
    baseline_completion_rate: float
    adaptive_total_collisions: int
    baseline_total_collisions: int
    avg_replanning_latency_ms: float
    scenarios: List[ScenarioBenchmarkResult]
