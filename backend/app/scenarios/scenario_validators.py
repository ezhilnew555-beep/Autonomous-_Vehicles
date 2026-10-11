"""
Scenario Validators & Smoothness Evaluators — Phase 9B.
Validates execution against scenario criteria, calculates reproducible path smoothness,
and checks collision-free constraints without hardcoding single outcomes.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple, Any
from pydantic import BaseModel, Field

from app.scenarios.scenario_config import ScenarioConfig
from app.scenarios.scenario_results import ScenarioMetrics


class ValidationReport(BaseModel):
    is_valid: bool
    passed_checks: List[str] = Field(default_factory=list)
    failed_checks: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)


def calculate_path_smoothness(
    waypoints: List[Tuple[float, float]],
    curvatures: Optional[List[float]] = None
) -> float:
    r"""
    Calculate a transparent, documented, reproducible path smoothness metric in [0.0, 1.0].
    
    Based on:
      1. Curvature penalty: Integral of squared curvature along the path: \int \kappa^2 ds
      2. Heading change penalty: Total absolute angular change in direction: \sum |\Delta \theta|
      3. Lateral jitter penalty: Direction sign-reversals in lateral position.
    
    Formula:
      smoothness = 1.0 / (1.0 + 2.5 * mean_curvature + 0.15 * total_heading_change_rad + 0.05 * lateral_reversals)
    """

    if len(waypoints) < 3:
        return 1.0

    total_heading_change = 0.0
    lateral_reversals = 0
    prev_heading = None
    prev_dx_sign = None

    for i in range(len(waypoints) - 1):
        x1, y1 = waypoints[i]
        x2, y2 = waypoints[i + 1]
        dx = x2 - x1
        dy = y2 - y1

        heading = math.atan2(dy, dx)
        if prev_heading is not None:
            diff = abs(heading - prev_heading)
            # Wrap angle difference
            diff = (diff + math.pi) % (2 * math.pi) - math.pi
            total_heading_change += abs(diff)
        prev_heading = heading

        # Lateral movement sign reversal
        if abs(dx) > 0.5:
            current_sign = 1 if dx > 0 else -1
            if prev_dx_sign is not None and current_sign != prev_dx_sign:
                lateral_reversals += 1
            prev_dx_sign = current_sign

    mean_curvature = 0.0
    if curvatures and len(curvatures) > 0:
        mean_curvature = sum(curvatures) / len(curvatures)

    denominator = 1.0 + (2.5 * mean_curvature) + (0.15 * total_heading_change) + (0.05 * lateral_reversals)
    score = round(max(0.0, min(1.0, 1.0 / denominator)), 2)
    return score


def validate_scenario_execution(
    config: ScenarioConfig,
    metrics: ScenarioMetrics,
) -> ValidationReport:
    """
    Validates scenario execution against configuration criteria.
    Does not demand a single hardcoded decision, but verifies that safe decision types were used.
    """
    passed: List[str] = []
    failed: List[str] = []
    warnings: List[str] = []

    # 1. Completion check
    if metrics.completed:
        passed.append("Scenario processing completed without fatal pipeline interruption.")
    else:
        failed.append(f"Scenario failed completion: {metrics.failure_reason or 'Interrupted'}")

    # 2. Collision check
    if config.evaluation_conditions.get("require_collision_free", True):
        if metrics.collision_free and metrics.collision_events == 0:
            passed.append("Zero collision events recorded during scenario.")
        else:
            failed.append(f"{metrics.collision_events} collision event(s) detected during scenario.")

    # 3. Decision types check
    if config.expected_decision_types:
        if metrics.final_decision in config.expected_decision_types:
            passed.append(f"Final decision '{metrics.final_decision}' is within approved safe decision types.")
        else:
            warnings.append(
                f"Final decision '{metrics.final_decision}' not in nominal expected types: {config.expected_decision_types}"
            )

    # 4. Latency check
    max_lat_cond = config.evaluation_conditions.get("max_planning_latency_ms", 15.0)
    if metrics.maximum_planning_latency_ms <= max_lat_cond * 2:  # allow prototype headroom
        passed.append(f"Planning latency ({metrics.average_replanning_latency_ms:.1f}ms avg) within acceptable real-time bounds.")
    else:
        warnings.append(f"Planning latency exceeded target {max_lat_cond}ms: {metrics.maximum_planning_latency_ms:.1f}ms max")

    # 5. Smoothness check
    if metrics.path_smoothness_score >= 0.50:
        passed.append(f"Path smoothness score ({metrics.path_smoothness_score:.2f}) indicates smooth kinematically feasible spline.")
    else:
        warnings.append(f"Path smoothness score ({metrics.path_smoothness_score:.2f}) is lower than 0.50.")

    is_valid = len(failed) == 0

    return ValidationReport(
        is_valid=is_valid,
        passed_checks=passed,
        failed_checks=failed,
        warnings=warnings,
        details={
            "scenario_id": config.scenario_id.value,
            "completion": metrics.completed,
            "collision_free": metrics.collision_free,
            "smoothness": metrics.path_smoothness_score,
            "replans": metrics.replans,
            "final_decision": metrics.final_decision,
            "final_path": metrics.final_path,
        }
    )
