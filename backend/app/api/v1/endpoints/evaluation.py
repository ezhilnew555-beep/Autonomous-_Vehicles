"""
Evaluation & Scenario Benchmarking Endpoints — Phase 9.

Provides REST endpoints for:
- Listing required Indian road test scenarios
- Running deterministic comparative benchmarks (Baseline vs Adaptive Planner)
- Generating comprehensive system-level performance summaries
"""
from __future__ import annotations

from typing import List
from fastapi import APIRouter, HTTPException, status

from app.evaluation.schemas import (
    ScenarioId,
    ScenarioInfo,
    ScenarioBenchmarkResult,
    SystemEvaluationSummary,
)
from app.evaluation.scenarios import list_all_scenarios, get_scenario_info
from app.evaluation.benchmark_runner import BenchmarkRunner
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/evaluation", tags=["Evaluation & Benchmarks"])


@router.get(
    "/scenarios",
    response_model=List[ScenarioInfo],
    summary="List all Indian road benchmark scenarios",
)
def get_scenarios() -> List[ScenarioInfo]:
    """Retrieve catalog of the 5 required Indian road testing situations."""
    return list_all_scenarios()


@router.post(
    "/run/{scenario_id}",
    response_model=ScenarioBenchmarkResult,
    summary="Run comparative benchmark for a specific scenario",
)
def run_scenario_benchmark(scenario_id: ScenarioId) -> ScenarioBenchmarkResult:
    """
    Simulate scenario execution comparing BASELINE / FIXED PATH vs ADAPTIVE PATH PLANNER.
    Returns collision count, minimum safety distance, TTC statistics, replanning latency,
    path length, path smoothness, and trajectory comparison points.
    """
    try:
        logger.info(f"Running scenario benchmark for {scenario_id.value}...")
        result = BenchmarkRunner.run_scenario(scenario_id)
        return result
    except Exception as exc:
        logger.exception(f"Benchmark execution failed for {scenario_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Benchmark execution failed: {exc}",
        )


@router.get(
    "/summary",
    response_model=SystemEvaluationSummary,
    summary="Retrieve comprehensive system evaluation summary",
)
def get_evaluation_summary() -> SystemEvaluationSummary:
    """
    Returns full evaluation report across all 5 scenarios including detection performance,
    tracking performance, collision counts, completion rates, and average latencies.
    """
    try:
        return BenchmarkRunner.get_full_evaluation_summary()
    except Exception as exc:
        logger.exception(f"Failed to generate evaluation summary: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate evaluation summary: {exc}",
        )
