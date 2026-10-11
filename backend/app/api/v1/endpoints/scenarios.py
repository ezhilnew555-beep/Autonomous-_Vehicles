"""
Scenario API Endpoints — Phase 9B.
Allows users to list scenarios, execute scenario tests, retrieve comparison tables,
view performance metrics summaries, and export results as JSON/CSV.
"""
from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import PlainTextResponse

from app.scenarios.scenario_manager import ScenarioManager

router = APIRouter()


@router.get("")
async def list_scenarios():
    """List all five required Indian road testing scenarios and their evaluation status."""
    return ScenarioManager.list_scenarios()


@router.get("/summary")
async def get_scenarios_summary():
    """
    Get aggregated system performance summary:
    Completion rate (e.g. 5 / 5), collision-free rate, avg latency, avg smoothness.
    """
    return ScenarioManager.get_summary()


@router.get("/comparison")
async def get_scenarios_comparison():
    """
    Get 5-scenario comparison table with 'NOT TESTED' for untested scenarios.
    """
    return ScenarioManager.get_comparison_table()


@router.get("/export")
async def export_scenarios_results(
    format: str = Query("json", description="'json' or 'csv'")
):
    """Export all scenario test results as a downloadable JSON or CSV file."""
    content, media_type = ScenarioManager.export(format)
    filename = f"eareyes_scenario_evaluation.{format.lower()}"
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/{scenario_id}")
async def get_scenario(scenario_id: str):
    """Retrieve full configuration and latest test result for a given scenario."""
    details = ScenarioManager.get_scenario_details(scenario_id)
    if not details:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found.")
    return details


@router.post("/{scenario_id}/run")
async def run_scenario(
    scenario_id: str,
    video_path: Optional[str] = Query(None, description="Optional path to custom video file")
):
    """
    Execute a scenario through the complete 10-stage autonomous pipeline.
    Calculates actual replanning latency, path smoothness, and collision metrics.
    """
    try:
        metrics = ScenarioManager.run_scenario(scenario_id, video_path=video_path)
        return metrics.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Scenario execution failed: {exc}")


@router.post("/run-all")
async def run_all_scenarios():
    """Execute all five scenarios sequentially and return comparison table and summary."""
    try:
        return ScenarioManager.run_all_scenarios()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch execution failed: {exc}")
