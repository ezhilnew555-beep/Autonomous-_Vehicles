"""
Scenario Configurations — Phase 9B.
Defines the five required Indian road testing situations:
1. VILLAGE_ROAD: Unmarked village road
2. URBAN_UNSIGNALLED_INTERSECTION: Busy urban intersection without signals
3. HIGHWAY_MERGE: Highway merge with slow-moving vehicles
4. MARKET_AREA: Dense market area with mixed traffic
5. CATTLE_CROSSING: Sudden cattle-crossing event
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Any
from pydantic import BaseModel, Field


class ScenarioID(str, Enum):
    VILLAGE_ROAD                    = "VILLAGE_ROAD"
    URBAN_UNSIGNALLED_INTERSECTION  = "URBAN_UNSIGNALLED_INTERSECTION"
    HIGHWAY_MERGE                   = "HIGHWAY_MERGE"
    MARKET_AREA                     = "MARKET_AREA"
    CATTLE_CROSSING                 = "CATTLE_CROSSING"


class ScenarioConfig(BaseModel):
    scenario_id: ScenarioID
    scenario_name: str
    description: str
    challenges: List[str] = Field(default_factory=list)
    expected_environment: Dict[str, Any] = Field(default_factory=dict)
    expected_objects: List[str] = Field(default_factory=list)
    expected_risk_events: List[str] = Field(default_factory=list)
    expected_decision_types: List[str] = Field(default_factory=list)
    evaluation_conditions: Dict[str, Any] = Field(default_factory=dict)
    simulated_duration_sec: float = 6.0
    nominal_speed_kmh: float = 35.0


SCENARIO_CONFIGS: Dict[ScenarioID, ScenarioConfig] = {
    ScenarioID.VILLAGE_ROAD: ScenarioConfig(
        scenario_id=ScenarioID.VILLAGE_ROAD,
        scenario_name="Unmarked Village Road",
        description=(
            "Narrow rural corridor with degraded earthen shoulders, no painted lane markings, "
            "and unpredictable movement of oncoming bicycles and pedestrians."
        ),
        challenges=[
            "Absence of centerline and lane boundary markers",
            "Irregular road edge boundaries (loose earthen berms)",
            "Oncoming bicycle moving along the vehicle's nominal path",
            "Planner must dynamically stay within drivable corridor without relying on paint",
        ],
        expected_environment={
            "road_type": "VILLAGE_ROAD",
            "is_unmarked": True,
            "has_intersection": False,
            "speed_limit_kmh": 30,
            "drivable_boundaries_px": [120, 520],
        },
        expected_objects=["bicycle", "person", "animal"],
        expected_risk_events=["LOW", "MEDIUM", "HIGH"],
        expected_decision_types=["SLOW_DOWN", "AVOID", "REPLAN", "CONTINUE"],
        evaluation_conditions={
            "require_drivable_corridor_compliance": True,
            "require_collision_free": True,
            "max_planning_latency_ms": 15.0,
        },
        simulated_duration_sec=6.0,
        nominal_speed_kmh=30.0,
    ),

    ScenarioID.URBAN_UNSIGNALLED_INTERSECTION: ScenarioConfig(
        scenario_id=ScenarioID.URBAN_UNSIGNALLED_INTERSECTION,
        scenario_name="Busy Urban Intersection Without Signals",
        description=(
            "Chaotic tier-2 city 4-way junction operating without functioning signals. "
            "Auto-rickshaws and two-wheelers cross opportunistically. The system must not assume green."
        ),
        challenges=[
            "Unsignalized 4-way crossing with high directional entropy",
            "Cross-cutting auto-rickshaw and filtering motorcycles",
            "Conservative approach: no signal != permission to continue",
            "Multi-object trajectory conflict resolution and headway management",
        ],
        expected_environment={
            "road_type": "URBAN_INTERSECTION",
            "is_unmarked": False,
            "has_intersection": True,
            "geometry_type": "INTERSECTION_4WAY",
            "speed_limit_kmh": 30,
            "drivable_boundaries_px": [80, 560],
        },
        expected_objects=["car", "motorcycle", "autorickshaw", "person"],
        expected_risk_events=["MEDIUM", "HIGH"],
        expected_decision_types=["SLOW_DOWN", "STOP", "AVOID", "REPLAN"],
        evaluation_conditions={
            "prohibit_uncontrolled_acceleration": True,
            "require_collision_free": True,
            "max_planning_latency_ms": 15.0,
        },
        simulated_duration_sec=7.0,
        nominal_speed_kmh=35.0,
    ),

    ScenarioID.HIGHWAY_MERGE: ScenarioConfig(
        scenario_id=ScenarioID.HIGHWAY_MERGE,
        scenario_name="Highway Merge with Slow Vehicles",
        description=(
            "Cruising on high-speed National Highway corridor where an overloaded tractor "
            "merges abruptly from the left acceleration berm at 20 km/h."
        ),
        challenges=[
            "High speed differential (60-70 km/h cruising vs 20 km/h merging obstacle)",
            "Tractor cutting into primary cruising corridor",
            "Safe gap estimation and high-speed lateral corridor re-selection",
            "Avoid selecting any path that intersects predicted vehicle trajectories",
        ],
        expected_environment={
            "road_type": "HIGHWAY_MERGE",
            "is_unmarked": False,
            "has_intersection": False,
            "geometry_type": "HIGHWAY_MERGE",
            "speed_limit_kmh": 60,
            "drivable_boundaries_px": [60, 580],
        },
        expected_objects=["truck", "car", "tractor"],
        expected_risk_events=["HIGH", "CRITICAL"],
        expected_decision_types=["AVOID", "REPLAN", "SLOW_DOWN"],
        evaluation_conditions={
            "require_lateral_clearance_margin_m": 2.0,
            "require_collision_free": True,
            "max_planning_latency_ms": 15.0,
        },
        simulated_duration_sec=6.5,
        nominal_speed_kmh=60.0,
    ),

    ScenarioID.MARKET_AREA: ScenarioConfig(
        scenario_id=ScenarioID.MARKET_AREA,
        scenario_name="Dense Market Area with Mixed Traffic",
        description=(
            "Confined bazaar street with dense pedestrian crowd, street vendors, handcarts, "
            "and two-wheelers. Action is determined by actual risk and environmental context, not hardcoded stop."
        ),
        challenges=[
            "Dense pedestrian crowd and handcarts flanking both curbs",
            "Restricted lateral clearance (< 2.0m corridor gaps)",
            "Frequent micro-trajectory replanning based on pedestrian intent",
            "Strict throttle governance without unnecessary total stagnation",
        ],
        expected_environment={
            "road_type": "MARKET_AREA",
            "is_unmarked": True,
            "has_intersection": False,
            "speed_limit_kmh": 20,
            "drivable_boundaries_px": [140, 500],
        },
        expected_objects=["person", "bicycle", "motorcycle", "cart"],
        expected_risk_events=["LOW", "MEDIUM", "HIGH"],
        expected_decision_types=["SLOW_DOWN", "AVOID", "REPLAN", "CONTINUE", "STOP"],
        evaluation_conditions={
            "max_speed_kmh": 20.0,
            "require_collision_free": True,
            "max_planning_latency_ms": 15.0,
        },
        simulated_duration_sec=8.0,
        nominal_speed_kmh=20.0,
    ),

    ScenarioID.CATTLE_CROSSING: ScenarioConfig(
        scenario_id=ScenarioID.CATTLE_CROSSING,
        scenario_name="Sudden Cattle Crossing",
        description=(
            "Stray cow suddenly steps into the road corridor from the median with a critical 1.2s lead time. "
            "The system must track, predict trajectory, assess rising risk, and execute evasive avoidance or emergency stop."
        ),
        challenges=[
            "Non-standard animal gait and unpredictable road dwell time",
            "Sudden lateral path encroachment (< 1.5s critical TTC)",
            "Emergency braking coordination with evasive shoulder spline",
            "Strict safety precedence: obstacle collision avoidance overrides road context",
        ],
        expected_environment={
            "road_type": "VILLAGE_ROAD",
            "is_unmarked": True,
            "has_intersection": False,
            "speed_limit_kmh": 35,
            "drivable_boundaries_px": [120, 520],
        },
        expected_objects=["animal"],
        expected_risk_events=["HIGH", "CRITICAL"],
        expected_decision_types=["SLOW_DOWN", "STOP", "AVOID", "REPLAN", "EMERGENCY_STOP"],
        evaluation_conditions={
            "require_rapid_ttc_detection": True,
            "require_collision_free": True,
            "max_planning_latency_ms": 15.0,
        },
        simulated_duration_sec=5.5,
        nominal_speed_kmh=35.0,
    ),
}
