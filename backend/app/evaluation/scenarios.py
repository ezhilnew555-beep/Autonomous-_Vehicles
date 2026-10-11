"""
Scenario Definitions & Dynamic Generators — Phase 9.

Implements the five required Indian road testing situations:
1. Unmarked village road
2. Busy urban intersection without signals
3. Highway merge with slow-moving vehicles
4. Dense market area with mixed traffic
5. Sudden cattle-crossing event
"""
from __future__ import annotations

from typing import Dict, List
from app.evaluation.schemas import ScenarioId, ScenarioInfo


SCENARIOS_CATALOG: Dict[ScenarioId, ScenarioInfo] = {
    ScenarioId.UNMARKED_VILLAGE_ROAD: ScenarioInfo(
        id=ScenarioId.UNMARKED_VILLAGE_ROAD,
        title="Unmarked Village Road",
        subtitle="Narrow rural corridor with degraded earthen shoulders and oncoming bicycle",
        indian_road_context=(
            "Rural single-lane asphalt road with no painted lane markings, eroded gravel edges, "
            "and pedestrians walking alongside. Road width narrows abruptly from 5.2m to 3.8m."
        ),
        challenges=[
            "Absence of centerline and lane boundary markers",
            "Irregular road edge boundaries (earthen shoulders)",
            "Oncoming bicycle moving along the vehicle's nominal path",
            "Requires conservative speed governance and lateral offset tracking",
        ],
        road_geometry="UNMARKED_NARROW_SINGLE_LANE",
        obstacle_types=["bicycle", "person"],
        simulated_duration_sec=6.0,
        nominal_speed_kmh=35.0,
    ),

    ScenarioId.URBAN_INTERSECTION_NO_SIGNALS: ScenarioInfo(
        id=ScenarioId.URBAN_INTERSECTION_NO_SIGNALS,
        title="Busy Urban Intersection Without Signals",
        subtitle="Uncontrolled 4-way junction with cross-cutting auto-rickshaws and two-wheelers",
        indian_road_context=(
            "Chaotic tier-2 city four-way junction operating without functioning signals. "
            "Multiple road users navigate via opportunistic gaps rather than right-of-way rules."
        ),
        challenges=[
            "Unsignalized 4-way crossing with high directional entropy",
            "Auto-rickshaw making an aggressive right turn cutting directly across ego corridor",
            "Two-wheeler filtering across the front bumper",
            "Requires speed reduction to 15 km/h and dynamic conflict point resolution",
        ],
        road_geometry="INTERSECTION_4WAY_UNCONTROLLED",
        obstacle_types=["autorickshaw", "motorcycle", "person"],
        simulated_duration_sec=7.0,
        nominal_speed_kmh=40.0,
    ),

    ScenarioId.HIGHWAY_MERGE_SLOW_VEHICLES: ScenarioInfo(
        id=ScenarioId.HIGHWAY_MERGE_SLOW_VEHICLES,
        title="Highway Merge with Slow Vehicles",
        subtitle="High-speed arterial entry where an overloaded tractor merges at 20 km/h",
        indian_road_context=(
            "Multi-lane National Highway corridor where ego vehicle cruises at 70 km/h. "
            "An agricultural tractor towing an overloaded trailer merges abruptly from a side ramp at 20 km/h."
        ),
        challenges=[
            "High speed differential (50 km/h delta)",
            "Short stopping sight distance if vehicle only relies on straight-line braking",
            "Overloaded cargo protruding beyond vehicle footprint",
            "Demands high-speed smooth lane change replan without vehicle destabilization",
        ],
        road_geometry="HIGHWAY_MULTI_LANE_MERGE",
        obstacle_types=["tractor", "truck", "car"],
        simulated_duration_sec=5.5,
        nominal_speed_kmh=70.0,
    ),

    ScenarioId.DENSE_MARKET_MIXED_TRAFFIC: ScenarioInfo(
        id=ScenarioId.DENSE_MARKET_MIXED_TRAFFIC,
        title="Dense Market Area with Mixed Traffic",
        subtitle="Bustling commercial street with handcarts, pedestrians, and tight clearance",
        indian_road_context=(
            "High-density commercial bazaar street with mixed non-motorized and motorized traffic. "
            "Shop fronts encroach on roadway, leaving a variable clearance gap under 2.0 meters."
        ),
        challenges=[
            "Extreme spatial confinement with lateral clearance < 1.5m",
            "Hand-drawn vegetable cart stopped partially inside the travel corridor",
            "Pedestrians dynamically stepping into road from between parked two-wheelers",
            "Requires continuous low-speed spline micro-adjustments and maximum sensor vigilance",
        ],
        road_geometry="URBAN_SHARED_STREET_NARROW",
        obstacle_types=["person", "handcart", "motorcycle"],
        simulated_duration_sec=8.0,
        nominal_speed_kmh=25.0,
    ),

    ScenarioId.SUDDEN_CATTLE_CROSSING: ScenarioInfo(
        id=ScenarioId.SUDDEN_CATTLE_CROSSING,
        title="Sudden Cattle-Crossing Event",
        subtitle="Stray bovine emerges perpendicular to path with 1.2s critical lead time",
        indian_road_context=(
            "Suburban arterial road with median greenery. A cow emerges suddenly from behind "
            "median bushes into the vehicle's right-half travel corridor at a close distance of 14 meters."
        ),
        challenges=[
            "Short Time-To-Collision (TTC = 1.2s) with sudden occlusion exit",
            "High obstacle mass requiring extra safety buffer (safety envelope +45px)",
            "Unpredictable animal motion trajectory (may halt mid-lane or accelerate)",
            "Demands immediate emergency braking combined with lateral evasion spline to the open left shoulder",
        ],
        road_geometry="SUBURBAN_DIVIDED_ARTERIAL",
        obstacle_types=["cow", "motorcycle"],
        simulated_duration_sec=5.0,
        nominal_speed_kmh=45.0,
    ),
}


def get_scenario_info(scenario_id: ScenarioId) -> ScenarioInfo:
    return SCENARIOS_CATALOG[scenario_id]


def list_all_scenarios() -> List[ScenarioInfo]:
    return list(SCENARIOS_CATALOG.values())
