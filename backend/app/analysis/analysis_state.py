"""
Analysis state definitions, enums, and mapping helpers matching the non-technical UI presentation.
"""
from __future__ import annotations
from enum import Enum
from typing import Optional, Dict, Any


class RoadStatus(str, Enum):
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    HIGH_RISK = "HIGH RISK"
    EMERGENCY = "EMERGENCY"
    INSUFFICIENT_DATA = "INSUFFICIENT DATA"


ROAD_STATUS_DESCRIPTIONS = {
    RoadStatus.SAFE: "Road corridor is clear. Cruising safely with no immediate collision threats.",
    RoadStatus.CAUTION: "Caution advised: moderate traffic density, intersecting geometry, or road features ahead.",
    RoadStatus.HIGH_RISK: "An obstacle or road user is entering the predicted vehicle path. Evasive action active.",
    RoadStatus.EMERGENCY: "CRITICAL HAZARD: Immediate stop or emergency evasive maneuver in progress.",
    RoadStatus.INSUFFICIENT_DATA: "Insufficient visual data to reliably evaluate road conditions.",
}


def map_decision_to_presentation_action(decision_action: str) -> str:
    """Map internal DecisionAction to the canonical UI presentation action."""
    act = (decision_action or "").upper()
    if "EMERGENCY" in act:
        return "EMERGENCY STOP"
    elif "STOP" in act or "WAIT" in act:
        return "STOP"
    elif "AVOID" in act or "STEER" in act:
        return "AVOID OBSTACLE"
    elif "REPLAN" in act or "LANE" in act or "DETOUR" in act:
        return "CHANGE PATH"
    elif "SLOW" in act or "APPROACH" in act or "YIELD" in act or "CAUTION" in act:
        return "SLOW DOWN"
    elif "NO_SAFE" in act or "BLOCKED" in act:
        return "NO SAFE ACTION"
    else:
        return "CONTINUE"


map_decision_to_plain_action = map_decision_to_presentation_action


def map_planning_status_to_label(status_str: str, needs_replanning: bool) -> str:
    """Map planner status to Section 8 canonical states: SAFE / REPLANNING / BLOCKED / UNKNOWN."""
    s = (status_str or "").upper()
    if needs_replanning or "REPLAN" in s:
        return "REPLANNING"
    elif "BLOCKED" in s or "CRITICAL" in s or "NO_SAFE" in s:
        return "BLOCKED"
    elif "SAFE" in s or "OPTIMAL" in s or "CLEAR" in s or "CONTINUE" in s:
        return "SAFE"
    elif not status_str:
        return "UNKNOWN"
    else:
        return "SAFE"


def map_road_type_to_presentation(raw_road_type: Any) -> str:
    """Map internal RoadType enum or string to canonical human-readable road type."""
    t = str(getattr(raw_road_type, "value", raw_road_type) or "").upper()
    if "VILLAGE" in t:
        return "Village Road"
    elif "INTERSECTION" in t:
        return "Urban Intersection"
    elif "MERGE" in t:
        return "Highway Merge"
    elif "HIGHWAY" in t:
        return "Highway"
    elif "MARKET" in t:
        return "Market Area"
    elif "NARROW" in t:
        return "Narrow Road"
    elif "URBAN" in t:
        return "Urban Road"
    else:
        return "Unknown"


def determine_road_status(
    risk_levels_present: list[str],
    has_emergency_stop: bool,
    active_threat: bool,
    frame_count: int = 1,
) -> RoadStatus:
    """Determine overall road status from risk indicators."""
    if frame_count < 1:
        return RoadStatus.INSUFFICIENT_DATA
    if has_emergency_stop or "CRITICAL" in risk_levels_present:
        return RoadStatus.EMERGENCY
    if "HIGH" in risk_levels_present or active_threat:
        return RoadStatus.HIGH_RISK
    if "MEDIUM" in risk_levels_present:
        return RoadStatus.CAUTION
    return RoadStatus.SAFE
