"""
Environment Rule Engine & Priority System — Phase 8C.

Synthesizes traffic signals, road signs, road context, and map constraints
under the strict 6-level Functional Safety Priority System:
  1. EMERGENCY COLLISION SAFETY  — Highest priority: immediate collision hazard
  2. IMMEDIATE OBSTACLE SAFETY   — High priority: potential obstacle conflict
  3. TRAFFIC SIGNAL / SIGNS      — Regulatory: red light, stop sign, turn prohibition
  4. DRIVABLE ROAD BOUNDARY      — Physical: stay within road shoulders / boundaries
  5. PATH PROGRESS               — Cruising forward along travel lane
  6. PATH EFFICIENCY             — Minimal lateral deviation & energy efficiency
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.environment.schemas import (
    RoadType,
    TrafficLightColor,
    TrafficSignal,
    RoadSign,
    RoadSignType,
)
from app.environment.road_context import RoadContextModel
from app.environment.map_context import MapContext
from app.environment.sign_rules import map_sign_to_rule, extract_speed_limit_value
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EnvironmentRuleEngine:
    """
    Unified environmental rule synthesizer.
    Combines traffic signals, road signs, road context, and map constraints.
    """

    def __init__(self, min_sign_confidence: float = 0.60) -> None:
        self.min_sign_confidence = min_sign_confidence

    def evaluate_environment(
        self,
        road_context: RoadContextModel,
        traffic_signal: Optional[TrafficSignal],
        road_signs: List[RoadSign],
        map_context: MapContext,
    ) -> Dict[str, Any]:
        """
        Evaluate full environmental context and return synthesized rules & constraints.

        Returns:
            Dict matching Section 10 specification:
            {
                "active_rules": [...],
                "planning_constraints": [...],
                "speed_constraints": 40.0,
                "restricted_directions": [...]
            }
        """
        active_rules: List[str] = []
        planning_constraints: List[str] = []
        restricted_directions: List[str] = list(map_context.restricted_directions)
        speed_candidates: List[float] = []

        if map_context.speed_limit is not None:
            speed_candidates.append(map_context.speed_limit)

        # ── 1. Traffic Signals ────────────────────────────────────────────────
        if traffic_signal and traffic_signal.is_reliable:
            color = traffic_signal.color
            if color == TrafficLightColor.RED:
                active_rules.append("RED_SIGNAL")
                planning_constraints.append("STOP_BEFORE_STOP_LINE")
            elif color == TrafficLightColor.YELLOW:
                active_rules.append("YELLOW_SIGNAL")
                planning_constraints.append("PREPARE_TO_STOP")
            elif color == TrafficLightColor.GREEN:
                active_rules.append("GREEN_SIGNAL")
                planning_constraints.append("PROCEED_WITH_CAUTION")

        # ── 2. Road Signs ─────────────────────────────────────────────────────
        for sign in road_signs:
            if sign.confidence < self.min_sign_confidence or sign.sign_type == RoadSignType.UNKNOWN:
                continue

            rule = map_sign_to_rule(sign, min_confidence=self.min_sign_confidence)
            if rule != "NONE" and rule not in active_rules:
                active_rules.append(rule)

            if sign.sign_type == RoadSignType.STOP:
                if "STOP_BEFORE_INTERSECTION" not in planning_constraints:
                    planning_constraints.append("STOP_BEFORE_INTERSECTION")

            elif sign.sign_type == RoadSignType.NO_ENTRY:
                if "PROHIBIT_FORWARD_CORRIDOR" not in planning_constraints:
                    planning_constraints.append("PROHIBIT_FORWARD_CORRIDOR")
                if "FORWARD" not in restricted_directions:
                    restricted_directions.append("FORWARD")

            elif sign.sign_type == RoadSignType.NO_LEFT_TURN:
                if "LEFT_PATH_BLOCKED" not in planning_constraints:
                    planning_constraints.append("LEFT_PATH_BLOCKED")
                if "LEFT" not in restricted_directions:
                    restricted_directions.append("LEFT")

            elif sign.sign_type == RoadSignType.NO_RIGHT_TURN:
                if "RIGHT_PATH_BLOCKED" not in planning_constraints:
                    planning_constraints.append("RIGHT_PATH_BLOCKED")
                if "RIGHT" not in restricted_directions:
                    restricted_directions.append("RIGHT")

            elif sign.sign_type == RoadSignType.YIELD:
                if "YIELD_TO_CROSS_TRAFFIC" not in planning_constraints:
                    planning_constraints.append("YIELD_TO_CROSS_TRAFFIC")

            elif sign.sign_type == RoadSignType.SPEED_LIMIT:
                speed_val = extract_speed_limit_value(sign.value)
                if speed_val is not None:
                    speed_candidates.append(speed_val)

        # ── 3. Road Context & Road Type ───────────────────────────────────────
        rtype = road_context.road_type
        if rtype == RoadType.VILLAGE_ROAD:
            active_rules.append("VILLAGE_ROAD_CONTEXT")
            planning_constraints.append("ALLOW_FLEXIBLE_CORRIDOR")
            planning_constraints.append("WATCH_CATTLE_AND_PEDESTRIANS")

        elif rtype == RoadType.URBAN_INTERSECTION:
            active_rules.append("URBAN_INTERSECTION_CONTEXT")
            if not (traffic_signal and traffic_signal.is_reliable):
                active_rules.append("UNSIGNALIZED_INTERSECTION")
                planning_constraints.append("CONSERVATIVE_INTERSECTION_APPROACH")
            planning_constraints.append("MULTI_DIRECTION_MONITORING")

        elif rtype == RoadType.HIGHWAY_MERGE:
            active_rules.append("HIGHWAY_MERGE_CONTEXT")
            planning_constraints.append("MONITOR_MERGING_VEHICLES")
            planning_constraints.append("MAINTAIN_MERGE_CORRIDOR")

        elif rtype == RoadType.MARKET_AREA:
            active_rules.append("MARKET_AREA_CONTEXT")
            planning_constraints.append("MIXED_TRAFFIC_CAUTION")
            planning_constraints.append("REDUCE_SPEED_HIGH_DENSITY")

        elif rtype == RoadType.NARROW_ROAD:
            active_rules.append("NARROW_ROAD_CONTEXT")
            planning_constraints.append("RESTRICT_LATERAL_MARGINS")

        # ── 4. Drivable Boundary Constraints ─────────────────────────────────
        planning_constraints.append("STAY_WITHIN_DRIVABLE_BOUNDARIES")

        # ── 5. Speed Constraints Resolution ──────────────────────────────────
        final_speed = min(speed_candidates) if speed_candidates else 50.0

        return {
            "active_rules": active_rules,
            "planning_constraints": planning_constraints,
            "speed_constraints": final_speed,
            "restricted_directions": restricted_directions,
        }
