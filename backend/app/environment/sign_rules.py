"""
Road Sign Rules & Constraint Engine — Phase 8B.

Maps detected road signs to deterministic driving rules and trajectory constraints:
  • STOP          -> Require full vehicle stop
  • NO_ENTRY      -> Prohibit entry / affected travel direction
  • SPEED_LIMIT   -> Update maximum speed constraint (km/h)
  • NO_LEFT_TURN  -> Prohibit left-turn / left evasive candidate paths
  • NO_RIGHT_TURN -> Prohibit right-turn / right evasive candidate paths
  • YIELD         -> Increase caution / yield right-of-way
  • UNKNOWN       -> No active rule applied (below confidence threshold)
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from app.environment.schemas import RoadSign, RoadSignType


class SignRuleConstants:
    STOP = "STOP"
    PROHIBITED_CORRIDOR = "PROHIBITED_CORRIDOR"
    SPEED_CONSTRAINT = "SPEED_CONSTRAINT"
    REJECT_LEFT_PATH = "REJECT_LEFT_PATH"
    REJECT_RIGHT_PATH = "REJECT_RIGHT_PATH"
    YIELD_CAUTION = "YIELD_CAUTION"
    PEDESTRIAN_ZONE_CAUTION = "PEDESTRIAN_ZONE_CAUTION"
    ADVISORY_SLOW_DOWN = "ADVISORY_SLOW_DOWN"
    NONE = "NONE"


def map_sign_to_rule(sign: RoadSign, min_confidence: float = 0.60) -> str:
    """
    Map a detected road sign to its active operational rule.
    Returns 'NONE' if confidence is below threshold or sign is UNKNOWN.
    """
    if sign.confidence < min_confidence or sign.sign_type == RoadSignType.UNKNOWN:
        return SignRuleConstants.NONE

    mapping = {
        RoadSignType.STOP: SignRuleConstants.STOP,
        RoadSignType.NO_ENTRY: SignRuleConstants.PROHIBITED_CORRIDOR,
        RoadSignType.SPEED_LIMIT: SignRuleConstants.SPEED_CONSTRAINT,
        RoadSignType.NO_LEFT_TURN: SignRuleConstants.REJECT_LEFT_PATH,
        RoadSignType.NO_RIGHT_TURN: SignRuleConstants.REJECT_RIGHT_PATH,
        RoadSignType.YIELD: SignRuleConstants.YIELD_CAUTION,
        RoadSignType.PEDESTRIAN_CROSSING: SignRuleConstants.PEDESTRIAN_ZONE_CAUTION,
        RoadSignType.WARNING: SignRuleConstants.ADVISORY_SLOW_DOWN,
        RoadSignType.UNKNOWN: SignRuleConstants.NONE,
    }
    return mapping.get(sign.sign_type, SignRuleConstants.NONE)


def extract_speed_limit_value(value: Optional[Any]) -> Optional[float]:
    """
    Extract numeric speed limit value from value field (e.g. 40, "40 km/h", "SPEED 40").
    Returns float or None.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"(\d+(?:\.\d+)?)", value)
        if match:
            return float(match.group(1))
    return None


def evaluate_sign_constraints(
    signs: List[RoadSign],
    min_confidence: float = 0.60,
) -> Dict[str, Any]:
    """
    Synthesize all active road signs into structured constraint outputs for Phase 8B/8C.

    Returns:
        Dict matching Phase 8B structured output specification:
        {
            "road_signs": [
                {"type": "STOP", "confidence": 0.94, "value": None, "rule": "STOP"}
            ],
            "active_constraints": ["STOP"],
            "speed_limit": 40.0 | None,
            "restricted_directions": ["LEFT", "RIGHT"]
        }
    """
    structured_signs: List[Dict[str, Any]] = []
    active_constraints: List[str] = []
    restricted_directions: List[str] = []
    speed_limit: Optional[float] = None

    for s in signs:
        rule = map_sign_to_rule(s, min_confidence=min_confidence)
        numeric_val = extract_speed_limit_value(s.value) if s.sign_type == RoadSignType.SPEED_LIMIT else s.value

        structured_signs.append({
            "type": s.sign_type.value,
            "confidence": round(s.confidence, 2),
            "value": numeric_val,
            "rule": rule,
        })

        if rule == SignRuleConstants.NONE:
            continue

        if rule not in active_constraints:
            active_constraints.append(rule)

        if s.sign_type == RoadSignType.NO_LEFT_TURN:
            if "LEFT" not in restricted_directions:
                restricted_directions.append("LEFT")
        elif s.sign_type == RoadSignType.NO_RIGHT_TURN:
            if "RIGHT" not in restricted_directions:
                restricted_directions.append("RIGHT")
        elif s.sign_type == RoadSignType.NO_ENTRY:
            if "FORWARD" not in restricted_directions:
                restricted_directions.append("FORWARD")
        elif s.sign_type == RoadSignType.SPEED_LIMIT:
            val = extract_speed_limit_value(s.value)
            if val is not None:
                # If multiple speed signs, prioritize the lowest (most conservative)
                if speed_limit is None or val < speed_limit:
                    speed_limit = val

    return {
        "road_signs": structured_signs,
        "active_constraints": active_constraints,
        "speed_limit": speed_limit,
        "restricted_directions": restricted_directions,
    }
