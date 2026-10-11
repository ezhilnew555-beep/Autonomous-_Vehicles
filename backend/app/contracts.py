"""
Standardized Data Contracts — Phase 10: Final Full-System Integration.

Provides canonical serialization adapters for all 7 primary pipeline stages:
1. Detection:   object_id, class_name, confidence, bbox
2. Tracking:    object_id, class_name, position, velocity, direction
3. Prediction:  object_id, future_positions, prediction_confidence
4. Risk:        object_id, risk_level, ttc, trajectory_conflict
5. Decision:    action, reason, threat_object, risk_level
6. Planning:    path_type, selected_path, safe, replanned
7. Environment: road_type, traffic_signal, road_signs, planning_constraints
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def format_detection_contract(
    class_name: str,
    confidence: float,
    bbox: Tuple[float, float, float, float] | List[float],
    object_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Format standardized detection output."""
    return {
        "object_id": object_id,
        "class_name": str(class_name),
        "confidence": round(float(confidence), 3),
        "bbox": [round(float(coord), 1) for coord in bbox],
    }


def format_tracking_contract(
    object_id: int,
    class_name: str,
    position: Tuple[float, float] | List[float],
    velocity: Tuple[float, float] | List[float] = (0.0, 0.0),
    direction: str = "FORWARD",
) -> Dict[str, Any]:
    """Format standardized tracking output."""
    return {
        "object_id": int(object_id),
        "class_name": str(class_name),
        "position": [round(float(p), 1) for p in position],
        "velocity": [round(float(v), 2) for v in velocity],
        "direction": str(direction),
    }


def format_prediction_contract(
    object_id: int,
    future_positions: List[Tuple[float, float]] | List[List[float]],
    prediction_confidence: float = 1.0,
) -> Dict[str, Any]:
    """Format standardized motion prediction output."""
    return {
        "object_id": int(object_id),
        "future_positions": [[round(float(pt[0]), 1), round(float(pt[1]), 1)] for pt in future_positions],
        "prediction_confidence": round(float(prediction_confidence), 2),
    }


def format_risk_contract(
    object_id: int,
    risk_level: str,
    ttc: Optional[float] = None,
    trajectory_conflict: bool = False,
) -> Dict[str, Any]:
    """Format standardized collision risk output."""
    return {
        "object_id": int(object_id),
        "risk_level": str(risk_level),
        "ttc": round(float(ttc), 2) if ttc is not None else None,
        "trajectory_conflict": bool(trajectory_conflict),
    }


def format_decision_contract(
    action: str,
    reason: str,
    threat_object: Optional[int] = None,
    risk_level: str = "LOW",
) -> Dict[str, Any]:
    """Format standardized autonomous decision output."""
    return {
        "action": str(action),
        "reason": str(reason),
        "threat_object": int(threat_object) if threat_object is not None else None,
        "risk_level": str(risk_level),
    }


def format_planning_contract(
    path_type: str,
    selected_path: List[Tuple[float, float]] | List[List[float]],
    safe: bool = True,
    replanned: bool = False,
) -> Dict[str, Any]:
    """Format standardized path planning output."""
    return {
        "path_type": str(path_type),
        "selected_path": [[round(float(pt[0]), 1), round(float(pt[1]), 1)] for pt in selected_path],
        "safe": bool(safe),
        "replanned": bool(replanned),
    }


def format_environment_contract(
    road_type: str,
    traffic_signal: Optional[str] = None,
    road_signs: Optional[List[Dict[str, Any]]] = None,
    planning_constraints: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Format standardized environmental context output."""
    return {
        "road_type": str(road_type),
        "traffic_signal": str(traffic_signal) if traffic_signal else None,
        "road_signs": list(road_signs) if road_signs else [],
        "planning_constraints": list(planning_constraints) if planning_constraints else [],
    }
