"""
Environment & Road-Context Awareness Module — Phase 8.

Submodules:
- signals: Traffic light detection and color state recognition (RED, YELLOW, GREEN).
- road_signs: Road sign detection (STOP, No Entry, Speed Limit, Turn restrictions, Yield, Warning).
- sign_rules: Deterministic sign-to-rule mapping & constraint extraction.
- road_context: Road surface classification, drivable region modeling & operational contexts.
- map_context: Modular local map and topological routing interface.
- environment_state: Unified environmental state snapshot.
- environment_rules: Priority-driven environmental rule engine.
- schemas: Shared environmental data structures.
"""
from app.environment.schemas import (
    TrafficLightColor,
    TrafficSignal,
    RoadSignType,
    RoadSign,
    RoadGeometryType,
    RoadType,
    DrivableRegion,
    MapRoadConstraint,
    EnvironmentState,
)
from app.environment.signals import TrafficSignalDetector
from app.environment.road_signs import RoadSignDetector
from app.environment.sign_rules import map_sign_to_rule, evaluate_sign_constraints
from app.environment.road_context import RoadContextModel
from app.environment.map_context import MapContext, MapContextInterface
from app.environment.environment_state import UnifiedEnvironmentState
from app.environment.environment_rules import EnvironmentRuleEngine

__all__ = [
    "TrafficLightColor",
    "TrafficSignal",
    "RoadSignType",
    "RoadSign",
    "RoadGeometryType",
    "RoadType",
    "DrivableRegion",
    "MapRoadConstraint",
    "EnvironmentState",
    "TrafficSignalDetector",
    "RoadSignDetector",
    "map_sign_to_rule",
    "evaluate_sign_constraints",
    "RoadContextModel",
    "MapContext",
    "MapContextInterface",
    "UnifiedEnvironmentState",
    "EnvironmentRuleEngine",
]
