"""
Unified Environment State — Phase 8C.

Aggregates all environmental streams into a single coherent state snapshot:
  • road_context    : Road type, intersection state, drivable boundary limits
  • traffic_signal  : Active signal color, confidence, stop-line distance
  • road_signs      : Active detected signs, confidence, speed limit values, rules
  • map_context     : Map identifier, speed limit, geometry, direction constraints
  • unified rules   : Active synthesized operational rules and planning constraints
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.environment.schemas import (
    RoadType,
    TrafficSignal,
    RoadSign,
    DrivableRegion,
)
from app.environment.road_context import RoadContextModel
from app.environment.map_context import MapContext


@dataclass
class UnifiedEnvironmentState:
    """
    Unified environmental perception snapshot matching Phase 8C Section 9 specification.
    """
    frame_idx: int
    timestamp_sec: float
    road_context: RoadContextModel
    traffic_signal: Optional[TrafficSignal] = None
    road_signs: List[RoadSign] = field(default_factory=list)
    map_context: MapContext = field(default_factory=MapContext)
    active_rules: List[str] = field(default_factory=list)
    planning_constraints: List[str] = field(default_factory=list)
    effective_speed_limit: Optional[float] = None
    restricted_directions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize unified environment state matching Section 9 schema."""
        sig_dict = None
        if self.traffic_signal:
            sig_dict = {
                "state": self.traffic_signal.color.value,
                "confidence": round(self.traffic_signal.confidence, 2),
                "is_reliable": self.traffic_signal.is_reliable,
                "distance_px": self.traffic_signal.distance_px,
            }

        signs_list = [
            {
                "type": s.sign_type.value,
                "confidence": round(s.confidence, 2),
                "value": s.value,
                "rule": getattr(s, "rule", "") or s.action_directive,
            }
            for s in self.road_signs
        ]

        return {
            "road_context": self.road_context.to_dict(),
            "traffic_signal": sig_dict,
            "road_signs": signs_list,
            "map_context": self.map_context.to_dict(),
            "active_rules": list(self.active_rules),
            "planning_constraints": list(self.planning_constraints),
            "effective_speed_limit": self.effective_speed_limit,
            "restricted_directions": list(self.restricted_directions),
        }
