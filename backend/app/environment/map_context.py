"""
Map Context & Topology Interface — Phase 8C.

Provides a modular map representation without external paid GPS or network dependencies:
  • Local road identifier & metadata
  • Configurable RoadType (VILLAGE_ROAD, URBAN_INTERSECTION, HIGHWAY_MERGE, MARKET_AREA, etc.)
  • Directional routing and turn restrictions
  • Posted speed limits & lane counts
  • Intersections & distance to stop lines
  • Simulated GPS latitude / longitude hooks for future telematics integration
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.environment.schemas import RoadType, RoadGeometryType, MapRoadConstraint
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MapContext:
    """
    Representation of local map metadata matching Phase 8C specification.
    """
    road_id:               str = "LOCAL_ROAD_001"
    road_type:             RoadType = RoadType.URBAN_ROAD
    direction:             str = "FORWARD"
    speed_limit:           Optional[float] = 50.0
    restricted_directions: List[str] = field(default_factory=list)
    has_intersection:      bool = False
    intersection_dist_px:  Optional[float] = None
    lane_count:            int = 2
    gps_lat:               Optional[float] = None
    gps_lon:               Optional[float] = None
    road_notes:            str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize map context matching Section 8 & 9 specification."""
        return {
            "road_id": self.road_id,
            "road_type": self.road_type.value,
            "direction": self.direction,
            "speed_limit": self.speed_limit,
            "restricted_directions": list(self.restricted_directions),
            "has_intersection": self.has_intersection,
            "lane_count": self.lane_count,
            "gps_lat": self.gps_lat,
            "gps_lon": self.gps_lon,
        }


# ── Built-In Scenario Map Contexts ────────────────────────────────────────────
SCENARIO_MAP_CONTEXTS: Dict[str, MapContext] = {
    "village_road": MapContext(
        road_id="VR_PANCHAYAT_04",
        road_type=RoadType.VILLAGE_ROAD,
        direction="FORWARD",
        speed_limit=30.0,
        restricted_directions=[],
        has_intersection=False,
        lane_count=1,
        road_notes="Unmarked rural village road. Cattle and irregular pedestrian movement common.",
    ),
    "urban_intersection": MapContext(
        road_id="URB_INT_RING_02",
        road_type=RoadType.URBAN_INTERSECTION,
        direction="FORWARD",
        speed_limit=35.0,
        restricted_directions=[],
        has_intersection=True,
        intersection_dist_px=180.0,
        lane_count=4,
        road_notes="4-way urban intersection without active signal. Uncontrolled junction approach.",
    ),
    "highway_merge": MapContext(
        road_id="NH_MERGE_RAMP_11",
        road_type=RoadType.HIGHWAY_MERGE,
        direction="FORWARD",
        speed_limit=60.0,
        restricted_directions=[],
        has_intersection=False,
        lane_count=3,
        road_notes="Highway acceleration merge section. Watch for merging two-wheelers and trucks.",
    ),
    "market_area": MapContext(
        road_id="MKT_BAZAAR_CHOWK_01",
        road_type=RoadType.MARKET_AREA,
        direction="FORWARD",
        speed_limit=25.0,
        restricted_directions=[],
        has_intersection=False,
        lane_count=2,
        road_notes="Dense market road with mixed pedestrian, cart, and auto-rickshaw traffic.",
    ),
    "cattle_crossing": MapContext(
        road_id="VR_CATTLE_CROSS_07",
        road_type=RoadType.VILLAGE_ROAD,
        direction="FORWARD",
        speed_limit=25.0,
        restricted_directions=[],
        has_intersection=False,
        lane_count=1,
        road_notes="Village corridor adjacent to grazing land. High sudden cattle hazard probability.",
    ),
    "straight_road": MapContext(
        road_id="LOCAL_ROAD_001",
        road_type=RoadType.URBAN_ROAD,
        direction="FORWARD",
        speed_limit=50.0,
        restricted_directions=[],
        has_intersection=False,
        lane_count=2,
        road_notes="Standard bidirectional urban arterial corridor.",
    ),
}


class MapContextInterface:
    """
    Interface for providing and updating local map context.
    """

    def __init__(self, default_profile: str = "straight_road") -> None:
        self.active_context = SCENARIO_MAP_CONTEXTS.get(
            default_profile, SCENARIO_MAP_CONTEXTS["straight_road"]
        )

    def set_scenario(self, scenario_name: str) -> None:
        """Switch active scenario map context."""
        if scenario_name in SCENARIO_MAP_CONTEXTS:
            self.active_context = SCENARIO_MAP_CONTEXTS[scenario_name]
            logger.info(f"Map context switched to scenario: {scenario_name}")
        else:
            logger.warning(f"Unknown map scenario '{scenario_name}'. Retaining current context.")

    def get_context(self) -> MapContext:
        """Return the active MapContext."""
        return self.active_context
