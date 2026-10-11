"""
Modular Map & Road Layout Provider — Phase 8.

Supplies high-level geometric and topological road layout without requiring
external paid map APIs:
  • Road geometry (straight, curve, 4-way intersection, T-junction, unmarked road)
  • Intersection approach & stop line distances
  • Road travel direction & lane count
  • Navigation destination directive
  • Static / local road constraints (speed limit, no-entry restrictions)

Core Safety Axiom:
  Map information is advisory and regulatory. It NEVER overrides immediate
  obstacle safety or dynamic collision avoidance.
"""
from __future__ import annotations

from typing import Dict, Optional

from app.environment.schemas import MapRoadConstraint, RoadGeometryType, RoadType
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Built-In Scenario Map Profiles ───────────────────────────────────────────
DEFAULT_MAP_PROFILES: Dict[str, MapRoadConstraint] = {
    "straight_road": MapRoadConstraint(
        road_id               = "LOCAL_ROAD_001",
        road_type             = RoadType.URBAN_ROAD,
        geometry_type         = RoadGeometryType.STRAIGHT,
        lane_count            = 2,
        speed_limit_kmh       = 50.0,
        is_one_way            = False,
        is_no_entry           = False,
        has_intersection      = False,
        intersection_dist_px  = None,
        destination_direction = "STRAIGHT",
        road_notes            = "2-lane arterial road. Clear straight corridor.",
    ),
    "village_road": MapRoadConstraint(
        road_id               = "VR_PANCHAYAT_04",
        road_type             = RoadType.VILLAGE_ROAD,
        geometry_type         = RoadGeometryType.UNMARKED_ROAD,
        lane_count            = 1,
        speed_limit_kmh       = 30.0,
        is_one_way            = False,
        is_no_entry           = False,
        has_intersection      = False,
        intersection_dist_px  = None,
        destination_direction = "STRAIGHT",
        road_notes            = "Unmarked village road corridor. Dynamic corridor tracking active at conservative speed.",
    ),
    "urban_intersection": MapRoadConstraint(
        road_id               = "URB_INT_RING_02",
        road_type             = RoadType.URBAN_INTERSECTION,
        geometry_type         = RoadGeometryType.INTERSECTION_4WAY,
        lane_count            = 4,
        speed_limit_kmh       = 35.0,
        is_one_way            = False,
        is_no_entry           = False,
        has_intersection      = True,
        intersection_dist_px  = 180.0,
        destination_direction = "STRAIGHT",
        road_notes            = "Approaching 4-way urban intersection without signals. Controlled junction approach.",
    ),
    "highway_merge": MapRoadConstraint(
        road_id               = "NH_MERGE_RAMP_11",
        road_type             = RoadType.HIGHWAY_MERGE,
        geometry_type         = RoadGeometryType.STRAIGHT,
        lane_count            = 3,
        speed_limit_kmh       = 60.0,
        is_one_way            = False,
        is_no_entry           = False,
        has_intersection      = False,
        intersection_dist_px  = None,
        destination_direction = "STRAIGHT",
        road_notes            = "Highway acceleration merge section. Watch for merging vehicles.",
    ),
    "market_area": MapRoadConstraint(
        road_id               = "MKT_BAZAAR_CHOWK_01",
        road_type             = RoadType.MARKET_AREA,
        geometry_type         = RoadGeometryType.STRAIGHT,
        lane_count            = 2,
        speed_limit_kmh       = 25.0,
        is_one_way            = False,
        is_no_entry           = False,
        has_intersection      = False,
        intersection_dist_px  = None,
        destination_direction = "STRAIGHT",
        road_notes            = "Market area context: dense mixed traffic zone. Heightened pedestrian caution active.",
    ),
    "cattle_crossing": MapRoadConstraint(
        road_id               = "VR_CATTLE_CROSS_07",
        road_type             = RoadType.VILLAGE_ROAD,
        geometry_type         = RoadGeometryType.UNMARKED_ROAD,
        lane_count            = 1,
        speed_limit_kmh       = 25.0,
        is_one_way            = False,
        is_no_entry           = False,
        has_intersection      = False,
        intersection_dist_px  = None,
        destination_direction = "STRAIGHT",
        road_notes            = "Village corridor with active cattle crossing risk. Flexible replanning active.",
    ),
    "intersection_t": MapRoadConstraint(
        road_id               = "URB_INT_T_01",
        road_type             = RoadType.URBAN_INTERSECTION,
        geometry_type         = RoadGeometryType.INTERSECTION_T,
        lane_count            = 2,
        speed_limit_kmh       = 35.0,
        is_one_way            = False,
        is_no_entry           = False,
        has_intersection      = True,
        intersection_dist_px  = 220.0,
        destination_direction = "TURN_RIGHT",
        road_notes            = "Approaching T-junction intersection. Yield to cross traffic.",
    ),
    "intersection_4way": MapRoadConstraint(
        road_id               = "URB_INT_4WAY_01",
        road_type             = RoadType.URBAN_INTERSECTION,
        geometry_type         = RoadGeometryType.INTERSECTION_4WAY,
        lane_count            = 4,
        speed_limit_kmh       = 40.0,
        is_one_way            = False,
        is_no_entry           = False,
        has_intersection      = True,
        intersection_dist_px  = 180.0,
        destination_direction = "STRAIGHT",
        road_notes            = "Approaching 4-way urban intersection. Mandatory stop line ahead.",
    ),
    "no_entry": MapRoadConstraint(
        road_id               = "RESTRICTED_001",
        road_type             = RoadType.URBAN_ROAD,
        geometry_type         = RoadGeometryType.STRAIGHT,
        lane_count            = 1,
        speed_limit_kmh       = 20.0,
        is_one_way            = True,
        is_no_entry           = True,
        has_intersection      = False,
        intersection_dist_px  = None,
        destination_direction = "HALT",
        road_notes            = "Forward corridor legally prohibited (One-way wrong direction / No Entry).",
    ),
    "unmarked_road": MapRoadConstraint(
        road_id               = "UNMARKED_001",
        road_type             = RoadType.VILLAGE_ROAD,
        geometry_type         = RoadGeometryType.UNMARKED_ROAD,
        lane_count            = 1,
        speed_limit_kmh       = 30.0,
        is_one_way            = False,
        is_no_entry           = False,
        has_intersection      = False,
        intersection_dist_px  = None,
        destination_direction = "STRAIGHT",
        road_notes            = "Unmarked road: no lane painted boundaries. Using dynamic road edge tracking.",
    ),
}


class ModularMapProvider:
    """
    Modular road layout and local topology manager.
    Operates offline with zero external paid API dependencies.
    """

    def __init__(self, default_profile: str = "straight_road") -> None:
        self.current_profile_name = default_profile
        self.active_constraint = DEFAULT_MAP_PROFILES.get(
            default_profile, DEFAULT_MAP_PROFILES["straight_road"]
        )
        logger.info(f"ModularMapProvider initialised with profile '{default_profile}'.")

    def set_profile(self, profile_name: str) -> None:
        """Switch active road layout scenario profile."""
        if profile_name in DEFAULT_MAP_PROFILES:
            self.current_profile_name = profile_name
            self.active_constraint = DEFAULT_MAP_PROFILES[profile_name]
            logger.info(f"Map profile updated to: {profile_name}")
        else:
            logger.warning(f"Unknown map profile '{profile_name}'; retaining '{self.current_profile_name}'")

    def get_constraints(
        self,
        frame_idx: int,
        timestamp_sec: float,
        simulated_progress: float = 0.0,
    ) -> MapRoadConstraint:
        """
        Return road layout and constraints for current frame / position.

        Can dynamically transition scenarios (e.g. cruising straight -> approaching intersection).
        """
        # Return active constraint
        return self.active_constraint
