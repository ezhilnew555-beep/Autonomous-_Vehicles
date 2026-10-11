"""
Road Context Representation & Drivable Region Modeling — Phase 8C.

Models physical road surface attributes, operational operational contexts,
and drivable boundaries for Indian driving conditions:
  • VILLAGE_ROAD       — Unmarked road surface, cattle / pedestrian crossings, irregular motion
  • URBAN_ROAD         — Standard asphalt city roadway with moderate density
  • URBAN_INTERSECTION — Multi-direction road junction, stop lines, signal/sign control
  • HIGHWAY            — High-speed multi-lane divided carriageway
  • HIGHWAY_MERGE      — On-ramp / entry merge zones with slow-moving merging vehicles
  • MARKET_AREA        — High-density mixed traffic (pedestrians, bicycles, auto-rickshaws, carts)
  • NARROW_ROAD        — Restricted width lane requiring reduced speed & tight margins
  • UNKNOWN            — Unverified road context handled conservatively
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.environment.schemas import RoadType, DrivableRegion
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RoadContextModel:
    """
    Environmental road context and drivable boundary definition.
    """
    road_type: RoadType = RoadType.URBAN_ROAD
    intersection: bool = False
    intersection_status: str = "NONE"   # "NONE" | "APPROACHING" | "AT_INTERSECTION" | "UNKNOWN"
    drivable_region: DrivableRegion = field(default_factory=DrivableRegion)
    has_lane_markings: bool = True
    surface_quality: str = "PAVED"      # "PAVED" | "UNMARKED" | "GRAVEL" | "ROUGH"
    clutter_density: str = "NORMAL"     # "LOW" | "NORMAL" | "HIGH" | "VERY_HIGH"
    caution_factor: float = 1.0         # Multiplier on safety buffers (>1.0 = more cautious)
    recommended_speed_factor: float = 1.0

    @classmethod
    def create_for_scenario(cls, road_type: RoadType, frame_width: int = 640) -> "RoadContextModel":
        """Factory method to configure road context model based on road scenario type."""
        w = float(frame_width)
        if road_type == RoadType.VILLAGE_ROAD:
            return cls(
                road_type=RoadType.VILLAGE_ROAD,
                intersection=False,
                intersection_status="NONE",
                drivable_region=DrivableRegion(
                    left_boundary_px=20.0,
                    right_boundary_px=w - 20.0,
                    road_width_px=w - 40.0,
                    is_unmarked=True,
                    unknown_region=False,
                ),
                has_lane_markings=False,
                surface_quality="UNMARKED",
                clutter_density="HIGH",
                caution_factor=1.35,
                recommended_speed_factor=0.65,
            )
        elif road_type == RoadType.URBAN_INTERSECTION:
            return cls(
                road_type=RoadType.URBAN_INTERSECTION,
                intersection=True,
                intersection_status="APPROACHING",
                drivable_region=DrivableRegion(
                    left_boundary_px=35.0,
                    right_boundary_px=w - 35.0,
                    road_width_px=w - 70.0,
                    is_unmarked=False,
                    unknown_region=False,
                ),
                has_lane_markings=True,
                surface_quality="PAVED",
                clutter_density="HIGH",
                caution_factor=1.30,
                recommended_speed_factor=0.55,
            )
        elif road_type == RoadType.HIGHWAY_MERGE:
            return cls(
                road_type=RoadType.HIGHWAY_MERGE,
                intersection=False,
                intersection_status="NONE",
                drivable_region=DrivableRegion(
                    left_boundary_px=30.0,
                    right_boundary_px=w - 30.0,
                    road_width_px=w - 60.0,
                    is_unmarked=False,
                    unknown_region=False,
                ),
                has_lane_markings=True,
                surface_quality="PAVED",
                clutter_density="NORMAL",
                caution_factor=1.25,
                recommended_speed_factor=0.75,
            )
        elif road_type == RoadType.MARKET_AREA:
            return cls(
                road_type=RoadType.MARKET_AREA,
                intersection=False,
                intersection_status="NONE",
                drivable_region=DrivableRegion(
                    left_boundary_px=25.0,
                    right_boundary_px=w - 25.0,
                    road_width_px=w - 50.0,
                    is_unmarked=True,
                    unknown_region=False,
                ),
                has_lane_markings=False,
                surface_quality="PAVED",
                clutter_density="VERY_HIGH",
                caution_factor=1.50,
                recommended_speed_factor=0.50,
            )
        elif road_type == RoadType.NARROW_ROAD:
            return cls(
                road_type=RoadType.NARROW_ROAD,
                intersection=False,
                intersection_status="NONE",
                drivable_region=DrivableRegion(
                    left_boundary_px=60.0,
                    right_boundary_px=w - 60.0,
                    road_width_px=w - 120.0,
                    is_unmarked=True,
                    unknown_region=False,
                ),
                has_lane_markings=False,
                surface_quality="PAVED",
                clutter_density="NORMAL",
                caution_factor=1.40,
                recommended_speed_factor=0.50,
            )
        elif road_type == RoadType.HIGHWAY:
            return cls(
                road_type=RoadType.HIGHWAY,
                intersection=False,
                intersection_status="NONE",
                drivable_region=DrivableRegion(
                    left_boundary_px=40.0,
                    right_boundary_px=w - 40.0,
                    road_width_px=w - 80.0,
                    is_unmarked=False,
                    unknown_region=False,
                ),
                has_lane_markings=True,
                surface_quality="PAVED",
                clutter_density="LOW",
                caution_factor=1.0,
                recommended_speed_factor=1.0,
            )
        else: # UNKNOWN or URBAN_ROAD
            return cls(
                road_type=road_type,
                intersection=False,
                intersection_status="UNKNOWN" if road_type == RoadType.UNKNOWN else "NONE",
                drivable_region=DrivableRegion(
                    left_boundary_px=35.0,
                    right_boundary_px=w - 35.0,
                    road_width_px=w - 70.0,
                    is_unmarked=False,
                    unknown_region=(road_type == RoadType.UNKNOWN),
                ),
                has_lane_markings=True,
                surface_quality="PAVED",
                clutter_density="NORMAL",
                caution_factor=1.20 if road_type == RoadType.UNKNOWN else 1.0,
                recommended_speed_factor=0.60 if road_type == RoadType.UNKNOWN else 1.0,
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize road context matching Section 9 schema."""
        return {
            "road_type": self.road_type.value if hasattr(self.road_type, "value") else str(self.road_type),
            "intersection": self.intersection,
            "drivable_region": {
                "left_boundary_px": self.drivable_region.left_boundary_px,
                "right_boundary_px": self.drivable_region.right_boundary_px,
                "road_width_px": self.drivable_region.road_width_px,
                "is_unmarked": self.drivable_region.is_unmarked,
                "unknown_region": self.drivable_region.unknown_region,
            },
            "has_lane_markings": self.has_lane_markings,
            "clutter_density": self.clutter_density,
            "recommended_speed_factor": self.recommended_speed_factor,
        }
