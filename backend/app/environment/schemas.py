"""
Environment schemas — Phase 8: Road-Context Awareness.

Defines typed data structures for:
1. Traffic Signals (RED, YELLOW, GREEN, UNKNOWN) with confidence & reliability
2. Road Signs (STOP, No Entry, Speed Limit, Pedestrian Crossing, Warning)
3. Modular Map & Road Layout Constraints (Geometry, Intersections, Directions, Unmarked roads)
4. Unified EnvironmentState snapshot consumed by RoadContext, DecisionEngine, and PathPlanner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class TrafficLightColor(str, Enum):
    RED     = "RED"
    YELLOW  = "YELLOW"
    GREEN   = "GREEN"
    UNKNOWN = "UNKNOWN"


@dataclass
class TrafficSignal:
    """
    Recognized traffic light state.

    Attributes:
        color         : Recognized color (RED, YELLOW, GREEN, UNKNOWN).
        confidence    : Recognition confidence (0.0 to 1.0).
        is_reliable   : True if confidence exceeds safe decision threshold.
        box           : Bounding box [x1, y1, x2, y2].
        distance_px   : Approximate vertical distance from ego bumper.
        reason        : Explanatory text for logs / decision justification.
    """
    color:        TrafficLightColor
    confidence:   float
    is_reliable:  bool
    box:          Tuple[float, float, float, float]
    distance_px:  float = 0.0
    reason:       str = ""


class RoadSignType(str, Enum):
    STOP                = "STOP"
    NO_ENTRY            = "NO_ENTRY"
    SPEED_LIMIT         = "SPEED_LIMIT"
    NO_LEFT_TURN        = "NO_LEFT_TURN"
    NO_RIGHT_TURN       = "NO_RIGHT_TURN"
    YIELD               = "YIELD"
    PEDESTRIAN_CROSSING = "PEDESTRIAN_CROSSING"
    WARNING             = "WARNING"
    UNKNOWN             = "UNKNOWN"


@dataclass
class RoadSign:
    """
    Recognized traffic / regulatory road sign.

    Attributes:
        sign_type       : Categorized sign type.
        confidence      : Recognition confidence (0.0 to 1.0).
        box             : Bounding box [x1, y1, x2, y2].
        value           : Optional metadata (e.g. 40, "40 km/h", "School Ahead").
        action_directive: Guidance for decision engine (e.g. "MANDATORY_STOP").
        rule            : Active operational rule (e.g. "STOP", "SPEED_CONSTRAINT", "REJECT_LEFT_PATH").
    """
    sign_type:        RoadSignType
    confidence:       float
    box:              Tuple[float, float, float, float]
    value:            Optional[Any] = None
    action_directive: str = ""
    rule:             str = ""


class RoadType(str, Enum):
    VILLAGE_ROAD       = "VILLAGE_ROAD"
    URBAN_ROAD         = "URBAN_ROAD"
    URBAN_INTERSECTION = "URBAN_INTERSECTION"
    HIGHWAY            = "HIGHWAY"
    HIGHWAY_MERGE      = "HIGHWAY_MERGE"
    MARKET_AREA        = "MARKET_AREA"
    NARROW_ROAD        = "NARROW_ROAD"
    UNKNOWN            = "UNKNOWN"


class RoadGeometryType(str, Enum):
    STRAIGHT          = "STRAIGHT"
    CURVED            = "CURVED"
    INTERSECTION_4WAY = "INTERSECTION_4WAY"
    INTERSECTION_T    = "INTERSECTION_T"
    ROUNDABOUT        = "ROUNDABOUT"
    UNMARKED_ROAD     = "UNMARKED_ROAD"


@dataclass
class DrivableRegion:
    """
    Representation of the drivable road boundaries in camera coordinates.
    """
    left_boundary_px:  float = 35.0
    right_boundary_px: float = 605.0
    road_width_px:     float = 570.0
    is_unmarked:       bool  = False
    unknown_region:    bool  = False

    def is_within_bounds(self, x: float) -> bool:
        """Check if an x-coordinate is within legal drivable boundaries."""
        return self.left_boundary_px <= x <= self.right_boundary_px


@dataclass
class MapRoadConstraint:
    """
    Modular zero-cost map and road layout context.

    Attributes:
        road_id                : Identifier for map section (e.g. 'LOCAL_ROAD_001').
        road_type              : High-level road type (URBAN_ROAD, VILLAGE_ROAD, etc.).
        geometry_type          : Road structure (Highway, T-Junction, Unmarked road, etc.).
        lane_count             : Number of lanes.
        speed_limit_kmh        : Default legal speed limit on this section.
        is_one_way             : True if road is designated one-way.
        is_no_entry            : True if entry into forward corridor is legally prohibited.
        has_intersection       : True if approaching an intersection.
        intersection_dist_px   : Pixel distance to intersection stop line.
        destination_direction  : High-level navigation route ("STRAIGHT", "LEFT", "RIGHT").
        restricted_directions  : List of legally restricted navigation directions.
        road_notes             : Contextual road notes (e.g. "Rural unmarked single carriage").
    """
    road_id:               str = "LOCAL_ROAD_001"
    road_type:             RoadType = RoadType.URBAN_ROAD
    geometry_type:         RoadGeometryType = RoadGeometryType.STRAIGHT
    lane_count:            int = 2
    speed_limit_kmh:       float = 50.0
    is_one_way:            bool = False
    is_no_entry:           bool = False
    has_intersection:      bool = False
    intersection_dist_px:  Optional[float] = None
    destination_direction: str = "STRAIGHT"
    restricted_directions: List[str] = field(default_factory=list)
    road_notes:            str = "Standard bidirectional asphalt corridor."


@dataclass
class EnvironmentState:
    """
    Aggregated environmental perception and context snapshot for current frame.
    """
    frame_idx:                    int
    timestamp_sec:                float
    active_signal:                Optional[TrafficSignal] = None
    detected_signs:               List[RoadSign] = field(default_factory=list)
    map_constraints:              MapRoadConstraint = field(default_factory=MapRoadConstraint)
    effective_speed_limit_factor: float = 1.0  # Throttle cap (1.0 = normal, 0.5 = cautious)
    mandatory_stop_required:      bool = False
    stopping_reason:              Optional[str] = None
    active_speed_limit:           Optional[float] = None
    active_sign_constraints:      List[str] = field(default_factory=list)
    restricted_directions:        List[str] = field(default_factory=list)
    road_type:                    RoadType = RoadType.URBAN_ROAD
    has_intersection:             bool = False
    drivable_region:              DrivableRegion = field(default_factory=DrivableRegion)
    active_rules:                 List[str] = field(default_factory=list)
    planning_constraints:         List[str] = field(default_factory=list)
    road_context:                 Optional[Any] = None
