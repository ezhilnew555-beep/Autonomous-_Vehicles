"""
Compatibility alias module for road_signs.py — Phase 8.
"""
from app.environment.road_signs import (
    RoadSignDetector,
    _HSV_RED1,
    _HSV_RED2,
    _HSV_BLUE,
    _HSV_YELLOW,
)

__all__ = [
    "RoadSignDetector",
    "_HSV_RED1",
    "_HSV_RED2",
    "_HSV_BLUE",
    "_HSV_YELLOW",
]
