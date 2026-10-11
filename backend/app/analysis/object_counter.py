"""
ObjectCounter — tracks both current-frame object counts and total unique tracked counts
using persistent ByteTrack Track IDs. Categorizes detected objects into canonical categories.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any
from collections import defaultdict


CANONICAL_CATEGORIES = [
    "Cars",
    "Bikes",
    "Bicycles",
    "People",
    "Animals",
    "Buses",
    "Trucks",
    "Auto-rickshaws",
    "Other",
    "Unknown",
]

# Mapping from raw YOLO / detector class names to canonical categories
CLASS_MAPPING = {
    # Cars
    "car": "Cars",
    "automobile": "Cars",
    "sedan": "Cars",
    "suv": "Cars",
    "van": "Cars",
    "taxi": "Cars",
    
    # Bikes / Motorcycles
    "motorcycle": "Bikes",
    "motorbike": "Bikes",
    "motor_bike": "Bikes",
    "scooter": "Bikes",
    "moped": "Bikes",
    
    # Bicycles
    "bicycle": "Bicycles",
    "bike": "Bicycles",
    "cycle": "Bicycles",
    
    # People
    "person": "People",
    "pedestrian": "People",
    "human": "People",
    "walker": "People",
    
    # Animals
    "animal": "Animals",
    "dog": "Animals",
    "cat": "Animals",
    "cow": "Animals",
    "cattle": "Animals",
    "ox": "Animals",
    "bull": "Animals",
    "horse": "Animals",
    "sheep": "Animals",
    "goat": "Animals",
    "donkey": "Animals",
    "elephant": "Animals",
    "bear": "Animals",
    "zebra": "Animals",
    "giraffe": "Animals",
    
    # Buses
    "bus": "Buses",
    "minibus": "Buses",
    "coach": "Buses",
    
    # Trucks
    "truck": "Trucks",
    "lorry": "Trucks",
    "pickup": "Trucks",
    "trailer": "Trucks",
    
    # Auto-rickshaws (specialized Indian road class)
    "autorickshaw": "Auto-rickshaws",
    "auto_rickshaw": "Auto-rickshaws",
    "rickshaw": "Auto-rickshaws",
    "auto": "Auto-rickshaws",
    "tuk_tuk": "Auto-rickshaws",
    "tuktuk": "Auto-rickshaws",
}


def canonicalize_class(raw_class: str) -> str:
    """Normalize raw class name string to canonical category."""
    if not raw_class:
        return "Unknown"
    normalized = raw_class.lower().strip().replace("-", "_").replace(" ", "_")
    return CLASS_MAPPING.get(normalized, "Other")


class ObjectCounter:
    """
    Maintains:
    1. Current frame object count (visible entities in the latest processed frame)
    2. Total unique tracked count (persistent entities counted exactly once across the whole video)
    """

    def __init__(self, min_persistence_frames: int = 3):
        self.min_persistence_frames = min_persistence_frames
        # track_id -> dict of {canonical_category: vote_count}
        self.track_votes: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # track_id -> total frames observed
        self.track_frame_count: Dict[int, int] = defaultdict(int)
        # Resolved persistent category: track_id -> canonical_category
        self.resolved_tracks: Dict[int, str] = {}
        # Counts in the most recent frame: canonical_category -> count
        self.current_frame_counts: Dict[str, int] = {cat: 0 for cat in CANONICAL_CATEGORIES}
        # Total unique objects across video: canonical_category -> count
        self.total_unique_counts: Dict[str, int] = {cat: 0 for cat in CANONICAL_CATEGORIES}

    def update_frame(
        self,
        tracked_objects: List[Any],
        raw_detections: Optional[List[Any]] = None,
    ) -> Dict[str, int]:
        """
        Update counter with current frame's tracked objects and optional raw detections.
        tracked_objects can be objects with attributes (.track_id, .class_name) or dicts.
        """
        # Reset current frame counts
        frame_counts: Dict[str, int] = {cat: 0 for cat in CANONICAL_CATEGORIES}

        active_track_ids = set()

        for obj in tracked_objects:
            if isinstance(obj, dict):
                tid = obj.get("track_id", obj.get("id", -1))
                raw_cls = obj.get("class_name", obj.get("cls", "unknown"))
            else:
                tid = getattr(obj, "track_id", getattr(obj, "id", -1))
                raw_cls = getattr(obj, "class_name", "unknown")

            canonical_cat = canonicalize_class(raw_cls)
            frame_counts[canonical_cat] += 1

            if tid is not None and tid >= 0:
                active_track_ids.add(tid)
                self.track_votes[tid][canonical_cat] += 1
                self.track_frame_count[tid] += 1

                # Update dominant category for this track
                dominant_cat = max(self.track_votes[tid].items(), key=lambda x: x[1])[0]
                self.resolved_tracks[tid] = dominant_cat

        # If raw detections exist and no tracks were matched (fallback)
        if not tracked_objects and raw_detections:
            for d in raw_detections:
                raw_cls = d.get("class_name", "unknown") if isinstance(d, dict) else getattr(d, "class_name", "unknown")
                cat = canonicalize_class(raw_cls)
                frame_counts[cat] += 1

        self.current_frame_counts = frame_counts
        self._recompute_total_unique()
        return self.current_frame_counts

    def _recompute_total_unique(self) -> None:
        """Recompute total unique counts across all resolved persistent tracks."""
        totals: Dict[str, int] = {cat: 0 for cat in CANONICAL_CATEGORIES}
        for tid, dominant_cat in self.resolved_tracks.items():
            # Apply persistence filter if track has been seen enough frames
            if self.track_frame_count[tid] >= self.min_persistence_frames or len(self.resolved_tracks) <= 5:
                totals[dominant_cat] += 1

        self.total_unique_counts = totals

    def get_counts(self) -> Dict[str, Any]:
        """Return standardized counts structure."""
        total_unique_sum = sum(self.total_unique_counts.values())
        return {
            "current_frame": dict(self.current_frame_counts),
            "total_unique": dict(self.total_unique_counts),
            "total_unique_all": total_unique_sum,
            "active_frame_total": sum(self.current_frame_counts.values()),
        }

    def reset(self) -> None:
        """Reset internal counter state."""
        self.track_votes.clear()
        self.track_frame_count.clear()
        self.resolved_tracks.clear()
        self.current_frame_counts = {cat: 0 for cat in CANONICAL_CATEGORIES}
        self.total_unique_counts = {cat: 0 for cat in CANONICAL_CATEGORIES}
