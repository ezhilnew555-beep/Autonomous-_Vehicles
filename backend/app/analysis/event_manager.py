"""
EventManager — records a chronological, deduplicated timeline of critical state changes
and autonomous vehicle events with human-readable explanations.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import time


@dataclass
class TimelineEvent:
    event_id: int
    timestamp_sec: float
    frame_index: int
    event_type: str
    severity: str  # "INFO", "CAUTION", "WARNING", "CRITICAL"
    description: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp_sec": round(self.timestamp_sec, 2),
            "frame_index": self.frame_index,
            "event_type": self.event_type,
            "severity": self.severity,
            "description": self.description,
            "details": self.details,
        }


class EventManager:
    """
    Manages autonomous driving timeline events with state change detection,
    rate-limiting, and plain English descriptions.
    """

    def __init__(self, debounce_window_sec: float = 1.0):
        self.debounce_window_sec = debounce_window_sec
        self.events: List[TimelineEvent] = []
        self._next_id = 1
        # Track last timestamp by (event_type, primary_subject_id) to avoid spam
        self._last_event_time: Dict[tuple, float] = {}

    def record_event(
        self,
        event_type: str,
        frame_index: int,
        timestamp_sec: float,
        severity: str,
        description: str,
        subject_id: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> Optional[TimelineEvent]:
        """
        Record a timeline event if outside the debounce window or forced.
        """
        key = (event_type, subject_id if subject_id is not None else "global")
        last_t = self._last_event_time.get(key)

        if not force and last_t is not None:
            if (timestamp_sec - last_t) < self.debounce_window_sec:
                return None  # Debounced

        event = TimelineEvent(
            event_id=self._next_id,
            timestamp_sec=timestamp_sec,
            frame_index=frame_index,
            event_type=event_type,
            severity=severity,
            description=description,
            details=details or {},
        )
        self._next_id += 1
        self.events.append(event)
        self._last_event_time[key] = timestamp_sec
        return event

    def log_decision_change(
        self,
        frame_index: int,
        timestamp_sec: float,
        prev_action: str,
        new_action: str,
        reason: str,
    ) -> Optional[TimelineEvent]:
        """Record decision transition."""
        if prev_action == new_action:
            return None

        severity = "INFO"
        if "STOP" in new_action or "EMERGENCY" in new_action:
            severity = "CRITICAL"
        elif "AVOID" in new_action or "SLOW" in new_action:
            severity = "WARNING"

        desc = f"Vehicle action changed from {prev_action} to {new_action}: {reason}"
        return self.record_event(
            event_type="DECISION_CHANGED",
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            severity=severity,
            description=desc,
            subject_id=new_action,
            details={"prev_action": prev_action, "new_action": new_action, "reason": reason},
            force=False,
        )

    def log_path_replan(
        self,
        frame_index: int,
        timestamp_sec: float,
        path_status: str,
        reason: str,
        clearance_m: float,
    ) -> Optional[TimelineEvent]:
        """Record path replanning event."""
        desc = f"Path replanned: {reason} (clearance: {clearance_m:.1f}m)"
        return self.record_event(
            event_type="PATH_REPLANNED",
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            severity="WARNING" if clearance_m < 2.0 else "CAUTION",
            description=desc,
            subject_id="planner",
            details={"path_status": path_status, "reason": reason, "clearance_m": round(clearance_m, 2)},
        )

    def log_risk_detected(
        self,
        frame_index: int,
        timestamp_sec: float,
        track_id: int,
        class_name: str,
        risk_level: str,
        ttc_sec: Optional[float],
        reason: str,
    ) -> Optional[TimelineEvent]:
        """Record high or critical collision risk detection."""
        if risk_level not in ("HIGH", "CRITICAL"):
            return None

        ttc_str = f"with TTC {ttc_sec:.1f}s" if ttc_sec is not None else "in trajectory conflict"
        desc = f"{risk_level} collision risk: {class_name} #{track_id} {ttc_str} ({reason})"
        severity = "CRITICAL" if risk_level == "CRITICAL" else "WARNING"

        return self.record_event(
            event_type="HIGH_RISK_DETECTED",
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            severity=severity,
            description=desc,
            subject_id=track_id,
            details={"track_id": track_id, "class_name": class_name, "risk_level": risk_level, "ttc_sec": ttc_sec},
        )

    def log_object_entered_path(
        self,
        frame_index: int,
        timestamp_sec: float,
        track_id: int,
        class_name: str,
        distance_m: float,
    ) -> Optional[TimelineEvent]:
        """Record when an obstacle cuts into the ego vehicle driving corridor."""
        desc = f"{class_name.capitalize()} #{track_id} entered vehicle path corridor at {distance_m:.1f}m ahead"
        return self.record_event(
            event_type="OBJECT_ENTERED_PATH",
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            severity="CAUTION",
            description=desc,
            subject_id=track_id,
            details={"track_id": track_id, "class_name": class_name, "distance_m": round(distance_m, 1)},
        )

    def log_traffic_signal(
        self,
        frame_index: int,
        timestamp_sec: float,
        color: str,
        confidence: float,
    ) -> Optional[TimelineEvent]:
        """Record traffic signal state change."""
        desc = f"Traffic signal recognized: {color.upper()} (confidence: {int(confidence * 100)}%)"
        severity = "WARNING" if color.upper() == "RED" else ("CAUTION" if color.upper() == "YELLOW" else "INFO")
        return self.record_event(
            event_type="SIGNAL_CHANGED",
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            severity=severity,
            description=desc,
            subject_id=color,
            details={"color": color, "confidence": round(confidence, 2)},
        )

    def log_road_sign(
        self,
        frame_index: int,
        timestamp_sec: float,
        sign_type: str,
        action_directive: str,
    ) -> Optional[TimelineEvent]:
        """Record road sign detection."""
        desc = f"Road sign detected: {sign_type} -> Directive: {action_directive}"
        return self.record_event(
            event_type="ROAD_SIGN_DETECTED",
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            severity="INFO",
            description=desc,
            subject_id=sign_type,
            details={"sign_type": sign_type, "action_directive": action_directive},
        )

    def get_timeline(self) -> List[Dict[str, Any]]:
        """Return full timeline serialized as list of dicts."""
        return [e.to_dict() for e in self.events]

    def get_critical_events(self) -> List[Dict[str, Any]]:
        """Return only WARNING and CRITICAL events."""
        return [e.to_dict() for e in self.events if e.severity in ("WARNING", "CRITICAL")]

    def reset(self) -> None:
        """Clear all events."""
        self.events.clear()
        self._last_event_time.clear()
        self._next_id = 1
