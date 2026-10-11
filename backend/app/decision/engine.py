"""
DecisionEngine — Autonomous Vehicle Decision-Making Logic (Phase 6).

Consumes detections, tracking, short-term motion predictions, collision risks,
and spatial road context to generate deterministic, safety-critical vehicle commands:
  • CONTINUE
  • SLOW_DOWN
  • STOP
  • AVOID
  • REPLAN
  • EMERGENCY_STOP

Compliant with strict automotive functional safety standards: rule-based,
100% deterministic (no probabilistic LLMs for safety decisions), with auditable reasoning.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

from app.tracking.tracker import TrackedObject
from app.prediction.schemas import PredictionResult
from app.collision.schemas import RiskLevel, CollisionRisk
from app.decision.schemas import (
    VehicleAction,
    AvoidanceDirection,
    RoadContext,
    DecisionOutput,
)
from app.decision.context import evaluate_road_context
from app.environment.schemas import (
    TrafficLightColor,
    TrafficSignal,
    RoadSignType,
    RoadSign,
    RoadGeometryType,
    EnvironmentState,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DecisionEngine:
    """
    Modular autonomous decision-making engine.

    Args:
        corridor_half_width: Ego vehicle travel corridor half-width in pixels.
        hysteresis_frames: Number of consecutive clear frames required before
                           stepping down from a high-urgency braking state.
    """

    def __init__(
        self,
        corridor_half_width: float = 80.0,
        hysteresis_frames: int = 3,
    ) -> None:
        self.corridor_half_width = corridor_half_width
        self.hysteresis_frames = hysteresis_frames

        # State tracking for hysteresis
        self._last_action: VehicleAction = VehicleAction.CONTINUE
        self._consecutive_clear_frames: int = 0

        # Complete history of decisions across the processed video
        self._decision_history: List[DecisionOutput] = []

        logger.info(
            f"DecisionEngine initialised: corridor_half_width={corridor_half_width}px, "
            f"hysteresis={hysteresis_frames}f"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate_frame(
        self,
        frame_idx: int,
        fps: float,
        frame_width: int,
        frame_height: int,
        tracked_objects: List[TrackedObject],
        predictions: Optional[Dict[int, PredictionResult]] = None,
        risks: Optional[Dict[int, CollisionRisk]] = None,
        environment: Optional[EnvironmentState] = None,
    ) -> DecisionOutput:
        """
        Evaluate full scene context and compute the vehicle action command for the current frame.
        """
        safe_fps = max(1.0, fps)
        timestamp_sec = round(frame_idx / safe_fps, 2)
        safe_risks = risks or {}

        # 1. Evaluate spatial road context (corridor obstruction, lateral clearance & environment)
        road_context = evaluate_road_context(
            frame_width         = frame_width,
            frame_height        = frame_height,
            corridor_half_width = self.corridor_half_width,
            tracked_objects     = tracked_objects,
            risks               = safe_risks,
            environment         = environment,
        )

        # 2. Select primary threat obstacle
        primary_threat = self._identify_primary_threat(tracked_objects, safe_risks)

        # 3. Compute raw vehicle action from rule matrix
        raw_decision = self._apply_rule_matrix(
            frame_idx     = frame_idx,
            timestamp_sec = timestamp_sec,
            primary_threat= primary_threat,
            road_context  = road_context,
            frame_width   = frame_width,
        )

        # 4. Apply hysteresis to prevent erratic chattering between stop and go
        smoothed_action = self._apply_hysteresis(raw_decision.action)
        if smoothed_action != raw_decision.action:
            raw_decision.action = smoothed_action
            if smoothed_action == VehicleAction.EMERGENCY_STOP:
                raw_decision.target_speed_factor = 0.0
                raw_decision.reason += " (Holding emergency stop buffer)"
            elif smoothed_action == VehicleAction.STOP:
                raw_decision.target_speed_factor = 0.0
                raw_decision.reason += " (Holding controlled stop buffer)"

        self._last_action = raw_decision.action
        self._decision_history.append(raw_decision)
        return raw_decision

    def get_history(self) -> List[DecisionOutput]:
        """Return the complete chronological decision history."""
        return list(self._decision_history)

    def get_action_counts(self) -> Dict[str, int]:
        """Return distribution counts of all actions across the video."""
        counts = {action.value: 0 for action in VehicleAction}
        for d in self._decision_history:
            counts[d.action.value] = counts.get(d.action.value, 0) + 1
        return counts

    def get_dominant_action(self) -> str:
        """Return the most frequent non-CONTINUE action, or CONTINUE if all clear."""
        non_continue = [d.action.value for d in self._decision_history if d.action != VehicleAction.CONTINUE]
        if non_continue:
            return Counter(non_continue).most_common(1)[0][0]
        return VehicleAction.CONTINUE.value

    def reset(self) -> None:
        """Reset internal history and state machine between video jobs."""
        self._last_action = VehicleAction.CONTINUE
        self._consecutive_clear_frames = 0
        self._decision_history.clear()

    # ── Private Rule-Based Decision Logic ─────────────────────────────────────

    def _identify_primary_threat(
        self,
        objects: List[TrackedObject],
        risks: Dict[int, CollisionRisk],
    ) -> Optional[Tuple[TrackedObject, CollisionRisk]]:
        """Identify the single highest priority obstacle threatening vehicle path."""
        if not objects or not risks:
            return None

        candidates = []
        for obj in objects:
            risk = risks.get(obj.track_id)
            if risk:
                candidates.append((obj, risk))

        if not candidates:
            return None

        # Sort criteria:
        # 1. Risk severity rank (CRITICAL > HIGH > MEDIUM > LOW)
        # 2. TTC (smaller TTC = more urgent)
        # 3. Distance (closer = more urgent)
        def sort_key(item: Tuple[TrackedObject, CollisionRisk]):
            _, r = item
            ttc_val = r.ttc_sec if r.ttc_sec is not None else 999.0
            return (-r.risk_level.severity, ttc_val, r.distance_px)

        candidates.sort(key=sort_key)
        return candidates[0]

    def _apply_rule_matrix(
        self,
        frame_idx: int,
        timestamp_sec: float,
        primary_threat: Optional[Tuple[TrackedObject, CollisionRisk]],
        road_context: RoadContext,
        frame_width: int,
    ) -> DecisionOutput:
        """
        Deterministic decision-making rule table.
        """
        # Case 0: No obstacles or all obstacles are clear
        if primary_threat is None:
            if road_context.corridor_blocked:
                # Corridor has a static blockage
                if road_context.can_avoid_left:
                    return DecisionOutput(
                        frame_idx           = frame_idx,
                        timestamp_sec       = timestamp_sec,
                        action              = VehicleAction.REPLAN,
                        target_object_id    = road_context.blocking_object_id,
                        target_class        = road_context.blocking_object_class,
                        risk_level          = RiskLevel.HIGH,
                        ttc_sec             = None,
                        avoidance_direction = AvoidanceDirection.LEFT,
                        target_speed_factor = 0.6,
                        reason              = "Static obstacle blocks travel corridor. Re-routing path via clear left lane.",
                        road_context        = road_context,
                    )
                elif road_context.can_avoid_right:
                    return DecisionOutput(
                        frame_idx           = frame_idx,
                        timestamp_sec       = timestamp_sec,
                        action              = VehicleAction.REPLAN,
                        target_object_id    = road_context.blocking_object_id,
                        target_class        = road_context.blocking_object_class,
                        risk_level          = RiskLevel.HIGH,
                        ttc_sec             = None,
                        avoidance_direction = AvoidanceDirection.RIGHT,
                        target_speed_factor = 0.6,
                        reason              = "Static obstacle blocks travel corridor. Re-routing path via clear right lane.",
                        road_context        = road_context,
                    )
                else:
                    return DecisionOutput(
                        frame_idx           = frame_idx,
                        timestamp_sec       = timestamp_sec,
                        action              = VehicleAction.STOP,
                        target_object_id    = road_context.blocking_object_id,
                        target_class        = road_context.blocking_object_class,
                        risk_level          = RiskLevel.HIGH,
                        ttc_sec             = None,
                        avoidance_direction = AvoidanceDirection.NONE,
                        target_speed_factor = 0.0,
                        reason              = "Obstacle blocking travel corridor with both lateral escape routes obstructed. Controlled stop initiated.",
                        road_context        = road_context,
                    )

            # Environmental constraints check when corridor has no obstacle
            env_decision = self._check_environmental_constraints(frame_idx, timestamp_sec, road_context)
            if env_decision is not None:
                return env_decision

            return DecisionOutput(
                frame_idx           = frame_idx,
                timestamp_sec       = timestamp_sec,
                action              = VehicleAction.CONTINUE,
                target_object_id    = None,
                target_class        = None,
                risk_level          = RiskLevel.LOW,
                ttc_sec             = None,
                avoidance_direction = AvoidanceDirection.NONE,
                target_speed_factor = 1.0,
                reason              = "Corridor clear and all environmental constraints satisfied.",
                road_context        = road_context,
            )

        obj, risk = primary_threat
        tid = obj.track_id
        cls_name = obj.class_name.title()
        ttc = risk.ttc_sec
        dist = risk.distance_px
        lat_offset = risk.lateral_offset_px

        # ── 1. CRITICAL Risk ──────────────────────────────────────────────────
        if risk.risk_level == RiskLevel.CRITICAL:
            # Check if evasive steering is possible (obstacle is offset or leaving one side clear)
            can_escape_left = road_context.can_avoid_left and (lat_offset >= -20.0)
            can_escape_right = road_context.can_avoid_right and (lat_offset <= 20.0)

            # If emergency buffer is completely breached (< 65px) or no escape: EMERGENCY_STOP
            if dist < 65.0 or (not can_escape_left and not can_escape_right):
                ttc_str = f"TTC {ttc:.1f}s" if ttc is not None else f"Dist {dist:.0f}px"
                return DecisionOutput(
                    frame_idx           = frame_idx,
                    timestamp_sec       = timestamp_sec,
                    action              = VehicleAction.EMERGENCY_STOP,
                    target_object_id    = tid,
                    target_class        = obj.class_name,
                    risk_level          = RiskLevel.CRITICAL,
                    ttc_sec             = ttc,
                    avoidance_direction = AvoidanceDirection.NONE,
                    target_speed_factor = 0.0,
                    reason              = f"Immediate collision hazard: {cls_name} #{tid} ({ttc_str}). No safe evasive path; maximum emergency braking triggered.",
                    road_context        = road_context,
                )

            # Evasive steering maneuver possible
            chosen_dir = AvoidanceDirection.LEFT if can_escape_left else AvoidanceDirection.RIGHT
            return DecisionOutput(
                frame_idx           = frame_idx,
                timestamp_sec       = timestamp_sec,
                action              = VehicleAction.AVOID,
                target_object_id    = tid,
                target_class        = obj.class_name,
                risk_level          = RiskLevel.CRITICAL,
                ttc_sec             = ttc,
                avoidance_direction = chosen_dir,
                target_speed_factor = 0.35,
                reason              = f"Critical collision imminent with {cls_name} #{tid} (TTC {ttc:.1f}s). Initiating emergency evasive maneuver to {chosen_dir.value}.",
                road_context        = road_context,
            )

        # ── 2. HIGH Risk ──────────────────────────────────────────────────────
        if risk.risk_level == RiskLevel.HIGH:
            # Sub-scenario A: Stationary or slow-moving vehicle blocking corridor
            is_vehicle = obj.class_name in {"car", "truck", "bus"}
            speed = obj.motion.pixel_speed if obj.motion else 0.0

            if is_vehicle and speed < 3.0:
                if road_context.can_avoid_left:
                    return DecisionOutput(
                        frame_idx           = frame_idx,
                        timestamp_sec       = timestamp_sec,
                        action              = VehicleAction.REPLAN,
                        target_object_id    = tid,
                        target_class        = obj.class_name,
                        risk_level          = RiskLevel.HIGH,
                        ttc_sec             = ttc,
                        avoidance_direction = AvoidanceDirection.LEFT,
                        target_speed_factor = 0.6,
                        reason              = f"Slow-moving {cls_name} #{tid} blocks predicted path. Re-routing trajectory via left lane.",
                        road_context        = road_context,
                    )
                elif road_context.can_avoid_right:
                    return DecisionOutput(
                        frame_idx           = frame_idx,
                        timestamp_sec       = timestamp_sec,
                        action              = VehicleAction.REPLAN,
                        target_object_id    = tid,
                        target_class        = obj.class_name,
                        risk_level          = RiskLevel.HIGH,
                        ttc_sec             = ttc,
                        avoidance_direction = AvoidanceDirection.RIGHT,
                        target_speed_factor = 0.6,
                        reason              = f"Slow-moving {cls_name} #{tid} blocks predicted path. Re-routing trajectory via right lane.",
                        road_context        = road_context,
                    )
                else:
                    return DecisionOutput(
                        frame_idx           = frame_idx,
                        timestamp_sec       = timestamp_sec,
                        action              = VehicleAction.STOP,
                        target_object_id    = tid,
                        target_class        = obj.class_name,
                        risk_level          = RiskLevel.HIGH,
                        ttc_sec             = ttc,
                        avoidance_direction = AvoidanceDirection.NONE,
                        target_speed_factor = 0.0,
                        reason              = f"Preceding {cls_name} #{tid} stopped in travel corridor with no lateral clearance. Controlled stop initiated.",
                        road_context        = road_context,
                    )

            # Sub-scenario B: Crossing pedestrian / animal / bike
            is_crosser = obj.class_name in {"person", "bicycle", "motorcycle", "cow", "dog", "horse"}
            if is_crosser:
                # If short TTC (< 1.6s) and tight corridor: STOP to yield
                if (ttc is not None and ttc < 1.6) or not (road_context.can_avoid_left or road_context.can_avoid_right):
                    ttc_info = f"in {ttc:.1f}s" if ttc is not None else ""
                    return DecisionOutput(
                        frame_idx           = frame_idx,
                        timestamp_sec       = timestamp_sec,
                        action              = VehicleAction.STOP,
                        target_object_id    = tid,
                        target_class        = obj.class_name,
                        risk_level          = RiskLevel.HIGH,
                        ttc_sec             = ttc,
                        avoidance_direction = AvoidanceDirection.NONE,
                        target_speed_factor = 0.0,
                        reason              = f"Predicted {cls_name} #{tid} trajectory crosses vehicle corridor {ttc_info}. Braking to yield.",
                        road_context        = road_context,
                    )

                # Avoidance corridor available
                chosen_dir = AvoidanceDirection.LEFT if road_context.can_avoid_left else AvoidanceDirection.RIGHT
                ttc_info = f"in {ttc:.1f}s" if ttc is not None else ""
                return DecisionOutput(
                    frame_idx           = frame_idx,
                    timestamp_sec       = timestamp_sec,
                    action              = VehicleAction.AVOID,
                    target_object_id    = tid,
                    target_class        = obj.class_name,
                    risk_level          = RiskLevel.HIGH,
                    ttc_sec             = ttc,
                    avoidance_direction = chosen_dir,
                    target_speed_factor = 0.45,
                    reason              = f"Predicted {cls_name} #{tid} trajectory crosses vehicle path {ttc_info}. Steering {chosen_dir.value} to avoid.",
                    road_context        = road_context,
                )

            # Standard high closing rate vehicle
            ttc_str = f"TTC {ttc:.1f}s" if ttc is not None else f"Dist {dist:.0f}px"
            return DecisionOutput(
                frame_idx           = frame_idx,
                timestamp_sec       = timestamp_sec,
                action              = VehicleAction.SLOW_DOWN,
                target_object_id    = tid,
                target_class        = obj.class_name,
                risk_level          = RiskLevel.HIGH,
                ttc_sec             = ttc,
                avoidance_direction = AvoidanceDirection.NONE,
                target_speed_factor = 0.4,
                reason              = f"Rapid closing speed towards {cls_name} #{tid} ({ttc_str}). Decelerating strongly.",
                road_context        = road_context,
            )

        # ── 3. MEDIUM Risk ────────────────────────────────────────────────────
        if risk.risk_level == RiskLevel.MEDIUM:
            # Check if environment requires a higher-priority STOP or REPLAN
            env_decision = self._check_environmental_constraints(frame_idx, timestamp_sec, road_context)
            if env_decision and env_decision.action in (VehicleAction.STOP, VehicleAction.REPLAN):
                return env_decision

            ttc_str = f" (TTC ~ {ttc:.1f}s)" if ttc is not None else ""
            return DecisionOutput(
                frame_idx           = frame_idx,
                timestamp_sec       = timestamp_sec,
                action              = VehicleAction.SLOW_DOWN,
                target_object_id    = tid,
                target_class        = obj.class_name,
                risk_level          = RiskLevel.MEDIUM,
                ttc_sec             = ttc,
                avoidance_direction = AvoidanceDirection.NONE,
                target_speed_factor = 0.65,
                reason              = f"Caution: {cls_name} #{tid} approaching corridor{ttc_str}. Decelerating and monitoring.",
                road_context        = road_context,
            )

        # ── 4. LOW Risk ───────────────────────────────────────────────────────
        env_decision = self._check_environmental_constraints(frame_idx, timestamp_sec, road_context)
        if env_decision is not None:
            return env_decision

        return DecisionOutput(
            frame_idx           = frame_idx,
            timestamp_sec       = timestamp_sec,
            action              = VehicleAction.CONTINUE,
            target_object_id    = tid,
            target_class        = obj.class_name,
            risk_level          = RiskLevel.LOW,
            ttc_sec             = ttc,
            avoidance_direction = AvoidanceDirection.NONE,
            target_speed_factor = 1.0,
            reason              = f"Safe separation maintained with {cls_name} #{tid}; corridor clear.",
            road_context        = road_context,
        )

    def _check_environmental_constraints(
        self,
        frame_idx: int,
        timestamp_sec: float,
        road_context: RoadContext,
    ) -> Optional[DecisionOutput]:
        """
        Evaluate traffic signals, regulatory road signs, and map constraints.
        Precedence: RED signal / STOP sign > NO ENTRY > YELLOW / Warning / Ped Crossing > Speed Limits.
        """
        env: Optional[EnvironmentState] = getattr(road_context, "environment", None)
        if not env:
            return None

        # 1. Traffic Signal (RED = Compulsory stop before stop line)
        sig = env.active_signal
        if sig and sig.is_reliable:
            if sig.color == TrafficLightColor.RED:
                return DecisionOutput(
                    frame_idx           = frame_idx,
                    timestamp_sec       = timestamp_sec,
                    action              = VehicleAction.STOP,
                    target_object_id    = None,
                    target_class        = "traffic_light",
                    risk_level          = RiskLevel.HIGH,
                    ttc_sec             = None,
                    avoidance_direction = AvoidanceDirection.NONE,
                    target_speed_factor = 0.0,
                    reason              = f"Compulsory halt: RED traffic signal recognized ({sig.confidence*100:.0f}% confidence). Halting before stop line.",
                    road_context        = road_context,
                )
            elif sig.color == TrafficLightColor.YELLOW:
                return DecisionOutput(
                    frame_idx           = frame_idx,
                    timestamp_sec       = timestamp_sec,
                    action              = VehicleAction.SLOW_DOWN,
                    target_object_id    = None,
                    target_class        = "traffic_light",
                    risk_level          = RiskLevel.MEDIUM,
                    ttc_sec             = None,
                    avoidance_direction = AvoidanceDirection.NONE,
                    target_speed_factor = 0.40,
                    reason              = f"YELLOW traffic signal active ({sig.confidence*100:.0f}% confidence). Decelerating to prepare for stop.",
                    road_context        = road_context,
                )

        # 2. Road Signs: STOP & NO_ENTRY (Priority regulatory stops)
        valid_signs = [s for s in env.detected_signs if getattr(s, "confidence", 0.0) >= 0.60 and s.sign_type != RoadSignType.UNKNOWN]

        for sign in valid_signs:
            if sign.sign_type == RoadSignType.STOP:
                return DecisionOutput(
                    frame_idx           = frame_idx,
                    timestamp_sec       = timestamp_sec,
                    action              = VehicleAction.STOP,
                    target_object_id    = None,
                    target_class        = "road_sign",
                    risk_level          = RiskLevel.HIGH,
                    ttc_sec             = None,
                    avoidance_direction = AvoidanceDirection.NONE,
                    target_speed_factor = 0.0,
                    reason              = "Mandatory regulatory STOP sign observed. Vehicle halted at intersection.",
                    road_context        = road_context,
                )
            elif sign.sign_type == RoadSignType.NO_ENTRY:
                if road_context.can_avoid_left:
                    return DecisionOutput(
                        frame_idx           = frame_idx,
                        timestamp_sec       = timestamp_sec,
                        action              = VehicleAction.REPLAN,
                        target_object_id    = None,
                        target_class        = "road_sign",
                        risk_level          = RiskLevel.HIGH,
                        ttc_sec             = None,
                        avoidance_direction = AvoidanceDirection.LEFT,
                        target_speed_factor = 0.55,
                        reason              = "No-Entry road restriction on forward corridor. Re-routing path via left lane.",
                        road_context        = road_context,
                    )
                elif road_context.can_avoid_right:
                    return DecisionOutput(
                        frame_idx           = frame_idx,
                        timestamp_sec       = timestamp_sec,
                        action              = VehicleAction.REPLAN,
                        target_object_id    = None,
                        target_class        = "road_sign",
                        risk_level          = RiskLevel.HIGH,
                        ttc_sec             = None,
                        avoidance_direction = AvoidanceDirection.RIGHT,
                        target_speed_factor = 0.55,
                        reason              = "No-Entry road restriction on forward corridor. Re-routing path via right lane.",
                        road_context        = road_context,
                    )
                else:
                    return DecisionOutput(
                        frame_idx           = frame_idx,
                        timestamp_sec       = timestamp_sec,
                        action              = VehicleAction.STOP,
                        target_object_id    = None,
                        target_class        = "road_sign",
                        risk_level          = RiskLevel.HIGH,
                        ttc_sec             = None,
                        avoidance_direction = AvoidanceDirection.NONE,
                        target_speed_factor = 0.0,
                        reason              = "No-Entry road restriction: forward corridor legally prohibited with no alternative escape route. Halting.",
                        road_context        = road_context,
                    )

        # 3. Map No-Entry constraint
        if env.map_constraints and env.map_constraints.is_no_entry:
            return DecisionOutput(
                frame_idx           = frame_idx,
                timestamp_sec       = timestamp_sec,
                action              = VehicleAction.STOP,
                target_object_id    = None,
                target_class        = "map_restriction",
                risk_level          = RiskLevel.HIGH,
                ttc_sec             = None,
                avoidance_direction = AvoidanceDirection.NONE,
                target_speed_factor = 0.0,
                reason              = "Map restriction: Forward corridor designated one-way wrong-direction / no-entry.",
                road_context        = road_context,
            )

        # 4. Caution Signs: YIELD, PEDESTRIAN_CROSSING & WARNING
        for sign in valid_signs:
            if sign.sign_type == RoadSignType.YIELD:
                return DecisionOutput(
                    frame_idx           = frame_idx,
                    timestamp_sec       = timestamp_sec,
                    action              = VehicleAction.SLOW_DOWN,
                    target_object_id    = None,
                    target_class        = "road_sign",
                    risk_level          = RiskLevel.MEDIUM,
                    ttc_sec             = None,
                    avoidance_direction = AvoidanceDirection.NONE,
                    target_speed_factor = 0.55,
                    reason              = "Regulatory YIELD sign recognized. Yielding right-of-way and exercising increased caution.",
                    road_context        = road_context,
                )
            elif sign.sign_type == RoadSignType.PEDESTRIAN_CROSSING:
                return DecisionOutput(
                    frame_idx           = frame_idx,
                    timestamp_sec       = timestamp_sec,
                    action              = VehicleAction.SLOW_DOWN,
                    target_object_id    = None,
                    target_class        = "road_sign",
                    risk_level          = RiskLevel.MEDIUM,
                    ttc_sec             = None,
                    avoidance_direction = AvoidanceDirection.NONE,
                    target_speed_factor = 0.60,
                    reason              = "Pedestrian crossing sign recognized. Speed reduced and pedestrian caution active.",
                    road_context        = road_context,
                )
            elif sign.sign_type == RoadSignType.WARNING:
                return DecisionOutput(
                    frame_idx           = frame_idx,
                    timestamp_sec       = timestamp_sec,
                    action              = VehicleAction.SLOW_DOWN,
                    target_object_id    = None,
                    target_class        = "road_sign",
                    risk_level          = RiskLevel.MEDIUM,
                    ttc_sec             = None,
                    avoidance_direction = AvoidanceDirection.NONE,
                    target_speed_factor = 0.65,
                    reason              = "Advisory road hazard / warning sign observed. Precautionary speed reduction.",
                    road_context        = road_context,
                )

        # 5. Map & Intersection Approach (Phase 8C)
        from app.environment.schemas import RoadType

        is_intersection = (
            getattr(env, "has_intersection", False)
            or (getattr(env, "road_context", None) and getattr(env.road_context, "intersection", False))
            or (getattr(env, "map_constraints", None) and getattr(env.map_constraints, "has_intersection", False))
            or getattr(env, "road_type", None) == RoadType.URBAN_INTERSECTION
        )
        if is_intersection:
            geom = env.map_constraints.geometry_type.value if (getattr(env, "map_constraints", None) and getattr(env.map_constraints, "geometry_type", None)) else "urban"
            dist = getattr(env.map_constraints, "intersection_dist_px", None) if getattr(env, "map_constraints", None) else None
            dist_str = f" in {dist:.0f}px" if dist else ""
            return DecisionOutput(
                frame_idx           = frame_idx,
                timestamp_sec       = timestamp_sec,
                action              = VehicleAction.SLOW_DOWN,
                target_object_id    = None,
                target_class        = "intersection",
                risk_level          = RiskLevel.MEDIUM,
                ttc_sec             = None,
                avoidance_direction = AvoidanceDirection.NONE,
                target_speed_factor = 0.55,
                reason              = f"Approaching {geom} intersection{dist_str}. Controlled intersection approach; yielding to cross traffic.",
                road_context        = road_context,
            )

        # 6. Road Types: MARKET_AREA, HIGHWAY_MERGE, VILLAGE_ROAD
        if env.road_type == RoadType.MARKET_AREA:
            return DecisionOutput(
                frame_idx           = frame_idx,
                timestamp_sec       = timestamp_sec,
                action              = VehicleAction.SLOW_DOWN,
                target_object_id    = None,
                target_class        = "road_context",
                risk_level          = RiskLevel.MEDIUM,
                ttc_sec             = None,
                avoidance_direction = AvoidanceDirection.NONE,
                target_speed_factor = 0.50,
                reason              = "Market area context: dense mixed traffic zone. Heightened pedestrian caution active.",
                road_context        = road_context,
            )
        elif env.road_type == RoadType.HIGHWAY_MERGE:
            return DecisionOutput(
                frame_idx           = frame_idx,
                timestamp_sec       = timestamp_sec,
                action              = VehicleAction.SLOW_DOWN,
                target_object_id    = None,
                target_class        = "road_context",
                risk_level          = RiskLevel.MEDIUM,
                ttc_sec             = None,
                avoidance_direction = AvoidanceDirection.NONE,
                target_speed_factor = 0.65,
                reason              = "Highway merge section: monitoring entering traffic and maintaining merge buffer.",
                road_context        = road_context,
            )
        elif env.road_type == RoadType.VILLAGE_ROAD or (
            env.map_constraints and env.map_constraints.geometry_type == RoadGeometryType.UNMARKED_ROAD
        ):
            return DecisionOutput(
                frame_idx           = frame_idx,
                timestamp_sec       = timestamp_sec,
                action              = VehicleAction.CONTINUE,
                target_object_id    = None,
                target_class        = "road_context",
                risk_level          = RiskLevel.LOW,
                ttc_sec             = None,
                avoidance_direction = AvoidanceDirection.NONE,
                target_speed_factor = 0.70,
                reason              = "Unmarked road corridor. Dynamic corridor tracking active at conservative speed.",
                road_context        = road_context,
            )

        # 7. Speed Limit Sign
        for sign in valid_signs:
            if sign.sign_type == RoadSignType.SPEED_LIMIT:
                from app.environment.sign_rules import extract_speed_limit_value
                speed_val = extract_speed_limit_value(sign.value) or 40.0
                speed_factor = round(min(1.0, max(0.35, speed_val / 80.0)), 2)
                return DecisionOutput(
                    frame_idx           = frame_idx,
                    timestamp_sec       = timestamp_sec,
                    action              = VehicleAction.CONTINUE,
                    target_object_id    = None,
                    target_class        = "road_sign",
                    risk_level          = RiskLevel.LOW,
                    ttc_sec             = None,
                    avoidance_direction = AvoidanceDirection.NONE,
                    target_speed_factor = speed_factor,
                    reason              = f"Cruising within posted speed limit ({speed_val:.0f} km/h).",
                    road_context        = road_context,
                )

        return None

    def _apply_hysteresis(self, current_action: VehicleAction) -> VehicleAction:
        """
        Enforce safety buffer time so the autonomous vehicle doesn't instantaneously
        accelerate or release brakes on transient 1-frame gaps.
        """
        if self._last_action in (VehicleAction.EMERGENCY_STOP, VehicleAction.STOP):
            if current_action in (VehicleAction.CONTINUE, VehicleAction.SLOW_DOWN):
                self._consecutive_clear_frames += 1
                if self._consecutive_clear_frames < self.hysteresis_frames:
                    # Hold the braking state
                    return self._last_action
            else:
                self._consecutive_clear_frames = 0
        else:
            self._consecutive_clear_frames = 0

        return current_action
