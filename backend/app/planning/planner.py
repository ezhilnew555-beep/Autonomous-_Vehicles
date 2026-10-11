"""
AdaptivePathPlanner — Phase 7: Adaptive Collision-Free Path Planning.

Orchestrates local obstacle mapping, candidate trajectory generation, collision checks,
multi-objective cost scoring, and adaptive replanning when the vehicle's path is obstructed.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from app.tracking.tracker import TrackedObject
from app.prediction.schemas import PredictionResult
from app.decision.schemas import DecisionOutput, VehicleAction
from app.environment.schemas import EnvironmentState, RoadSignType, RoadGeometryType
from app.planning.schemas import (
    PlanningStatus,
    CandidatePath,
    PlanningResult,
    ReplanningEvent,
)
from app.planning.occupancy import LocalObstacleMap
from app.planning.generator import TrajectoryGenerator
from app.planning.scorer import TrajectoryScorer
from app.utils.logger import get_logger

logger = get_logger(__name__)


class AdaptivePathPlanner:
    """
    Adaptive path planning engine for autonomous driving prototype.
    """

    def __init__(
        self,
        lookahead_distance: float = 340.0,
        num_waypoints: int = 25,
        road_margin_px: float = 35.0,
    ) -> None:
        self.lookahead_distance = lookahead_distance
        self.num_waypoints = num_waypoints
        self.road_margin_px = road_margin_px

        self.generator = TrajectoryGenerator(
            num_waypoints      = num_waypoints,
            lookahead_distance = lookahead_distance,
            road_margin_px     = road_margin_px,
        )
        self.scorer = TrajectoryScorer()

        self._last_status: PlanningStatus = PlanningStatus.CONTINUE_ORIGINAL
        self._last_side: Optional[str] = None
        self._replanning_events: List[ReplanningEvent] = []

        logger.info(
            f"AdaptivePathPlanner ready: lookahead={lookahead_distance}px, "
            f"waypoints={num_waypoints}, margin={road_margin_px}px"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def plan_frame(
        self,
        frame_idx: int,
        fps: float,
        frame_width: int,
        frame_height: int,
        tracked_objects: List[TrackedObject],
        predictions: Optional[Dict[int, PredictionResult]] = None,
        decision: Optional[DecisionOutput] = None,
        environment: Optional[EnvironmentState] = None,
    ) -> PlanningResult:
        """
        Execute adaptive path planning cycle for current video frame.
        """
        t0 = time.perf_counter()
        safe_fps = max(1.0, fps)
        timestamp_sec = round(frame_idx / safe_fps, 2)

        origin_x = frame_width / 2.0
        origin_y = float(frame_height)

        # 1. Build local obstacle map with inflated safety zones
        obstacle_map = LocalObstacleMap(
            frame_width     = frame_width,
            frame_height    = frame_height,
            tracked_objects = tracked_objects,
            predictions     = predictions,
        )

        # 2. Generate candidate trajectory bundle
        candidates = self.generator.generate_candidate_bundle(
            origin_x     = origin_x,
            origin_y     = origin_y,
            frame_width  = frame_width,
            frame_height = frame_height,
        )

        # Nominal centerline trajectory (dx = 0)
        centerline_candidate = next((c for c in candidates if c.lateral_offset == 0.0), candidates[0])
        original_path = centerline_candidate.coordinates

        # 3. Check environmental constraints (No-entry, turn restrictions, emergency stop, stop line)
        is_corridor_prohibited = False
        is_no_left_turn = False
        is_no_right_turn = False

        if environment:
            map_c = getattr(environment, "map_constraints", None)
            if map_c and getattr(map_c, "is_no_entry", False):
                is_corridor_prohibited = True
            signs = getattr(environment, "detected_signs", getattr(environment, "road_signs", []))
            for s in signs:
                if getattr(s, "confidence", 0.0) >= 0.60:
                    st = getattr(s, "sign_type", None)
                    if st == RoadSignType.NO_ENTRY:
                        is_corridor_prohibited = True
                    elif st == RoadSignType.NO_LEFT_TURN:
                        is_no_left_turn = True
                    elif st == RoadSignType.NO_RIGHT_TURN:
                        is_no_right_turn = True


        # Check if nominal path is safe from obstacles and legally permissible
        is_center_free, center_clearance = obstacle_map.evaluate_path(original_path)
        if is_corridor_prohibited:
            is_center_free = False

        needs_replanning = False
        replanning_reason = "Nominal trajectory clear; safe separation maintained."
        selected_candidate: Optional[CandidatePath] = None
        status = PlanningStatus.CONTINUE_ORIGINAL

        # Emergency stop takes unconditional precedence
        if decision and decision.action == VehicleAction.EMERGENCY_STOP:
            status = PlanningStatus.NO_SAFE_PATH
            selected_candidate = None
            needs_replanning = True
            replanning_reason = f"EMERGENCY_STOP: {decision.reason}"
        elif is_center_free and not (decision and decision.action in (VehicleAction.AVOID, VehicleAction.REPLAN)) and center_clearance > 40.0:
            status = PlanningStatus.CONTINUE_ORIGINAL
            selected_candidate = centerline_candidate
            needs_replanning = False
            self._last_side = None
            if decision and decision.action == VehicleAction.STOP:
                replanning_reason = f"Controlled stop before intersection / line: {decision.reason}"
        else:
            # 4. Centerline compromised or evasion requested -> Filter candidates by legal sign rules
            viable_candidates: List[CandidatePath] = []
            for c in candidates:
                # Prohibit forward corridor if NO_ENTRY active
                if is_corridor_prohibited and abs(c.lateral_offset) <= 10.0:
                    continue
                # Prohibit left evasive turns if NO_LEFT_TURN active
                if is_no_left_turn and c.lateral_offset < -10.0:
                    continue
                # Prohibit right evasive turns if NO_RIGHT_TURN active
                if is_no_right_turn and c.lateral_offset > 10.0:
                    continue
                # Prohibit paths that breach physical drivable road boundaries (Phase 8C)
                drivable = getattr(environment, "drivable_region", None)
                if drivable:
                    if any(pt[0] < drivable.left_boundary_px or pt[0] > drivable.right_boundary_px for pt in c.coordinates):
                        continue
                viable_candidates.append(c)

            best_candidate = self.scorer.score_candidates(
                candidates     = viable_candidates,
                obstacle_map   = obstacle_map,
                preferred_side = self._last_side,
            )

            if best_candidate is not None:
                selected_candidate = best_candidate
                if best_candidate.lateral_offset < -10.0:
                    status = PlanningStatus.REPLANNED_AVOID_LEFT
                    self._last_side = "LEFT"
                    needs_replanning = True
                    replanning_reason = (
                        f"Alternative trajectory generated: deviating {abs(best_candidate.lateral_offset):.0f}px LEFT "
                        f"around obstacle (clearance {best_candidate.min_obstacle_distance:.0f}px)."
                    )
                elif best_candidate.lateral_offset > 10.0:
                    status = PlanningStatus.REPLANNED_AVOID_RIGHT
                    self._last_side = "RIGHT"
                    needs_replanning = True
                    replanning_reason = (
                        f"Alternative trajectory generated: deviating {best_candidate.lateral_offset:.0f}px RIGHT "
                        f"around obstacle (clearance {best_candidate.min_obstacle_distance:.0f}px)."
                    )
                else:
                    status = PlanningStatus.CONTINUE_ORIGINAL
                    self._last_side = None
                    needs_replanning = False
            else:
                # All candidates blocked
                status = PlanningStatus.NO_SAFE_PATH
                selected_candidate = None
                needs_replanning = True
                replanning_reason = "No collision-free trajectory found. Full road corridor obstructed."

        # Truncate path if stopping before a stop line or controlled halt
        if selected_candidate and decision and decision.action == VehicleAction.STOP:
            # Shorten trajectory ahead to represent bringing vehicle to halt before stopping boundary
            halt_len = max(4, len(selected_candidate.coordinates) // 2)
            safe_path = selected_candidate.coordinates[:halt_len]
        else:
            safe_path = selected_candidate.coordinates if selected_candidate else []

        planning_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        min_clearance = selected_candidate.min_obstacle_distance if selected_candidate else 0.0

        # Log replanning event when path status transitions or evasion activates
        if (status != self._last_status and status != PlanningStatus.CONTINUE_ORIGINAL) or (
            needs_replanning and frame_idx % 24 == 0
        ):
            target_id = decision.target_object_id if decision else None
            target_cls = decision.target_class if decision else None
            self._replanning_events.append(ReplanningEvent(
                timestamp_sec    = timestamp_sec,
                frame_idx        = frame_idx,
                trigger_reason   = replanning_reason,
                previous_status  = self._last_status.value,
                new_status       = status.value,
                planning_time_ms = planning_time_ms,
                obstacle_id      = target_id,
                obstacle_class   = target_cls,
            ))

        self._last_status = status

        return PlanningResult(
            frame_idx         = frame_idx,
            timestamp_sec     = timestamp_sec,
            status            = status,
            needs_replanning  = needs_replanning,
            replanning_reason = replanning_reason,
            original_path     = original_path,
            safe_path         = safe_path,
            all_candidates    = candidates,
            selected_path     = selected_candidate,
            min_clearance_px  = min_clearance,
            planning_time_ms  = planning_time_ms,
        )

    def get_replanning_events(self) -> List[ReplanningEvent]:
        """Return all logged path replanning occurrences."""
        return list(self._replanning_events)

    def reset(self) -> None:
        """Reset internal state between jobs."""
        self._last_status = PlanningStatus.CONTINUE_ORIGINAL
        self._last_side = None
        self._replanning_events.clear()
