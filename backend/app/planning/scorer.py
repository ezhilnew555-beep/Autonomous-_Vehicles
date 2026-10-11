"""
Trajectory Candidate Scorer — Phase 7.

Scores candidate paths based on collision clearance, inverse-distance safety margins,
squared curvature smoothness (preventing unnecessary sharp turns), and lateral lane deviation.
"""
from __future__ import annotations

from typing import List, Optional

from app.planning.schemas import CandidatePath
from app.planning.occupancy import LocalObstacleMap


class TrajectoryScorer:
    """
    Multi-objective trajectory cost scorer for autonomous vehicle evasive planning.

    Cost formulation:
      J = w_safety * J_safety + w_smooth * J_smooth + w_dev * J_dev
    """

    def __init__(
        self,
        w_safety: float = 1.2,
        w_smoothness: float = 450.0,
        w_deviation: float = 0.35,
    ) -> None:
        self.w_safety = w_safety
        self.w_smoothness = w_smoothness
        self.w_deviation = w_deviation

    def score_candidates(
        self,
        candidates: List[CandidatePath],
        obstacle_map: LocalObstacleMap,
        preferred_side: Optional[str] = None,  # "LEFT" | "RIGHT" | None
    ) -> Optional[CandidatePath]:
        """
        Evaluate all candidate paths, compute cost components, and select the optimal path.

        Args:
            candidates     : Bundle of generated trajectories.
            obstacle_map   : Local obstacle map with safety envelopes.
            preferred_side : Optional side to favor for temporal hysteresis.

        Returns:
            The optimal collision-free CandidatePath with minimum cost,
            or None if all candidates collide with obstacles.
        """
        feasible_candidates: List[CandidatePath] = []

        for c in candidates:
            pts = c.coordinates
            is_free, min_clearance = obstacle_map.evaluate_path(pts)
            c.is_collision_free = is_free
            c.min_obstacle_distance = round(min_clearance, 1)

            if not is_free:
                c.safety_cost = 99999.0
                c.total_cost = 99999.0
                continue

            # 1. Safety margin cost (higher penalty as clearance shrinks)
            # Normalised so clearance of 60px yields moderate cost, clearance < 20px yields high cost
            safe_val = max(1.0, min_clearance)
            safety_cost = (120.0 / safe_val) ** 1.6

            # 2. Smoothness cost (penalizes unnecessary sharp turns and high curvature)
            smooth_cost = sum(p.curvature ** 2 for p in c.points) * 1000.0

            # 3. Deviation cost (encourages returning to centerline when safe)
            dev_cost = (abs(c.lateral_offset) / 60.0) ** 1.8 * 15.0

            # 4. Hysteresis bias (slight discount if adhering to previous evasive choice)
            bias = 0.0
            if preferred_side == "LEFT" and c.lateral_offset < 0:
                bias = -12.0
            elif preferred_side == "RIGHT" and c.lateral_offset > 0:
                bias = -12.0

            total = (
                (self.w_safety * safety_cost)
                + (self.w_smoothness * smooth_cost)
                + (self.w_deviation * dev_cost)
                + bias
            )

            c.safety_cost = round(safety_cost, 2)
            c.smoothness_cost = round(smooth_cost, 2)
            c.deviation_cost = round(dev_cost, 2)
            c.total_cost = round(max(0.0, total), 2)
            feasible_candidates.append(c)

        if not feasible_candidates:
            return None

        # Select minimum cost collision-free path
        feasible_candidates.sort(key=lambda x: x.total_cost)
        return feasible_candidates[0]
