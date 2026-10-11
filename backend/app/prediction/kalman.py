"""
KalmanFilter2D — constant-velocity 2D Kalman filter.

State vector : [x, y, vx, vy]
Observation  : [x, y]           (position only; velocity is latent)

Model assumptions (constant velocity):
    x_k+1 = x_k + vx_k * dt
    y_k+1 = y_k + vy_k * dt
    vx_k+1 = vx_k  (velocity is constant between frames)
    vy_k+1 = vy_k

Implementation uses pure NumPy — no scipy dependency.
"""
from __future__ import annotations

import numpy as np


class KalmanFilter2D:
    """
    Constant-velocity Kalman filter for 2D position tracking.

    Parameters control the process noise (how much the model trusts that
    motion is constant) and measurement noise (how much it trusts the
    raw centroid observations). Tuning these per object class lets the
    filter reflect how predictable that class is:

        Vehicles  → low process noise (smooth, physics-constrained movement)
        Persons   → higher process noise (can change direction suddenly)
        Animals   → highest process noise (least predictable)

    Args:
        sigma_process_pos : Standard deviation of position process noise (pixels).
        sigma_process_vel : Standard deviation of velocity process noise (px/frame).
        sigma_measure     : Standard deviation of measurement noise (pixels).
        dt                : Time step in frames (default 1).
    """

    def __init__(
        self,
        sigma_process_pos: float = 1.0,
        sigma_process_vel: float = 2.0,
        sigma_measure:     float = 5.0,
        dt:                float = 1.0,
    ) -> None:
        # ── State transition matrix  F  (4×4) ─────────────────────────────────
        # [1  0  dt  0 ]
        # [0  1  0   dt]
        # [0  0  1   0 ]
        # [0  0  0   1 ]
        self.F = np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1],
        ], dtype=np.float64)

        # ── Observation matrix  H  (2×4) ──────────────────────────────────────
        # We observe only position, not velocity.
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # ── Process noise covariance  Q  (4×4) ────────────────────────────────
        self.Q = np.diag([
            sigma_process_pos ** 2,
            sigma_process_pos ** 2,
            sigma_process_vel ** 2,
            sigma_process_vel ** 2,
        ]).astype(np.float64)

        # ── Measurement noise covariance  R  (2×2) ────────────────────────────
        self.R = np.diag([
            sigma_measure ** 2,
            sigma_measure ** 2,
        ]).astype(np.float64)

        # Identity matrix (pre-computed)
        self._I4 = np.eye(4, dtype=np.float64)

        # State vector and covariance (uninitialised until first observation)
        self.x: np.ndarray | None = None   # shape (4,) = [x, y, vx, vy]
        self.P: np.ndarray | None = None   # shape (4, 4)

        self._is_initialised = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def initialise(
        self,
        position: tuple[float, float],
        velocity: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """
        Set initial state. Call this with the first observed position.
        Velocity is estimated from the first two observations if not provided.
        Initial covariance is intentionally large so the filter converges quickly.
        """
        self.x = np.array(
            [position[0], position[1], velocity[0], velocity[1]],
            dtype=np.float64,
        )
        # Large initial uncertainty — converges within ~5 observations
        self.P = np.diag([100.0, 100.0, 25.0, 25.0]).astype(np.float64)
        self._is_initialised = True

    def update(self, measurement: tuple[float, float]) -> None:
        """
        Ingest one new position observation and update the state estimate.
        If the filter has not been initialised yet, initialise it here.
        """
        if not self._is_initialised:
            self.initialise(measurement)
            return

        # ── Predict ───────────────────────────────────────────────────────────
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # ── Update (Kalman gain + state correction) ───────────────────────────
        z     = np.array(measurement, dtype=np.float64)
        S     = self.H @ P_pred @ self.H.T + self.R       # Innovation covariance
        K     = P_pred @ self.H.T @ np.linalg.inv(S)      # Kalman gain
        y_err = z - self.H @ x_pred                        # Innovation (residual)

        self.x = x_pred + K @ y_err
        self.P = (self._I4 - K @ self.H) @ P_pred

    def predict_ahead(
        self,
        n_steps: int,
    ) -> tuple[list[tuple[float, float]], list[float]]:
        """
        Project state n_steps into the future WITHOUT updating the filter.

        Returns:
            positions     : List of (x, y) predicted positions.
            uncertainties : List of 1-sigma positional uncertainty radii (pixels).
                            Computed as sqrt(trace of 2×2 position block of P).
        """
        if not self._is_initialised or self.x is None or self.P is None:
            return [], []

        x = self.x.copy()
        P = self.P.copy()

        positions:     list[tuple[float, float]] = []
        uncertainties: list[float]               = []

        for _ in range(n_steps):
            x = self.F @ x
            P = self.F @ P @ self.F.T + self.Q

            positions.append((float(x[0]), float(x[1])))
            # 1-sigma radius from position covariance block (top-left 2×2)
            uncertainty = float(np.sqrt(P[0, 0] + P[1, 1]))
            uncertainties.append(uncertainty)

        return positions, uncertainties

    @property
    def state(self) -> tuple[float, float, float, float] | None:
        """Current (x, y, vx, vy) estimate, or None if uninitialised."""
        if self.x is None:
            return None
        return tuple(float(v) for v in self.x)

    @property
    def estimated_velocity(self) -> tuple[float, float] | None:
        """Estimated (vx, vy) in pixels/frame, or None."""
        if self.x is None:
            return None
        return (float(self.x[2]), float(self.x[3]))
