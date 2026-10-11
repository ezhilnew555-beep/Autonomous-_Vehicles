"""
Configuration settings for Phase 4: Short-Term Motion Prediction.
"""
from dataclasses import dataclass


@dataclass
class PredictionConfig:
    # Total future time horizon to predict ahead (seconds)
    PREDICTION_HORIZON_SEC: float = 2.0

    # Interval between predicted trajectory waypoints (seconds)
    # E.g. 0.2s interval over 2.0s horizon generates 10 predicted points
    PREDICTION_INTERVAL_SEC: float = 0.2

    # Minimum historical frames required before producing a prediction
    MIN_HISTORY_FRAMES: int = 5

    # Minimum detection confidence threshold to trust for prediction
    MIN_CONFIDENCE_THRESHOLD: float = 0.35

    # Minimum pixel speed (px/frame) below which object is treated as stationary
    STATIONARY_SPEED_THRESHOLD_PX: float = 0.75

    # Class predictability multiplier (higher = more predictable / tighter confidence)
    # Used to scale prediction confidence score [0.0 - 1.0]
    CLASS_PREDICTABILITY: dict[str, float] = None

    def __post_init__(self):
        if self.CLASS_PREDICTABILITY is None:
            self.CLASS_PREDICTABILITY = {
                # Vehicles: highly predictable, constrained to road/lanes
                "bus": 0.95,
                "truck": 0.95,
                "car": 0.90,
                "motorcycle": 0.80,
                "bicycle": 0.75,
                # Pedestrians: can suddenly turn or halt
                "person": 0.60,
                # Animals: unpredictable sudden darting/stopping
                "cow": 0.45,
                "horse": 0.45,
                "dog": 0.35,
                "cat": 0.35,
                "sheep": 0.40,
                "bird": 0.30,
                "_default": 0.65,
            }
