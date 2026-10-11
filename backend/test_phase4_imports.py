import sys
sys.path.insert(0, '.')

print('--- Phase 4 import verification ---')

# Core modules
from app.config import get_settings
s = get_settings()
print(f'[OK] config: horizon={s.PREDICTION_HORIZON}f  min_history={s.PREDICTION_MIN_HISTORY}')

# Prediction schemas
from app.prediction.schemas import PredictedPoint, PredictionResult
print('[OK] prediction.schemas')

# Kalman filter unit test
from app.prediction.kalman import KalmanFilter2D
kf = KalmanFilter2D(sigma_process_pos=1.0, sigma_process_vel=2.0, sigma_measure=5.0)
for i in range(6):
    kf.update((100.0 + i * 8, 200.0 + i * 3))
positions, uncertainties = kf.predict_ahead(10)
print(f'[OK] kalman: 6 obs, predict 10 steps')
print(f'     first  = ({positions[0][0]:.1f}, {positions[0][1]:.1f})  u={uncertainties[0]:.1f}px')
print(f'     last   = ({positions[9][0]:.1f}, {positions[9][1]:.1f})  u={uncertainties[9]:.1f}px')
print(f'     velest = {kf.estimated_velocity}')

# Kalman uncertainty grows with time
assert uncertainties[9] > uncertainties[0], "Uncertainty must grow with forecast horizon"
print(f'[OK] uncertainty growth: {uncertainties[0]:.1f} -> {uncertainties[9]:.1f} px (good)')

# Prediction engine
from app.prediction.predictor import PredictionEngine
engine = PredictionEngine(horizon=15, min_history=4)
print('[OK] prediction.predictor: PredictionEngine instantiated')

# Writer import
from app.video.writer import AnnotatedVideoWriter
print('[OK] video.writer')

# Service import
from app.services.video_service import register_job, get_job
print('[OK] video_service')

# Updated schema
from app.models.schemas import TrackedObjectSummary, DetectionStats
obj = TrackedObjectSummary(
    track_id=1, class_name='car', avg_confidence=0.91,
    frame_count=30, avg_pixel_speed=12.3, last_direction='E',
    has_prediction=True, prediction_method='kalman', predicted_direction='NE'
)
assert obj.has_prediction is True
assert obj.prediction_method == 'kalman'
print(f'[OK] schemas: prediction fields present and valid')

print()
print('=== All Phase 4 imports OK ===')
