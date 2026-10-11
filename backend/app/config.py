"""Settings loaded from environment variables / .env file."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Server ────────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # ── Storage ───────────────────────────────────────────────────────────────
    UPLOAD_DIR: str = "storage/uploads"
    OUTPUT_DIR: str = "storage/outputs"

    # ── Pipeline Mode ─────────────────────────────────────────────────────────
    # "detection"  = Phase 2 Pure YOLO Detection (no tracking/TTC/planning)
    # "tracking"   = Phase 3 Object Tracking + Movement Estimation (ByteTrack)
    # "prediction" = Phase 4 Short-Term Motion Prediction (Kalman 2.0s baseline)
    # "full"       = Advanced Autonomous Driving Pipeline (Phases 3-9)
    PIPELINE_MODE: str = "prediction"

    # ── YOLO ──────────────────────────────────────────────────────────────────
    YOLO_MODEL_PATH: str = "models/yolo11n.pt"
    YOLO_CONFIDENCE: float = 0.25
    YOLO_IMG_SIZE: int = 640
    YOLO_DEVICE: str = "cpu"

    # ── ByteTrack ─────────────────────────────────────────────────────────────
    TRACK_ACTIVATION_THRESHOLD: float = 0.25
    TRACK_BUFFER: int = 30
    MATCH_THRESHOLD: float = 0.80
    HISTORY_LENGTH: int = 50

    # ── Motion Prediction (Phase 4) ───────────────────────────────────────────
    PREDICTION_HORIZON_SEC: float = 2.0      # 2.0 seconds future horizon
    PREDICTION_INTERVAL_SEC: float = 0.2     # 0.2 second intervals (10 points)
    PREDICTION_HORIZON: int = 30             # frames fallback
    PREDICTION_MIN_HISTORY: int = 5          # minimum history frames before predicting

    # ── Collision Risk Analysis (Phase 5) ─────────────────────────────────────
    TTC_CRITICAL_THRESHOLD: float = 1.0     # seconds
    TTC_HIGH_THRESHOLD: float = 2.5         # seconds
    TTC_MEDIUM_THRESHOLD: float = 4.5       # seconds
    EGO_CORRIDOR_HALF_WIDTH_PX: float = 80.0 # half-width of ego vehicle corridor in pixels
    CRITICAL_DISTANCE_PX: float = 70.0       # immediate emergency proximity threshold (pixels)

    # ── Road Signs (Phase 8B) ─────────────────────────────────────────────────
    SIGN_CONFIDENCE_THRESHOLD: float = 0.60
    SIGN_UNKNOWN_THRESHOLD: float = 0.40
    DEFAULT_SPEED_LIMIT_KMH: float = 50.0
_settings: Settings | None = None


def get_settings() -> Settings:
    """Singleton accessor — safe to call from FastAPI Depends."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
