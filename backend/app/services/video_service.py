"""
VideoService — orchestrates the full detection + tracking + prediction + collision + decision + planning pipeline.

Pipeline (Phases 1-7):
    FrameReader
    → YOLODetector (Phase 2)
    → ObjectTracker (Phase 3)
    → PredictionEngine (Phase 4)
    → CollisionRiskAnalyzer (Phase 5)
    → DecisionEngine (Phase 6)
    → AdaptivePathPlanner (Phase 7)
    → AnnotatedVideoWriter (Phases 3 + 4 + 5 + 6 + 7)

Each video job runs as a FastAPI BackgroundTask so HTTP responses return instantly.
Job state is stored in-process (sufficient for single-worker Phase 1–7).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List

from app.config import Settings
from app.detection.detector import YOLODetector, Detection
from app.tracking.tracker import ObjectTracker
from app.tracking.track_manager import TrackManager
from app.prediction.config import PredictionConfig
from app.prediction.predictor import PredictionEngine
from app.collision.analyzer import CollisionRiskAnalyzer
from app.collision.schemas import RiskLevel
from app.decision.engine import DecisionEngine
from app.decision.schemas import DecisionOutput
from app.planning.planner import AdaptivePathPlanner
from app.planning.schemas import PlanningResult
from app.environment.signals import TrafficSignalDetector
from app.environment.signs import RoadSignDetector
from app.environment.map_provider import ModularMapProvider
from app.environment.road_context import RoadContextModel
from app.environment.map_context import MapContext
from app.environment.environment_rules import EnvironmentRuleEngine
from app.environment.schemas import EnvironmentState, TrafficLightColor, RoadType
from app.video.reader import FrameReader
from app.video.writer import AnnotatedVideoWriter
from app.models.schemas import (
    JobStatus,
    DetectionStats,
    TrackedObjectSummary,
    ActiveTrackItem,
    ActivePredictionItem,
    RiskEventItem,
    DecisionItem,
    ReplanningEventItem,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ─── In-process job registry ──────────────────────────────────────────────────

@dataclass
class JobRecord:
    video_id:          str
    original_filename: str
    input_path:        str
    output_path:       str
    status:            JobStatus = JobStatus.PENDING
    progress:          float     = 0.0
    message:           str       = ""
    size_bytes:        int       = 0
    stats: Optional[DetectionStats] = None


_jobs: dict[str, JobRecord] = {}


def register_job(
    video_id:          str,
    original_filename: str,
    input_path:        str,
    output_path:       str,
    size_bytes:        int,
) -> JobRecord:
    job = JobRecord(
        video_id          = video_id,
        original_filename = original_filename,
        input_path        = input_path,
        output_path       = output_path,
        size_bytes        = size_bytes,
    )
    _jobs[video_id] = job
    logger.info(f"Job registered: {video_id} | '{original_filename}' ({size_bytes:,} bytes)")
    return job


def get_job(video_id: str) -> Optional[JobRecord]:
    if video_id in _jobs:
        return _jobs[video_id]

    try:
        from app.config import get_settings
        from app.utils.file_utils import get_output_path
        from pathlib import Path
        settings = get_settings()
        upload_dir = Path(settings.UPLOAD_DIR)
        for p in upload_dir.glob(f"{video_id}.*"):
            output_path = get_output_path(settings.OUTPUT_DIR, video_id)
            job = register_job(
                video_id=video_id,
                original_filename=p.name,
                input_path=str(p),
                output_path=str(output_path),
                size_bytes=p.stat().st_size,
            )
            if Path(output_path).exists():
                job.status = JobStatus.DONE
                job.progress = 1.0
            return job
    except Exception as e:
        logger.warning(f"Failed to auto-recover job for {video_id}: {e}")
    return None


def list_jobs() -> list[JobRecord]:
    return list(_jobs.values())


def _is_ego_vehicle_hood(d, frame_width: int, frame_height: int) -> bool:
    """
    Filter out bounding boxes corresponding ONLY to the camera vehicle's own hood or
    interior dashboard visible as a sliver at the extreme bottom frame boundary in dashcam videos.
    Never filters out vehicles driving ahead on the road.
    """
    if d.class_name in ("car", "truck", "bus"):
        is_touching_bottom = d.y2 >= (0.97 * frame_height)
        is_bottom_sliver = d.y1 >= (0.88 * frame_height)
        is_wide_hood = d.width >= (0.70 * frame_width)
        if is_touching_bottom and is_bottom_sliver and is_wide_hood:
            return True
    return False


def _deg_to_compass(deg: float) -> str:
    """Convert degrees [0, 360) to 8-point compass string."""
    dirs = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    idx = int((deg + 22.5) / 45.0) % 8
    return dirs[idx]


# ─── In-process job registry ──────────────────────────────────────────────────

def process_video(video_id: str, settings: Settings, mode: Optional[str] = None) -> None:
    """
    Video processing pipeline:
      - mode='detection' : Phase 2 pure YOLO detection with bounding boxes (no tracking/TTC/planning).
      - mode='full'      : Full autonomous pipeline (Phases 3-9).

    Runs inside a FastAPI BackgroundTasks call.
    Updates job.progress (0–100) and job.status throughout execution.
    """
    job = _jobs.get(video_id)
    if job is None:
        logger.error(f"process_video called for unknown video_id: {video_id}")
        return

    pipeline_mode = (mode or settings.PIPELINE_MODE).lower().strip()
    job.status   = JobStatus.PROCESSING
    job.progress = 0.0
    job.message  = f"Initialising pipeline ({pipeline_mode} mode)…"
    t_start      = time.time()

    try:
        # ── 1. Initialise YOLO Detector & FrameReader ───────────────────────────
        logger.info(f"[{video_id}] Initialising YOLO detector ({pipeline_mode} mode)…")
        detector = YOLODetector(
            model_path          = settings.YOLO_MODEL_PATH,
            confidence          = settings.YOLO_CONFIDENCE,
            imgsz               = settings.YOLO_IMG_SIZE,
            device              = settings.YOLO_DEVICE,
            filter_road_classes = True,
        )

        reader = FrameReader(job.input_path)
        fps    = max(1.0, reader.fps)

        # ─── PHASE 2: PURE DETECTION MODE (Enhanced with temporal stability) ──
        if pipeline_mode == "detection":
            tracker = ObjectTracker(
                track_activation_threshold = settings.TRACK_ACTIVATION_THRESHOLD,
                lost_track_buffer          = settings.TRACK_BUFFER,
                minimum_matching_threshold = settings.MATCH_THRESHOLD,
                frame_rate                 = max(1, int(fps)),
                history_length             = settings.HISTORY_LENGTH,
            )
            raw_class_counter: dict[str, int] = {}
            track_class_votes: dict[int, dict[str, int]] = {}
            track_frame_count: dict[int, int] = {}
            total_detections_count = 0
            peak_concurrent = 0
            processed_frames = 0
            job.message = "Detecting road objects frame-by-frame…"

            with AnnotatedVideoWriter(job.output_path, fps, reader.width, reader.height) as writer:
                for frame_idx, frame in reader:
                    try:
                        raw_dets = detector.detect(frame)
                    except Exception as det_err:
                        logger.warning(f"[{video_id}] Frame #{frame_idx} detection warning: {det_err}")
                        raw_dets = []

                    # Exclude camera vehicle dashboard/hood
                    detections = [
                        d for d in raw_dets
                        if not _is_ego_vehicle_hood(d, reader.width, reader.height)
                    ]

                    # Track to associate detections across frames and stabilize classes
                    tracked = tracker.update(detections)
                    for obj in tracked:
                        tid = obj.track_id
                        if tid < 0:
                            continue
                        if tid not in track_class_votes:
                            track_class_votes[tid] = {}
                            track_frame_count[tid] = 0
                        cls = obj.class_name
                        track_class_votes[tid][cls] = track_class_votes[tid].get(cls, 0) + 1
                        track_frame_count[tid] += 1

                    if len(detections) > peak_concurrent:
                        peak_concurrent = len(detections)

                    # Smooth detection display labels using track dominant class if available
                    track_dominant = {}
                    for obj in tracked:
                        tid = obj.track_id
                        if tid in track_class_votes and track_class_votes[tid]:
                            track_dominant[tid] = max(track_class_votes[tid].items(), key=lambda x: x[1])[0]

                    smoothed_detections = []
                    for i, d in enumerate(detections):
                        assigned_cls = d.class_name
                        if i < len(tracked) and tracked[i].track_id in track_dominant:
                            assigned_cls = track_dominant[tracked[i].track_id]

                        raw_class_counter[assigned_cls] = raw_class_counter.get(assigned_cls, 0) + 1
                        total_detections_count += 1
                        smoothed_detections.append(Detection(
                            class_id   = d.class_id,
                            class_name = assigned_cls,
                            confidence = d.confidence,
                            x1 = d.x1, y1 = d.y1, x2 = d.x2, y2 = d.y2,
                        ))

                    writer.write_detection_frame(frame, smoothed_detections, frame_idx=frame_idx, fps=fps)

                    processed_frames += 1
                    if reader.frame_count > 0:
                        job.progress = (processed_frames / reader.frame_count) * 100.0

            reader.close()
            elapsed = time.time() - t_start

            # Compute unique persistent objects using track majority class voting
            min_persistence = min(10, max(3, int(fps * 0.3)))
            unique_objects_by_class: dict[str, int] = {}
            for tid, votes in track_class_votes.items():
                if track_frame_count.get(tid, 0) >= min_persistence:
                    dominant_cls = max(votes.items(), key=lambda x: x[1])[0]
                    unique_objects_by_class[dominant_cls] = unique_objects_by_class.get(dominant_cls, 0) + 1

            total_unique = sum(unique_objects_by_class.values())
            if total_unique == 0 and track_class_votes:
                for tid, votes in track_class_votes.items():
                    dominant_cls = max(votes.items(), key=lambda x: x[1])[0]
                    unique_objects_by_class[dominant_cls] = unique_objects_by_class.get(dominant_cls, 0) + 1
                total_unique = sum(unique_objects_by_class.values())

            job.stats = DetectionStats(
                total_frames              = reader.frame_count,
                processed_frames          = processed_frames,
                processing_fps            = round(processed_frames / elapsed, 1) if elapsed > 0 else 0.0,
                video_fps                 = round(fps, 2),
                duration_seconds          = round(reader.duration, 2),
                unique_track_ids          = total_unique,
                objects_by_class          = unique_objects_by_class,
                tracked_objects           = [],
                pipeline_mode             = "detection",
                total_objects_detected    = total_unique,
                total_frame_detections    = total_detections_count,
                peak_concurrent_objects   = peak_concurrent,
                frame_detections_by_class = raw_class_counter,
                processing_time_sec       = round(elapsed, 2),
            )

            job.status   = JobStatus.DONE
            job.progress = 100.0
            job.message  = (
                f"Phase 2 Done — {processed_frames} frames in {elapsed:.1f}s "
                f"({job.stats.processing_fps} fps) | {total_unique} unique objects ({total_detections_count} detections evaluated)"
            )
            logger.info(f"[{video_id}] {job.message}")
            return

        # ─── PHASE 3: OBJECT TRACKING + MOVEMENT ESTIMATION MODE ─────────────
        if pipeline_mode == "tracking":
            logger.info(f"[{video_id}] Initialising TrackManager (Phase 3 mode)…")
            track_manager = TrackManager(
                track_activation_threshold = settings.TRACK_ACTIVATION_THRESHOLD,
                lost_track_buffer          = settings.TRACK_BUFFER,
                minimum_matching_threshold = settings.MATCH_THRESHOLD,
                frame_rate                 = max(1, int(fps)),
                history_length             = settings.HISTORY_LENGTH,
            )
            raw_class_counter: dict[str, int] = {}
            total_detections_count = 0
            peak_concurrent = 0
            processed_frames = 0
            job.message = "Tracking objects and estimating movement frame-by-frame…"

            with AnnotatedVideoWriter(job.output_path, fps, reader.width, reader.height) as writer:
                for frame_idx, frame in reader:
                    try:
                        raw_dets = detector.detect(frame)
                    except Exception as det_err:
                        logger.warning(f"[{video_id}] Frame #{frame_idx} detection warning: {det_err}")
                        raw_dets = []

                    # Exclude camera vehicle dashboard/hood
                    detections = [
                        d for d in raw_dets
                        if not _is_ego_vehicle_hood(d, reader.width, reader.height)
                    ]

                    # Update ByteTrack tracker + movement estimation
                    track_manager.update(detections, frame_idx=frame_idx)
                    tracked_objects = track_manager.tracked_objects

                    for d in detections:
                        raw_class_counter[d.class_name] = raw_class_counter.get(d.class_name, 0) + 1
                        total_detections_count += 1

                    if len(tracked_objects) > peak_concurrent:
                        peak_concurrent = len(tracked_objects)

                    # Write Phase 3 annotated frame with persistent IDs, trajectory history, and direction arrows
                    writer.write_tracking_frame(
                        frame,
                        tracked_objects,
                        frame_idx=frame_idx,
                        fps=fps,
                        total_unique=track_manager.total_unique_tracks,
                    )

                    processed_frames += 1
                    if reader.frame_count > 0:
                        job.progress = (processed_frames / reader.frame_count) * 100.0

            reader.close()
            elapsed = time.time() - t_start

            # Build ActiveTrackItem list for active tracks in the final frame
            active_items = []
            for st in track_manager.structured_tracks:
                active_items.append(ActiveTrackItem(
                    id                 = st["id"],
                    class_name         = st["class_name"],
                    confidence         = st["confidence"],
                    direction          = st["direction"],
                    direction_symbol   = st["direction_symbol"],
                    speed_px_per_frame = st["velocity"],
                    speed_px_per_sec   = st["velocity_sec"],
                    center             = list(st["center"]),
                    bbox               = st["bbox"],
                    frames_seen        = len(st["history"]),
                ))

            # Build all tracks summary for table display
            all_summaries = track_manager.get_all_tracks_summary()
            tracked_obj_summaries = [
                TrackedObjectSummary(
                    track_id        = s["track_id"],
                    class_name      = s["class_name"],
                    avg_confidence  = s["avg_confidence"],
                    frame_count     = s["frame_count"],
                    avg_pixel_speed = s["avg_pixel_speed"],
                    last_direction  = "—",
                )
                for s in all_summaries
            ]

            # Objects by class
            unique_by_class: dict[str, int] = {}
            for s in all_summaries:
                cls = s["class_name"]
                unique_by_class[cls] = unique_by_class.get(cls, 0) + 1

            total_unique = track_manager.total_unique_tracks

            job.stats = DetectionStats(
                total_frames              = reader.frame_count,
                processed_frames          = processed_frames,
                processing_fps            = round(processed_frames / elapsed, 1) if elapsed > 0 else 0.0,
                video_fps                 = round(fps, 2),
                duration_seconds          = round(reader.duration, 2),
                unique_track_ids          = total_unique,
                objects_by_class          = unique_by_class,
                tracked_objects           = tracked_obj_summaries,
                pipeline_mode             = "tracking",
                total_objects_detected    = total_unique,
                total_frame_detections    = total_detections_count,
                peak_concurrent_objects   = peak_concurrent,
                frame_detections_by_class = raw_class_counter,
                processing_time_sec       = round(elapsed, 2),
                active_tracks             = active_items,
                active_tracks_count       = len(active_items),
                total_tracks_count        = total_unique,
                tracking_duration_sec     = round(reader.duration, 2),
            )

            job.status   = JobStatus.DONE
            job.progress = 100.0
            job.message  = (
                f"Phase 3 Done — {processed_frames} frames in {elapsed:.1f}s "
                f"({job.stats.processing_fps} fps) | {total_unique} persistent tracked objects"
            )
            logger.info(f"[{video_id}] {job.message}")
            return

        # ─── PHASE 4: SHORT-TERM MOTION PREDICTION MODE ───────────────────────
        if pipeline_mode == "prediction":
            logger.info(f"[{video_id}] Initialising TrackManager & PredictionEngine (Phase 4 mode)…")
            track_manager = TrackManager(
                track_activation_threshold = settings.TRACK_ACTIVATION_THRESHOLD,
                lost_track_buffer          = settings.TRACK_BUFFER,
                minimum_matching_threshold = settings.MATCH_THRESHOLD,
                frame_rate                 = max(1, int(fps)),
                history_length             = settings.HISTORY_LENGTH,
            )
            pred_config = PredictionConfig(
                PREDICTION_HORIZON_SEC  = settings.PREDICTION_HORIZON_SEC,
                PREDICTION_INTERVAL_SEC = settings.PREDICTION_INTERVAL_SEC,
                MIN_HISTORY_FRAMES      = settings.PREDICTION_MIN_HISTORY,
            )
            predictor = PredictionEngine(
                config = pred_config,
                fps    = fps,
            )

            raw_class_counter: dict[str, int] = {}
            total_detections_count = 0
            peak_concurrent = 0
            processed_frames = 0
            tracks_with_valid_pred: set[int] = set()
            job.message = "Tracking objects & predicting short-term trajectories…"

            with AnnotatedVideoWriter(job.output_path, fps, reader.width, reader.height) as writer:
                for frame_idx, frame in reader:
                    try:
                        raw_dets = detector.detect(frame)
                    except Exception as det_err:
                        logger.warning(f"[{video_id}] Frame #{frame_idx} detection warning: {det_err}")
                        raw_dets = []

                    # Exclude camera vehicle dashboard/hood
                    detections = [
                        d for d in raw_dets
                        if not _is_ego_vehicle_hood(d, reader.width, reader.height)
                    ]

                    # Update ByteTrack tracker + movement estimation
                    track_manager.update(detections, frame_idx=frame_idx)
                    tracked_objects = track_manager.tracked_objects

                    for d in detections:
                        raw_class_counter[d.class_name] = raw_class_counter.get(d.class_name, 0) + 1
                        total_detections_count += 1

                    if len(tracked_objects) > peak_concurrent:
                        peak_concurrent = len(tracked_objects)

                    # Update Kalman filter and generate short-term predicted trajectories
                    predictions = predictor.update_and_predict(tracked_objects)

                    for tid, pred in predictions.items():
                        if pred.has_prediction:
                            tracks_with_valid_pred.add(tid)

                    # Write Phase 4 annotated frame
                    writer.write_prediction_frame(
                        frame,
                        tracked_objects,
                        predictions,
                        frame_idx   = frame_idx,
                        fps         = fps,
                        horizon_sec = settings.PREDICTION_HORIZON_SEC,
                    )

                    processed_frames += 1
                    if reader.frame_count > 0:
                        job.progress = (processed_frames / reader.frame_count) * 100.0

            reader.close()
            elapsed = time.time() - t_start

            # Build ActivePredictionItem list for active tracks in the final frame
            active_predictions: list[ActivePredictionItem] = []
            final_predictions = predictor.update_and_predict(track_manager.tracked_objects)
            conf_sum = 0.0
            conf_count = 0

            for obj in track_manager.tracked_objects:
                tid = obj.track_id
                pred = final_predictions.get(tid)
                dir_str = obj.motion.direction if obj.motion else "UNKNOWN"
                dir_sym = obj.motion.direction_symbol if obj.motion else "?"
                spd_f = obj.motion.pixel_speed if obj.motion else 0.0
                spd_s = obj.motion.pixel_speed_sec if obj.motion else 0.0

                if pred and pred.is_available:
                    conf_pct = round(pred.prediction_confidence * 100.0, 1)
                    conf_sum += conf_pct
                    conf_count += 1
                    active_predictions.append(ActivePredictionItem(
                        id                    = tid,
                        class_name            = obj.class_name,
                        direction             = dir_str,
                        direction_symbol      = dir_sym,
                        speed_px_per_frame    = spd_f,
                        speed_px_per_sec      = spd_s,
                        horizon_sec           = pred.prediction_horizon_sec,
                        confidence_pct        = conf_pct,
                        is_available          = True,
                        unavailability_reason = None,
                        prediction_method     = pred.method,
                        waypoints_count       = len(pred.predicted_points),
                        waypoints             = [[round(p.x, 1), round(p.y, 1)] for p in pred.predicted_points],
                        current_center        = [round(obj.center[0], 1), round(obj.center[1], 1)],
                    ))
                else:
                    reason = pred.unavailability_reason if pred else "Insufficient history"
                    active_predictions.append(ActivePredictionItem(
                        id                    = tid,
                        class_name            = obj.class_name,
                        direction             = dir_str,
                        direction_symbol      = dir_sym,
                        speed_px_per_frame    = spd_f,
                        speed_px_per_sec      = spd_s,
                        horizon_sec           = settings.PREDICTION_HORIZON_SEC,
                        confidence_pct        = 0.0,
                        is_available          = False,
                        unavailability_reason = reason,
                        prediction_method     = "none",
                        waypoints_count       = 0,
                        waypoints             = [],
                        current_center        = [round(obj.center[0], 1), round(obj.center[1], 1)],
                    ))

            avg_conf = round(conf_sum / conf_count, 1) if conf_count > 0 else 0.0

            # Build all tracks summary for table display
            all_summaries = track_manager.get_all_tracks_summary()
            tracked_obj_summaries = [
                TrackedObjectSummary(
                    track_id            = s["track_id"],
                    class_name          = s["class_name"],
                    avg_confidence      = s["avg_confidence"],
                    frame_count         = s["frame_count"],
                    avg_pixel_speed     = s["avg_pixel_speed"],
                    last_direction      = "—",
                    has_prediction      = s["track_id"] in tracks_with_valid_pred,
                    prediction_method   = "kalman" if s["track_id"] in tracks_with_valid_pred else "none",
                    predicted_direction = "—",
                )
                for s in all_summaries
            ]

            # Objects by class
            unique_by_class: dict[str, int] = {}
            for s in all_summaries:
                cls = s["class_name"]
                unique_by_class[cls] = unique_by_class.get(cls, 0) + 1

            total_unique = track_manager.total_unique_tracks

            job.stats = DetectionStats(
                total_frames              = reader.frame_count,
                processed_frames          = processed_frames,
                processing_fps            = round(processed_frames / elapsed, 1) if elapsed > 0 else 0.0,
                video_fps                 = round(fps, 2),
                duration_seconds          = round(reader.duration, 2),
                unique_track_ids          = total_unique,
                objects_by_class          = unique_by_class,
                tracked_objects           = tracked_obj_summaries,
                pipeline_mode             = "prediction",
                total_objects_detected    = total_unique,
                total_frame_detections    = total_detections_count,
                peak_concurrent_objects   = peak_concurrent,
                frame_detections_by_class = raw_class_counter,
                processing_time_sec       = round(elapsed, 2),
                active_tracks_count       = len(track_manager.tracked_objects),
                total_tracks_count        = total_unique,
                tracking_duration_sec     = round(reader.duration, 2),
                objects_with_predictions  = len(tracks_with_valid_pred),
                prediction_horizon_frames = int(round(settings.PREDICTION_HORIZON_SEC * fps)),
                prediction_horizon_sec    = settings.PREDICTION_HORIZON_SEC,
                prediction_interval_sec   = settings.PREDICTION_INTERVAL_SEC,
                active_predictions        = active_predictions,
                active_predictions_count  = sum(1 for p in active_predictions if p.is_available),
                avg_prediction_confidence = avg_conf,
            )

            job.status   = JobStatus.DONE
            job.progress = 100.0
            job.message  = (
                f"Phase 4 Done — {processed_frames} frames in {elapsed:.1f}s "
                f"({job.stats.processing_fps} fps) | {total_unique} tracked objects | "
                f"{len(tracks_with_valid_pred)} predicted trajectories ({settings.PREDICTION_HORIZON_SEC}s horizon)"
            )
            logger.info(f"[{video_id}] {job.message}")
            return

        logger.info(f"[{video_id}] Initialising ByteTrack tracker…")
        tracker = ObjectTracker(
            track_activation_threshold = settings.TRACK_ACTIVATION_THRESHOLD,
            lost_track_buffer          = settings.TRACK_BUFFER,
            minimum_matching_threshold = settings.MATCH_THRESHOLD,
            frame_rate                 = max(1, int(fps)),
            history_length             = settings.HISTORY_LENGTH,
        )

        logger.info(f"[{video_id}] Initialising PredictionEngine…")
        predictor = PredictionEngine(
            horizon     = settings.PREDICTION_HORIZON,
            min_history = settings.PREDICTION_MIN_HISTORY,
        )

        logger.info(f"[{video_id}] Initialising CollisionRiskAnalyzer (Phase 5)…")
        risk_analyzer = CollisionRiskAnalyzer(
            ttc_critical_thresh = settings.TTC_CRITICAL_THRESHOLD,
            ttc_high_thresh     = settings.TTC_HIGH_THRESHOLD,
            ttc_medium_thresh   = settings.TTC_MEDIUM_THRESHOLD,
            corridor_half_width = settings.EGO_CORRIDOR_HALF_WIDTH_PX,
            critical_distance   = settings.CRITICAL_DISTANCE_PX,
        )

        logger.info(f"[{video_id}] Initialising DecisionEngine (Phase 6)…")
        decision_engine = DecisionEngine(
            corridor_half_width = settings.EGO_CORRIDOR_HALF_WIDTH_PX,
            hysteresis_frames   = 3,
        )

        logger.info(f"[{video_id}] Initialising AdaptivePathPlanner (Phase 7)…")
        path_planner = AdaptivePathPlanner(
            lookahead_distance = 340.0,
            num_waypoints      = 25,
            road_margin_px     = 35.0,
        )

        logger.info(f"[{video_id}] Initialising Environment Awareness (Phase 8)…")
        signal_detector = TrafficSignalDetector(history_len=5)
        sign_detector = RoadSignDetector()
        map_provider = ModularMapProvider(default_profile="straight_road")
        env_rule_engine = EnvironmentRuleEngine(min_sign_confidence=0.60)

        # ── Aggregate state ────────────────────────────────────────────────────
        raw_class_counter:     dict[str, int]  = {}
        total_detections_count: int            = 0
        peak_concurrent:       int             = 0
        track_records:         dict[int, dict] = {}
        risk_counter:          dict[str, int]  = {
            RiskLevel.LOW.value:      0,
            RiskLevel.MEDIUM.value:   0,
            RiskLevel.HIGH.value:     0,
            RiskLevel.CRITICAL.value: 0,
        }
        total_planning_ms = 0.0
        processed_frames = 0
        last_planning_result: Optional[PlanningResult] = None
        detected_signals_list: list[dict] = []
        detected_signs_list: list[dict] = []
        last_active_signal: Optional[dict] = None
        last_map_constraints = None

        has_logged_detection_start = False
        has_logged_tracking_start = False
        has_logged_prediction_start = False
        prev_decision_action = None
        prev_path_status = None
        logged_risk_track_ids = set()
        last_tracked: list[TrackedObject] = []
        last_valid_dets: list[Detection] = []
        cached_active_signal = None
        cached_detected_signs = []

        job.message = "Processing frames…"
        logger.info(f"[{video_id}] VIDEO_STARTED | Input: '{job.original_filename}', Duration: {reader.duration:.1f}s, FPS: {fps:.1f}")

        # ── Frame loop ─────────────────────────────────────────────────────────
        with AnnotatedVideoWriter(
            job.output_path, fps, reader.width, reader.height
        ) as writer:

            for frame_idx, frame in reader:

                # 1. Detection (High-Speed Adaptive Cadence: run YOLO on keyframes, extrapolate on intermediate)
                should_detect = (frame_idx % 2 == 0) or (processed_frames == 0)
                if should_detect:
                    raw_dets = detector.detect(frame)
                    detections = [
                        d for d in raw_dets
                        if not _is_ego_vehicle_hood(d, reader.width, reader.height)
                    ]
                    last_valid_dets = detections
                else:
                    if last_tracked:
                        extrapolated = []
                        for obj in last_tracked:
                            m = obj.motion
                            dx = m.dx if (m and hasattr(m, "dx")) else 0.0
                            dy = m.dy if (m and hasattr(m, "dy")) else 0.0
                            extrapolated.append(Detection(
                                class_id=obj.class_id,
                                class_name=obj.class_name,
                                confidence=max(0.25, obj.confidence * 0.98),
                                x1=max(0.0, min(float(reader.width), obj.x1 + dx)),
                                y1=max(0.0, min(float(reader.height), obj.y1 + dy)),
                                x2=max(0.0, min(float(reader.width), obj.x2 + dx)),
                                y2=max(0.0, min(float(reader.height), obj.y2 + dy)),
                            ))
                        detections = extrapolated
                    else:
                        detections = last_valid_dets

                if not has_logged_detection_start:
                    logger.info(f"[{video_id}] DETECTION_STARTED | Initial frame #{frame_idx} ({len(detections)} objects detected)")
                    has_logged_detection_start = True

                if len(detections) > peak_concurrent:
                    peak_concurrent = len(detections)

                # 2. Tracking + motion estimation
                tracked_objects = tracker.update(detections)
                last_tracked = tracked_objects
                if tracked_objects and not has_logged_tracking_start:
                    logger.info(f"[{video_id}] TRACKING_STARTED | Frame #{frame_idx}: Tracking active with {len(tracked_objects)} initial tracks")
                    has_logged_tracking_start = True

                # 3. Motion prediction (Phase 4)
                predictions = predictor.update_and_predict(tracked_objects)
                if predictions and not has_logged_prediction_start:
                    logger.info(f"[{video_id}] PREDICTION_STARTED | Frame #{frame_idx}: Trajectory prediction engine active")
                    has_logged_prediction_start = True

                # 4. Collision Risk Analysis (Phase 5)
                risks = risk_analyzer.analyze_frame(
                    frame_idx       = frame_idx,
                    fps             = fps,
                    frame_width     = reader.width,
                    frame_height    = reader.height,
                    tracked_objects = tracked_objects,
                    predictions     = predictions,
                )
                for r_tid, r_item in risks.items():
                    if r_item.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
                        r_key = (r_tid, r_item.risk_level.value)
                        if r_key not in logged_risk_track_ids:
                            logger.warning(
                                f"[{video_id}] RISK_DETECTED | Frame #{frame_idx} Track #{r_tid} "
                                f"({r_item.class_name}): {r_item.risk_level.value} (TTC: {r_item.ttc_sec}s)"
                            )
                            logged_risk_track_ids.add(r_key)

                # 4.5. Environment Perception (Phase 8: Traffic Signals, Road Signs & Map)
                # Evaluated every 3 frames for high-speed throughput
                if frame_idx % 3 == 0 or cached_active_signal is None:
                    cached_active_signal = signal_detector.detect_signal(frame, yolo_detections=detections)
                    cached_detected_signs = sign_detector.detect_signs(frame, yolo_detections=detections)
                active_signal = cached_active_signal
                detected_signs = cached_detected_signs

                timestamp_sec = round(frame_idx / max(1.0, fps), 2)
                map_constraints = map_provider.get_constraints(frame_idx, timestamp_sec)
                last_map_constraints = map_constraints

                road_ctx_model = RoadContextModel.create_for_scenario(
                    map_constraints.road_type,
                    frame_width=reader.width
                )
                m_ctx = MapContext(
                    road_id=map_constraints.road_id,
                    road_type=map_constraints.road_type,
                    direction=map_constraints.destination_direction,
                    speed_limit=map_constraints.speed_limit_kmh,
                    restricted_directions=map_constraints.restricted_directions,
                    has_intersection=map_constraints.has_intersection,
                    intersection_dist_px=map_constraints.intersection_dist_px,
                    lane_count=map_constraints.lane_count,
                    road_notes=map_constraints.road_notes,
                )
                env_rules_result = env_rule_engine.evaluate_environment(
                    road_context=road_ctx_model,
                    traffic_signal=active_signal,
                    road_signs=detected_signs,
                    map_context=m_ctx,
                )

                environment = EnvironmentState(
                    frame_idx             = frame_idx,
                    timestamp_sec         = timestamp_sec,
                    active_signal         = active_signal,
                    detected_signs        = detected_signs,
                    map_constraints       = map_constraints,
                    road_type             = map_constraints.road_type,
                    has_intersection      = map_constraints.has_intersection,
                    drivable_region       = road_ctx_model.drivable_region,
                    active_rules          = env_rules_result["active_rules"],
                    planning_constraints  = env_rules_result["planning_constraints"],
                    active_speed_limit    = env_rules_result["speed_constraints"],
                    restricted_directions = env_rules_result["restricted_directions"],
                )

                if active_signal and active_signal.is_reliable:
                    sig_dict = {
                        "color": active_signal.color.value,
                        "confidence": active_signal.confidence,
                        "is_reliable": active_signal.is_reliable,
                        "reason": active_signal.reason,
                        "distance_px": active_signal.distance_px,
                    }
                    last_active_signal = sig_dict
                    if not any(s["color"] == sig_dict["color"] for s in detected_signals_list):
                        detected_signals_list.append(sig_dict)

                for s in detected_signs:
                    s_dict = {
                        "sign_type": s.sign_type.value,
                        "confidence": s.confidence,
                        "value": s.value,
                        "action_directive": s.action_directive,
                        "rule": getattr(s, "rule", "") or s.action_directive,
                        "frame_idx": frame_idx,
                        "timestamp_sec": timestamp_sec,
                    }
                    if not any(existing["sign_type"] == s_dict["sign_type"] for existing in detected_signs_list):
                        detected_signs_list.append(s_dict)

                # 5. Autonomous Decision-Making (Phase 6 + Phase 8 Environment)
                decision = decision_engine.evaluate_frame(
                    frame_idx       = frame_idx,
                    fps             = fps,
                    frame_width     = reader.width,
                    frame_height    = reader.height,
                    tracked_objects = tracked_objects,
                    predictions     = predictions,
                    risks           = risks,
                    environment     = environment,
                )
                if decision.action != prev_decision_action:
                    logger.info(
                        f"[{video_id}] DECISION_CHANGED | Frame #{frame_idx}: "
                        f"{prev_decision_action.value if prev_decision_action else 'INIT'} -> {decision.action.value} "
                        f"({decision.reason})"
                    )
                    prev_decision_action = decision.action
                    if decision.action.value == "EMERGENCY_STOP":
                        logger.error(f"[{video_id}] EMERGENCY_STOP | Frame #{frame_idx}: {decision.reason}")

                # 6. Adaptive Collision-Free Path Planning (Phase 7 + Phase 8 Constraints)
                planning = path_planner.plan_frame(
                    frame_idx       = frame_idx,
                    fps             = fps,
                    frame_width     = reader.width,
                    frame_height    = reader.height,
                    tracked_objects = tracked_objects,
                    predictions     = predictions,
                    decision        = decision,
                    environment     = environment,
                )
                if planning.status != prev_path_status:
                    logger.info(f"[{video_id}] PATH_SELECTED | Frame #{frame_idx}: {planning.status.value}")
                    prev_path_status = planning.status

                if planning.needs_replanning:
                    logger.info(f"[{video_id}] REPLAN_TRIGGERED | Frame #{frame_idx}: {planning.replanning_reason}")

                last_planning_result = planning
                total_planning_ms += planning.planning_time_ms

                # 7. Write annotated frame (All Phase 3-8 visual layers)
                writer.write_frame(frame, tracked_objects, predictions, risks, decision, planning, environment)

                # 8. Accumulate stats
                for obj in tracked_objects:
                    tid = obj.track_id
                    cls = obj.class_name
                    raw_class_counter[cls] = raw_class_counter.get(cls, 0) + 1
                    total_detections_count += 1

                    if tid not in track_records:
                        track_records[tid] = {
                            "class_votes":        {},
                            "confidences":        [],
                            "speeds":             [],
                            "direction":          "—",
                            "frame_count":        0,
                            "had_prediction":     False,
                            "prediction_method":  "none",
                            "predicted_dir_deg":  None,
                            "max_risk_level":     RiskLevel.LOW,
                            "min_ttc":            None,
                            "last_risk_reason":   "—",
                        }
                    rec = track_records[tid]
                    rec["class_votes"][cls] = rec["class_votes"].get(cls, 0) + 1
                    rec["confidences"].append(obj.confidence)
                    rec["frame_count"] += 1
                    if obj.motion:
                        rec["speeds"].append(obj.motion.pixel_speed)
                        rec["direction"] = obj.motion.direction_label

                    # Prediction metadata
                    pred = predictions.get(tid)
                    if pred and pred.has_prediction:
                        rec["had_prediction"]    = True
                        rec["prediction_method"] = pred.method
                        if pred.predicted_direction_deg is not None:
                            rec["predicted_dir_deg"] = pred.predicted_direction_deg

                    # Collision risk metadata
                    risk = risks.get(tid)
                    if risk:
                        if risk.risk_level.severity > rec["max_risk_level"].severity:
                            rec["max_risk_level"] = risk.risk_level
                        if risk.ttc_sec is not None:
                            if rec["min_ttc"] is None or risk.ttc_sec < rec["min_ttc"]:
                                rec["min_ttc"] = risk.ttc_sec
                        rec["last_risk_reason"] = risk.reason
                        risk_counter[risk.risk_level.value] = risk_counter.get(risk.risk_level.value, 0) + 1

                processed_frames += 1
                if reader.frame_count > 0:
                    job.progress = (processed_frames / reader.frame_count) * 100.0

        reader.close()

        # ── Build final stats ──────────────────────────────────────────────────
        elapsed = time.time() - t_start

        tracked_summaries: list[TrackedObjectSummary] = []
        objects_with_predictions = 0
        min_persistence = min(8, max(3, int(fps * 0.25)))
        unique_objects_by_class: dict[str, int] = {}

        for tid, rec in track_records.items():
            confs  = rec["confidences"]
            speeds = rec["speeds"]

            deg = rec["predicted_dir_deg"]
            predicted_dir_label = _deg_to_compass(deg) if deg is not None else "—"

            if rec["had_prediction"]:
                objects_with_predictions += 1

            votes = rec.get("class_votes", {})
            dominant_cls = max(votes.items(), key=lambda x: x[1])[0] if votes else "unknown"

            if rec["frame_count"] >= min_persistence:
                unique_objects_by_class[dominant_cls] = unique_objects_by_class.get(dominant_cls, 0) + 1

            tracked_summaries.append(TrackedObjectSummary(
                track_id            = tid,
                class_name          = dominant_cls,
                avg_confidence      = round(sum(confs) / len(confs), 3) if confs else 0.0,
                frame_count         = rec["frame_count"],
                avg_pixel_speed     = round(sum(speeds) / len(speeds), 2) if speeds else 0.0,
                last_direction      = rec["direction"],
                has_prediction      = rec["had_prediction"],
                prediction_method   = rec["prediction_method"],
                predicted_direction = predicted_dir_label,
                max_risk_level      = rec["max_risk_level"].value,
                min_ttc_sec         = round(rec["min_ttc"], 2) if rec["min_ttc"] is not None else None,
                last_risk_reason    = rec["last_risk_reason"],
            ))

        # Format recorded risk events (Phase 5)
        raw_events = risk_analyzer.get_recorded_events()
        formatted_events = [
            RiskEventItem(
                timestamp_sec = ev.timestamp_sec,
                frame_idx     = ev.frame_idx,
                object_id     = ev.object_id,
                object_class  = ev.object_class,
                distance_px   = ev.distance_px,
                ttc_sec       = ev.ttc_sec,
                risk_level    = ev.risk_level.value,
                reason        = ev.reason,
            )
            for ev in raw_events
        ]

        # Format decision history (Phase 6)
        raw_decisions = decision_engine.get_history()
        formatted_decisions = [
            DecisionItem(
                frame_idx           = d.frame_idx,
                timestamp_sec       = d.timestamp_sec,
                action              = d.action.value,
                target_object_id    = d.target_object_id,
                target_class        = d.target_class,
                risk_level          = d.risk_level.value if d.risk_level else None,
                ttc_sec             = d.ttc_sec,
                avoidance_direction = d.avoidance_direction.value,
                target_speed_factor = d.target_speed_factor,
                reason              = d.reason,
            )
            for d in raw_decisions
        ]

        # Format replanning events (Phase 7)
        raw_replans = path_planner.get_replanning_events()
        formatted_replans = [
            ReplanningEventItem(
                timestamp_sec    = r.timestamp_sec,
                frame_idx        = r.frame_idx,
                trigger_reason   = r.trigger_reason,
                previous_status  = r.previous_status,
                new_status       = r.new_status,
                planning_time_ms = r.planning_time_ms,
                obstacle_id      = r.obstacle_id,
                obstacle_class   = r.obstacle_class,
            )
            for r in raw_replans
        ]

        action_counts = decision_engine.get_action_counts()
        dominant_action = decision_engine.get_dominant_action()
        avg_planning_time = round(total_planning_ms / max(1, processed_frames), 2)
        final_status = last_planning_result.status.value if last_planning_result else "CONTINUE_ORIGINAL"

        crit_count = sum(1 for e in formatted_events if e.risk_level == RiskLevel.CRITICAL.value)
        high_count = sum(1 for e in formatted_events if e.risk_level == RiskLevel.HIGH.value)
        total_unique = sum(unique_objects_by_class.values()) or len(track_records)

        job.stats = DetectionStats(
            total_frames              = reader.frame_count,
            processed_frames          = processed_frames,
            processing_fps            = round(processed_frames / elapsed, 1) if elapsed > 0 else 0.0,
            video_fps                 = round(fps, 2),
            duration_seconds          = round(reader.duration, 2),
            unique_track_ids          = total_unique,
            objects_by_class          = unique_objects_by_class,
            tracked_objects           = sorted(tracked_summaries, key=lambda x: x.track_id),
            pipeline_mode             = "full",
            total_objects_detected    = total_unique,
            total_frame_detections    = total_detections_count,
            peak_concurrent_objects   = peak_concurrent,
            frame_detections_by_class = raw_class_counter,
            objects_with_predictions  = objects_with_predictions,
            prediction_horizon_frames = settings.PREDICTION_HORIZON,
            risk_counts               = risk_counter,
            critical_events_count     = crit_count,
            high_events_count         = high_count,
            risk_events               = formatted_events,
            action_counts             = action_counts,
            dominant_action           = dominant_action,
            decision_history          = formatted_decisions,
            replanning_count          = len(formatted_replans),
            avg_planning_time_ms      = avg_planning_time,
            final_path_status         = final_status,
            replanning_events         = formatted_replans,
            active_traffic_signal    = last_active_signal,
            traffic_signals_detected = detected_signals_list,
            road_signs_detected      = detected_signs_list,
            active_map_profile       = {
                "road_id": getattr(last_map_constraints, "road_id", "LOCAL_ROAD_001"),
                "road_type": getattr(last_map_constraints, "road_type", RoadType.URBAN_ROAD).value,
                "geometry_type": last_map_constraints.geometry_type.value,
                "lane_count": last_map_constraints.lane_count,
                "speed_limit_kmh": last_map_constraints.speed_limit_kmh,
                "has_intersection": last_map_constraints.has_intersection,
                "destination_direction": last_map_constraints.destination_direction,
                "restricted_directions": getattr(last_map_constraints, "restricted_directions", []),
                "road_notes": last_map_constraints.road_notes,
            } if last_map_constraints else None,
        )

        job.status   = JobStatus.DONE
        job.progress = 100.0
        job.message  = (
            f"Done — {processed_frames} frames in {elapsed:.1f}s "
            f"({job.stats.processing_fps} fps) | "
            f"Path: {final_status} ({avg_planning_time}ms avg plan) | "
            f"Action: {dominant_action}"
        )
        logger.info(f"[{video_id}] SCENARIO_COMPLETED | {job.message}")

    except Exception as exc:
        elapsed = time.time() - t_start
        logger.error(f"[{video_id}] SCENARIO_FAILED | Pipeline error after {elapsed:.1f}s: {exc}", exc_info=True)
        job.status  = JobStatus.ERROR
        job.message = f"Processing error: {exc}"


# ─── Utilities ────────────────────────────────────────────────────────────────

_COMPASS_LABELS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]


def _deg_to_compass(deg: float) -> str:
    """Convert angle in degrees (0=right, CCW positive) to 8-point compass label."""
    idx = int((deg + 22.5) / 45.0) % 8
    return _COMPASS_LABELS[idx]
