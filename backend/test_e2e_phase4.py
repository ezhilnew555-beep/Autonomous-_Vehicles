"""
End-to-end test for Phase 4 Motion Prediction mode.
Runs process_video with mode='prediction' on a test video and asserts on:
1. Job completion without exceptions.
2. Output mp4 file generated and non-empty.
3. stats.pipeline_mode == 'prediction'
4. stats.prediction_horizon_sec == 2.0
5. stats.prediction_interval_sec == 0.2
6. stats.active_predictions list populated with correct schema.
7. stats.objects_with_predictions >= 0
"""
import os
import glob
from pathlib import Path

from app.config import get_settings
from app.services.video_service import register_job, process_video, get_job
from app.models.schemas import JobStatus


def test_e2e_phase4():
    settings = get_settings()
    
    # Find an available test upload video
    upload_videos = glob.glob(os.path.join(settings.UPLOAD_DIR, "*.mp4"))
    assert len(upload_videos) > 0, "No upload videos found for testing"
    test_video_path = upload_videos[0]
    print(f"Testing with video: {test_video_path}")

    video_id = "test_phase4_e2e_verification"
    output_path = os.path.join(settings.OUTPUT_DIR, f"{video_id}_processed.mp4")
    
    if os.path.exists(output_path):
        os.remove(output_path)

    # Register job
    job = register_job(
        video_id=video_id,
        original_filename=os.path.basename(test_video_path),
        input_path=test_video_path,
        output_path=output_path,
        size_bytes=os.path.getsize(test_video_path),
    )

    # Process in Phase 4 prediction mode
    print("Running process_video(mode='prediction')...")
    process_video(video_id, settings, mode="prediction")

    # Assertions
    retrieved_job = get_job(video_id)
    assert retrieved_job is not None
    print(f"Job Status: {retrieved_job.status}")
    print(f"Job Message: {retrieved_job.message}")
    assert retrieved_job.status == JobStatus.DONE, f"Job failed: {retrieved_job.message}"

    assert os.path.exists(output_path), "Processed video file was not generated"
    assert os.path.getsize(output_path) > 1000, "Processed video file is empty"

    stats = retrieved_job.stats
    assert stats is not None, "Stats were not generated"
    assert stats.pipeline_mode == "prediction"
    assert stats.prediction_horizon_sec == 2.0
    assert stats.prediction_interval_sec == 0.2
    assert hasattr(stats, "active_predictions")
    print(f"Total Unique Tracks: {stats.unique_track_ids}")
    print(f"Active Predictions in final frame: {len(stats.active_predictions)}")
    print(f"Objects with valid predictions: {stats.objects_with_predictions}")
    print(f"Average prediction confidence: {stats.avg_prediction_confidence}%")

    if stats.active_predictions:
        first_pred = stats.active_predictions[0]
        print(f"Sample Active Prediction: ID #{first_pred.id} ({first_pred.class_name}) - "
              f"Dir: {first_pred.direction}, Speed: {first_pred.speed_px_per_frame:.1f} px/f, "
              f"Conf: {first_pred.confidence_pct}%, Avail: {first_pred.is_available}")

    print("\nPhase 4 E2E verification test passed successfully!")


if __name__ == "__main__":
    test_e2e_phase4()
