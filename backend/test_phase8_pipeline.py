"""
Phase 8 End-to-End Pipeline & Video Processing Verification.

Uploads synthetic dynamic road video with Traffic Signal and Road Sign to FastAPI,
processes through full Phase 1-8 pipeline:
YOLO Detection -> ByteTrack -> Prediction -> Collision Analysis -> Environment Perception -> Decision Engine -> Path Planner -> Video Annotation Writer.
"""
import os
import sys
import time
import requests
import cv2
import numpy as np

# ── 1. Create a 3-second synthetic road scenario with Traffic Light & Road Sign
out_path = "test_phase8_e2e_video.mp4"
fps = 24
w, h = 640, 480
n_frames = fps * 3  # 72 frames

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

for i in range(n_frames):
    # Dark asphalt road
    frame = np.full((h, w, 3), [45, 45, 45], dtype=np.uint8)

    # Road edges / lane lines
    cv2.line(frame, (w // 2 - 140, h), (w // 2 - 40, 120), (220, 220, 220), 2)
    cv2.line(frame, (w // 2 + 140, h), (w // 2 + 40, 120), (220, 220, 220), 2)

    # 1. Traffic Light fixture on right gantry/pole (x: 540..570, y: 50..140)
    # Black housing
    cv2.rectangle(frame, (535, 45), (575, 145), (20, 20, 20), -1)
    cv2.rectangle(frame, (535, 45), (575, 145), (100, 100, 100), 2)
    # Upper Red lamp lit (bright red BGR: (0, 0, 255))
    cv2.circle(frame, (555, 68), 12, (0, 0, 255), -1)
    # Middle Yellow lamp off (dark)
    cv2.circle(frame, (555, 95), 10, (10, 40, 40), -1)
    # Bottom Green lamp off (dark)
    cv2.circle(frame, (555, 122), 10, (10, 40, 10), -1)

    # 2. Road Sign on left shoulder: Red circle with white horizontal bar (NO ENTRY)
    # (x: 70, y: 90), radius 26
    cv2.circle(frame, (70, 90), 26, (0, 0, 220), -1)
    cv2.rectangle(frame, (52, 85), (88, 95), (255, 255, 255), -1)

    # 3. Dynamic obstacle: Pedestrian crossing slowly
    ped_x = int(120 + i * 1.5)
    cv2.circle(frame, (ped_x, 340), 12, (230, 160, 45), -1)

    cv2.putText(frame, f"Phase 8 E2E Frame {i+1}/{n_frames}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    writer.write(frame)

writer.release()
print(f"[1] Created Phase 8 test video: {out_path} ({n_frames} frames @ {fps}fps)")

# ── 2. Upload video to FastAPI ────────────────────────────────────────────────
BASE = "http://localhost:8000/api/v1/video"

with open(out_path, "rb") as f:
    r = requests.post(f"{BASE}/upload", files={"file": (out_path, f, "video/mp4")})

assert r.status_code == 201, f"Upload failed: {r.status_code} {r.text}"
video_id = r.json()["video_id"]
print(f"[2] Uploaded video successfully -> video_id = {video_id}")

# ── 3. Trigger processing ─────────────────────────────────────────────────────
r = requests.post(f"{BASE}/process/{video_id}")
assert r.status_code == 200, f"Process start failed: {r.status_code} {r.text}"
print(f"[3] Background pipeline started for video {video_id}")

# ── 4. Poll status until complete ─────────────────────────────────────────────
print("[4] Polling status...")
for attempt in range(60):
    time.sleep(2)
    r = requests.get(f"{BASE}/status/{video_id}")
    data = r.json()
    status = data["status"]
    prog = data["progress"]
    print(f"    [{attempt+1:2d}] status={status:<12s} progress={prog:5.1f}%")
    if status in ("done", "error"):
        break

assert status == "done", f"Pipeline failed with message: {data.get('message')}"

# ── 5. Verify Phase 8 Results & Schemas ───────────────────────────────────────
print("\n[5] Verifying Phase 8 Results:")
stats = data["stats"]
assert stats is not None, "Missing stats in API response"

print(f"  Processed Frames          : {stats['processed_frames']} / {stats['total_frames']}")
print(f"  Processing FPS            : {stats['processing_fps']}")
print(f"  Final Path Status         : {stats.get('final_path_status')}")
print(f"  Dominant Action           : {stats.get('dominant_action')}")
print(f"  Traffic Signals Detected  : {len(stats.get('traffic_signals_detected', []))}")
print(f"  Road Signs Detected       : {len(stats.get('road_signs_detected', []))}")
print(f"  Active Map Profile        : {stats.get('active_map_profile')}")

assert "traffic_signals_detected" in stats, "traffic_signals_detected missing from DetectionStats"
assert "road_signs_detected" in stats, "road_signs_detected missing from DetectionStats"
assert "active_map_profile" in stats, "active_map_profile missing from DetectionStats"
assert stats["active_map_profile"] is not None, "active_map_profile is None"

# Check output video file existence
output_files = os.listdir("storage/outputs")
matching = [f for f in output_files if video_id in f]
assert matching, f"No output video file generated for {video_id}"
out_size = os.path.getsize(f"storage/outputs/{matching[0]}")
assert out_size > 5000, f"Output video is too small ({out_size} bytes)"

print(f"  Annotated Video File      : storage/outputs/{matching[0]} ({out_size:,} bytes)")
print("\n[SUCCESS] Phase 8 End-to-End Pipeline fully verified!")
