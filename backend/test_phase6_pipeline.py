"""
Phase 6 Full Pipeline & Video Processing Verification.
"""
import os
import time
import requests
import cv2
import numpy as np

# ── 1. Create a 5-second multi-object video ───────────────────────────────────
out_path = "test_phase6_e2e_video.mp4"
fps = 24
w, h = 640, 480
n_frames = fps * 5  # 120 frames

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

for i in range(n_frames):
    frame = np.full((h, w, 3), [40, 40, 40], dtype=np.uint8)
    cv2.line(frame, (w // 2 - 80, h), (w // 2 - 50, 100), (200, 200, 200), 2)
    cv2.line(frame, (w // 2 + 80, h), (w // 2 + 50, 100), (200, 200, 200), 2)
    car_y = int(120 + i * 2.0)
    cv2.rectangle(frame, (w // 2 - 35, car_y), (w // 2 + 35, car_y + 60), (30, 100, 220), -1)
    ped_x = int(80 + i * 2.2)
    cv2.circle(frame, (ped_x, 340), 16, (220, 150, 40), -1)
    cv2.putText(frame, f"Phase 6 Frame {i}/{n_frames}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    writer.write(frame)

writer.release()
print(f"[1] Created test video: {out_path} ({n_frames} frames @ {fps}fps)")

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

# ── 5. Verify Phase 6 Results & Schemas ───────────────────────────────────────
print("\n[5] Verifying Phase 6 Results:")
stats = data["stats"]
assert stats is not None, "Missing stats in API response"

print(f"  Processed Frames   : {stats['processed_frames']} / {stats['total_frames']}")
print(f"  Processing FPS     : {stats['processing_fps']}")
print(f"  Dominant Action    : {stats.get('dominant_action')}")
print(f"  Action Counts      : {stats.get('action_counts')}")
print(f"  Decisions Recorded : {len(stats.get('decision_history', []))}")

assert "action_counts" in stats, "action_counts missing from DetectionStats"
assert "dominant_action" in stats, "dominant_action missing from DetectionStats"
assert "decision_history" in stats, "decision_history missing from DetectionStats"
assert len(stats["decision_history"]) == stats["processed_frames"], "Decision history frame count mismatch"

# Check output video file existence
output_files = os.listdir("storage/outputs")
matching = [f for f in output_files if video_id in f]
assert matching, f"No output video file generated for {video_id}"
out_size = os.path.getsize(f"storage/outputs/{matching[0]}")
print(f"  Output Video       : {matching[0]} ({out_size:,} bytes)")

print("\n" + "=" * 70)
print("PHASE 6 END-TO-END PIPELINE VERIFICATION PASSED!")
print("=" * 70)
