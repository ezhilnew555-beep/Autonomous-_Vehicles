"""
Eareyes 2k26 — Complete End-to-End Integration Test (Full Pipeline, Phases 1-9).

Tests the full autonomous driving pipeline:
  Video Upload (Phase 1)
  → YOLO Detection (Phase 2)
  → ByteTrack Object Tracking (Phase 3)
  → Kalman Motion Prediction (Phase 4)
  → Collision Risk / TTC Analysis (Phase 5)
  → Autonomous Decision-Making (Phase 6)
  → Adaptive Path Planning (Phase 7)
  → Environment Awareness — Traffic Signals, Road Signs, Map (Phase 8)
  → Annotated Video Output

Usage:
  python test_full_pipeline_e2e.py

Requires backend to be running: uvicorn app.main:app --port 8000
"""
import os
import sys
import time
import requests
import cv2
import numpy as np

BASE = "http://localhost:8000/api/v1/video"

# ── 1. Synthesize a rich 5-second test road video ─────────────────────────────
OUT_PATH = "test_full_e2e_video.mp4"
FPS   = 24
W, H  = 640, 480
N_FRAMES = FPS * 5  # 120 frames

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
vwriter = cv2.VideoWriter(OUT_PATH, fourcc, FPS, (W, H))

for i in range(N_FRAMES):
    frame = np.full((H, W, 3), [45, 45, 45], dtype=np.uint8)

    # Road surface
    cv2.rectangle(frame, (W//2 - 140, 0), (W//2 + 140, H), (55, 55, 55), -1)

    # Lane lines
    cv2.line(frame, (W//2 - 130, H), (W//2 - 50, 120), (210, 210, 210), 2)
    cv2.line(frame, (W//2 + 130, H), (W//2 + 50, 120), (210, 210, 210), 2)
    # Dashed center line
    for seg in range(0, H, 50):
        cv2.line(frame, (W//2, seg), (W//2, min(seg+30, H)), (150, 150, 100), 1)

    # Moving car (blue box) — drives forward (y decreases)
    car_y = max(120, 380 - i * 2)
    cv2.rectangle(frame, (W//2 - 30, car_y), (W//2 + 30, car_y + 55), (200, 100, 30), -1)
    cv2.putText(frame, "CAR", (W//2 - 15, car_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    # Pedestrian crossing — walks rightward across road
    ped_x = min(W - 20, 80 + i * 2)
    cv2.circle(frame, (ped_x, 300), 14, (50, 200, 230), -1)
    cv2.putText(frame, "P", (ped_x - 5, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

    # Motorcycle on right shoulder
    moto_y = max(150, 400 - i * 2)
    cv2.rectangle(frame, (W//2 + 70, moto_y), (W//2 + 100, moto_y + 30), (100, 60, 200), -1)

    # Traffic light fixture (upper-right)
    cv2.rectangle(frame, (560, 40), (590, 130), (15, 15, 15), -1)
    cv2.rectangle(frame, (560, 40), (590, 130), (90, 90, 90), 1)
    # Red lamp lit
    cv2.circle(frame, (575, 60), 11, (0, 0, 240), -1)
    # Yellow lamp off
    cv2.circle(frame, (575, 85), 9, (10, 30, 30), -1)
    # Green lamp off
    cv2.circle(frame, (575, 110), 9, (10, 25, 10), -1)

    # Road sign (left shoulder): stop sign octagon
    cx, cy = 55, 90
    pts = np.array([
        [cx, cy - 22], [cx + 16, cy - 16], [cx + 22, cy], [cx + 16, cy + 16],
        [cx, cy + 22], [cx - 16, cy + 16], [cx - 22, cy], [cx - 16, cy - 16]
    ], np.int32)
    cv2.fillPoly(frame, [pts], (0, 0, 190))
    cv2.putText(frame, "STOP", (cx - 15, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    # Frame label
    cv2.putText(frame, f"E2E Test Frame {i+1}/{N_FRAMES}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    vwriter.write(frame)

vwriter.release()
print(f"[1] Synthetic test video created: {OUT_PATH} ({N_FRAMES} frames @ {FPS}fps)")

# ── 2. Upload video ───────────────────────────────────────────────────────────
with open(OUT_PATH, "rb") as f:
    r = requests.post(f"{BASE}/upload", files={"file": (OUT_PATH, f, "video/mp4")})

assert r.status_code == 201, f"Upload failed: {r.status_code} {r.text}"
video_id = r.json()["video_id"]
print(f"[2] Upload OK -> video_id = {video_id}")

# ── 3. Start processing in FULL pipeline mode ─────────────────────────────────
r = requests.post(f"{BASE}/process/{video_id}", params={"mode": "full"})
assert r.status_code == 200, f"Process start failed: {r.status_code} {r.text}"
print(f"[3] Full pipeline started (mode=full)")

# ── 4. Poll until done ────────────────────────────────────────────────────────
print("[4] Polling status...")
data = {}
for attempt in range(90):
    time.sleep(2)
    r = requests.get(f"{BASE}/status/{video_id}")
    data = r.json()
    status   = data["status"]
    progress = data["progress"]
    print(f"    [{attempt+1:2d}] status={status:<12s}  progress={progress:5.1f}%")
    if status in ("done", "error"):
        break

assert status == "done", f"Pipeline failed: {data.get('message')}"

# ── 5. Verify all phase stats ─────────────────────────────────────────────────
print("\n[5] Verifying full pipeline results:")
stats = data.get("stats")
assert stats is not None, "stats is None in API response"

print(f"  pipeline_mode            = {stats.get('pipeline_mode')}")
assert stats["pipeline_mode"] == "full", f"Expected mode=full, got {stats['pipeline_mode']}"

print(f"  processed_frames         = {stats['processed_frames']}/{stats['total_frames']}")
print(f"  processing_fps           = {stats['processing_fps']}")
print(f"  unique_track_ids         = {stats['unique_track_ids']}")

# Phase 5 — Collision
assert "risk_counts"           in stats, "Missing: risk_counts"
assert "critical_events_count" in stats, "Missing: critical_events_count"
assert "high_events_count"     in stats, "Missing: high_events_count"
assert "risk_events"           in stats, "Missing: risk_events"
print(f"  risk_counts              = {stats['risk_counts']}")
print(f"  critical_events          = {stats['critical_events_count']}")
print(f"  high_events              = {stats['high_events_count']}")
print(f"  risk_events logged       = {len(stats['risk_events'])}")

# Phase 6 — Decision
assert "dominant_action"  in stats, "Missing: dominant_action"
assert "action_counts"    in stats, "Missing: action_counts"
assert "decision_history" in stats, "Missing: decision_history"
print(f"  dominant_action          = {stats['dominant_action']}")
print(f"  decision_history         = {len(stats['decision_history'])} frames")

# Phase 7 — Path Planning
assert "final_path_status"   in stats, "Missing: final_path_status"
assert "replanning_count"    in stats, "Missing: replanning_count"
assert "avg_planning_time_ms" in stats, "Missing: avg_planning_time_ms"
assert "replanning_events"   in stats, "Missing: replanning_events"
print(f"  final_path_status        = {stats['final_path_status']}")
print(f"  replanning_count         = {stats['replanning_count']}")
print(f"  avg_planning_time_ms     = {stats['avg_planning_time_ms']} ms")

# Phase 8 — Environment
assert "active_traffic_signal"    in stats, "Missing: active_traffic_signal"
assert "traffic_signals_detected" in stats, "Missing: traffic_signals_detected"
assert "road_signs_detected"      in stats, "Missing: road_signs_detected"
assert "active_map_profile"       in stats, "Missing: active_map_profile"
assert stats["active_map_profile"] is not None, "active_map_profile is None — map provider not reached"
print(f"  active_traffic_signal    = {stats['active_traffic_signal']}")
print(f"  traffic_signals_detected = {len(stats['traffic_signals_detected'])}")
print(f"  road_signs_detected      = {len(stats['road_signs_detected'])}")
print(f"  active_map_profile       = {stats['active_map_profile']}")

# Tracked objects
tracked = stats.get("tracked_objects", [])
print(f"  tracked_objects          = {len(tracked)} unique")
if tracked:
    for obj in sorted(tracked, key=lambda x: -x["frame_count"])[:5]:
        print(f"    #{obj['track_id']:<3} {obj['class_name']:<12} conf={obj['avg_confidence']:.0%} "
              f"frames={obj['frame_count']}  max_risk={obj.get('max_risk_level','?')}")

# ── 6. Verify annotated output video ─────────────────────────────────────────
output_files = os.listdir("storage/outputs")
matching = [f for f in output_files if video_id in f]
assert matching, f"No output video for {video_id}"
out_size = os.path.getsize(f"storage/outputs/{matching[0]}")
assert out_size > 10000, f"Output video suspiciously small: {out_size} bytes"
print(f"\n[6] Annotated output video: storage/outputs/{matching[0]} ({out_size:,} bytes)")

# ── 7. Verify safety fallbacks ────────────────────────────────────────────────
print("\n[7] Safety fallback checks:")
valid_actions = {"CONTINUE", "SLOW_DOWN", "STOP", "AVOID", "REPLAN", "EMERGENCY_STOP"}
assert stats["dominant_action"] in valid_actions, f"Invalid action: {stats['dominant_action']}"
print(f"  dominant action '{stats['dominant_action']}' is in valid action set ✓")

valid_path_statuses = {
    "CONTINUE_ORIGINAL", "SAFE_PATH_FOUND", "REPLANNED_AVOID_LEFT",
    "REPLANNED_AVOID_RIGHT", "NO_SAFE_PATH", "STOPPED_NO_PATH"
}
assert stats["final_path_status"] in valid_path_statuses, f"Invalid path status: {stats['final_path_status']}"
print(f"  final path status '{stats['final_path_status']}' is valid ✓")

# risk events have correct schema
for ev in stats["risk_events"][:3]:
    assert "timestamp_sec"  in ev, "risk_event missing timestamp_sec"
    assert "frame_idx"      in ev, "risk_event missing frame_idx"
    assert "object_id"      in ev, "risk_event missing object_id"
    assert "object_class"   in ev, "risk_event missing object_class"
    assert "risk_level"     in ev, "risk_event missing risk_level"
    assert "reason"         in ev, "risk_event missing reason"
    assert ev["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}, f"Invalid risk_level: {ev['risk_level']}"
print(f"  {len(stats['risk_events'])} risk event(s) schema validated ✓")

print("\n" + "=" * 70)
print(" EAREYES 2K26 — FULL PIPELINE E2E INTEGRATION TEST PASSED!")
print(f" {N_FRAMES} frames | {stats['processing_fps']} fps | mode=full")
print("=" * 70)
