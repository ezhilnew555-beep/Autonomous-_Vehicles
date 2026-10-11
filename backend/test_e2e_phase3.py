"""
End-to-End API Integration test for Phase 3: Object Tracking + Movement Estimation.
"""
import time
import cv2
import numpy as np
import requests

BASE_URL = "http://127.0.0.1:8000"


def generate_test_video(filename="test_p3_input.mp4", num_frames=20):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, 10, (320, 240))
    for i in range(num_frames):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # Moving box simulating a moving road vehicle
        cv2.rectangle(frame, (40 + i * 4, 80), (100 + i * 4, 140), (0, 255, 0), -1)
        out.write(frame)
    out.release()
    return filename


def test_phase3_api():
    video_file = generate_test_video()

    # 1. Upload video
    with open(video_file, "rb") as f:
        up_res = requests.post(f"{BASE_URL}/api/v1/video/upload", files={"file": (video_file, f, "video/mp4")})
    assert up_res.status_code == 201, f"Upload failed: {up_res.text}"
    vid_id = up_res.json()["video_id"]
    print(f"Uploaded video: {vid_id}")

    # 2. Trigger Phase 3 tracking mode
    proc_res = requests.post(f"{BASE_URL}/api/v1/video/process/{vid_id}?mode=tracking")
    assert proc_res.status_code == 200, f"Process start failed: {proc_res.text}"
    print("Started processing with mode=tracking")

    # 3. Poll for completion
    for i in range(30):
        time.sleep(1)
        st = requests.get(f"{BASE_URL}/api/v1/video/status/{vid_id}").json()
        print(f"Poll {i}: status={st.get('status')}, progress={st.get('progress')}%")
        if st.get("status") in ("done", "error"):
            assert st.get("status") == "done", f"Job errored: {st.get('message')}"
            stats = st.get("stats", {})
            print("Job finished! Stats:", stats)
            assert stats.get("pipeline_mode") == "tracking"
            assert "active_tracks" in stats
            assert "total_tracks_count" in stats
            assert "tracking_duration_sec" in stats
            assert stats.get("processing_fps", 0) > 0
            print("Phase 3 API verification PASSED!")
            break
    else:
        raise TimeoutError("Processing timed out after 30 seconds")

    # 4. Verify result video is downloadable
    res_url = f"{BASE_URL}/api/v1/video/result/{vid_id}"
    res_get = requests.get(res_url, stream=True)
    assert res_get.status_code in (200, 206), f"Result video check failed: {res_get.status_code}"
    print("Result video verified accessible!")


if __name__ == "__main__":
    test_phase3_api()
    print("ALL PHASE 3 E2E TESTS PASSED!")
