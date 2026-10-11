"""Full end-to-end test: upload synthetic video → process → verify stats."""
import requests
import time

BASE = 'http://localhost:8000/api/v1/video'

# 1. Upload
print("1. Uploading test video...")
with open('test_road_video.mp4', 'rb') as f:
    r = requests.post(f'{BASE}/upload', files={'file': ('test_road_video.mp4', f, 'video/mp4')})

assert r.status_code == 201, f"Upload failed: {r.status_code} {r.text}"
video_id = r.json()['video_id']
print(f"   video_id = {video_id}")

# 2. Start processing
print("\n2. Starting processing...")
r = requests.post(f'{BASE}/process/{video_id}')
assert r.status_code == 200, f"Process start failed: {r.status_code} {r.text}"
print(f"   {r.json()['message']}")

# 3. Poll until done
print("\n3. Polling status...")
for attempt in range(100):
    time.sleep(3)
    r = requests.get(f'{BASE}/status/{video_id}')
    data = r.json()
    status = data['status']
    progress = data['progress']
    print(f"   [{attempt+1:3d}] {status:12s}  {progress:5.1f}%")
    if status == 'done':
        print("\n4. DONE! Statistics:")
        stats = data['stats']
        print(f"   Frames processed : {stats['processed_frames']} / {stats['total_frames']}")
        print(f"   Pipeline FPS     : {stats['processing_fps']}")
        print(f"   Unique tracks    : {stats['unique_track_ids']}")
        print(f"   By class         : {stats['objects_by_class']}")
        print(f"   Tracked objects  : {len(stats['tracked_objects'])}")
        if stats['tracked_objects']:
            print()
            print("   ID  | Class      | Confidence | Frames | Speed   | Direction")
            print("   " + "-" * 60)
            for obj in sorted(stats['tracked_objects'], key=lambda x: -x['frame_count'])[:10]:
                print(f"   #{obj['track_id']:<3} | {obj['class_name']:<10} | "
                      f"{obj['avg_confidence']:.0%}       | {obj['frame_count']:<6} | "
                      f"{obj['avg_pixel_speed']:<7.2f} | {obj['last_direction']}")
        break
    elif status == 'error':
        print(f"\n   ERROR: {data.get('message')}")
        break
