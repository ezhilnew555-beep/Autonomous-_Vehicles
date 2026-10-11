import cv2
import numpy as np
import requests
import time

def test_phase2_e2e():
    # 1. Create a dummy test video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_input.mp4', fourcc, 10, (320, 240))
    for i in range(15):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(frame, (50 + i * 5, 80), (120 + i * 5, 150), (0, 255, 0), -1)
        out.write(frame)
    out.release()

    # 2. Upload video
    with open('test_input.mp4', 'rb') as f:
        up_res = requests.post(
            'http://127.0.0.1:8000/api/v1/video/upload',
            files={'file': ('test_input.mp4', f, 'video/mp4')},
        )
    print('Upload status:', up_res.status_code)
    assert up_res.status_code == 201
    vid_id = up_res.json()['video_id']
    print('Video ID:', vid_id)

    # 3. Start processing with mode=detection
    proc_res = requests.post(
        f'http://127.0.0.1:8000/api/v1/video/process/{vid_id}?mode=detection'
    )
    print('Process status:', proc_res.status_code, proc_res.json())
    assert proc_res.status_code == 200

    # 4. Poll status
    for i in range(30):
        time.sleep(1)
        st = requests.get(f'http://127.0.0.1:8000/api/v1/video/status/{vid_id}').json()
        print(f"Poll {i}: status={st.get('status')}, progress={st.get('progress')}%")
        if st.get('status') in ('done', 'error'):
            stats = st.get('stats', {})
            print('DONE!')
            print('Pipeline Mode:', stats.get('pipeline_mode'))
            print('Total objects detected:', stats.get('total_objects_detected'))
            print('Objects by class:', stats.get('objects_by_class'))
            print('Processing time:', stats.get('processing_time_sec'), 'sec')
            print('Processing FPS:', stats.get('processing_fps'))
            assert st.get('status') == 'done'
            assert stats.get('pipeline_mode') == 'detection'
            assert 'car' in stats.get('objects_by_class', {}) or stats.get('total_objects_detected') is not None
            print('ALL ASSERTIONS PASSED!')
            return

    raise TimeoutError('Processing did not finish in 30 seconds.')

if __name__ == '__main__':
    test_phase2_e2e()
