"""
FrameReader — OpenCV-based frame-by-frame video reader.

Wraps cv2.VideoCapture as a context manager and iterator.
Never loads the entire video into memory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from app.utils.logger import get_logger

logger = get_logger(__name__)


class FrameReader:
    """
    Opens a video file and yields BGR frames one at a time.

    Usage::

        with FrameReader("path/to/video.mp4") as reader:
            for frame_idx, frame in reader:
                process(frame)

    Attributes:
        fps (float):         Frames per second reported by the container.
        frame_count (int):   Total frame count (may be approximate for some formats).
        width (int):         Frame width in pixels.
        height (int):        Frame height in pixels.
        duration (float):    Estimated duration in seconds.
    """

    def __init__(self, video_path: str | Path) -> None:
        self.path = Path(video_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")

        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise IOError(f"OpenCV could not open video: {self.path}")

        self.fps:         float = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_count: int   = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width:       int   = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height:      int   = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration:    float = (self.frame_count / self.fps) if self.fps > 0 else 0.0

        logger.info(
            f"Opened '{self.path.name}': "
            f"{self.width}×{self.height} @ {self.fps:.1f} fps, "
            f"~{self.frame_count} frames ({self.duration:.1f}s)"
        )

    # ─── Iterator ─────────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (frame_index, bgr_frame) pairs until the video ends."""
        frame_idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1

    # ─── Context manager ──────────────────────────────────────────────────────

    def close(self) -> None:
        if self._cap.isOpened():
            self._cap.release()

    def __enter__(self) -> "FrameReader":
        return self

    def __exit__(self, *_) -> None:
        self.close()
