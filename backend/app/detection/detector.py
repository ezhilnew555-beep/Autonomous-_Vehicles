"""
YOLODetector — wraps Ultralytics YOLO for road-object detection.

Responsibilities:
  - Load and hold the model (instantiated once at app startup).
  - Run per-frame inference.
  - Return structured Detection objects filtered to road-relevant COCO classes.

Tracking and motion estimation are handled downstream in app/tracking/.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from app.detection.road_classes import ROAD_CLASS_IDS
from app.detection.model_manager import ModelManager
from app.detection.config import get_detection_config
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ─── Data structure ───────────────────────────────────────────────────────────

@dataclass
class Detection:
    """
    Single detection result for one object in one frame.

    Coordinates are absolute pixels in the original frame space.
    """
    class_id:   int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height


# ─── Detector ─────────────────────────────────────────────────────────────────

class YOLODetector:
    """
    Wraps an Ultralytics YOLO model for road-user detection.

    The model is loaded once and reused across all video frames.
    Auto-downloads the weights from Ultralytics CDN if the file is not found.

    Args:
        model_path: Path to .pt weights file (e.g. "models/yolo11n.pt").
        confidence: Detection confidence threshold (0–1). Default 0.40.
        imgsz: Inference image size in pixels. Default 640.
        device: PyTorch device string — "cpu", "0" (GPU 0), "cuda", "mps".
        filter_road_classes: When True, only COCO classes in ROAD_CLASS_IDS are
            returned. Set False to return all 80 COCO classes.
    """

    def __init__(
        self,
        model_path: str = "models/yolo11n.pt",
        confidence: float = 0.40,
        imgsz: int = 640,
        device: str = "cpu",
        filter_road_classes: bool = True,
    ) -> None:
        cfg = get_detection_config()
        self.confidence          = confidence if confidence is not None else cfg.CONFIDENCE_THRESHOLD
        self.imgsz               = imgsz if imgsz is not None else cfg.IMAGE_SIZE
        req_dev                  = device if device is not None else cfg.DEVICE
        self.device              = ModelManager.resolve_device(req_dev)
        self.filter_road_classes = filter_road_classes

        self.model = ModelManager.load_model(model_path or cfg.MODEL_NAME, self.device)
        self.class_names: dict[int, str] = self.model.names

        if filter_road_classes:
            self._allowed_ids = ROAD_CLASS_IDS & set(self.class_names.keys())
            logger.info(
                f"Model loaded — {len(self.class_names)} classes total, "
                f"filtering to {len(self._allowed_ids)} road-relevant classes."
            )
        else:
            self._allowed_ids = set(self.class_names.keys())
            logger.info(f"Model loaded — returning all {len(self.class_names)} classes.")

    # ─── Public API ───────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference on a single BGR frame (as returned by cv2.VideoCapture).

        Returns:
            List of Detection objects. May be empty if no road objects are found.
        """
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        if not results:
            return []

        result = results[0]     # Single-image inference always returns one result
        if result.boxes is None or len(result.boxes) == 0:
            return []

        detections: list[Detection] = []
        boxes = result.boxes

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())

            if self.filter_road_classes and cls_id not in self._allowed_ids:
                continue

            conf        = float(boxes.conf[i].item())
            x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i].tolist())
            class_name  = self.class_names.get(cls_id, f"class_{cls_id}")

            detections.append(Detection(
                class_id   = cls_id,
                class_name = class_name,
                confidence = conf,
                x1 = x1, y1 = y1, x2 = x2, y2 = y2,
            ))

        return detections

    @property
    def model_info(self) -> dict:
        """Return a summary dict for logging / API responses."""
        return {
            "model_path": str(self.model.model),
            "device":     self.device,
            "confidence": self.confidence,
            "imgsz":      self.imgsz,
            "num_classes": len(self.class_names),
        }
