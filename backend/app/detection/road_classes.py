"""
Road-user class definitions for COCO-80 pretrained YOLO models.

COCO class IDs are fixed for yolo11*.pt weights.
Auto-rickshaw is NOT in COCO-80 — a placeholder comment marks the injection
point for future custom-model classes (Phase 4+).
"""

# ─── COCO-80 indices relevant to Indian road scenarios ────────────────────────
ROAD_CLASS_IDS: frozenset[int] = frozenset({
    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    5,   # bus
    7,   # truck
    14,  # bird       (animal on road)
    15,  # cat        (animal on road)
    16,  # dog        (animal on road)
    17,  # horse      (animal on road)
    18,  # sheep      (animal on road)
    19,  # cow        (very common in Indian roads)
    # Future custom classes (non-COCO) — add IDs here when using a custom model:
    # 80: "auto_rickshaw"
    # 81: "tractor"
    # 82: "e_rickshaw"
})

# ─── Human-readable display names ────────────────────────────────────────────
CLASS_DISPLAY_NAMES: dict[str, str] = {
    "person":     "Person",
    "bicycle":    "Bicycle",
    "car":        "Car",
    "motorcycle": "Motorcycle",
    "bus":        "Bus",
    "truck":      "Truck",
    "bird":       "Bird",
    "cat":        "Cat",
    "dog":        "Dog",
    "horse":      "Horse",
    "sheep":      "Sheep",
    "cow":        "Cow",
}

# ─── Per-class BGR annotation colours (OpenCV convention) ─────────────────────
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "person":     (0,   220,   0),    # vivid green
    "bicycle":    (255, 128,   0),    # orange
    "car":        (50,  150, 255),    # blue
    "motorcycle": (255,   0, 200),    # magenta
    "bus":        (0,   230, 230),    # cyan
    "truck":      (0,   200, 255),    # light blue
    "cow":        (60,  130, 200),    # steel blue
    "horse":      (20,   80, 180),    # dark blue
    "dog":        (80,  200,  80),    # light green
    "sheep":      (120, 200, 120),    # pale green
    "bird":       (200, 180,  60),    # gold
    "cat":        (180, 100, 200),    # purple
}
_DEFAULT_COLOR: tuple[int, int, int] = (180, 180, 180)  # grey


def get_color(class_name: str) -> tuple[int, int, int]:
    return CLASS_COLORS.get(class_name.lower(), _DEFAULT_COLOR)


def is_road_class(class_id: int) -> bool:
    return class_id in ROAD_CLASS_IDS


def get_display_name(class_name: str) -> str:
    return CLASS_DISPLAY_NAMES.get(class_name.lower(), class_name.title())
