"""ML-based detection: YOLO object proposals + dominant-color matching.

YOLO (pretrained on COCO) finds objects; each detection's crop is reduced to
its dominant colors via k-means, and those are snapped to color categories
with the same LAB-based classifier the classical pipeline uses. A detection
is kept when a sufficiently large color cluster matches the requested
category.

The model is loaded lazily on first use (import + weights download can take
seconds) and guarded by a lock for the WebSocket path.
"""

import threading

import cv2
import numpy as np

from .color_detector import COLOR_CATEGORIES, Detection, classify_bgr

MODEL_NAME = "yolov8n.pt"  # nano: ~6 MB, CPU-friendly
CONF_THRESHOLD = 0.3
CROP_SHRINK = 0.15  # trim box edges before color analysis (background pollution)
CLUSTER_MIN_WEIGHT = 0.25  # a color must own this fraction of the crop to count

_model = None
_model_lock = threading.Lock()
_predict_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from ultralytics import YOLO

                _model = YOLO(MODEL_NAME)
    return _model


def warm_up() -> None:
    """Load the model and run one dummy inference so the first user request is fast."""
    model = _get_model()
    model.predict(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)


def _dominant_colors(crop_bgr: np.ndarray, k: int = 3, sample: int = 2000):
    """Top color clusters of a crop as [(bgr, weight), ...], heaviest first."""
    pixels = crop_bgr.reshape(-1, 3).astype(np.float32)
    if len(pixels) > sample:
        idx = np.random.default_rng(0).choice(len(pixels), sample, replace=False)
        pixels = pixels[idx]
    k = min(k, len(pixels))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    counts = np.bincount(labels.flatten(), minlength=k)
    weights = counts / counts.sum()
    order = np.argsort(-weights)
    return [
        (tuple(int(c) for c in centers[i]), float(weights[i])) for i in order
    ]


def detect_category_ml(
    frame_bgr: np.ndarray,
    category: str,
    min_area: int = 500,
    conf: float = CONF_THRESHOLD,
) -> list[Detection]:
    """Detect objects of the requested color category using YOLO + color match."""
    category = category.lower()
    if category not in COLOR_CATEGORIES:
        raise ValueError(
            f"Unknown color category: {category!r} (choose from {', '.join(COLOR_CATEGORIES)})"
        )

    model = _get_model()
    with _predict_lock:  # ultralytics predict is not thread-safe
        result = model.predict(frame_bgr, conf=conf, verbose=False)[0]

    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        if w * h < min_area:
            continue

        dx, dy = int(w * CROP_SHRINK), int(h * CROP_SHRINK)
        crop = frame_bgr[y1 + dy : y2 - dy, x1 + dx : x2 - dx]
        if crop.size == 0:
            crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        matched = any(
            weight >= CLUSTER_MIN_WEIGHT and classify_bgr(bgr) == category
            for bgr, weight in _dominant_colors(crop)
        )
        if not matched:
            continue

        detections.append(
            Detection(
                x1, y1, w, h,
                area=float(w * h),
                label=model.names[int(box.cls[0])],
                confidence=float(box.conf[0]),
            )
        )

    detections.sort(key=lambda d: d.area, reverse=True)
    return detections
