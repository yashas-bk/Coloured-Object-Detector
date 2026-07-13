"""Late fusion of the classical and ML detection engines.

Both engines run on the same frame; their outputs are matched by IoU and
every detection lands in one of three buckets:

- agreement:      both engines found it (matched pair). Strongest evidence —
                  two independent mechanisms corroborate. Uses the ML box
                  (tighter edges) and label, with boosted confidence.
- ml-only:        YOLO found a color-verified object the classical mask
                  missed (usually lighting pushed pixels out of the hue
                  band). Recovered recall the classical engine can't provide.
- classical-only: a colored region YOLO didn't recognize (non-COCO object).
                  YOLO's silence is absence of evidence, not a veto — kept at
                  moderate confidence, demoted when the region looks like a
                  background surface (huge or stringy) rather than an object.

Composite confidence lives in [0, 1]; detections below MIN_CONFIDENCE are
dropped. The weights are starting guesses meant to be tuned against a
labeled benchmark.
"""

import numpy as np

from .color_detector import Detection, detect_category
from .ml_detector import detect_category_ml

IOU_MATCH = 0.4  # boxes overlapping at least this much are the same object
CONTAINMENT_MATCH = 0.65  # or: the smaller box is mostly inside the larger one
MIN_CONFIDENCE = 0.25
LARGE_REGION_FRAC = 0.25  # classical regions above this frame fraction get demoted


def _iou(a: Detection, b: Detection) -> float:
    x1, y1 = max(a.x, b.x), max(a.y, b.y)
    x2, y2 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / (a.w * a.h + b.w * b.h - inter)


def _match_score(a: Detection, b: Detection) -> float:
    """Match quality in [0, 1]; 0 means 'not the same object'.

    IoU alone misses a common real case: the classical engine masks an
    object's colored BODY while YOLO boxes the WHOLE object (car body vs car
    incl. windows/wheels) — heavy containment, low IoU. Either signal
    qualifies a pair.
    """
    x1, y1 = max(a.x, b.x), max(a.y, b.y)
    x2, y2 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    iou = inter / (a.w * a.h + b.w * b.h - inter)
    containment = inter / min(a.w * a.h, b.w * b.h)
    if iou >= IOU_MATCH or containment >= CONTAINMENT_MATCH:
        return max(iou, containment)
    return 0.0


def _fuse(
    classical: list[Detection], ml: list[Detection], frame_area: int
) -> list[Detection]:
    # Greedy one-to-one matching, best overlap first.
    pairs = sorted(
        (
            (_match_score(c, m), ci, mi)
            for ci, c in enumerate(classical)
            for mi, m in enumerate(ml)
        ),
        reverse=True,
    )
    matched_c: set[int] = set()
    matched_m: set[int] = set()
    fused: list[Detection] = []

    for score, ci, mi in pairs:
        if score <= 0:
            break
        if ci in matched_c or mi in matched_m:
            continue
        matched_c.add(ci)
        matched_m.add(mi)
        m = ml[mi]
        fused.append(
            Detection(
                m.x, m.y, m.w, m.h, m.area,
                label=m.label,
                confidence=min(1.0, 0.9 + 0.1 * m.confidence),
                source="agreement",
            )
        )

    for mi, m in enumerate(ml):
        if mi in matched_m:
            continue
        fused.append(
            Detection(
                m.x, m.y, m.w, m.h, m.area,
                label=m.label,
                confidence=m.confidence * 0.85,
                source="ml-only",
            )
        )

    for ci, c in enumerate(classical):
        if ci in matched_c:
            continue
        box_area = max(c.w * c.h, 1)
        solidity = min(1.0, c.area / box_area)  # dense blob = object-like
        frac = box_area / max(frame_area, 1)
        size_factor = (
            1.0 if frac <= LARGE_REGION_FRAC
            else max(0.15, 1.0 - (frac - LARGE_REGION_FRAC) * 2.5)
        )
        fused.append(
            Detection(
                c.x, c.y, c.w, c.h, c.area,
                confidence=0.7 * solidity * size_factor,
                source="classical-only",
            )
        )

    fused = [d for d in fused if d.confidence >= MIN_CONFIDENCE]
    fused.sort(key=lambda d: d.confidence, reverse=True)
    return fused


def detect_category_fusion(
    frame_bgr: np.ndarray,
    category: str,
    min_area: int = 500,
) -> list[Detection]:
    """Detect objects of a color category using both engines fused."""
    classical, _ = detect_category(frame_bgr, category, min_area)
    ml = detect_category_ml(frame_bgr, category, min_area)
    h, w = frame_bgr.shape[:2]
    return _fuse(classical, ml, w * h)
