"""Color-agnostic object detection using classical CV.

Pipeline: blur -> HSV + LAB color spaces -> category mask -> morphology
(open to remove specks, close to heal highlight/shadow gaps) -> contours
-> nearby-box merging.

Design decisions that matter:
- Chromatic categories are HSV hue bands (red gets two bands because its hue
  wraps the 0/179 boundary), additionally gated on LAB chroma so washed-out
  pixels don't sneak in.
- Achromatic categories (white/gray/black) are defined by LAB lightness and
  chroma, NOT low HSV saturation. In HSV every dim or washed-out pixel looks
  "gray"; in LAB only genuinely colorless pixels have near-zero chroma.
- One physical object often fragments into several mask blobs (specular
  highlights, shadows). Morphological closing plus a bounding-box merge pass
  reunite them before results are reported.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

DEFAULT_MIN_AREA = 500

BOX_COLOR = (0, 255, 0)  # green, BGR

# LAB thresholds (OpenCV 8-bit LAB: L in 0-255, a/b centered at 128).
BLACK_MAX_L = 60       # black must be at most this light...
BLACK_MAX_CHROMA = 20  # ...and colorless: dark red/navy are colors, not black
                       # (real blacks measure chroma <= ~4; dark colors >= ~31)
WHITE_MIN_L = 195     # white needs to be at least this light...
WHITE_MAX_CHROMA = 20  # ...and close to colorless
GRAY_MAX_CHROMA = 14  # gray must be genuinely colorless
COLOR_MIN_CHROMA = 18  # chromatic matches must carry at least this much color

# HSV floors for chromatic hue bands.
SAT_FLOOR = 60
VAL_FLOOR = 50

# Named color categories, in UI display order. Simple categories are plain
# hue bands; red/orange/brown/pink/purple need composite rules (hue +
# saturation + lightness) implemented in _chromatic_masks, because those
# color NAMES don't map to hues alone:
#   - pink is a light, softened red (a tint), not just magenta hues
#   - brown is dark OR muted warm (orange-ish) color
# Calibrated against CSS named colors (see benchmark of classify_bgr).
BROWN_MAX_V = 185   # warm colors darker than this are brown, not orange
MUTED_MAX_S = 150   # warm colors more muted than this read as brown/tan...
MUTED_MAX_V = 235   # ...unless they are nearly white-bright
PINK_MIN_L = 160    # light...
PINK_MAX_S = 130    # ...and softened reds are pink

COLOR_CATEGORIES: dict[str, dict] = {
    "red":    {"swatch": "#E53935"},
    "orange": {"swatch": "#FB8C00"},
    "yellow": {"swatch": "#FDD835"},
    "brown":  {"swatch": "#795548"},
    "green":  {"swatch": "#43A047"},
    "cyan":   {"swatch": "#00ACC1"},
    "blue":   {"swatch": "#1E88E5"},
    "purple": {"swatch": "#8E24AA"},
    "pink":   {"swatch": "#EC407A"},
    "white":  {"swatch": "#FFFFFF"},
    "black":  {"swatch": "#212121"},
}


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    area: float
    label: str = ""  # semantic class name (ML detections only)
    confidence: float = 0.0  # model confidence (ML detections only)
    source: str = ""  # fusion bucket: "agreement" | "classical-only" | "ml-only"
    # Exact mask outlines (cv2 contours) this detection was built from;
    # classical detections only. A merged detection carries one contour per
    # fragment. Display-only: excluded from to_dict / the JSON API.
    contours: list = field(default_factory=list, repr=False, compare=False)

    def to_dict(self):
        return {
            "x": self.x, "y": self.y, "w": self.w, "h": self.h, "area": self.area,
            "label": self.label, "confidence": round(self.confidence, 3),
            "source": self.source,
        }


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """'#RRGGBB' -> (B, G, R)"""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Invalid hex color: {hex_color!r} (expected #RRGGBB)")
    try:
        r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        raise ValueError(f"Invalid hex color: {hex_color!r} (expected #RRGGBB)") from None
    return (b, g, r)


def _spaces(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blur then convert once: returns (hsv, L, chroma) arrays."""
    if frame_bgr.shape[0] >= 3 and frame_bgr.shape[1] >= 3:
        frame_bgr = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.int32) - 128
    b = lab[:, :, 2].astype(np.int32) - 128
    chroma = np.sqrt(a * a + b * b)
    return hsv, lab[:, :, 0], chroma


def _chromatic_keep(
    hsv: np.ndarray, L: np.ndarray, chroma: np.ndarray, category: str,
    sat_floor: int = SAT_FLOOR, val_floor: int = VAL_FLOOR,
    chroma_floor: float = COLOR_MIN_CHROMA,
) -> np.ndarray:
    """Boolean array: pixels belonging to a chromatic category.

    All chromatic categories are mutually exclusive by construction.
    """
    H = hsv[:, :, 0].astype(np.int32)
    S = hsv[:, :, 1].astype(np.int32)
    V = hsv[:, :, 2].astype(np.int32)
    base = (S >= sat_floor) & (V >= val_floor) & (chroma >= chroma_floor)

    red_hue = (H <= 6) | (H >= 170)
    warm_hue = (7 <= H) & (H <= 22)
    # Light softened reds are pink, not red (classic pink #FFC0CB etc.)
    pink_tint = red_hue & (L >= PINK_MIN_L) & (S <= PINK_MAX_S)
    # Muted warm colors are brown/tan even when bright (tan, peru, wood)
    muted_warm = (S <= MUTED_MAX_S) & (V <= MUTED_MAX_V)

    if category == "red":
        return base & red_hue & ~pink_tint
    if category == "orange":
        return base & warm_hue & (V > BROWN_MAX_V) & ~muted_warm
    if category == "brown":
        return base & warm_hue & ((V <= BROWN_MAX_V) | muted_warm)
    if category == "yellow":
        return base & (23 <= H) & (H <= 35)
    if category == "green":
        return base & (36 <= H) & (H <= 80)
    if category == "cyan":
        return base & (81 <= H) & (H <= 95)
    if category == "blue":
        return base & (96 <= H) & (H <= 128)
    if category == "purple":
        # magenta hues up to 155 read as purple (CSS purple/violet ~ hue 150)
        return base & (129 <= H) & (H <= 155)
    if category == "pink":
        return base & (((156 <= H) & (H <= 169)) | pink_tint)
    raise ValueError(f"Not a chromatic category: {category!r}")


def _category_mask(
    hsv: np.ndarray, L: np.ndarray, chroma: np.ndarray, category: str
) -> np.ndarray:
    category = category.lower()
    if category not in COLOR_CATEGORIES:
        raise ValueError(
            f"Unknown color category: {category!r} (choose from {', '.join(COLOR_CATEGORIES)})"
        )

    if category == "black":
        keep = (L <= BLACK_MAX_L) & (chroma <= BLACK_MAX_CHROMA)
    elif category == "white":
        keep = (L >= WHITE_MIN_L) & (chroma <= WHITE_MAX_CHROMA)
    else:
        keep = _chromatic_keep(hsv, L, chroma, category)
    return keep.astype(np.uint8) * 255


# One object cut by a horizontal occluder (water line, shadow band, strap)
# leaves vertically stacked fragments whose x-ranges nest. Those merge across
# a much larger gap than plain proximity allows. Deliberately asymmetric: two
# DISTINCT objects usually sit side by side (gravity), almost never floating
# one above the other, so the same generosity horizontally would glue
# neighbours together.
SPLIT_X_OVERLAP_FRAC = 0.8  # x-overlap required, fraction of the narrower box
SPLIT_MAX_GAP_FRAC = 0.5    # vertical gap allowed, fraction of the shorter box


def _vertical_split(a: Detection, b: Detection) -> bool:
    """Do these boxes look like one object cut by a horizontal occluder?"""
    x_overlap = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
    if x_overlap < SPLIT_X_OVERLAP_FRAC * min(a.w, b.w):
        return False
    v_gap = max(a.y, b.y) - min(a.y + a.h, b.y + b.h)
    return v_gap <= SPLIT_MAX_GAP_FRAC * min(a.h, b.h)


def _merge_nearby(boxes: list[Detection], gap: int) -> list[Detection]:
    """Union bounding boxes that overlap or sit within `gap` px of each other,
    plus vertically split fragments of one occluded object (_vertical_split).

    One object frequently yields several mask blobs; this reunites them.
    """
    boxes = list(boxes)
    changed = True
    while changed:
        changed = False
        merged: list[Detection] = []
        for d in boxes:
            hit = None
            for m in merged:
                if (
                    d.x - gap < m.x + m.w
                    and m.x - gap < d.x + d.w
                    and d.y - gap < m.y + m.h
                    and m.y - gap < d.y + d.h
                ) or _vertical_split(d, m):
                    hit = m
                    break
            if hit is None:
                merged.append(Detection(d.x, d.y, d.w, d.h, d.area,
                                        contours=list(d.contours)))
            else:
                # Fragments separated by an occluder: the object plausibly
                # continues behind it, so credit the hidden band to the mask
                # area — otherwise the merged box's solidity (used by fusion
                # scoring) punishes the occlusion itself.
                x_overlap = min(hit.x + hit.w, d.x + d.w) - max(hit.x, d.x)
                v_gap = max(hit.y, d.y) - min(hit.y + hit.h, d.y + d.h)
                if x_overlap > 0 and v_gap > 0:
                    hit.area += x_overlap * v_gap
                x1, y1 = min(hit.x, d.x), min(hit.y, d.y)
                x2 = max(hit.x + hit.w, d.x + d.w)
                y2 = max(hit.y + hit.h, d.y + d.h)
                hit.x, hit.y, hit.w, hit.h = x1, y1, x2 - x1, y2 - y1
                hit.area += d.area
                hit.contours.extend(d.contours)
                changed = True
        boxes = merged
    return boxes


def _mask_to_detections(mask: np.ndarray, min_area: int) -> tuple[list[Detection], np.ndarray]:
    h, w = mask.shape[:2]

    # Open kills speck noise; close heals highlight/shadow gaps inside objects.
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_size = max(9, (int(min(h, w) * 0.035) // 2) * 2 + 1)
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fragments = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 25:  # sub-speck; not even worth merging
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        fragments.append(Detection(x, y, bw, bh, float(area), contours=[contour]))

    # Merge first, THEN apply min_area: fragments of one large object may each
    # be individually small.
    gap = max(8, int(min(h, w) * 0.02))
    detections = [d for d in _merge_nearby(fragments, gap) if d.area >= min_area]
    detections.sort(key=lambda d: d.area, reverse=True)
    return detections, mask


def detect_category(
    frame_bgr: np.ndarray,
    category: str,
    min_area: int = DEFAULT_MIN_AREA,
) -> tuple[list[Detection], np.ndarray]:
    """Detect regions belonging to a named color category."""
    hsv, L, chroma = _spaces(frame_bgr)
    mask = _category_mask(hsv, L, chroma, category)
    return _mask_to_detections(mask, min_area)


CHROMATIC = [c for c in COLOR_CATEGORIES if c not in ("white", "black")]


def classify_bgr(bgr: tuple[int, int, int]) -> str:
    """Snap a single BGR color to its nearest category name.

    Runs the exact same mask logic as detection on a 1x1 image, so the
    eyedropper can never disagree with what detection would match.
    """
    pixel = np.uint8([[list(bgr)]])
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)[0, 0]
    L_arr = np.array([[int(lab[0])]], dtype=np.int32)
    chroma = ((int(lab[1]) - 128) ** 2 + (int(lab[2]) - 128) ** 2) ** 0.5
    chroma_arr = np.array([[chroma]])
    L = int(lab[0])

    if L <= BLACK_MAX_L and chroma <= BLACK_MAX_CHROMA:
        return "black"
    if L >= WHITE_MIN_L and chroma <= WHITE_MAX_CHROMA:
        return "white"
    if chroma <= GRAY_MAX_CHROMA:
        # Gray isn't a category; snap to the nearer achromatic one.
        return "white" if L >= 128 else "black"

    for name in CHROMATIC:
        if _chromatic_keep(hsv, L_arr, chroma_arr, name)[0][0]:
            return name
    # Below the saturation/value floors (e.g. dusty pastels): the eyedropper
    # must still answer — retry without floors, then achromatic as last resort.
    for name in CHROMATIC:
        if _chromatic_keep(hsv, L_arr, chroma_arr, name,
                           sat_floor=0, val_floor=0, chroma_floor=0)[0][0]:
            return name
    return "white" if L >= 128 else "black"


def detect_color(
    frame_bgr: np.ndarray,
    target_bgr: tuple[int, int, int],
    min_area: int = DEFAULT_MIN_AREA,
) -> tuple[list[Detection], np.ndarray]:
    """Detect regions matching an exact color by snapping it to its category."""
    return detect_category(frame_bgr, classify_bgr(target_bgr), min_area)


def annotate(
    frame_bgr: np.ndarray, detections: list[Detection], label: str = "Object"
) -> np.ndarray:
    """Draw detection outlines and labels. Returns a copy.

    Classical detections carry their exact mask contours and are outlined
    with them (boxes cover neighbours when objects overlap); ML detections
    only have boxes. Labels: ML detections carry their own class label +
    confidence; classical ones fall back to the caller-supplied label.
    """
    out = frame_bgr.copy()
    for det in detections:
        if det.label:
            text = f"{label} {det.label} {det.confidence:.0%}"
        elif det.confidence > 0:
            text = f"{label} object {det.confidence:.0%}"
        else:
            text = label
        if det.contours:
            cv2.drawContours(out, det.contours, -1, BOX_COLOR, 2)
        else:
            cv2.rectangle(out, (det.x, det.y), (det.x + det.w, det.y + det.h), BOX_COLOR, 2)
        cv2.putText(
            out,
            text,
            (det.x, max(det.y - 10, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            BOX_COLOR,
            2,
        )
    return out
