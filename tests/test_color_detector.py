"""Tests for the classical detection engine: categories, classification,
and regression cases discovered during development."""

import numpy as np
import pytest

from detection import COLOR_CATEGORIES, classify_bgr, detect_category, detect_color, hex_to_bgr

# One representative BGR sample per detectable category
SAMPLES = {
    "red": (0, 0, 220),
    "orange": (0, 128, 255),
    "brown": (19, 69, 139),
    "yellow": (0, 230, 230),
    "green": (0, 200, 0),
    "cyan": (200, 200, 0),
    "blue": (230, 80, 0),
    "purple": (200, 0, 150),
    "pink": (150, 0, 230),
    "white": (250, 250, 250),
    "black": (15, 15, 15),
}


def scene(*rects, bg=120, size=(400, 400)):
    img = np.full((*size, 3), bg, dtype=np.uint8)
    for x, y, w, h, bgr in rects:
        img[y : y + h, x : x + w] = bgr
    return img


# ---- category matrix ----

@pytest.mark.parametrize("category,bgr", SAMPLES.items())
def test_detects_own_sample(category, bgr):
    img = scene((50, 50, 100, 100, bgr))
    detections, _ = detect_category(img, category)
    assert len(detections) == 1


@pytest.mark.parametrize("category,bgr", SAMPLES.items())
def test_classifies_own_sample(category, bgr):
    assert classify_bgr(bgr) == category


def test_yellow_ignores_red():
    img = scene((50, 50, 100, 100, SAMPLES["red"]))
    detections, _ = detect_category(img, "yellow")
    assert detections == []


@pytest.mark.parametrize("category", ["red", "yellow", "blue", "green"])
def test_plain_gray_scene_no_matches(category):
    detections, _ = detect_category(scene(), category)
    assert detections == []


def test_unknown_category_raises():
    with pytest.raises(ValueError, match="Unknown color category"):
        detect_category(scene(), "chartreuse")


def test_detect_color_hex_path():
    img = scene((150, 150, 100, 100, (0, 230, 230)))
    detections, _ = detect_color(img, hex_to_bgr("#FFFF00"))
    assert len(detections) == 1


# ---- regression: fragmented object must yield ONE box ----

def test_fragmented_object_single_box():
    """Specular highlights / shadows split the mask; morphology + box
    merging must reunite one physical object into one detection."""
    img = scene((100, 150, 200, 100, (0, 230, 230)))
    img[150:250, 180:192] = (245, 245, 245)  # specular stripe
    img[150:250, 250:262] = (60, 90, 95)     # shadow stripe
    detections, _ = detect_category(img, "yellow")
    assert len(detections) == 1
    d = detections[0]
    assert d.x <= 105 and d.x + d.w >= 295  # spans the full object


def test_object_split_by_horizontal_occluder_single_box():
    """A water line / shadow band across an object cuts its mask into two
    stacked blobs too far apart for plain proximity merging (the rubber-duck
    photo). Nested x-ranges + small vertical gap must reunite them."""
    img = scene((120, 80, 200, 120, (0, 230, 230)))   # head/top part
    img[200:240, 0:400] = (200, 120, 30)              # blue "water" band
    img[240:340, 100:340] = (0, 230, 230)             # body/bottom part
    detections, _ = detect_category(img, "yellow")
    assert len(detections) == 1
    d = detections[0]
    assert d.y <= 85 and d.y + d.h >= 335  # spans head through body


def test_side_by_side_objects_stay_separate():
    """Two distinct objects sitting next to each other must NOT be glued
    together by the occlusion-split rule (it only applies vertically)."""
    img = scene((60, 150, 120, 120, (0, 230, 230)),
                (240, 150, 120, 120, (0, 230, 230)))
    detections, _ = detect_category(img, "yellow")
    assert len(detections) == 2


def test_vertically_distant_objects_stay_separate():
    """Stacked boxes merge only across small gaps relative to their size;
    a shelf-like arrangement with a wide gap stays two objects."""
    img = scene((100, 40, 200, 100, (0, 230, 230)),
                (100, 260, 200, 100, (0, 230, 230)))  # gap 120 > 50% of 100
    detections, _ = detect_category(img, "yellow")
    assert len(detections) == 2


def test_noise_specks_not_merged_into_phantoms():
    img = scene()
    rng = np.random.default_rng(42)
    for _ in range(20):
        x, y = rng.integers(0, 396, 2)
        img[y : y + 3, x : x + 3] = (0, 230, 230)
    detections, _ = detect_category(img, "yellow")
    assert detections == []


# ---- exact-shape outlines ----

def test_detections_carry_contours():
    detections, _ = detect_category(scene((50, 50, 100, 100, (0, 230, 230))), "yellow")
    assert len(detections) == 1
    assert len(detections[0].contours) >= 1


def test_merged_detection_carries_all_fragment_contours():
    img = scene((120, 80, 200, 120, (0, 230, 230)))
    img[200:240, 0:400] = (200, 120, 30)
    img[240:340, 100:340] = (0, 230, 230)
    detections, _ = detect_category(img, "yellow")
    assert len(detections) == 1
    assert len(detections[0].contours) == 2  # head + body fragments


def test_contours_not_in_api_dict():
    detections, _ = detect_category(scene((50, 50, 100, 100, (0, 230, 230))), "yellow")
    assert "contours" not in detections[0].to_dict()


# ---- regression: muted colors are colors, not achromatic ----

@pytest.mark.parametrize("bgr", [
    (140, 110, 90),   # dull blue
    (150, 170, 200),  # tan / skin
    (60, 100, 110),   # olive
    (150, 140, 180),  # dusty pink
    (40, 130, 140),   # dim yellow
])
def test_muted_colors_are_chromatic(bgr):
    assert classify_bgr(bgr) not in ("white", "black")


# ---- regression: dark colors are not black ----

@pytest.mark.parametrize("bgr,expected", [
    ((31, 0, 103), "red"),    # #67001F dark red
    ((16, 8, 74), "red"),     # dark maroon
    ((74, 31, 0), "blue"),    # navy
    ((11, 61, 11), "green"),  # dark forest green
])
def test_dark_colors_keep_their_hue(bgr, expected):
    assert classify_bgr(bgr) == expected


@pytest.mark.parametrize("bgr", [(30, 28, 28), (54, 51, 51)])
def test_true_blacks_are_black(bgr):
    assert classify_bgr(bgr) == "black"


def test_black_ignores_dark_red_region():
    img = scene((50, 50, 100, 100, (31, 0, 103)), (250, 250, 100, 100, (25, 25, 25)))
    detections, _ = detect_category(img, "black")
    assert len(detections) == 1 and detections[0].x > 200
    detections, _ = detect_category(img, "red")
    assert len(detections) == 1 and detections[0].x < 200


# ---- gray removed; achromatic snapping ----

def test_gray_not_a_category():
    assert "gray" not in COLOR_CATEGORIES
    with pytest.raises(ValueError):
        detect_category(scene(), "gray")


def test_gray_pixels_snap_to_nearest_achromatic():
    assert classify_bgr((128, 128, 128)) == "white"
    assert classify_bgr((70, 70, 70)) == "black"


# ---- CSS named-color calibration (the color-boundary contract) ----

CSS_EXPECTATIONS = {
    "#FF0000": "red", "#DC143C": "red", "#B22222": "red", "#8B0000": "red", "#FF6347": "red",
    "#FFC0CB": "pink", "#FF69B4": "pink", "#FFB6C1": "pink", "#FF1493": "pink",
    "#FFA500": "orange", "#FF8C00": "orange", "#FF7F50": "orange",
    "#8B4513": "brown", "#A0522D": "brown", "#D2B48C": "brown",
    "#FFFF00": "yellow", "#FFD700": "yellow", "#F0E68C": "yellow",
    "#008000": "green", "#00FF00": "green", "#228B22": "green", "#2E8B57": "green",
    "#00FFFF": "cyan", "#40E0D0": "cyan", "#008080": "cyan",
    "#0000FF": "blue", "#000080": "blue", "#4169E1": "blue", "#87CEEB": "blue",
    "#800080": "purple", "#EE82EE": "purple", "#8A2BE2": "purple", "#4B0082": "purple",
    "#FFFFFF": "white", "#000000": "black",
}


@pytest.mark.parametrize("hex_color,expected", CSS_EXPECTATIONS.items())
def test_css_named_colors(hex_color, expected):
    assert classify_bgr(hex_to_bgr(hex_color)) == expected
