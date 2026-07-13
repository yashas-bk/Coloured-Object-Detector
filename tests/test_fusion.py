"""Tests for the fusion engine's matching, buckets, and confidence scoring.

These test the fusion logic directly with fabricated detections — no YOLO
inference, so they run fast and offline.
"""

import pytest

from detection.color_detector import Detection
from detection.fusion import MIN_CONFIDENCE, _fuse, _iou, _match_score

FRAME_AREA = 640 * 480


def det(x, y, w, h, area=None, **kw):
    return Detection(x, y, w, h, area if area is not None else float(w * h), **kw)


# ---- geometry ----

def test_iou_half_overlap():
    assert _iou(det(0, 0, 100, 100), det(50, 0, 100, 100)) == pytest.approx(1 / 3, abs=0.01)


def test_iou_disjoint():
    assert _iou(det(0, 0, 100, 100), det(200, 200, 50, 50)) == 0.0


def test_containment_qualifies_as_match():
    """Classical masks an object's BODY; YOLO boxes the WHOLE object.
    Low IoU but heavy containment must still match."""
    body = det(100, 100, 100, 50)     # colored body panel
    whole = det(80, 60, 300, 200)     # full object incl. windows etc.
    assert _iou(body, whole) < 0.4    # IoU alone would miss it
    assert _match_score(body, whole) > 0


# ---- buckets ----

def test_agreement_bucket():
    fused = _fuse(
        [det(100, 100, 100, 100, 9500)],
        [det(105, 102, 95, 98, label="cup", confidence=0.8)],
        FRAME_AREA,
    )
    assert len(fused) == 1
    d = fused[0]
    assert d.source == "agreement"
    assert d.label == "cup"
    assert d.confidence >= 0.9


def test_classical_only_kept_at_moderate_confidence():
    fused = _fuse([det(50, 50, 80, 80, 6000)], [], FRAME_AREA)
    assert len(fused) == 1
    assert fused[0].source == "classical-only"
    assert 0.4 < fused[0].confidence < 0.75


def test_ml_only_recovered_with_discount():
    fused = _fuse([], [det(10, 10, 60, 60, label="banana", confidence=0.7)], FRAME_AREA)
    assert len(fused) == 1
    assert fused[0].source == "ml-only"
    assert fused[0].confidence == pytest.approx(0.7 * 0.85, abs=0.01)


def test_blob_split_by_two_yolo_boxes():
    """One classical blob spanning two touching objects: YOLO splits it."""
    fused = _fuse(
        [det(100, 100, 220, 100, 20000)],
        [det(100, 100, 100, 100, label="cup", confidence=0.8),
         det(220, 100, 100, 100, label="cup", confidence=0.75)],
        FRAME_AREA,
    )
    assert sorted(d.source for d in fused) == ["agreement", "ml-only"]


# ---- false-positive suppression ----

def test_wall_sized_region_suppressed():
    huge = det(0, 0, 600, 440, 600 * 440 * 0.9)  # ~86% of the frame
    assert _fuse([huge], [], FRAME_AREA) == []


def test_stringy_region_suppressed():
    stringy = det(50, 50, 200, 200, 4000)  # solidity 0.1: lighting artifact
    assert _fuse([stringy], [], FRAME_AREA) == []


def test_weak_ml_only_below_floor():
    fused = _fuse([], [det(10, 10, 60, 60, label="cup", confidence=0.25)], FRAME_AREA)
    assert fused == []


def test_all_survivors_meet_confidence_floor():
    fused = _fuse(
        [det(50, 50, 80, 80, 6000), det(300, 300, 200, 100, 2500)],
        [det(400, 50, 90, 90, label="cup", confidence=0.5)],
        FRAME_AREA,
    )
    assert all(d.confidence >= MIN_CONFIDENCE for d in fused)
