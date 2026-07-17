"""Zero-shot naming of unrecognized detections with CLIP.

The fusion engine's classical-only bucket contains coloured regions YOLO
couldn't name (non-COCO objects). CLIP closes that gap without training:
each crop is embedded and compared against a bank of text embeddings ("a
photo of a rubber duck", ...); the best match above a similarity threshold
becomes the detection's label.

Design notes:
- CLIP cannot generate names, only rank candidates — hence the vocabulary
  (see vocabulary.py) plus optional per-request user labels.
- Text embeddings for the default vocabulary are computed once and cached
  in memory; custom labels are embedded on the fly (a few ms each).
- The similarity threshold keeps nearest-neighbour ranking honest: below it
  the detection stays an unnamed "object" instead of receiving the least-bad
  guess.
- Model loads lazily (~350 MB download on first ever use) and inference is
  serialised behind a lock, mirroring ml_detector.
"""

import threading

import cv2
import numpy as np

from .color_detector import Detection
from .vocabulary import DEFAULT_VOCABULARY

# -quickgelu variant matches the activation the OpenAI weights were trained
# with; plain ViT-B-32 loads but open_clip warns about the mismatch.
MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED = "openai"
PROMPT = "a photo of a {}"
MIN_SIMILARITY = 0.22  # cosine; below this the crop stays unnamed
# Image-to-image similarity (few-shot memory) runs far higher than
# image-to-text, hence the separate, stricter threshold.
MEMORY_MIN_SIMILARITY = 0.60
CROP_PAD = 0.08  # expand the box slightly: CLIP likes a little context

# RLock: _vocab_embeddings holds the lock while _embed_texts -> _load
# re-acquires it on the same thread; a plain Lock deadlocks there.
_lock = threading.RLock()
_model = None
_preprocess = None
_tokenizer = None
_vocab_bank: np.ndarray | None = None  # (N, D) normalized text embeddings


def _load():
    global _model, _preprocess, _tokenizer
    if _model is None:
        with _lock:
            if _model is None:
                import open_clip

                model, _, preprocess = open_clip.create_model_and_transforms(
                    MODEL_NAME, pretrained=PRETRAINED
                )
                model.eval()
                _tokenizer = open_clip.get_tokenizer(MODEL_NAME)
                _preprocess = preprocess
                _model = model
    return _model, _preprocess, _tokenizer


def _embed_texts(labels: list[str]) -> np.ndarray:
    import torch

    model, _, tokenizer = _load()
    tokens = tokenizer([PROMPT.format(lbl) for lbl in labels])
    with torch.no_grad():
        emb = model.encode_text(tokens)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.numpy().astype(np.float32)


def _vocab_embeddings() -> np.ndarray:
    global _vocab_bank
    if _vocab_bank is None:
        with _lock:
            if _vocab_bank is None:
                _vocab_bank = _embed_texts(DEFAULT_VOCABULARY)
    return _vocab_bank


def warm_up() -> None:
    """Load the model and precompute the vocabulary bank."""
    _vocab_embeddings()


def _embed_crop(crop_bgr: np.ndarray) -> np.ndarray:
    import torch
    from PIL import Image

    model, preprocess, _ = _load()
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = preprocess(Image.fromarray(rgb)).unsqueeze(0)
    with torch.no_grad():
        emb = model.encode_image(tensor)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.numpy().astype(np.float32)[0]


def embed_bgr(image_bgr: np.ndarray) -> np.ndarray:
    """Public, lock-serialised CLIP embedding of a BGR image (normalized)."""
    with _lock:
        return _embed_crop(image_bgr)


def name_crop(
    crop_bgr: np.ndarray, extra_labels: list[str] | None = None
) -> tuple[str, float]:
    """Best name for a crop: (label, cosine similarity).

    Taught objects (few-shot memory) are checked first and win over the
    text vocabulary when they clear MEMORY_MIN_SIMILARITY; otherwise the
    best vocabulary match applies. Returns ("", score) when nothing clears
    its threshold.
    """
    img_emb = embed_bgr(crop_bgr)

    from .object_memory import get_memory

    mem_name, mem_score = get_memory().match(img_emb)
    if mem_name and mem_score >= MEMORY_MIN_SIMILARITY:
        return mem_name, mem_score

    labels = list(DEFAULT_VOCABULARY)
    bank = _vocab_embeddings()
    if extra_labels:
        extras = [lbl.strip() for lbl in extra_labels if lbl.strip()]
        if extras:
            bank = np.vstack([bank, _embed_texts(extras)])
            labels += extras

    sims = bank @ img_emb
    best = int(np.argmax(sims))
    score = float(sims[best])
    if score < MIN_SIMILARITY:
        return "", score
    return labels[best], score


def name_detections(
    frame_bgr: np.ndarray,
    detections: list[Detection],
    extra_labels: list[str] | None = None,
) -> list[Detection]:
    """Fill in labels for unnamed detections (fusion's classical-only bucket).

    Mutates and returns the list. Named detections (from YOLO) are untouched.
    """
    h, w = frame_bgr.shape[:2]
    for det in detections:
        if det.label:
            continue
        px, py = int(det.w * CROP_PAD), int(det.h * CROP_PAD)
        x1, y1 = max(0, det.x - px), max(0, det.y - py)
        x2, y2 = min(w, det.x + det.w + px), min(h, det.y + det.h + py)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        label, _ = name_crop(crop, extra_labels)
        if label:
            det.label = label
    return detections
