---
title: Coloured Object Detector
emoji: 🟡
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
---

# Coloured Object Detector

A web application that finds objects of a chosen colour in images, uploaded
videos, and a live webcam feed. It implements the same task three ways — a
classical computer-vision pipeline, a deep-learning pipeline, and a fusion of
the two — and lets you compare them side by side on the same input.

The project started as a single-file OpenCV script that drew boxes around
yellow objects on a webcam. It grew into a comparison platform built around
one question: when does classical image processing hold up against a neural
network, and when does it fall apart?

## Contents

- [How detection works](#how-detection-works)
  - [Classical engine](#classical-engine)
  - [ML engine](#ml-engine)
  - [Fusion engine](#fusion-engine)
- [The web app](#the-web-app)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Running](#running)
- [API](#api)
- [Benchmark](#benchmark)
- [Tests](#tests)
- [Docker and deployment](#docker-and-deployment)
- [Known limitations](#known-limitations)

## How detection works

The user picks one of eleven colour categories: red, orange, yellow, brown,
green, cyan, blue, purple, pink, white, black. All three engines answer the
same question — "where are the objects of this colour?" — and return the same
`Detection` structure (box, area, optional class label and confidence), so
they are interchangeable behind a single `method` parameter.

### Classical engine

`detection/color_detector.py`. No machine learning; pixels in, boxes out.

1. **Preprocessing.** The frame is Gaussian-blurred to suppress sensor noise,
   then converted to HSV and LAB colour spaces in one pass. LAB chroma
   (distance from the neutral axis) is computed per pixel.

2. **Category mask.** Each category is a rule over hue, saturation, value,
   LAB lightness and chroma. Most categories are plain hue bands, but a few
   need composite rules because colour *names* do not map to hues alone:

   - Red wraps around the hue circle, so it is the union of two bands.
   - Brown is not a hue — it is dark or muted orange. Warm hues split into
     orange (bright and vivid) and brown (dark, or muted even when bright).
   - Pink is partly a tint: light, softened reds count as pink alongside the
     magenta-pink hue band. The purple/pink boundary sits at hue 312 degrees
     so that CSS purple, violet and orchid all read as purple.
   - White and black are judged in LAB only: white is high lightness with
     near-zero chroma, black is low lightness with near-zero chroma. Judging
     them by HSV saturation is a classic mistake — every dim or washed-out
     pixel has low saturation, so "grey/white/black by saturation" matches
     half the scene. Chroma separates truly colourless pixels from muted
     colours. For the same reason, a chroma floor is applied to all chromatic
     categories, and a dark red (`#67001F`) classifies as red, not black.

   The boundaries were calibrated against CSS named colours: 40 of 42
   unambiguous names classify to the expected category (the two misses,
   `chocolate` and `peru`, sit on the genuinely fuzzy brown/orange border).

3. **Morphology.** The binary mask is opened (removes speck noise) and then
   closed with a generous elliptical kernel. Closing matters more than it
   sounds: specular highlights and shadows punch holes through an object's
   mask, and without it one object fragments into several detections.

4. **Contours and merging.** External contours become candidate boxes.
   Boxes that overlap or sit within roughly 2% of the frame of each other are
   merged, and only then is the minimum-area filter applied — fragments of a
   large object can each be individually small.

Typical latency is 10–60 ms per frame on CPU depending on resolution.

### ML engine

`detection/ml_detector.py`. YOLOv8-nano (pretrained on COCO, 80 classes)
proposes objects; colour is then verified per proposal:

1. YOLO runs on the frame and returns class-labelled boxes with confidences.
2. Each box is shrunk by 15% per side to cut background pixels, and the crop
   is reduced to its dominant colours with k-means (k=3) over a pixel sample.
3. Each dominant colour is classified with the *same* LAB-based classifier
   the classical engine uses, so "yellow" means the same thing in both
   engines. A detection is kept if a cluster owning at least 25% of the crop
   matches the requested category.

The result is semantic: "yellow car, 87%" rather than an anonymous yellow
region. The trade-offs are the inverse of the classical engine's: it only
sees objects in COCO's vocabulary (a yellow sticky note returns nothing), and
inference costs roughly 100–200 ms per frame on CPU.

The model loads lazily and is warmed up in a background thread at server
start; inference is serialised behind a lock because Ultralytics predict is
not thread-safe.

### Fusion engine

`detection/fusion.py`. Both engines run on the frame and their outputs are
reconciled. Boxes are matched greedily, best overlap first; a pair counts as
the same object if IoU is at least 0.4, or if the smaller box is at least 65%
contained in the larger one. Containment matters because the classical engine
masks an object's coloured *body* while YOLO boxes the *whole* object — car
paint versus car-with-windows overlaps heavily but has low IoU.

Every detection lands in one of three buckets:

| Bucket         | Meaning                                       | Confidence            |
| -------------- | --------------------------------------------- | --------------------- |
| agreement      | both engines found it; strongest evidence     | 0.9 + 0.1 × YOLO conf |
| ml-only        | YOLO found it, colour mask missed it (usually lighting) | YOLO conf × 0.85 |
| classical-only | coloured region YOLO cannot name (non-COCO object) | 0.7 × solidity × size factor |

Classical-only regions are demoted when they look like background rather than
an object: very large regions (a wall lit warm) and low-solidity, stringy
regions (lighting artifacts) fall below the 0.25 confidence floor and are
dropped. YOLO's silence is treated as absence of evidence, not a veto — the
non-COCO bucket is what preserves the ability to find arbitrary coloured
things.

The scoring weights are initial estimates intended to be tuned against the
benchmark below.

## The web app

FastAPI backend, vanilla HTML/JS frontend, four tabs:

- **Image** — upload a photo, get annotated results. An eyedropper lets you
  click any pixel and snaps it to the nearest colour category.
- **Video** — uploads are processed asynchronously in a worker thread
  (progress polled by the client) and re-encoded to H.264 with a bundled
  ffmpeg, because OpenCV's default mp4 codec does not play in browsers.
  Videos are capped at roughly 60 seconds and downscaled to 640 px wide.
- **Camera** — the browser captures webcam frames with `getUserMedia` and
  streams JPEGs over a WebSocket; the server replies with detection boxes as
  JSON, which the client draws over the live video. The client only sends the
  next frame after the previous reply arrives, so a slow server can never
  build a queue of stale frames.
- **Arena** — one upload, all three engines, side-by-side annotated results
  with per-engine counts and timings.

The engine (classical / ml / fusion) is switchable from the toolbar in every
tab, including live mid-stream on the camera.

## Project structure

```
.
├── app/
│   └── main.py               FastAPI app: REST endpoints, WebSocket handler,
│                             async video jobs, benchmark/labeling endpoints
├── detection/
│   ├── color_detector.py     classical engine + colour category definitions +
│                             classify_bgr (shared colour vocabulary)
│   ├── ml_detector.py        YOLO engine + dominant-colour verification
│   └── fusion.py             cross-engine matching, buckets, confidence scoring
├── static/
│   ├── index.html            the app UI (four tabs)
│   └── label.html            benchmark labeling tool
├── benchmark/
│   ├── collect_images.py     downloads test images from Wikimedia Commons
│   ├── prelabel.py           pre-labels images with fusion output (as drafts)
│   ├── run_benchmark.py      scores all engines against ground truth
│   ├── images/               test images (git-ignored; re-downloadable)
│   └── annotations/          hand-reviewed ground truth (tracked)
├── samples/                  demo images, including a fusion test scene
├── tests/
│   ├── test_color_detector.py
│   └── test_fusion.py
├── yellow_detector.py        standalone webcam script (the original project)
└── requirements.txt
```

## Installation

Requires Python 3.10 or newer.

```
pip install -r requirements.txt
```

This pulls in OpenCV, FastAPI, and Ultralytics (which installs PyTorch; CPU
build is sufficient). The YOLOv8n weights (~6 MB) download automatically on
first use.

## Running

```
python -m uvicorn app.main:app --port 8000
```

Open http://127.0.0.1:8000. The labeling tool lives at /label.

The original standalone webcam script still works without the server:

```
python yellow_detector.py --color yellow
```

## API

| Method | Path                      | Description                                        |
| ------ | ------------------------- | -------------------------------------------------- |
| GET    | /colors                   | available colour categories with UI swatches       |
| GET    | /classify?color=%23RRGGBB | snap an exact colour to its nearest category       |
| POST   | /detect/image             | multipart: file, color, min_area, method           |
| POST   | /detect/video             | same fields; returns a job_id                      |
| GET    | /jobs/{id}                | job progress / summary                             |
| GET    | /jobs/{id}/result         | annotated H.264 video                              |
| POST   | /detect/arena             | run all three engines on one image                 |
| WS     | /ws/detect                | binary JPEG frames in, detection JSON out; JSON    |
|        |                           | text messages update color/min_area/method live    |
| GET    | /bench/*                  | labeling endpoints (list, image, annotations, suggest) |

## Benchmark

The point of the benchmark is not a winner but an operating envelope: under
which conditions does each engine hold up, and at what latency.

Ground truth is hand-labeled through the tool at /label: boxes with a colour
category each, per-object COCO/non-COCO flags, and per-image condition tags
(dim-light, cluttered, outdoor). To speed labeling up, `prelabel.py` seeds
every image with fusion's own output as a draft — drafts are excluded from
scoring until a human reviews and saves them, since unreviewed model output
as ground truth would let the model grade its own homework.

```
python benchmark/collect_images.py    # fetch test images (Wikimedia Commons)
python benchmark/prelabel.py          # seed draft annotations
# review drafts at http://127.0.0.1:8000/label
python benchmark/run_benchmark.py     # score and print the table
```

The scorer matches predictions to ground truth greedily at IoU ≥ 0.5 within
each colour, and reports precision, recall, F1 and mean latency — overall,
per condition tag, and recall split by COCO/non-COCO objects. Results land in
`benchmark/results.json`.

## Tests

```
python -m pytest tests/
```

91 tests cover the category matrix (every colour detects and classifies its
own sample), the CSS named-colour calibration, regression cases found during
development (object fragmentation, muted colours misread as grey, dark red
misread as black), and the fusion matcher's buckets and false-positive
suppression. The fusion tests use fabricated detections, so the suite runs in
about a second with no model download.

## Docker and deployment

```
docker build -t color-detector .
docker run -p 8000:8000 color-detector
```

The image uses CPU-only PyTorch (the default Linux wheel bundles CUDA and
inflates the image by several gigabytes) and bakes the YOLO weights in at
build time so the first request does not wait on a download. The server
honours a `PORT` environment variable, which is what most hosts set.

To deploy on Hugging Face Spaces: create a Space with the Docker SDK, push
this repository to it, and it builds and serves automatically. Render and
Fly.io work the same way from the Dockerfile. Note that the camera tab
requires HTTPS in production — browsers only allow `getUserMedia` on secure
origins — which all of the hosts above provide by default.

The video tab is CPU-heavy; on free tiers expect uploads near the 60-second
cap to process slowly, especially in ml or fusion mode.

## Known limitations

- Colour is judged on pixel values, and pixels do not respect human colour
  constancy. A yellow taxi photographed at dusk is, in its pixels, dark amber
  — humans call it yellow because the brain discounts the lighting, but the
  classical engine will honestly report brown/orange. The samples/ directory
  keeps one such photo as a standing test case.
- The ML engine cannot see anything outside COCO's 80 classes. That is by
  design in the comparison, and is what the classical-only fusion bucket
  compensates for.
- Video jobs and their results are held in process memory and the temp
  directory; a server restart forgets them. Fine for a single-user tool, not
  for multi-user deployment.
- ML inference on CPU costs 100–200 ms per frame, which caps the camera tab
  at roughly 5–10 FPS in ml/fusion mode.

Planned next: open-vocabulary naming of classical-only detections with CLIP,
few-shot object memory, and a hosted deployment.
