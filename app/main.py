"""FastAPI backend for the color object detector."""

import asyncio
import base64
import json
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from imageio_ffmpeg import get_ffmpeg_exe
from starlette.websockets import WebSocketDisconnect

from detection import (
    COLOR_CATEGORIES,
    annotate,
    classify_bgr,
    detect_category,
    detect_category_fusion,
    detect_category_ml,
    hex_to_bgr,
    warm_up_ml,
)

METHODS = ("classical", "ml", "fusion")


def _detect(frame, color: str, min_area: int, method: str):
    """Dispatch to the requested engine; returns a list of Detections."""
    if method == "ml":
        return detect_category_ml(frame, color, min_area)
    if method == "fusion":
        return detect_category_fusion(frame, color, min_area)
    detections, _ = detect_category(frame, color, min_area)
    return detections

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_VIDEO_BYTES = 100 * 1024 * 1024  # 100 MB
MAX_VIDEO_FRAMES = 1800  # ~60 s at 30 fps; longer videos are truncated
PROCESS_WIDTH = 640  # frames are downscaled to this width for processing speed

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

JOBS: dict[str, dict] = {}  # in-memory job registry (single-process server)
JOBS_DIR = Path(tempfile.gettempdir()) / "color-detector-jobs"
JOBS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Color Object Detector")


@app.on_event("startup")
async def _warm_ml_model():
    # Load YOLO in the background so the first ML request doesn't stall ~10 s.
    threading.Thread(target=warm_up_ml, daemon=True).start()


@app.get("/colors")
async def colors():
    """Available color categories and their UI swatch colors."""
    return [{"name": name, "swatch": spec["swatch"]} for name, spec in COLOR_CATEGORIES.items()]


@app.get("/classify")
async def classify(color: str):
    """Snap an exact hex color (e.g. from the eyedropper) to its nearest category."""
    try:
        bgr = hex_to_bgr(color)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"category": classify_bgr(bgr)}


@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    color: str = Form("yellow"),
    min_area: int = Form(500),
    method: str = Form("classical"),
):
    data = await file.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image too large (max 10 MB)")

    buf = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode image — is it a valid image file?")

    min_area = max(1, min_area)
    if method not in METHODS:
        raise HTTPException(400, f"Unknown method: {method!r} (choose from {', '.join(METHODS)})")

    try:
        detections = _detect(frame, color, min_area, method)
    except ValueError as e:
        raise HTTPException(400, str(e))

    annotated = annotate(frame, detections, label=color.lower())

    ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(500, "Failed to encode annotated image")

    return {
        "count": len(detections),
        "method": method,
        "detections": [d.to_dict() for d in detections],
        "annotated_image": "data:image/jpeg;base64,"
        + base64.b64encode(encoded.tobytes()).decode(),
    }


@app.post("/detect/arena")
def detect_arena(
    file: UploadFile = File(...),
    color: str = Form("yellow"),
    min_area: int = Form(500),
):
    """Run BOTH engines on the same image; returns per-method results + timing.

    Sync endpoint on purpose: FastAPI runs it in a worker thread, so the
    ~150 ms ML inference doesn't stall the event loop.
    """
    data = file.file.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image too large (max 10 MB)")

    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode image — is it a valid image file?")
    if color.lower() not in COLOR_CATEGORIES:
        raise HTTPException(400, f"Unknown color category: {color!r}")

    min_area = max(1, min_area)
    out = {"color": color.lower()}
    for method in METHODS:
        t0 = time.perf_counter()
        detections = _detect(frame, color, min_area, method)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        annotated = annotate(frame, detections, label=color.lower())
        ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise HTTPException(500, "Failed to encode annotated image")
        out[method] = {
            "count": len(detections),
            "ms": ms,
            "detections": [d.to_dict() for d in detections],
            "annotated_image": "data:image/jpeg;base64,"
            + base64.b64encode(encoded.tobytes()).decode(),
        }
    return out


def _process_video(job_id: str, src_path: Path, color: str, min_area: int, method: str) -> None:
    """Worker thread: detect + annotate every frame, encode H.264 via ffmpeg."""
    job = JOBS[job_id]
    encoder = None
    try:
        cap = cv2.VideoCapture(str(src_path))
        if not cap.isOpened():
            raise RuntimeError("Could not open video — unsupported format?")

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if 1 <= fps <= 120 else 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total = min(total, MAX_VIDEO_FRAMES) if total > 0 else MAX_VIDEO_FRAMES
        job["total_frames"] = total

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = min(1.0, PROCESS_WIDTH / max(w, 1))
        # H.264 yuv420p requires even dimensions
        ow, oh = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
        if ow < 2 or oh < 2:
            raise RuntimeError("Video dimensions too small")
        # min_area is in original-resolution pixels; areas shrink by scale^2
        scaled_min_area = max(1, int(min_area * scale * scale))

        out_path = JOBS_DIR / f"{job_id}.mp4"
        encoder = subprocess.Popen(
            [get_ffmpeg_exe(), "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{ow}x{oh}", "-r", f"{fps:.3f}", "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             str(out_path)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        frames_done = frames_with_detections = total_detections = max_concurrent = 0
        while frames_done < total:
            ret, frame = cap.read()
            if not ret:
                break
            if (frame.shape[1], frame.shape[0]) != (ow, oh):
                frame = cv2.resize(frame, (ow, oh))
            detections = _detect(frame, color, scaled_min_area, method)
            annotated = annotate(frame, detections, label=color)
            encoder.stdin.write(annotated.tobytes())
            frames_done += 1
            if detections:
                frames_with_detections += 1
                total_detections += len(detections)
                max_concurrent = max(max_concurrent, len(detections))
            job["progress"] = frames_done

        cap.release()
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("Video encoding failed")
        if frames_done == 0:
            raise RuntimeError("Video contained no readable frames")

        job.update(
            status="done",
            progress=frames_done,
            total_frames=frames_done,
            summary={
                "frames": frames_done,
                "frames_with_detections": frames_with_detections,
                "total_detections": total_detections,
                "max_concurrent": max_concurrent,
            },
        )
    except Exception as e:
        if encoder is not None and encoder.poll() is None:
            encoder.stdin.close()
            encoder.wait()
        job.update(status="error", error=str(e))
    finally:
        src_path.unlink(missing_ok=True)


@app.post("/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    color: str = Form("yellow"),
    min_area: int = Form(500),
    method: str = Form("classical"),
):
    if color.lower() not in COLOR_CATEGORIES:
        raise HTTPException(400, f"Unknown color category: {color!r}")
    if method not in METHODS:
        raise HTTPException(400, f"Unknown method: {method!r} (choose from {', '.join(METHODS)})")

    data = await file.read()
    if len(data) > MAX_VIDEO_BYTES:
        raise HTTPException(413, "Video too large (max 100 MB)")
    if not data:
        raise HTTPException(400, "Empty upload")

    job_id = uuid.uuid4().hex[:12]
    src_path = JOBS_DIR / f"{job_id}.src"
    src_path.write_bytes(data)

    JOBS[job_id] = {"status": "processing", "progress": 0, "total_frames": 0}
    threading.Thread(
        target=_process_video,
        args=(job_id, src_path, color.lower(), max(1, min_area), method),
        daemon=True,
    ).start()
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    return job


@app.get("/jobs/{job_id}/result")
async def job_result(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    if job.get("status") != "done":
        raise HTTPException(409, "Job not finished")
    return FileResponse(JOBS_DIR / f"{job_id}.mp4", media_type="video/mp4")


@app.websocket("/ws/detect")
async def ws_detect(ws: WebSocket):
    """Live detection stream.

    Protocol: text messages are JSON config updates ({"color", "min_area"});
    binary messages are JPEG frames. Each frame gets one JSON reply with
    detection boxes (in the sent frame's coordinates) and processing time.
    """
    await ws.accept()
    color = "yellow"
    min_area = 500
    method = "classical"

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            if msg.get("text") is not None:
                try:
                    cfg = json.loads(msg["text"])
                    color = str(cfg.get("color", color))
                    min_area = max(1, int(cfg.get("min_area", min_area)))
                    m = str(cfg.get("method", method))
                    method = m if m in METHODS else method
                except (json.JSONDecodeError, TypeError, ValueError):
                    await ws.send_json({"error": "Bad config message"})
                continue

            frame_bytes = msg.get("bytes")
            if not frame_bytes:
                continue

            t0 = time.perf_counter()
            buf = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if frame is None:
                await ws.send_json({"error": "Could not decode frame"})
                continue

            try:
                # Off the event loop: ML inference takes ~150 ms per frame.
                detections = await asyncio.get_event_loop().run_in_executor(
                    None, _detect, frame, color, min_area, method
                )
            except ValueError as e:
                await ws.send_json({"error": str(e)})
                continue

            await ws.send_json({
                "count": len(detections),
                "method": method,
                "detections": [d.to_dict() for d in detections],
                "ms": round((time.perf_counter() - t0) * 1000, 1),
            })
    except WebSocketDisconnect:
        pass


# ---- Benchmark / labeling ----

BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmark"
BENCH_IMAGES = BENCH_DIR / "images"
BENCH_ANNOTATIONS = BENCH_DIR / "annotations"


def _bench_image_path(name: str) -> Path:
    p = BENCH_IMAGES / Path(name).name  # Path().name blocks traversal
    if not p.is_file():
        raise HTTPException(404, f"No such benchmark image: {name}")
    return p


@app.get("/bench/images")
async def bench_images():
    """All benchmark images with labeled/unlabeled status."""
    BENCH_ANNOTATIONS.mkdir(parents=True, exist_ok=True)
    out = []
    if BENCH_IMAGES.is_dir():
        for p in sorted(BENCH_IMAGES.glob("*.jpg")):
            ann_path = BENCH_ANNOTATIONS / f"{p.stem}.json"
            labeled = False
            if ann_path.is_file():
                # model-generated drafts don't count as labeled until reviewed
                labeled = not json.loads(ann_path.read_text(encoding="utf-8")).get("draft", False)
            out.append({"name": p.name, "labeled": labeled})
    return out


@app.get("/bench/image/{name}")
async def bench_image(name: str):
    return FileResponse(_bench_image_path(name), media_type="image/jpeg")


@app.get("/bench/annotations/{name}")
async def bench_get_annotation(name: str):
    p = BENCH_ANNOTATIONS / f"{Path(name).stem}.json"
    if not p.is_file():
        return {"image": Path(name).name, "tags": [], "objects": []}
    return json.loads(p.read_text(encoding="utf-8"))


@app.post("/bench/annotations/{name}")
async def bench_save_annotation(name: str, payload: dict):
    _bench_image_path(name)  # must correspond to a real image
    BENCH_ANNOTATIONS.mkdir(parents=True, exist_ok=True)
    record = {
        "image": Path(name).name,
        "draft": False,  # saving from the labeling tool means human-reviewed
        "tags": [str(t) for t in payload.get("tags", [])],
        "objects": [
            {
                "box": [int(v) for v in o["box"][:4]],
                "color": str(o["color"]).lower(),
                "coco": bool(o.get("coco", False)),
            }
            for o in payload.get("objects", [])
        ],
    }
    p = BENCH_ANNOTATIONS / f"{Path(name).stem}.json"
    p.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return {"saved": p.name, "objects": len(record["objects"])}


@app.get("/bench/suggest/{name}")
def bench_suggest(name: str):
    """Model-assisted pre-labels: fusion detections across every category.

    Sync endpoint (worker thread) — runs YOLO once per category, ~3 s total.
    """
    frame = cv2.imread(str(_bench_image_path(name)))
    if frame is None:
        raise HTTPException(500, "Could not read image")
    suggestions = []
    for category in COLOR_CATEGORIES:
        try:
            for d in detect_category_fusion(frame, category):
                suggestions.append({
                    "box": [d.x, d.y, d.w, d.h],
                    "color": category,
                    "coco": bool(d.label),
                    "label": d.label,
                    "confidence": round(d.confidence, 2),
                })
        except ValueError:
            continue
    return {"suggestions": suggestions}


@app.get("/label")
async def label_page():
    return FileResponse(STATIC_DIR / "label.html")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
