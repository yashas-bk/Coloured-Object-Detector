"""Benchmark runner: scores all three detection methods against ground truth.

For every labeled image, each method's predictions are matched to ground-truth
boxes (greedy, best-IoU-first, IoU >= 0.5, same color category). Matched
pairs are true positives; leftover predictions are false positives; leftover
ground truth are false negatives.

Reports precision/recall/F1 + latency overall and per slice:
- per condition tag (dim-light, cluttered, outdoor)
- per object kind (COCO vs non-COCO) — recall only, since false positives
  can't be attributed to an object subset

Usage:  python benchmark/run_benchmark.py
Writes: benchmark/results.json
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection import detect_category, detect_category_fusion, detect_category_ml  # noqa: E402

BENCH = Path(__file__).parent
IOU_THRESHOLD = 0.5
METHODS = ("classical", "ml", "fusion")


def _predict(method: str, frame, color: str):
    if method == "classical":
        return detect_category(frame, color)[0]
    if method == "ml":
        return detect_category_ml(frame, color)
    return detect_category_fusion(frame, color)


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / (aw * ah + bw * bh - inter)


def _match(preds: list[tuple], truths: list[dict]):
    """Greedy IoU matching. Returns (tp_truth_indices, n_fp)."""
    pairs = sorted(
        ((_iou(p, t["box"]), pi, ti)
         for pi, p in enumerate(preds)
         for ti, t in enumerate(truths)),
        reverse=True,
    )
    used_p, used_t = set(), set()
    for iou, pi, ti in pairs:
        if iou < IOU_THRESHOLD:
            break
        if pi in used_p or ti in used_t:
            continue
        used_p.add(pi)
        used_t.add(ti)
    return used_t, len(preds) - len(used_p)


class Tally:
    def __init__(self):
        self.tp = self.fp = self.fn = 0

    def add(self, tp, fp, fn):
        self.tp += tp
        self.fp += fp
        self.fn += fn

    def metrics(self):
        p = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        r = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
                "tp": self.tp, "fp": self.fp, "fn": self.fn}


def main():
    annotations = []
    drafts = 0
    for f in sorted((BENCH / "annotations").glob("*.json")):
        ann = json.loads(f.read_text(encoding="utf-8"))
        if ann.get("draft"):
            drafts += 1  # unreviewed model output is NOT ground truth
            continue
        annotations.append(ann)  # human-reviewed ("no objects" is valid truth)
    if drafts:
        print(f"skipping {drafts} unreviewed draft annotations "
              f"(review them at http://127.0.0.1:8000/label)")
    if not annotations:
        print("No reviewed annotations found — label images at http://127.0.0.1:8000/label")
        return

    print(f"Scoring {len(annotations)} labeled images...\n")

    overall = {m: Tally() for m in METHODS}
    by_tag = {m: defaultdict(Tally) for m in METHODS}
    recall_by_kind = {m: {"coco": Tally(), "non-coco": Tally()} for m in METHODS}
    latency = {m: [] for m in METHODS}

    for n, ann in enumerate(annotations, 1):
        img_path = BENCH / "images" / ann["image"]
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  skipping unreadable {ann['image']}")
            continue

        # group ground truth by color: each color is scored independently
        truths_by_color = defaultdict(list)
        for obj in ann["objects"]:
            truths_by_color[obj["color"]].append(obj)

        for method in METHODS:
            for color, truths in truths_by_color.items():
                t0 = time.perf_counter()
                preds = [(d.x, d.y, d.w, d.h) for d in _predict(method, frame, color)]
                latency[method].append((time.perf_counter() - t0) * 1000)

                tp_truth_idx, n_fp = _match(preds, truths)
                tp, fn = len(tp_truth_idx), len(truths) - len(tp_truth_idx)
                overall[method].add(tp, n_fp, fn)
                for tag in ann.get("tags", []):
                    by_tag[method][tag].add(tp, n_fp, fn)
                for ti, t in enumerate(truths):
                    kind = "coco" if t.get("coco") else "non-coco"
                    if ti in tp_truth_idx:
                        recall_by_kind[method][kind].add(1, 0, 0)
                    else:
                        recall_by_kind[method][kind].add(0, 0, 1)
        print(f"  [{n}/{len(annotations)}] {ann['image']}")

    # ---- report ----
    results = {"images": len(annotations), "iou_threshold": IOU_THRESHOLD, "methods": {}}
    header = f"\n{'':10s}{'slice':14s}{'prec':>7s}{'recall':>8s}{'F1':>7s}{'ms/img':>9s}"
    print(header)
    print("-" * len(header))
    for method in METHODS:
        m = overall[method].metrics()
        avg_ms = sum(latency[method]) / max(len(latency[method]), 1)
        results["methods"][method] = {"overall": m, "avg_ms": round(avg_ms, 1),
                                      "tags": {}, "recall_by_kind": {}}
        print(f"{method:10s}{'overall':14s}{m['precision']:7.2f}{m['recall']:8.2f}"
              f"{m['f1']:7.2f}{avg_ms:9.0f}")
        for tag, tally in sorted(by_tag[method].items()):
            tm = tally.metrics()
            results["methods"][method]["tags"][tag] = tm
            print(f"{'':10s}{tag:14s}{tm['precision']:7.2f}{tm['recall']:8.2f}{tm['f1']:7.2f}")
        for kind, tally in recall_by_kind[method].items():
            km = tally.metrics()
            results["methods"][method]["recall_by_kind"][kind] = km["recall"]
            print(f"{'':10s}{kind + ' (rec)':14s}{'':7s}{km['recall']:8.2f}")
        print()

    out = BENCH / "results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
