"""Pre-label benchmark images with fusion suggestions, saved as DRAFTS.

Drafts load into the labeling tool as starting points but do not count as
labeled, and the benchmark scorer skips them — a human must review each one
(fix boxes, set tags, save) before it becomes ground truth. Blind-accepting
model output as truth would let fusion grade its own homework.

Usage:  python benchmark/prelabel.py     (skips images that already have any annotation)
"""

import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection import COLOR_CATEGORIES, detect_category_fusion  # noqa: E402

BENCH = Path(__file__).parent
IMAGES = BENCH / "images"
ANNOTATIONS = BENCH / "annotations"

# Filename prefixes from collect_images.py -> starter tags (review can change them)
PREFIX_TAGS = {"dim": ["dim-light"], "coco": ["outdoor"], "noncoco": []}


def main():
    ANNOTATIONS.mkdir(parents=True, exist_ok=True)
    images = sorted(IMAGES.glob("*.jpg"))
    done = skipped = 0
    t_start = time.time()

    for i, p in enumerate(images, 1):
        ann_path = ANNOTATIONS / f"{p.stem}.json"
        if ann_path.exists():
            skipped += 1
            continue
        frame = cv2.imread(str(p))
        if frame is None:
            print(f"  [{i}/{len(images)}] UNREADABLE {p.name}")
            continue

        objects = []
        for category in COLOR_CATEGORIES:
            for d in detect_category_fusion(frame, category):
                objects.append({
                    "box": [d.x, d.y, d.w, d.h],
                    "color": category,
                    "coco": bool(d.label),
                })

        prefix = p.name.split("__", 1)[0]
        record = {
            "image": p.name,
            "draft": True,
            "tags": PREFIX_TAGS.get(prefix, []),
            "objects": objects,
        }
        ann_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        done += 1
        print(f"  [{i}/{len(images)}] {p.name}: {len(objects)} draft boxes")

    print(f"\npre-labeled {done}, skipped {skipped} existing, "
          f"in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
