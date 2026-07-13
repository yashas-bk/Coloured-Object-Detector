"""Collect benchmark images from Wikimedia Commons (freely licensed).

Downloads a spread of images across the benchmark's condition cells:
COCO-object colors, non-COCO colored objects, and difficult lighting.
Re-runnable; skips files that already exist.

Usage:  python benchmark/collect_images.py
"""

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

IMAGES_DIR = Path(__file__).parent / "images"
UA = {"User-Agent": "color-detector-benchmark/1.0 (personal project)"}
THUMB_WIDTH = 1024

# (cell prefix, search query, how many to take)
QUERIES = [
    # COCO objects with strong colors
    ("coco", "yellow taxi cab street", 4),
    ("coco", "red double decker bus london", 3),
    ("coco", "bananas market stall", 3),
    ("coco", "red apples basket", 3),
    ("coco", "oranges fruit bowl", 3),
    ("coco", "blue car parked", 3),
    ("coco", "red fire hydrant", 3),
    ("coco", "person red jacket", 3),
    ("coco", "green tractor field", 3),
    ("coco", "black cat sitting", 3),
    # non-COCO colored objects
    ("noncoco", "sticky notes wall", 3),
    ("noncoco", "lego bricks pile", 3),
    ("noncoco", "colored pencils", 3),
    ("noncoco", "balloons bunch colorful", 3),
    ("noncoco", "rubber duck yellow", 3),
    ("noncoco", "traffic cone street", 3),
    ("noncoco", "yellow tulips field", 3),
    ("noncoco", "red mailbox", 3),
    # difficult lighting
    ("dim", "street night neon", 3),
    ("dim", "sunset city street cars", 3),
    ("dim", "dim room interior lamp", 3),
    ("dim", "foggy street morning", 3),
]


def _get(url: str, timeout: int = 30) -> bytes:
    """GET with retry/backoff for 429 rate limits."""
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers=UA)
            return urllib.request.urlopen(req, timeout=timeout).read()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 4:
                wait = 5 * (attempt + 1)
                print(f"    rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("unreachable")


def api(params: dict) -> dict:
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    return json.loads(_get(url, timeout=25))


def search_images(query: str, n: int):
    data = api({
        "action": "query", "format": "json", "generator": "search",
        "gsrnamespace": 6, "gsrlimit": n * 3, "gsrsearch": query,
        "prop": "imageinfo", "iiprop": "url|size", "iiurlwidth": THUMB_WIDTH,
    })
    out = []
    for page in data.get("query", {}).get("pages", {}).values():
        ii = page.get("imageinfo", [{}])[0]
        if not ii.get("thumburl"):
            continue
        if not page["title"].lower().endswith((".jpg", ".jpeg")):
            continue
        if (ii.get("width") or 0) < 500 or (ii.get("height") or 0) < 400:
            continue
        out.append((page["title"], ii["thumburl"]))
    return out[:n]


def slug(title: str) -> str:
    name = title.replace("File:", "").rsplit(".", 1)[0]
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")[:60].lower()


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    seen_titles = set()
    downloaded = skipped = failed = 0

    for cell, query, n in QUERIES:
        try:
            results = search_images(query, n)
        except Exception as e:
            print(f"  search failed for {query!r}: {e}")
            continue
        for title, url in results:
            if title in seen_titles:
                continue
            seen_titles.add(title)
            dest = IMAGES_DIR / f"{cell}__{slug(title)}.jpg"
            if dest.exists():
                skipped += 1
                continue
            try:
                dest.write_bytes(_get(url))
                downloaded += 1
                print(f"  {dest.name}")
            except Exception as e:
                failed += 1
                print(f"  FAILED {title[:50]}: {e}")
            time.sleep(1.5)  # be polite to the API
        time.sleep(2.5)

    total = len(list(IMAGES_DIR.glob("*.jpg")))
    print(f"\ndownloaded {downloaded}, skipped {skipped} existing, {failed} failed")
    print(f"total images in {IMAGES_DIR}: {total}")


if __name__ == "__main__":
    main()
