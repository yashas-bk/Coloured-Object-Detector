"""Few-shot object memory: recognize user-taught objects from one example.

CLIP embeddings make this almost free: teaching stores the embedding of an
example image under a user-chosen name; recognition compares a detection
crop's embedding against every stored example (cosine similarity, exact
nearest neighbour — no training, no index needed at this scale).

Image-to-image similarity runs much higher than CLIP's image-to-text scores
(same object across views ~0.6-0.9 vs text matches ~0.2-0.35), so memory
has its own threshold in clip_namer and, above it, beats the vocabulary.

Storage is a JSON file next to the repo root: human-inspectable, and a
512-float vector per example is tiny. The file is user data, not code —
it is gitignored.
"""

import json
import threading
from pathlib import Path

import numpy as np

MEMORY_PATH = Path(__file__).resolve().parent.parent / "object_memory.json"
MAX_OBJECTS = 50
MAX_EXAMPLES_PER_OBJECT = 10
MAX_NAME_LEN = 40


class ObjectMemory:
    """Thread-safe store of named example embeddings with disk persistence."""

    def __init__(self, path: Path = MEMORY_PATH):
        self._path = path
        self._lock = threading.RLock()
        self._objects: dict[str, list[np.ndarray]] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            if not self._path.is_file():
                return
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                for name, examples in raw.get("objects", {}).items():
                    self._objects[str(name)] = [
                        np.asarray(e, dtype=np.float32) for e in examples
                    ]
            except (json.JSONDecodeError, ValueError, TypeError):
                # Corrupt memory file: start empty rather than crash detection.
                self._objects = {}

    def _save(self) -> None:
        data = {
            "objects": {
                name: [e.round(6).tolist() for e in examples]
                for name, examples in self._objects.items()
            }
        }
        self._path.write_text(json.dumps(data), encoding="utf-8")

    def teach(self, name: str, embedding: np.ndarray) -> int:
        """Store one example under `name`; returns the example count for it."""
        name = " ".join(name.split())[:MAX_NAME_LEN]
        if not name:
            raise ValueError("Object name must not be empty")
        self._ensure_loaded()
        with self._lock:
            if name not in self._objects and len(self._objects) >= MAX_OBJECTS:
                raise ValueError(f"Memory is full ({MAX_OBJECTS} objects max)")
            examples = self._objects.setdefault(name, [])
            if len(examples) >= MAX_EXAMPLES_PER_OBJECT:
                raise ValueError(
                    f"{name!r} already has {MAX_EXAMPLES_PER_OBJECT} examples"
                )
            examples.append(np.asarray(embedding, dtype=np.float32))
            self._save()
            return len(examples)

    def forget(self, name: str) -> bool:
        self._ensure_loaded()
        with self._lock:
            if name not in self._objects:
                return False
            del self._objects[name]
            self._save()
            return True

    def summary(self) -> list[dict]:
        self._ensure_loaded()
        with self._lock:
            return [
                {"name": name, "examples": len(examples)}
                for name, examples in sorted(self._objects.items())
            ]

    def match(self, embedding: np.ndarray) -> tuple[str, float]:
        """Best-matching taught object: (name, cosine similarity).

        Returns ("", 0.0) when nothing is taught. Thresholding is the
        caller's job — image-image similarity scales differ from text.
        """
        self._ensure_loaded()
        with self._lock:
            if not self._objects:
                return "", 0.0
            emb = np.asarray(embedding, dtype=np.float32)
            best_name, best_score = "", -1.0
            for name, examples in self._objects.items():
                score = float(max(ex @ emb for ex in examples))
                if score > best_score:
                    best_name, best_score = name, score
            return best_name, best_score


_memory = ObjectMemory()


def get_memory() -> ObjectMemory:
    return _memory
