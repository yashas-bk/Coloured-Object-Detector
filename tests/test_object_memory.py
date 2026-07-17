"""Object memory store: teach/match/forget logic and disk persistence.

Uses small synthetic unit vectors instead of real CLIP embeddings — the
store is pure nearest-neighbour math, so dimensionality doesn't matter.
"""

import numpy as np
import pytest

from detection.object_memory import (
    MAX_EXAMPLES_PER_OBJECT,
    MAX_OBJECTS,
    ObjectMemory,
)


def unit(*values):
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def memory(tmp_path):
    return ObjectMemory(path=tmp_path / "memory.json")


def test_empty_memory_matches_nothing(memory):
    name, score = memory.match(unit(1, 0, 0))
    assert name == ""
    assert score == 0.0


def test_match_returns_closest_taught_object(memory):
    memory.teach("red mug", unit(1, 0, 0))
    memory.teach("blue pen", unit(0, 1, 0))

    name, score = memory.match(unit(0.9, 0.1, 0))
    assert name == "red mug"
    assert score > 0.95


def test_multiple_examples_use_best_match(memory):
    # Two quite different views of the same object.
    memory.teach("mascot", unit(1, 0, 0))
    memory.teach("mascot", unit(0, 0, 1))

    name, score = memory.match(unit(0.1, 0, 0.99))
    assert name == "mascot"
    assert score > 0.95


def test_teach_returns_example_count_and_normalizes_name(memory):
    assert memory.teach("  desk   toy ", unit(1, 0, 0)) == 1
    assert memory.teach("desk toy", unit(0, 1, 0)) == 2
    assert memory.summary() == [{"name": "desk toy", "examples": 2}]


def test_empty_name_rejected(memory):
    with pytest.raises(ValueError):
        memory.teach("   ", unit(1, 0, 0))


def test_forget(memory):
    memory.teach("mug", unit(1, 0, 0))
    assert memory.forget("mug") is True
    assert memory.forget("mug") is False
    assert memory.summary() == []
    assert memory.match(unit(1, 0, 0)) == ("", 0.0)


def test_persistence_round_trip(tmp_path):
    path = tmp_path / "memory.json"
    first = ObjectMemory(path=path)
    first.teach("mug", unit(1, 0, 0))
    first.teach("mug", unit(0, 1, 0))
    first.teach("pen", unit(0, 0, 1))

    reloaded = ObjectMemory(path=path)
    assert reloaded.summary() == [
        {"name": "mug", "examples": 2},
        {"name": "pen", "examples": 1},
    ]
    name, score = reloaded.match(unit(0, 0, 1))
    assert name == "pen"
    assert score > 0.99


def test_corrupt_file_starts_empty(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text("{not json", encoding="utf-8")
    memory = ObjectMemory(path=path)
    assert memory.summary() == []
    # And it recovers: teaching after corruption works.
    memory.teach("mug", unit(1, 0, 0))
    assert memory.summary() == [{"name": "mug", "examples": 1}]


def test_example_limit_enforced(memory):
    for _ in range(MAX_EXAMPLES_PER_OBJECT):
        memory.teach("mug", unit(1, 0, 0))
    with pytest.raises(ValueError):
        memory.teach("mug", unit(1, 0, 0))


def test_object_limit_enforced(memory):
    for i in range(MAX_OBJECTS):
        memory.teach(f"object {i}", unit(1, 0, 0))
    with pytest.raises(ValueError):
        memory.teach("one too many", unit(1, 0, 0))
    # Existing objects can still gain examples at the cap.
    assert memory.teach("object 0", unit(0, 1, 0)) == 2
