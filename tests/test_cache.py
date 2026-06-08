"""Unit tests for ninetoothed._cache.Cache.

Covers the public surface of the Cache class:
  - in-memory mode (cache_dir=None)
  - disk-backed mode (cache_dir=tmp_path)
  - FIFO eviction at max_memory limit
  - L2 -> L1 promotion on hit
  - clear_memory() preserves L2
  - thread-safety (concurrent put/get from multiple threads)
  - contains() reflects L1 + L2
"""
import threading
import time

import pytest

from ninetoothed._cache import Cache


# ---------- in-memory mode ----------

def test_memory_only_put_and_get():
    c = Cache(max_memory=16)
    c.put("k1", {"v": 1})
    assert c.get("k1") == {"v": 1}


def test_memory_only_get_missing_returns_default():
    c = Cache(max_memory=16)
    assert c.get("missing") is None
    assert c.get("missing", default="fallback") == "fallback"


def test_memory_only_is_memory_only_property():
    c = Cache(max_memory=16)
    assert c.is_memory_only is True
    assert c.cache_dir is None


def test_memory_only_does_not_persist_across_instances():
    c1 = Cache(max_memory=16)
    c1.put("k", "v")
    c2 = Cache(max_memory=16)
    assert c2.get("k") is None  # not shared across instances


# ---------- disk-backed mode ----------

def test_disk_backed_persists(tmp_path):
    disk = tmp_path / "cache"
    c1 = Cache(cache_dir=disk, suffix=".json", max_memory=16)
    c1.put("k1", [1, 2, 3])
    assert c1.get("k1") == [1, 2, 3]

    c2 = Cache(cache_dir=disk, suffix=".json", max_memory=16)
    assert c2.get("k1") == [1, 2, 3]  # reloaded from disk


def test_disk_backed_creates_directory(tmp_path):
    disk = tmp_path / "deep" / "nested" / "cache"
    Cache(cache_dir=disk, suffix=".json", max_memory=4)
    assert disk.exists()


def test_disk_backed_is_not_memory_only(tmp_path):
    c = Cache(cache_dir=tmp_path, suffix=".json", max_memory=4)
    assert c.is_memory_only is False
    assert c.cache_dir == tmp_path


def test_disk_backed_default_serializer_is_json(tmp_path):
    import json
    c = Cache(cache_dir=tmp_path, suffix=".json", max_memory=4)
    c.put("k", {"a": 1})
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text()) == {"a": 1}


def test_disk_backed_custom_serializer(tmp_path):
    """identity serializer for plain strings; file is the string verbatim."""
    c = Cache(
        cache_dir=tmp_path,
        suffix=".txt",
        serializer=lambda v: v,
        deserializer=lambda s: s,
        max_memory=4,
    )
    c.put("k", "hello world")
    files = list(tmp_path.glob("*.txt"))
    assert len(files) == 1
    assert files[0].read_text() == "hello world"
    assert c.get("k") == "hello world"


# ---------- L1 + L2 promotion ----------

def test_l2_hit_promotes_to_l1(tmp_path):
    c = Cache(cache_dir=tmp_path, suffix=".json", max_memory=4)
    c.put("k", "v1")
    c.clear_memory()
    assert c.memory_size == 0
    # get() should pull from disk and promote to L1
    assert c.get("k") == "v1"
    assert c.memory_size == 1


def test_clear_memory_preserves_disk(tmp_path):
    c = Cache(cache_dir=tmp_path, suffix=".json", max_memory=4)
    c.put("k", "v1")
    c.clear_memory()
    # New instance can still read the value from disk
    c2 = Cache(cache_dir=tmp_path, suffix=".json", max_memory=4)
    assert c2.get("k") == "v1"


def test_contains_reflects_l1_and_l2(tmp_path):
    c = Cache(cache_dir=tmp_path, suffix=".json", max_memory=4)
    assert c.contains("k") is False
    c.put("k", "v")
    assert c.contains("k") is True
    c.clear_memory()
    assert c.contains("k") is True  # still in L2


# ---------- FIFO eviction ----------

def test_fifo_eviction_at_max_memory():
    c = Cache(max_memory=3)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)
    assert c.memory_size == 3
    # 4th insert should evict "a" (FIFO)
    c.put("d", 4)
    assert c.memory_size == 3
    assert c.get("a") is None
    assert c.get("b") == 2
    assert c.get("c") == 3
    assert c.get("d") == 4


def test_fifo_eviction_with_disk(tmp_path):
    c = Cache(cache_dir=tmp_path, suffix=".json", max_memory=2)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)
    # "a" was evicted from L1
    assert c.get("a") == 1  # should be promoted from L2 again
    # After this promotion, L1 holds {b, a} (insertion order); "c" still in L1
    assert c.memory_size == 2


# ---------- delete ----------

def test_delete_removes_from_l1_and_l2(tmp_path):
    c = Cache(cache_dir=tmp_path, suffix=".json", max_memory=4)
    c.put("k", "v")
    assert c.contains("k") is True
    c.delete("k")
    assert c.contains("k") is False
    assert c.get("k") is None


def test_delete_missing_key_is_noop():
    c = Cache(max_memory=4)
    c.delete("never_existed")  # should not raise


# ---------- disk corruption / bad data ----------

def test_corrupt_disk_file_returns_default(tmp_path):
    c = Cache(cache_dir=tmp_path, suffix=".json", max_memory=4)
    c.put("k", "v")
    # Corrupt the file on disk
    path = c._path_for("k")
    path.write_text("this is not valid json {{{")
    c.clear_memory()
    # get() should swallow the JSONDecodeError and return default
    assert c.get("k") is None
    assert c.get("k", default="safe") == "safe"


# ---------- thread safety ----------

def test_concurrent_put_get_no_race(tmp_path):
    """Many threads putting + getting on disjoint keys must not corrupt state."""
    c = Cache(cache_dir=tmp_path, suffix=".json", max_memory=1024)

    errors = []

    def worker(i):
        try:
            for j in range(50):
                c.put(f"k_{i}_{j}", j)
                v = c.get(f"k_{i}_{j}")
                assert v == j
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"


def test_concurrent_eviction_under_contention():
    """Stress eviction: many threads force frequent FIFO eviction."""
    c = Cache(max_memory=4)
    c.put("seed", 0)

    errors = []

    def worker():
        try:
            for j in range(200):
                c.put(f"k_{j}", j)
                _ = c.get(f"k_{j}")
                _ = c.memory_size
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"


def test_concurrent_disk_writes_no_partial_files(tmp_path):
    """After concurrent writes + force-flush, every persisted key must be readable."""
    c = Cache(cache_dir=tmp_path, suffix=".json", max_memory=4)

    def writer(prefix, n):
        for j in range(n):
            c.put(f"{prefix}_{j}", {"v": j})
            # Force L1 eviction by over-filling
            c.clear_memory()
            c.get(f"{prefix}_{j}")  # re-promote to L1, then we don't really need it

    threads = [threading.Thread(target=writer, args=(f"p{i}", 30)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Every key we wrote must round-trip from disk via a fresh instance
    c2 = Cache(cache_dir=tmp_path, suffix=".json", max_memory=4)
    for i in range(4):
        for j in range(30):
            assert c2.get(f"p{i}_{j}") == {"v": j}


# ---------- introspection ----------

def test_memory_size_reports_l1_length():
    c = Cache(max_memory=16)
    assert c.memory_size == 0
    c.put("a", 1)
    assert c.memory_size == 1
    c.put("b", 2)
    assert c.memory_size == 2
