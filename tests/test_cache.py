"""Unit tests for ninetoothed._cache.Cache.

Covers the public surface of the Cache class:
  - in-memory mode (cache_dir=None)
  - disk-backed mode (cache_dir=tmp_path)
  - FIFO eviction at max_memory limit
  - L2 -> L1 promotion on hit
  - clear_memory() preserves L2
  - thread-safety (concurrent put/get from multiple threads)
  - contains() reflects L1 + L2
  - integration: make() cache key is sensitive to non-Tensor
    elements in the tensors tuple (e.g. ceil_mode), even when the
    output shape happens to be the same for both values.
"""

import threading

import pytest
import torch
import torch.nn.functional as F

from ninetoothed._cache import Cache
from ninetoothed.make import _HANDLE_CACHE
from tests.test_max_pool2d import max_pool2d
from tests.utils import get_available_devices

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


def test_updating_existing_key_does_not_evict_other_entries():
    c = Cache(max_memory=2)
    c.put("a", 1)
    c.put("b", 2)

    c.put("b", 3)

    assert c.memory_size == 2
    assert c.get("a") == 1
    assert c.get("b") == 3


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


# ---------- integration: make() cache key sensitivity ----------
# These tests verify that _HANDLE_CACHE in ninetoothed.make produces
# DISTINCT cache keys for arrangements that differ only in non-Tensor
# elements of the tensors tuple (e.g. ceil_mode). This guards against
# regressions where the cache would mistake two semantically-different
# kernels for the same cache entry.
#
# The two interesting cases:
#   (A) the differing argument produces a different output shape -- a
#       shape-naive cache would still correctly miss here, so this is
#       just a sanity check.
#   (B) the differing argument produces an IDENTICAL output shape -- a
#       shape-naive cache would mistakenly HIT and return the wrong
#       kernel. This is the key regression test.
#   (C) verify the cache HIT path: a second call with the same arguments
#       must reuse the cached entry.


@pytest.fixture(autouse=True)
def _clear_handle_cache():
    """Each test starts (and ends) with an empty _HANDLE_CACHE L1."""
    _HANDLE_CACHE.clear_memory()
    yield
    _HANDLE_CACHE.clear_memory()


@pytest.mark.parametrize("device", get_available_devices())
def test_make_cache_distinguishes_ceil_mode_different_shapes(device):
    """Sanity case: h=64, r=3 gives DIFFERENT output shapes for the two
    ceil_mode values (False -> (21,21), True -> (22,22)). Cache must MISS
    on the second call. (Even a shape-naive cache would miss here.)"""
    torch.manual_seed(0)
    x = torch.randn(32, 3, 64, 64, dtype=torch.float16, device=device)

    out_false = max_pool2d(x, (3, 3), ceil_mode=False)
    out_true = max_pool2d(x, (3, 3), ceil_mode=True)

    assert out_false.shape == (32, 3, 21, 21)
    assert out_true.shape == (32, 3, 22, 22)
    assert torch.allclose(out_false, F.max_pool2d(x, (3, 3), ceil_mode=False))
    assert torch.allclose(out_true, F.max_pool2d(x, (3, 3), ceil_mode=True))
    # Two distinct ceil_mode values -> two L1 entries
    assert _HANDLE_CACHE.memory_size == 2


@pytest.mark.parametrize("device", get_available_devices())
def test_make_cache_distinguishes_ceil_mode_same_shape(device):
    """Key regression test: h=63, r=3 gives IDENTICAL output shape (21,21)
    for both ceil_mode values. A shape-naive cache would mistakenly HIT
    on the second call and return the wrong kernel (the False kernel,
    which uses floor_mode=True, when True was requested, which uses
    floor_mode=False).

    The cache MUST still MISS: tensors tuple contains the raw bool
    ceil_mode, whose repr() distinguishes True from False, and the
    cache key is content-sensitive to that repr()."""
    torch.manual_seed(0)
    x = torch.randn(32, 3, 63, 63, dtype=torch.float16, device=device)

    out_false = max_pool2d(x, (3, 3), ceil_mode=False)
    out_true = max_pool2d(x, (3, 3), ceil_mode=True)

    # Output shapes are identical -- shape alone cannot disambiguate
    assert out_false.shape == out_true.shape == (32, 3, 21, 21)
    # But values DO differ between the two kernels (different floor_mode
    # in the arrangement), and each matches its reference
    assert torch.allclose(out_false, F.max_pool2d(x, (3, 3), ceil_mode=False))
    assert torch.allclose(out_true, F.max_pool2d(x, (3, 3), ceil_mode=True))
    # Cache must hold 2 distinct entries (one per ceil_mode value)
    assert _HANDLE_CACHE.memory_size == 2


@pytest.mark.parametrize("device", get_available_devices())
def test_make_cache_reuses_unchanged_ceil_mode(device):
    """Verify cache HIT path: a second call with the same ceil_mode
    must reuse the cached entry (L1 size unchanged)."""
    torch.manual_seed(0)
    x = torch.randn(32, 3, 64, 64, dtype=torch.float16, device=device)

    out1 = max_pool2d(x, (3, 3), ceil_mode=False)
    assert _HANDLE_CACHE.memory_size == 1  # first call: miss + put

    out2 = max_pool2d(x, (3, 3), ceil_mode=False)
    assert _HANDLE_CACHE.memory_size == 1  # second call: hit, no new entry
    assert torch.allclose(out1, out2)
