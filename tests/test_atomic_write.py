"""Tests for atomic disk writes in ninetoothed._cache.Cache.put.

Verifies the contract:
  - Successful put leaves exactly one file at the target path (no .tmp residue).
  - A mid-write failure (simulated by patching the file system) leaves the
    previous file intact and no half-written replacement.
  - The in-process L1 cache is unaffected by disk failure.
"""

import json

from ninetoothed._cache import Cache


def test_successful_put_leaves_no_tmp_file(tmp_path):
    c = Cache(cache_dir=tmp_path, suffix=".json", max_memory=4)
    c.put("k", {"v": 1})

    # Exactly one .json file, no .tmp residue
    files = list(tmp_path.glob("*"))
    assert len(files) == 1
    assert files[0].suffix == ".json"
    assert not list(tmp_path.glob("*.tmp"))


def test_put_overwrites_existing_file_atomically(tmp_path):
    c = Cache(cache_dir=tmp_path, suffix=".json", max_memory=4)
    c.put("k", "first")
    c.put("k", "second")

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text()) == "second"


def test_disk_failure_preserves_l1(tmp_path):
    """If the serializer raises, L1 must still have the value."""

    def bad_serializer(_):
        raise TypeError("nope")

    c = Cache(
        cache_dir=tmp_path,
        suffix=".json",
        serializer=bad_serializer,
        max_memory=4,
    )
    c.put("k", "v")
    # L1 was updated before the (failing) disk write, so it's still in memory
    assert c.get("k") == "v"


def test_disk_failure_leaves_no_tmp_residue(tmp_path):
    def bad_serializer(_):
        raise TypeError("nope")

    c = Cache(
        cache_dir=tmp_path,
        suffix=".json",
        serializer=bad_serializer,
        max_memory=4,
    )
    c.put("k", "v")
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob("*.json"))


def test_oserror_during_rename_keeps_old_file(tmp_path):
    """Simulate a rename() failure: original file must survive."""
    # First, write a real value through the cache to establish a real file.
    c = Cache(cache_dir=tmp_path, suffix=".json", max_memory=4)
    c.put("k", "original")

    path = c._path_for("k")
    assert path.read_text()  # file exists

    # Now patch os.replace to fail
    import ninetoothed._cache as cache_mod

    original_replace = cache_mod.os.replace

    def fail_replace(src, dst):
        raise OSError("simulated rename failure")

    cache_mod.os.replace = fail_replace
    try:
        c.put("k", "new_value")
    finally:
        cache_mod.os.replace = original_replace

    # The on-disk file should still contain the ORIGINAL value
    assert json.loads(path.read_text()) == "original"
    # And no .tmp residue
    assert not list(tmp_path.glob("*.tmp"))


def test_no_tmp_residue_under_concurrent_writes(tmp_path):
    """Many threads writing distinct keys should not leave .tmp files lying around."""
    c = Cache(cache_dir=tmp_path, suffix=".json", max_memory=4)

    def writer(i):
        for j in range(20):
            c.put(f"k_{i}_{j}", j)

    import threading

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Every key persisted; no .tmp residue
    assert not list(tmp_path.glob("*.tmp"))
    json_files = list(tmp_path.glob("*.json"))
    assert len(json_files) > 0

    # A fresh instance can read them all
    c2 = Cache(cache_dir=tmp_path, suffix=".json", max_memory=4)
    for i in range(4):
        for j in range(20):
            assert c2.get(f"k_{i}_{j}") == j
