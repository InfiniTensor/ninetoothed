"""Unified cache infrastructure for ninetoothed.

This module is the single source of truth for cache key derivation and the
two-tier (memory L1 + filesystem L2) Cache class. All ninetoothed cache
points should be built on top of it so the policy (content-sensitivity,
eviction, thread-safety, atomic disk writes) lives in one place.

Public API:
    - Cache: two-tier key/value cache.
    - hash_tensor_signature: content-sensitive hash of a ninetoothed.Tensor.
    - hash_function_source: content-sensitive hash of a callable.
    - hash_value: generic repr-based hash.
    - project_files_fingerprint: hash of all files in a directory.
"""

import functools
import hashlib
import inspect
import json
import os
import pathlib
import threading
from typing import Any, Callable, Optional

# Default cache root. Mirrors ninetoothed.generation.CACHE_DIR.
DEFAULT_CACHE_ROOT = pathlib.Path.home() / ".ninetoothed"


# ---------- Tensor fingerprint ----------


def hash_tensor_signature(tensor) -> tuple:
    """Content-stable fingerprint of a ninetoothed.Tensor.

    Captures only the *structural* identity (ndim, jagged_dim, padding value)
    of the tensor's symbolic layout -- NOT the symbol names bound to
    `Tensor.name` (which are instance-counter-bound and would give every
    freshly constructed Tensor a unique identity).

    Two Tensors that share the same structural layout hash equal, even if
    they were constructed in different `ninetoothed.make()` calls.
    """
    return (
        tensor.ndim,
        tensor.jagged_dim,
        tensor.other,
    )


# ---------- Function fingerprint ----------


def hash_function_source(func) -> str:
    """Content-sensitive SHA256 hash of a callable.

    `functools.partial` is unwrapped: the inner function's source is hashed
    together with `repr()` of the bound args/kwargs, so
    `partial(jagged_dim=1)` differs from `partial(jagged_dim=2)`.

    Falls back to a stable `id:` token (module, qualname, id) when
    `inspect.getsource` fails (lambdas, C builtins, REPL-defined code).

    Returns a 64-char hex digest prefixed with `src:` or `id:`.
    """
    partial_args = []
    while isinstance(func, functools.partial):
        partial_args.append((func.args, func.keywords))
        func = func.func

    try:
        src = inspect.getsource(func)
    except (OSError, TypeError):
        module = getattr(func, "__module__", "?")
        qualname = getattr(func, "__qualname__", "?")
        return "id:" + module + "." + qualname + "@" + str(id(func))

    h = hashlib.sha256()
    h.update(src.encode("utf-8"))
    if partial_args:
        h.update(repr(partial_args).encode("utf-8"))
    return "src:" + h.hexdigest()


def hash_value(value: Any) -> str:
    """Generic repr-based hash, stable for any repr-able Python value."""
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


# ---------- Cache class ----------


class Cache:
    """Two-tier cache: in-memory L1 (FIFO eviction) + filesystem L2 (optional).

    Usage:
        cache = Cache(namespace="auto_tuning", suffix=".json")
        value = cache.get(key, default=None)
        if value is None:
            value = ...expensive computation...
            cache.put(key, value)

    Disk layout: <cache_dir>/<sha256(repr(key))><suffix>. Disk format
    controlled by `serializer` / `deserializer` (default JSON).

    Pass neither `cache_dir` nor `namespace` to get a memory-only cache
    (no disk writes -- useful for per-process JIT handle caches whose
    values are not serializable).

    Thread-safe.
    """

    def __init__(
        self,
        namespace: Optional[str] = None,
        *,
        suffix: str = ".json",
        serializer: Optional[Callable[[Any], str]] = None,
        deserializer: Optional[Callable[[str], Any]] = None,
        cache_dir: Optional[pathlib.Path] = None,
        max_memory: int = 256,
    ):
        self._suffix = suffix
        self._serializer = serializer if serializer is not None else json.dumps
        self._deserializer = deserializer if deserializer is not None else json.loads
        self._max_memory = max_memory

        # Resolve disk directory.
        if cache_dir is not None:
            self._cache_dir = pathlib.Path(cache_dir)
        elif namespace is not None:
            self._cache_dir = DEFAULT_CACHE_ROOT / namespace
        else:
            self._cache_dir = None  # memory-only mode

        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._mem: dict = {}
        self._lock = threading.Lock()

    # ----- introspection -----

    @property
    def cache_dir(self) -> Optional[pathlib.Path]:
        """Disk directory backing this cache, or None for memory-only."""
        return self._cache_dir

    @property
    def is_memory_only(self) -> bool:
        return self._cache_dir is None

    @property
    def memory_size(self) -> int:
        with self._lock:
            return len(self._mem)

    # ----- key -> path -----

    def _path_for(self, key: Any) -> Optional[pathlib.Path]:
        if self._cache_dir is None:
            return None
        h = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()
        return self._cache_dir / (h + self._suffix)

    # ----- core API -----

    def contains(self, key: Any) -> bool:
        """True iff `key` is in L1 or L2."""
        with self._lock:
            if key in self._mem:
                return True
        path = self._path_for(key)
        return path is not None and path.exists()

    def get(self, key: Any, default: Any = None) -> Any:
        """L1 (mem) then L2 (disk). L2 hits are promoted to L1."""
        with self._lock:
            if key in self._mem:
                return self._mem[key]

        path = self._path_for(key)
        if path is None or not path.exists():
            return default

        try:
            value = self._deserializer(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return default

        with self._lock:
            if len(self._mem) >= self._max_memory:
                self._mem.pop(next(iter(self._mem)))
            self._mem[key] = value
        return value

    def put(self, key: Any, value: Any) -> None:
        """Write L1 (mem) and L2 (disk). Disk failure leaves L1 intact.

        Disk writes are atomic: serialize to a sibling `.tmp` file, fsync,
        then rename over the target. This prevents a concurrent reader from
        observing a half-written value if the writer is killed mid-write
        (e.g. crash, OOM kill, or power loss).
        """
        with self._lock:
            if len(self._mem) >= self._max_memory:
                self._mem.pop(next(iter(self._mem)))
            self._mem[key] = value

        path = self._path_for(key)
        if path is not None:
            self._atomic_write(path, value)

    def _atomic_write(self, path: pathlib.Path, value: Any) -> None:
        """Serialize, write to `<path>.tmp`, fsync, then rename over `path`.

        Failure modes (OSError, TypeError, ValueError from serializer) leave
        both L1 and the on-disk state intact -- no half-written file is
        visible to readers, and the L1 entry is still authoritative in-process.
        """
        try:
            payload = self._serializer(value)
        except (TypeError, ValueError):
            return
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except OSError:
            # Best-effort cleanup of the leftover .tmp file.
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass

    def delete(self, key: Any) -> None:
        """Remove from L1 and L2. Missing entries are silently ignored."""
        with self._lock:
            self._mem.pop(key, None)
        path = self._path_for(key)
        if path is not None and path.exists():
            try:
                path.unlink()
            except OSError:
                pass

    def clear_memory(self) -> None:
        """Clear L1 only; L2 (disk) is preserved for cross-process sharing."""
        with self._lock:
            self._mem.clear()


# ---------- Project fingerprint ----------


def project_files_fingerprint(
    directory: pathlib.Path, exclude_suffixes=(".pyc",)
) -> str:
    """SHA256 fingerprint over all files under `directory`.

    Used to namespace caches by the ninetoothed installation version: when
    the package code changes, the fingerprint changes and old cache files
    are effectively ignored (different subdir, different key prefix).

    `rglob` is sorted for determinism; `.pyc` is excluded by default.
    """
    h = hashlib.sha256()
    paths = sorted(
        p
        for p in pathlib.Path(directory).rglob("*")
        if p.is_file() and p.suffix not in exclude_suffixes
    )
    for p in paths:
        h.update(str(p.relative_to(directory)).encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()
