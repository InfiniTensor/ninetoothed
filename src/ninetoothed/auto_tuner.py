"""Auto-tuner for ninetoothed kernels.

Migrated to the unified Cache API (ninetoothed._cache.Cache). All timings
are stored as a single JSON per (project, triton-version) directory; the
prior per-func split-file layout is gone -- users with existing caches
should run `rm -rf ~/.ninetoothed/auto_tuning/`.
"""

import os

import triton

from ninetoothed._cache import Cache, project_files_fingerprint
from ninetoothed.aot import _KernelLaunchError
from ninetoothed.generation import CACHE_DIR


class AutoTuner:
    def __init__(self, funcs, keys):
        self._funcs = funcs

        self._keys = keys

        self._func_to_key = {func: key for func, key in zip(self._funcs, self._keys)}

        # Disk layout: <CACHE_DIR>/auto_tuning/<project_key>_triton_<ver>/
        # The project_key isolates caches across ninetoothed versions.
        subdir = f"{_project_key()}_triton_{triton.__version__.replace('.', '_')}"
        disk_dir = CACHE_DIR / "auto_tuning" / subdir

        self._cache = Cache(
            cache_dir=disk_dir,
            suffix=".json",
            max_memory=64,
        )

        # The full timings dict is stored under a single sentinel key.
        self._disk_key = ("_all_timings_",)
        loaded = self._cache.get(self._disk_key, default={})
        if not loaded:
            loaded = {key: {} for key in self._keys}
        self._timings = loaded

        self._best_func = {}

    def __call__(self, *args, **kwargs):
        if (arg_key := type(self)._make_arg_key(args, kwargs)) in self._best_func:
            return self._best_func[arg_key](*args, **kwargs)

        timings = self._get_timings(args, kwargs)

        best_timing = min(timings)
        best_timing_index = timings.index(best_timing)
        best_func = self._funcs[best_timing_index]

        self._best_func[arg_key] = best_func

        return best_func(*args, **kwargs)

    def _get_timings(self, args, kwargs):
        if (arg_key := type(self)._make_arg_key(args, kwargs)) in self._timings:
            return self._timings[arg_key]

        timings = [self._get_timing(func, args, kwargs) for func in self._funcs]

        self._timings[arg_key] = timings
        self._save()
        return timings

    def _get_timing(self, func, args, kwargs):
        func_key = self._func_to_key[func]

        data = self._timings[func_key]

        if (arg_key := type(self)._make_arg_key(args, kwargs)) in data:
            return data[arg_key]

        try:
            timing = triton.testing.do_bench(lambda: func(*args, **kwargs))
        except _KernelLaunchError:
            timing = float("inf")

        data[arg_key] = timing
        self._save()
        return timing

    def _save(self):
        """Persist the full timings dict (L1 + L2)."""
        self._cache.put(self._disk_key, self._timings)

    @staticmethod
    def _make_arg_key(args, kwargs):
        key_parts = []

        def _make_key(arg):
            if hasattr(arg, "shape") and hasattr(arg, "dtype"):
                return AutoTuner._make_tensor_key(arg)

            return str(arg)

        for arg in args:
            key_parts.append(_make_key(arg))

        for key, arg in sorted(kwargs.items()):
            key_parts.append(f"{key}={_make_key(arg)}")

        arg_key = ", ".join(key_parts)

        return arg_key

    @staticmethod
    def _make_tensor_key(tensor):
        return f"tensor(shape={tuple(tensor.shape)}, dtype={str(tensor.dtype).split('.')[-1]})"


def _project_key():
    """Fingerprint of the ninetoothed source tree, used to namespace caches
    across ninetoothed installation versions."""
    return project_files_fingerprint(os.path.dirname(os.path.abspath(__file__)))
