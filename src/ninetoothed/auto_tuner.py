import hashlib
import json
import os

import triton

from ninetoothed.generation import CACHE_DIR


class AutoTuner:
    def __init__(self, funcs, keys):
        self._funcs = funcs
        self._keys = keys
        self._best_func = {}
        self._timings = {}
        self._cache_results = True
        self._cache_dir = (
            _AUTO_TUNING_CACHE_DIR
            / f"{_project_key()}_triton_{triton.__version__.replace('.', '_')}"
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def run(self, *args, **kwargs):
        if len(self._funcs) > 1:
            param_key = self._make_param_key(args, kwargs)
            keys = tuple(list(self._keys) + [param_key])

            if keys not in self._best_func:

                def benchmark():
                    best_time = float("inf")
                    best_func = None

                    for idx, func in enumerate(self._funcs):
                        key = tuple([self._keys[idx], param_key])

                        if key in self._timings:
                            func_time = self._timings[key]
                        else:
                            func_time = triton.testing.do_bench(
                                lambda: func(*args, **kwargs)
                            )
                            self._timings[key] = func_time

                        if func_time < best_time:
                            best_time = func_time
                            best_func = func

                    self._best_func[keys] = best_func

                if self._cache_results:
                    self._check_disk_cache(keys, benchmark)
                else:
                    benchmark()

            return self._best_func[keys](*args, **kwargs)

        else:
            return self._funcs[0](*args, **kwargs)

    def _check_disk_cache(self, tuning_key, bench_fn):
        cache_key = hashlib.sha256(str(tuning_key).encode("utf-8")).hexdigest()
        cache_path = self._cache_dir / f"{cache_key}.json"

        if cache_path.exists():
            data = json.loads(cache_path.read_text())
            best_func_idx = data.get("best_func_idx")
            if best_func_idx is not None and 0 <= best_func_idx < len(self._funcs):
                self._best_func[tuning_key] = self._funcs[best_func_idx]

            timings = data.get("timings", {})

            for key_str, timing in timings.items():
                idx = int(key_str)
                if 0 <= idx < len(self._keys):
                    self._timings[tuple([self._keys[idx]])] = timing

            return True

        bench_fn()

        best_func = self._best_func.get(tuning_key)
        if best_func is not None:
            best_func_idx = None
            for idx, func in enumerate(self._funcs):
                if func is best_func:
                    best_func_idx = idx
                    break

            if best_func_idx is not None:
                timings = {}
                for idx, key in enumerate(self._keys):
                    func_key = tuple([key])
                    if func_key in self._timings:
                        timings[str(idx)] = self._timings[func_key]

                data = {
                    "tuning_key": str(tuning_key),
                    "best_func_idx": best_func_idx,
                    "timings": timings,
                }
                cache_path.write_text(json.dumps(data))

        return False

    def _make_param_key(self, args, kwargs):
        key_parts = []

        for arg in args:
            if hasattr(arg, "shape") and hasattr(arg, "dtype"):
                key_parts.append(f"shape{tuple(arg.shape)}")
                key_parts.append(f"dtype{str(arg.dtype).split('.')[-1]}")
            elif isinstance(arg, (int, float, str, bool)):
                key_parts.append(f"{type(arg).__name__}{arg}")

        for k, v in sorted(kwargs.items()):
            if hasattr(v, "shape") and hasattr(v, "dtype"):
                key_parts.append(f"{k}_shape{tuple(v.shape)}")
                key_parts.append(f"{k}_dtype{str(v.dtype).split('.')[-1]}")
            elif isinstance(v, (int, float, str, bool)):
                key_parts.append(f"{k}_{v}")

        return "_".join(key_parts) if key_parts else "default"


_AUTO_TUNING_CACHE_DIR = CACHE_DIR / "auto_tuning"

_FILE_PATH = os.path.abspath(__file__)

_PARENT_DIR = os.path.dirname(_FILE_PATH)


def _project_key():
    consolidated_hash = hashlib.sha256()

    for dirpath, dirnames, filenames in os.walk(_PARENT_DIR):
        dirnames.sort()
        filenames.sort()

        for filename in filenames:
            file_path = os.path.join(dirpath, filename)

            if (
                not os.path.isfile(file_path)
                or os.path.splitext(file_path)[1] == ".pyc"
            ):
                continue

            file_hash = _calculate_file_hash(file_path)
            consolidated_hash.update(file_hash.encode("utf-8"))

    return consolidated_hash.hexdigest()


def _calculate_file_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
