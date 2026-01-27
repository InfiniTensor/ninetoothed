import hashlib
import json
import os

import triton

from ninetoothed.generation import CACHE_DIR


class AutoTuner:
    def __init__(self, funcs, keys):
        self._funcs = funcs

        self._keys = keys

        self._func_to_key = {func: key for func, key in zip(self._funcs, self._keys)}

        self._cache_dir = (
            _AUTO_TUNING_CACHE_DIR
            / f"{_project_key()}_triton_{triton.__version__.replace('.', '_')}"
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        auto_tuner_key = tuple(self._keys)
        cache_key = hashlib.sha256(str(auto_tuner_key).encode("utf-8")).hexdigest()
        self._cache_path = self._cache_dir / f"{cache_key}.json"

        if self._cache_path.exists():
            self._timings = json.loads(self._cache_path.read_text())
        else:
            self._timings = {key: {} for key in self._keys}

        self._best_func = {}

    def run(self, *args, **kwargs):
        if (arg_key := self._make_arg_key(args, kwargs)) in self._best_func:
            return self._best_func[arg_key](*args, **kwargs)

        timings = self._get_timings(args, kwargs)

        best_timing = min(timings)
        best_timing_index = timings.index(best_timing)
        best_func = self._funcs[best_timing_index]

        self._best_func[arg_key] = best_func

        return best_func(*args, **kwargs)

    def _get_timings(self, args, kwargs):
        if (arg_key := self._make_arg_key(args, kwargs)) in self._timings:
            return self._timings[arg_key]

        timings = [self._get_timing(func, args, kwargs) for func in self._funcs]

        self._timings[arg_key] = timings

        self._cache_path.write_text(json.dumps(self._timings))

        return timings

    def _get_timing(self, func, args, kwargs):
        func_key = self._func_to_key[func]

        data = self._timings[func_key]

        if (arg_key := self._make_arg_key(args, kwargs)) in data:
            return data[arg_key]

        cache_key = hashlib.sha256(str(func_key).encode("utf-8")).hexdigest()
        cache_path = self._cache_dir / f"{cache_key}.json"

        if cache_path.exists():
            data |= json.loads(cache_path.read_text())

        if arg_key in data:
            return data[arg_key]

        timing = triton.testing.do_bench(lambda: func(*args, **kwargs))

        data[arg_key] = timing

        cache_path.write_text(json.dumps(data))

        return timing

    def _make_arg_key(self, args, kwargs):
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
