import math
import threading

from ninetoothed.compiler.cache import (
    CACHE_DIR,
    cache_lock,
    read_manifest,
    stable_digest,
    write_manifest,
)


class AutoTuner:
    """Select a runtime candidate using an injectable benchmark strategy."""

    def __init__(
        self,
        funcs,
        keys,
        *,
        benchmark=None,
        cache_namespace=None,
        validator=None,
    ):
        self._funcs = tuple(funcs)
        self._keys = tuple(keys)

        if not self._funcs or len(self._funcs) != len(self._keys):
            raise ValueError("AutoTuner requires one key for every candidate.")

        self._benchmark = benchmark or _default_benchmark
        self._validator = validator
        self._lock = threading.Lock()
        self._key_ids = tuple(_candidate_id(key) for key in self._keys)
        self._func_to_key = {func: key for func, key in zip(self._funcs, self._key_ids)}
        namespace = cache_namespace or _default_cache_namespace()
        self._cache_dir = _AUTO_TUNING_CACHE_DIR / stable_digest(
            {
                "schema": 2,
                "namespace": namespace,
                "candidate_ids": self._key_ids,
            }
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path = self._cache_dir / "selection.json"

        manifest = read_manifest(self._cache_path) or {}
        self._selection_timings = dict(manifest.get("timings", {}))
        self._candidate_timings = {key: {} for key in self._key_ids}

        self._best_func = {}

    def __call__(self, *args, **kwargs):
        if self._validator is not None:
            self._validator(args, kwargs)

        arg_key = type(self)._make_arg_key(args, kwargs)

        with self._lock:
            best_func = self._best_func.get(arg_key)

            if best_func is None:
                timings = self._get_timings(args, kwargs)

                if all(math.isinf(timing) for timing in timings):
                    raise RuntimeError(self._all_candidates_failed_message(arg_key))

                best_timing = min(timings)
                best_timing_index = timings.index(best_timing)
                best_func = self._funcs[best_timing_index]
                result = best_func(*args, **kwargs)
                self._best_func[arg_key] = best_func

                return result

        return best_func(*args, **kwargs)

    def _get_timings(self, args, kwargs):
        arg_key = type(self)._make_arg_key(args, kwargs)

        update_lock = self._cache_path.with_suffix(".update")

        with cache_lock(update_lock):
            manifest = read_manifest(self._cache_path) or {}
            self._selection_timings |= dict(manifest.get("timings", {}))

            if arg_key in self._selection_timings:
                cached = self._selection_timings[arg_key]

                if not all(math.isinf(timing) for timing in cached):
                    return cached

                del self._selection_timings[arg_key]

            timings = [self._get_timing(func, args, kwargs) for func in self._funcs]
            self._selection_timings[arg_key] = timings
            write_manifest(
                self._cache_path,
                {"schema": 2, "timings": self._selection_timings},
            )

        return timings

    def _get_timing(self, func, args, kwargs):
        func_key = self._func_to_key[func]

        data = self._candidate_timings[func_key]

        arg_key = type(self)._make_arg_key(args, kwargs)

        if arg_key in data and not math.isinf(data[arg_key]):
            return data[arg_key]

        data.pop(arg_key, None)

        cache_path = self._get_func_cache_path(func)

        update_lock = cache_path.with_suffix(".update")

        with cache_lock(update_lock):
            manifest = read_manifest(cache_path) or {}
            data |= dict(manifest.get("timings", {}))

            if arg_key in data and not math.isinf(data[arg_key]):
                return data[arg_key]

            data.pop(arg_key, None)

            failure = None

            try:
                timing = self._benchmark(func, args, kwargs)
            except Exception as exc:  # noqa: BLE001
                timing = float("inf")
                failure = f"{type(exc).__name__}: {exc}"

            data[arg_key] = timing
            failures = dict((read_manifest(cache_path) or {}).get("failures", {}))

            if failure is not None:
                failures[arg_key] = failure
            else:
                failures.pop(arg_key, None)

            write_manifest(
                cache_path,
                {"schema": 2, "timings": data, "failures": failures},
            )

        return timing

    def _get_func_cache_path(self, func):
        func_key = self._func_to_key[func]
        cache_key = stable_digest({"schema": 2, "candidate_id": func_key})
        cache_path = self._cache_dir / f"{cache_key}.json"

        return cache_path

    def _all_candidates_failed_message(self, arg_key):
        failures = []

        for func, candidate_id in zip(self._funcs, self._key_ids):
            manifest = read_manifest(self._get_func_cache_path(func)) or {}
            reason = dict(manifest.get("failures", {})).get(
                arg_key, "benchmark returned an infinite timing"
            )
            failures.append(f"{candidate_id}: {reason}")

        return "All auto-tuning candidates failed: " + "; ".join(failures)

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
        stride = tuple(tensor.stride()) if hasattr(tensor, "stride") else None
        device = str(getattr(tensor, "device", None))

        return (
            f"tensor(shape={tuple(tensor.shape)}, "
            f"stride={stride}, "
            f"dtype={str(tensor.dtype).split('.')[-1]}, "
            f"device={device})"
        )


_AUTO_TUNING_CACHE_DIR = CACHE_DIR / "auto_tuning"


def _default_benchmark(function, args, kwargs):
    import torch

    for _ in range(3):
        function(*args, **kwargs)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    for _ in range(10):
        function(*args, **kwargs)

    end.record()
    end.synchronize()

    return start.elapsed_time(end) / 10


def _default_cache_namespace():
    import torch

    return f"cuda_event_torch_{torch.__version__.replace('.', '_')}"


def _candidate_id(key):
    if isinstance(key, (str, int, float, bool, type(None))):
        return str(key)
    return repr(key)
