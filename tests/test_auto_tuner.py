import time
import uuid

import pytest

from ninetoothed.auto_tuner import AutoTuner
from tests.utils import get_available_devices


@pytest.mark.parametrize("_", get_available_devices())
@pytest.mark.parametrize("kwargs", ({"a": 2, "b": 4}, {"a": 2, "b": 4, "c": 6, "d": 8}))
@pytest.mark.parametrize("args", ((1,), (1, 3, 5)))
def test_auto_tuner(args, kwargs, _):
    benchmark_calls = []

    def benchmark(function, candidate_args, candidate_kwargs):
        benchmark_calls.append(function)

        return _deterministic_benchmark(function, candidate_args, candidate_kwargs)

    namespace = f"test_{uuid.uuid4().hex}"
    auto_tuner = AutoTuner(
        (_foo, _bar),
        (_foo.__name__, _bar.__name__),
        benchmark=benchmark,
        cache_namespace=namespace,
    )

    assert not auto_tuner._get_func_cache_path(_foo).exists()

    assert not auto_tuner._get_func_cache_path(_bar).exists()

    assert not auto_tuner._cache_path.exists()

    auto_tuner(*args, **kwargs)
    assert benchmark_calls == [_foo, _bar]

    cache_hit_benchmarks = []
    cached_tuner = AutoTuner(
        (_foo, _bar),
        (_foo.__name__, _bar.__name__),
        benchmark=lambda *call: cache_hit_benchmarks.append(call) or 0.0,
        cache_namespace=namespace,
    )
    cached_tuner(*args, **kwargs)
    assert not cache_hit_benchmarks

    assert auto_tuner._get_func_cache_path(_foo).exists()

    assert auto_tuner._get_func_cache_path(_bar).exists()

    assert auto_tuner._cache_path.exists()

    auto_tuner(*args, **kwargs)
    assert benchmark_calls == [_foo, _bar]

    auto_tuner._get_func_cache_path(_foo).unlink()

    auto_tuner._get_func_cache_path(_bar).unlink()

    auto_tuner._cache_path.unlink()

    best_func = auto_tuner._best_func[auto_tuner._make_arg_key(args, kwargs)]

    if _foo_delay(*args, **kwargs) < _bar_delay(*args, **kwargs):
        assert best_func is _foo
    else:
        assert best_func is _bar


@pytest.mark.parametrize("_", get_available_devices())
def test_auto_tuner_reports_every_failed_candidate(_):
    def fail(function, args, kwargs):
        del args, kwargs
        raise RuntimeError(f"Candidate {function.__name__} failed.")

    tuner = AutoTuner(
        (_foo, _bar),
        ("first", "second"),
        benchmark=fail,
        cache_namespace=f"failure_{uuid.uuid4().hex}",
    )

    with pytest.raises(RuntimeError, match="first.*_foo failed.*second.*_bar failed"):
        tuner(1)


def _foo_delay(*args, **kwargs):
    return 0.001 * (2 * len(args) + len(kwargs))


def _bar_delay(*args, **kwargs):
    return 0.001 * (len(args) + 2 * len(kwargs))


def _foo(*args, **kwargs):
    time.sleep(_foo_delay(*args, **kwargs))


def _bar(*args, **kwargs):
    time.sleep(_bar_delay(*args, **kwargs))


def _deterministic_benchmark(function, args, kwargs):
    if function is _foo:
        return _foo_delay(*args, **kwargs)

    if function is _bar:
        return _bar_delay(*args, **kwargs)

    raise AssertionError(f"Unexpected tuning candidate: {function!r}.")
