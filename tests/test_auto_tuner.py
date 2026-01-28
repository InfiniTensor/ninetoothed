import time

import pytest

from ninetoothed.auto_tuner import AutoTuner
from tests.utils import get_available_devices


@pytest.mark.parametrize("_", get_available_devices())
@pytest.mark.parametrize("kwargs", ({"a": 2, "b": 4}, {"a": 2, "b": 4, "c": 6, "d": 8}))
@pytest.mark.parametrize("args", ((1,), (1, 3, 5)))
def test_auto_tuner(args, kwargs, _):
    auto_tuner = AutoTuner((_foo, _bar), (_foo.__name__, _bar.__name__))

    assert not auto_tuner._get_func_cache_path(_foo).exists()

    assert not auto_tuner._get_func_cache_path(_bar).exists()

    assert not auto_tuner._cache_path.exists()

    first_time_start_time = time.perf_counter()

    auto_tuner(*args, **kwargs)

    first_time_end_time = time.perf_counter()

    first_time_elapsed_time = first_time_end_time - first_time_start_time

    assert auto_tuner._get_func_cache_path(_foo).exists()

    assert auto_tuner._get_func_cache_path(_bar).exists()

    assert auto_tuner._cache_path.exists()

    second_time_start_time = time.perf_counter()

    auto_tuner(*args, **kwargs)

    second_time_end_time = time.perf_counter()

    second_time_elapsed_time = second_time_end_time - second_time_start_time

    assert second_time_elapsed_time < first_time_elapsed_time

    auto_tuner._get_func_cache_path(_foo).unlink()

    auto_tuner._get_func_cache_path(_bar).unlink()

    auto_tuner._cache_path.unlink()

    best_func = auto_tuner._best_func[auto_tuner._make_arg_key(args, kwargs)]

    if _foo_delay(*args, **kwargs) < _bar_delay(*args, **kwargs):
        assert best_func is _foo
    else:
        assert best_func is _bar


def _foo_delay(*args, **kwargs):
    return 0.001 * (2 * len(args) + len(kwargs))


def _bar_delay(*args, **kwargs):
    return 0.001 * (len(args) + 2 * len(kwargs))


def _foo(*args, **kwargs):
    time.sleep(_foo_delay(*args, **kwargs))


def _bar(*args, **kwargs):
    time.sleep(_bar_delay(*args, **kwargs))
