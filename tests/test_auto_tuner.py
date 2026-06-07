import time

import pytest

from ninetoothed.auto_tuner import AutoTuner
from tests.utils import get_available_devices


@pytest.fixture
def auto_tuner_factory(tmp_path, monkeypatch):
    """Factory: each call yields a fresh AutoTuner backed by tmp_path.

    Avoids polluting ~/.ninetoothed across pytest runs / parametrize rows.
    """
    import ninetoothed.auto_tuner as at_mod
    monkeypatch.setattr(at_mod, "CACHE_DIR", tmp_path / ".ninetoothed")
    (tmp_path / ".ninetoothed").mkdir()
    yield lambda: AutoTuner((_foo, _bar), (_foo.__name__, _bar.__name__))


@pytest.mark.parametrize("_", get_available_devices())
@pytest.mark.parametrize("kwargs", ({"a": 2, "b": 4}, {"a": 2, "b": 4, "c": 6, "d": 8}))
@pytest.mark.parametrize("args", ((1,), (1, 3, 5)))
def test_auto_tuner(args, kwargs, _, auto_tuner_factory):
    auto_tuner = auto_tuner_factory()

    # Initial state: timings dict is empty (fresh cache).
    assert auto_tuner._timings == {key: {} for key in auto_tuner._keys}

    first_time_start_time = time.perf_counter()

    auto_tuner(*args, **kwargs)

    first_time_end_time = time.perf_counter()

    first_time_elapsed_time = first_time_end_time - first_time_start_time

    # After benchmarking, timings should be populated for every func + arg.
    arg_key = auto_tuner._make_arg_key(args, kwargs)

    for func_key in auto_tuner._keys:
        assert arg_key in auto_tuner._timings[func_key]
    assert arg_key in auto_tuner._best_func

    second_time_start_time = time.perf_counter()

    auto_tuner(*args, **kwargs)

    second_time_end_time = time.perf_counter()

    second_time_elapsed_time = second_time_end_time - second_time_start_time

    # Cached second call must be substantially faster.
    assert second_time_elapsed_time < first_time_elapsed_time

    best_func = auto_tuner._best_func[arg_key]

    if _foo_delay(*args, **kwargs) < _bar_delay(*args, **kwargs):
        assert best_func is _foo
    else:
        assert best_func is _bar


def test_auto_tuner_persists_across_instances(auto_tuner_factory):
    """Re-instantiation should load timings from disk, skipping re-benchmark."""
    tuner1 = auto_tuner_factory()
    tuner1(1, 2, 3, a=4)

    tuner2 = auto_tuner_factory()

    # Both instances see the same persisted timings loaded from disk.
    assert tuner2._timings == tuner1._timings
    assert tuner2._timings  # not empty


def _foo_delay(*args, **kwargs):
    return 0.001 * (2 * len(args) + len(kwargs))


def _bar_delay(*args, **kwargs):
    return 0.001 * (len(args) + 2 * len(kwargs))


def _foo(*args, **kwargs):
    time.sleep(_foo_delay(*args, **kwargs))


def _bar(*args, **kwargs):
    time.sleep(_bar_delay(*args, **kwargs))
