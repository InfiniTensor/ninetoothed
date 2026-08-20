import time
import uuid
from types import SimpleNamespace

import pytest
import torch

from ninetoothed.auto_tuner import AutoTuner, _default_benchmark
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

    def fail_after_benchmark(*_args, **_kwargs):
        raise RuntimeError("Winner invocation failed.")

    tuner = AutoTuner(
        (fail_after_benchmark,),
        ("winner",),
        benchmark=lambda *_args: 1.0,
        cache_namespace=f"winner_failure_{uuid.uuid4().hex}",
    )

    with pytest.raises(RuntimeError, match="Winner invocation failed"):
        tuner(1)

    assert tuner._best_func == {}


def _runtime_tensor():
    return SimpleNamespace(
        shape=(1,),
        dtype="float32",
        device=SimpleNamespace(type="vendor_test"),
        stride=lambda: (1,),
    )


def _event_runtime(
    calls,
    *,
    elapsed_ms=10.0,
    elapsed_error=None,
    synchronize=None,
    event_synchronize=None,
):
    events = []

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing
            events.append(self)

        def record(self):
            calls.append("record")

        def synchronize(self):
            calls.append("event-synchronize")

            if event_synchronize is not None:
                event_synchronize()

        def elapsed_time(self, other):
            assert other in events

            if elapsed_error is not None:
                raise elapsed_error
            return elapsed_ms

    attributes = {"Event": Event}

    if synchronize is not None:
        attributes["synchronize"] = synchronize
    return SimpleNamespace(**attributes)


def test_default_benchmark_uses_device_events(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch,
        "vendor_test",
        _event_runtime(calls, elapsed_ms=25.0),
        raising=False,
    )

    elapsed = _default_benchmark(
        lambda *_: calls.append("kernel"),
        (_runtime_tensor(),),
        {},
    )

    assert elapsed == 2.5
    assert calls.count("kernel") == 13
    assert calls.count("record") == 2
    assert calls.count("event-synchronize") == 1


@pytest.mark.parametrize("event_fallback", (False, True))
def test_default_benchmark_uses_synchronized_wall_clock(monkeypatch, event_fallback):
    calls = []

    def synchronize():
        calls.append("synchronize")

    runtime = SimpleNamespace(synchronize=synchronize)

    if event_fallback:
        runtime = _event_runtime(
            calls,
            elapsed_error=NotImplementedError(),
            synchronize=synchronize,
        )

    monkeypatch.setattr(torch, "vendor_test", runtime, raising=False)
    elapsed = _default_benchmark(
        lambda *_: calls.append("kernel"),
        (_runtime_tensor(),),
        {},
    )

    assert elapsed >= 0
    assert calls.count("kernel") == (23 if event_fallback else 13)
    assert calls.count("synchronize") == (3 if event_fallback else 2)


def test_default_benchmark_prefers_accelerator_after_cpu_scalar(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch,
        "vendor_test",
        _event_runtime(
            calls,
            synchronize=lambda: calls.append("accelerator-synchronize"),
        ),
        raising=False,
    )

    elapsed = _default_benchmark(
        lambda *_: calls.append("kernel"),
        (torch.tensor(1.0), _runtime_tensor()),
        {},
    )

    assert elapsed == 1.0
    assert calls.count("accelerator-synchronize") == 1


@pytest.mark.parametrize("synchronize_with", ("runtime", "event"))
def test_async_candidate_error_cannot_be_cached_as_winner(
    monkeypatch, synchronize_with
):
    calls = []
    pending_error = [False]

    def synchronize():
        if pending_error[0]:
            pending_error[0] = False
            raise RuntimeError("Asynchronous launch failed.")

    runtime_options = (
        {"synchronize": synchronize}
        if synchronize_with == "runtime"
        else {"event_synchronize": synchronize}
    )
    monkeypatch.setattr(
        torch,
        "vendor_test",
        _event_runtime(calls, **runtime_options),
        raising=False,
    )
    tensor = _runtime_tensor()

    def bad_candidate(*_args):
        pending_error[0] = True

    def good_candidate(*_args):
        pass

    tuner = AutoTuner(
        (bad_candidate, good_candidate),
        ("bad", "good"),
        cache_namespace=f"async_failure_{synchronize_with}_{uuid.uuid4().hex}",
    )
    tuner(tensor)

    arg_key = tuner._make_arg_key((tensor,), {})
    assert tuner._best_func[arg_key] is good_candidate
    assert tuner._candidate_timings["bad"][arg_key] == float("inf")


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
