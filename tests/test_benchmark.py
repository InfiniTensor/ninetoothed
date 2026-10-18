"""Verify the shared measurement contract without requiring accelerator hardware."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import ninetoothed.compiler.benchmark as timing


@pytest.mark.parametrize("method", ("graph", "event", "python", "unsynchronized"))
def test_benchmark_capability_fallbacks(monkeypatch, method):
    state = SimpleNamespace(
        active=None,
        elapsed=0.0,
        captures=[],
        calls=0,
        waits=0,
        device=0,
        stream="caller",
    )

    @contextmanager
    def device_scope(device):
        previous = state.device
        state.device = device.index

        try:
            yield
        finally:
            state.device = previous

    def new_stream():
        assert state.device == 1

        return "capture-on-device-1"

    @contextmanager
    def stream_scope(stream):
        previous = state.stream
        state.stream = stream

        try:
            yield
        finally:
            state.stream = previous

    class Graph:
        def __init__(self):
            self.calls = 0

        def replay(self):
            assert state.device == 1 and state.stream == "capture-on-device-1"
            state.elapsed = self.calls * 0.001 + 0.005

    class Event:
        def __init__(self, **_kwargs):
            pass

        def record(self):
            assert state.device == 1
            assert state.stream == (
                "capture-on-device-1" if method == "graph" else "caller"
            )

            if method == "python":
                raise RuntimeError("Events are unsupported on this external stream.")

        def synchronize(self):
            state.waits += 1

        def elapsed_time(self, _other):
            return state.elapsed

    @contextmanager
    def capture(graph, *, stream):
        assert stream == "capture-on-device-1"
        state.active = graph

        try:
            yield
        finally:
            state.captures.append(graph.calls)
            state.active = None

            if method == "event":
                # Model a graph context whose capture_end raises before it
                # restores the stream. The timer's outer scope must restore it.
                state.stream = "leaked-graph-stream"
                raise RuntimeError("Graph capture failed during exit.")

    def synchronize():
        if method == "unsynchronized":
            raise NotImplementedError("Device synchronization is unavailable.")

        state.waits += 1

    def function(_tensor):
        assert state.device == 1
        state.calls += 1

        if state.active is not None:
            state.active.calls += 1
        else:
            state.elapsed = 0.01

    runtime = SimpleNamespace(
        Graph=Graph,
        graph=capture,
        Event=Event,
        synchronize=synchronize,
        Stream=new_stream,
        stream=stream_scope,
        device=device_scope,
    )

    if method == "unsynchronized":
        runtime.Event = None
    elif method == "python":
        runtime.Graph = None

    monkeypatch.setattr(timing, "_BENCHMARKS", {})
    tensor = SimpleNamespace(device=SimpleNamespace(type="vendor_test", index=1))
    result = timing.benchmark_kernel(
        function, (tensor,), runtime=runtime, min_total_ms=0.01
    )
    assert isinstance(result, float) and result > 0
    assert state.calls >= 123
    assert state.device == 0 and state.stream == "caller"

    if method == "graph":
        assert state.captures[0] == 8
        assert state.captures[-1] >= 128
        assert result == pytest.approx(0.001, rel=0.01)
    elif method == "event":
        assert result == pytest.approx(0.001)
        assert state.waits >= 13
    elif method == "python":
        assert state.waits >= 13
    else:
        assert state.waits == 0


def test_benchmark_adapter_and_python_contract(monkeypatch):
    device_type = "vendor_test"
    monkeypatch.setattr(timing, "_BENCHMARKS", {})
    tensor = SimpleNamespace(device=SimpleNamespace(type=device_type))
    kwargs = {"output": tensor}
    calls = []

    def function(*, output):
        assert output is tensor
        calls.append(output)

    def adapter(function, args, supplied_kwargs, *, min_total_ms):
        assert args == () and supplied_kwargs is kwargs
        assert min_total_ms == 0.01
        function(**supplied_kwargs)

        return timing.BenchmarkResult(
            milliseconds=0.002, method="custom", synchronized=True
        )

    timing.register_device_benchmark(device_type, adapter)
    assert (
        timing.benchmark_kernel(
            function, (), kwargs, min_total_ms=0.01, runtime=SimpleNamespace()
        )
        == 0.002
    )
    assert len(calls) == 4

    def unsupported(*_args, **_kwargs):
        raise NotImplementedError("Use the standard fallback chain.")

    timing.register_device_benchmark(device_type, unsupported)
    result = timing.benchmark_kernel(
        function, (), kwargs, min_total_ms=0.01, runtime=SimpleNamespace()
    )
    assert isinstance(result, float) and result > 0

    with pytest.raises(ValueError, match="positive"):
        timing.benchmark_kernel(function, (), kwargs, min_total_ms=0)

    timing.register_device_benchmark(
        device_type, lambda *_args, **_kwargs: float("nan")
    )

    with pytest.raises(RuntimeError, match="invalid duration"):
        timing.benchmark_kernel(function, (), kwargs, runtime=SimpleNamespace())

    monkeypatch.setattr(timing, "_BENCHMARKS", {})
    import_module = __import__

    def without_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("The optional tensor package is not installed.")

        return import_module(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", without_torch)
    result = timing.benchmark_kernel(function, (), kwargs, min_total_ms=0.01)
    assert isinstance(result, float) and result > 0
