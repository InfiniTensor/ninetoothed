"""Internal timing with graph, device-event, and synchronized Python fallbacks.

``benchmark_kernel`` returns the measured milliseconds per invocation.
Device-graph batching amortizes submission gaps for short kernels.
Event/Python fallbacks preserve usability on new platforms without promising
the same short-kernel precision.
"""

import time
from contextlib import nullcontext
from dataclasses import dataclass
from math import ceil, isfinite
from statistics import median

_BENCHMARKS = {}


@dataclass(frozen=True, kw_only=True)
class BenchmarkResult:
    """Milliseconds per invocation and the method actually used to measure it."""

    milliseconds: float
    method: str
    synchronized: bool | None


def register_device_benchmark(device_type, benchmark):
    """Register an internal platform timer before tuning on this device type.

    The adapter returns milliseconds or a BenchmarkResult. Raise
    NotImplementedError to request the standard fallback chain.
    """
    if not callable(benchmark):
        raise TypeError("A device benchmark must be callable.")

    _BENCHMARKS[str(device_type).lower()] = benchmark


def benchmark_kernel(function, args, kwargs=None, *, min_total_ms=20, runtime=None):
    """Warm up, measure, and return milliseconds per invocation.

    The preference is a registered adapter, device graphs, device events, then
    Python perf_counter. Available synchronization is used around host timing.
    Without synchronization, Python timing measures submission rather than
    completed asynchronous work.
    The caller owns buffers and must supply a repeatable operation.
    """
    if not isfinite(min_total_ms) or min_total_ms <= 0:
        raise ValueError("Min_total_ms must be finite and positive.")

    kwargs = {} if kwargs is None else kwargs
    argument_devices = [
        device
        for value in (*args, *kwargs.values())
        if (device := getattr(value, "device", None)) is not None
    ]
    devices = [device for device in argument_devices if _device_type(device) != "cpu"]

    if devices and any(device != devices[0] for device in devices):
        raise ValueError("Timing requires arguments on one accelerator.")

    device = devices[0] if devices else next(iter(argument_devices), None)

    if runtime is None:
        runtime = _device_runtime(args, kwargs)

    context = getattr(runtime, "device", None)
    scope = context(device) if devices and callable(context) else nullcontext()

    with scope:
        for _ in range(3):
            function(*args, **kwargs)

        _synchronize(runtime)
        adapter = _BENCHMARKS.get(_device_type(device))
        result = None

        if adapter is not None:
            try:
                result = adapter(function, args, kwargs, min_total_ms=min_total_ms)

                if not isinstance(result, BenchmarkResult):
                    result = BenchmarkResult(
                        milliseconds=result, method="custom", synchronized=None
                    )
            except NotImplementedError:
                pass

        if result is None:
            try:
                result = _graph_benchmark(function, args, kwargs, runtime, min_total_ms)
            except NotImplementedError:
                try:
                    result = _event_benchmark(
                        function, args, kwargs, runtime, min_total_ms
                    )
                except NotImplementedError:
                    result = _python_benchmark(
                        function, args, kwargs, runtime, min_total_ms
                    )

    if not isfinite(result.milliseconds) or result.milliseconds <= 0:
        raise RuntimeError("The benchmark returned an invalid duration.")

    return result.milliseconds


def _samples(measure, calls, min_total_ms):
    samples = []
    total_ms = 0.0

    while total_ms < min_total_ms or len(samples) < 12:
        elapsed = measure()

        if not isfinite(elapsed) or elapsed <= 0:
            raise NotImplementedError("The timer did not report a positive duration.")

        samples.append(elapsed / calls)
        total_ms += elapsed

    return median(samples)


def _graph_benchmark(function, args, kwargs, runtime, min_total_ms):
    factory = getattr(runtime, "Graph", None) or getattr(runtime, "CUDAGraph", None)

    if factory is None or not all(
        callable(getattr(runtime, name, None))
        for name in ("graph", "Event", "synchronize", "Stream", "stream")
    ):
        raise NotImplementedError("Device graphs are unavailable.")

    try:
        capture_stream = runtime.Stream()
    except (NotImplementedError, RuntimeError, TypeError) as exc:
        raise NotImplementedError("Graph capture streams are unavailable.") from exc

    # The enclosing device scope selects the input device. This separate stream
    # scope restores the caller even if the graph context fails during exit.
    with runtime.stream(capture_stream):
        return _graph_stream_benchmark(
            function, args, kwargs, runtime, min_total_ms, factory, capture_stream
        )


def _graph_stream_benchmark(
    function, args, kwargs, runtime, min_total_ms, factory, capture_stream
):
    def capture(calls):
        try:
            graph = factory()

            with runtime.graph(graph, stream=capture_stream):
                for _ in range(calls):
                    function(*args, **kwargs)
        except (NotImplementedError, RuntimeError, TypeError) as exc:
            raise NotImplementedError("This call cannot be captured.") from exc

        return graph

    try:
        start = runtime.Event(enable_timing=True)
        end = runtime.Event(enable_timing=True)
    except (NotImplementedError, RuntimeError, TypeError) as exc:
        raise NotImplementedError("Graph timing events are unavailable.") from exc

    if not all(
        callable(getattr(start, name, None)) for name in ("record", "elapsed_time")
    ):
        raise NotImplementedError("Graph timing events are incomplete.")

    def measure(graph):
        start.record()
        graph.replay()
        end.record()
        wait = getattr(end, "synchronize", None)

        if callable(wait):
            wait()
        else:
            runtime.synchronize()

        elapsed = start.elapsed_time(end)

        if not isfinite(elapsed) or elapsed <= 0:
            raise NotImplementedError(
                "Device events did not report a positive duration."
            )

        return elapsed

    pilot = capture(8)
    pilot.replay()
    elapsed = measure(pilot)
    calls = min(1024, max(1, 1 << max(0, ceil(8 / elapsed) - 1).bit_length()))
    graph = pilot if calls == 8 else capture(calls)

    for _ in range(3):
        graph.replay()

    runtime.synchronize()
    elapsed = _samples(lambda: measure(graph), calls, min_total_ms)

    return BenchmarkResult(milliseconds=elapsed, method="graph", synchronized=True)


def _event_benchmark(function, args, kwargs, runtime, min_total_ms):
    factory = getattr(runtime, "Event", None)

    if not callable(factory):
        raise NotImplementedError("Device events are unavailable.")

    try:
        start = factory(enable_timing=True)
        end = factory(enable_timing=True)
    except (NotImplementedError, RuntimeError, TypeError) as exc:
        raise NotImplementedError("Timing events are unavailable.") from exc

    if not all(
        callable(getattr(start, name, None)) for name in ("record", "elapsed_time")
    ):
        raise NotImplementedError("Timing events are incomplete.")

    wait = getattr(end, "synchronize", None)
    synchronized = False
    recording_supported = False

    def measure():
        nonlocal synchronized, recording_supported

        try:
            start.record()
        except (NotImplementedError, TypeError) as exc:
            raise NotImplementedError("Event recording is unavailable.") from exc
        except RuntimeError as exc:
            # Some external streams reject events with RuntimeError. Only the
            # first record is a capability probe, before any measured launches.
            # A successful device wait is required to rule out pending failures.
            if not recording_supported and _synchronize(runtime):
                raise NotImplementedError("Event recording is unavailable.") from exc

            raise

        recording_supported = True

        for _ in range(10):
            function(*args, **kwargs)

        try:
            end.record()
        except (NotImplementedError, TypeError) as exc:
            raise NotImplementedError("Event recording is unavailable.") from exc

        if callable(wait):
            wait()
            synchronized = True
        else:
            synchronized = _synchronize(runtime)

        try:
            return start.elapsed_time(end)
        except (NotImplementedError, TypeError) as exc:
            raise NotImplementedError("Elapsed device time is unavailable.") from exc

    elapsed = _samples(measure, 10, min_total_ms)

    return BenchmarkResult(
        milliseconds=elapsed, method="event", synchronized=synchronized
    )


def _python_benchmark(function, args, kwargs, runtime, min_total_ms):
    synchronized = False

    def measure():
        nonlocal synchronized
        before = _synchronize(runtime)
        started = time.perf_counter()

        for _ in range(10):
            function(*args, **kwargs)

        synchronized = _synchronize(runtime) and before

        return (time.perf_counter() - started) * 1000

    elapsed = _samples(measure, 10, min_total_ms)

    return BenchmarkResult(
        milliseconds=elapsed, method="python", synchronized=synchronized
    )


def _synchronize(runtime):
    synchronize = getattr(runtime, "synchronize", None)

    if callable(synchronize):
        try:
            synchronize()
        except NotImplementedError:
            return False

        return True

    return False


def _device_type(device):
    return str(getattr(device, "type", str(device).split(":", 1)[0])).lower()


def _device_runtime(args, kwargs):
    """Resolve optional framework interfaces without requiring a tensor package."""
    try:
        import torch
    except ImportError:
        return None

    cpu_runtime = None

    for value in (*args, *kwargs.values()):
        device = getattr(value, "device", None)

        if device is None:
            continue

        kind = _device_type(device)
        runtime = getattr(torch, kind, None)

        if kind != "cpu" and runtime is not None:
            return runtime

        if kind == "cpu":
            cpu_runtime = runtime

    return cpu_runtime
