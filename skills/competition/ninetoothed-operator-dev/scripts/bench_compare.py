#!/usr/bin/env python3
"""Benchmarking + Roofline helpers for NineToothed operators.

Import these from your own bench file (keeps this script free of any dynamic
import of your kernel):

    from bench_compare import benchmark, roofline, compare
    ms = benchmark(lambda: my_op(x))
    print(roofline(flops=2*M*N*K, bytes_moved=(M*K+K*N+M*N)*2))  # gpu auto-detected

Timing prefers `triton.testing.do_bench` — the same primitive NineToothed itself
uses for autotuning (`src/ninetoothed/auto_tuner.py`, `build.py`) and the Triton
ecosystem standard (quantile timing + L2-cache flush between reps). It falls back
to `torch.cuda.Event`, then to `perf_counter` on CPU, so the script still runs in
a CUDA-less or Triton-less environment.
"""

from __future__ import annotations

import os
import statistics
import time
from typing import Callable, Optional

# Ridge point (FLOP/byte) = peak fp16 FLOPs / peak HBM bandwidth; below => memory-bound.
# LABEL-ONLY fallback: this table only sets the compute/memory-bound *label* in
# roofline(). It is NOT a safety bound — the reward-hacking guard measures bandwidth
# on the live device instead (evaluation/skill_eval/robust_bench.measure_peak_bw).
# Peak fp16 FLOPs can't be probed at runtime, so an unlisted card (RTX 5090, etc.)
# just yields verdict `unknown` — a missing label, never a wrong verdict. Kept tiny
# on purpose: two well-known anchors, not a speculative catalog.
GPU_RIDGE = {
    "H100": 295.0,  # SXM fp16 ~989 TFLOP/s / ~3.35 TB/s
    "A100": 156.0,  # fp16 ~312 TFLOP/s / ~2.0 TB/s.
}


def _match_spec(name: str) -> Optional[str]:
    """Case-insensitive substring match of a device name against GPU_RIDGE."""
    up = name.upper()

    return next((key for key in GPU_RIDGE if key in up), None)


def detect_gpu(gpu: Optional[str] = None) -> Optional[str]:
    """Resolve a GPU_RIDGE key for the roofline *label*.

    Priority: explicit `gpu` arg > $NT_GPU env override > live
    `torch.cuda.get_device_name(0)`. Returns None when nothing matches — roofline
    then reports verdict "unknown" rather than guessing a device. (The
    reward-hacking guard does not rely on this at all; it measures the device's
    bandwidth directly.)
    """
    for candidate in (gpu, os.environ.get("NT_GPU")):
        if candidate:
            return _match_spec(candidate)

    try:
        import torch

        if torch.cuda.is_available():
            return _match_spec(torch.cuda.get_device_name(0))
    except Exception:
        pass
    return None


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _try_do_bench(fn: Callable[[], object], warmup: int, iters: int) -> Optional[dict]:
    """Preferred path: triton.testing.do_bench.

    Returns None if triton is absent or no GPU. Note: under do_bench
    `warmup`/`iters` are *millisecond budgets* (do_bench auto-sizes the iteration
    count to fill them), not raw counts.
    """
    if not _has_cuda():
        return None

    try:
        import triton  # noqa: PLC0415

        # Quantiles -> [median, p20, p80]; robust central estimate + spread.
        p50, p20, p80 = triton.testing.do_bench(
            fn, warmup=warmup, rep=iters, quantiles=[0.5, 0.2, 0.8]
        )
    except Exception:
        return None
    return {
        "mean_ms": p50,  # Median (do_bench's robust central value).
        "std_ms": (p80 - p20) / 2.0,  # Inter-quantile spread proxy.
        "min_ms": p20,
        "iters": -1,  # Auto (time-budgeted by do_bench).
        "timer": "triton.do_bench(p50; p20-p80 spread)",
    }


def benchmark(fn: Callable[[], object], warmup: int = 25, iters: int = 100) -> dict:
    """Time `fn` and return {'mean_ms','std_ms','min_ms','iters','timer'}.

    Prefers triton.testing.do_bench (see module docstring); falls back to
    torch.cuda.Event, then perf_counter.
    """
    dobench = _try_do_bench(fn, warmup, iters)

    if dobench is not None:
        return dobench

    if _has_cuda():
        import torch

        for _ in range(warmup):
            fn()

        torch.cuda.synchronize()
        times = []

        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))  # Ms.

        timer = "cuda.Event"
    else:
        print("[bench_compare] WARNING: CUDA unavailable; using perf_counter (wall).")

        for _ in range(min(warmup, 3)):
            fn()

        times = []

        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1e3)

        timer = "perf_counter"

    return {
        "mean_ms": statistics.fmean(times),
        "std_ms": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "min_ms": min(times),
        "iters": len(times),
        "timer": timer,
    }


def throughput(mean_ms: float, *, bytes_moved: int = 0, flops: int = 0) -> dict:
    """Return GB/s and TFLOPS from a mean latency."""
    sec = mean_ms * 1e-3
    out = {}

    if bytes_moved:
        out["GB_s"] = bytes_moved / sec / 1e9

    if flops:
        out["TFLOPS"] = flops / sec / 1e12
    return out


def roofline(*, flops: int, bytes_moved: int, gpu: Optional[str] = None) -> dict:
    """Classify compute- vs memory-bound by arithmetic intensity vs ridge.

    `gpu` is auto-detected (see `detect_gpu`) when omitted. If the device can't
    be resolved to a known spec, returns verdict 'unknown' with ridge_point None
    rather than guessing a device — a wrong ridge flips the bound conclusion.
    """
    if bytes_moved <= 0:
        raise ValueError("Bytes_moved must be > 0.")

    ai = flops / bytes_moved
    key = detect_gpu(gpu)

    if key is None:
        return {
            "arithmetic_intensity": ai,
            "ridge_point": None,
            "gpu": None,
            "verdict": "unknown",
            "note": "no ridge anchor for this device; set $NT_GPU or pass gpu= to label it",
        }

    ridge = GPU_RIDGE[key]
    verdict = "compute-bound" if ai >= ridge else "memory-bound"

    return {
        "arithmetic_intensity": ai,
        "ridge_point": ridge,
        "gpu": key,
        "verdict": verdict,
    }


def compare(candidates: dict[str, Callable[[], object]], **bench_kw) -> dict:
    """Benchmark several callables; return name -> result, with speedup vs the slowest as a convenience field."""
    results = {name: benchmark(fn, **bench_kw) for name, fn in candidates.items()}
    slowest = max(r["mean_ms"] for r in results.values())

    for r in results.values():
        r["speedup_vs_slowest"] = (
            slowest / r["mean_ms"] if r["mean_ms"] else float("inf")
        )
    return results


if __name__ == "__main__":
    # Self-test that runs anywhere (no CUDA / no torch required for roofline).
    print(
        f"roofline demo (fp16 GEMM 4096^3; detected gpu={detect_gpu() or 'unknown'}):"
    )
    M = N = K = 4096
    print(roofline(flops=2 * M * N * K, bytes_moved=(M * K + K * N + M * N) * 2))
    print("benchmark demo (CPU sleep 1ms):")
    print(benchmark(lambda: time.sleep(0.001), warmup=2, iters=5))
