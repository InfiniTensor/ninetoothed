#!/usr/bin/env python3
"""Roofline-aware benchmark comparison for NineToothed vs PyTorch operators.

Measures GPU kernel time via CUDA events, computes bandwidth (GB/s) and
throughput (TFLOPS), classifies compute-bound vs memory-bound using
hardware-specific ridge points, and outputs a comparison table.

NOTE: Currently covers 4 operators (add, silu, softmax, matmul).
For other operators, use scripts/benchmark.py or scripts/diag_overhead.py.

Hardware ridge points (GFLOP/s per GB/s bandwidth):
  - NVIDIA H100: 295
  - NVIDIA A100: 156
  - NVIDIA L4:   121
  - MetaX C500:  ~100 (estimated)

Usage:
    python scripts/bench_compare.py --op add --shapes 1024 4096 65536 1048576
    python scripts/bench_compare.py --op softmax --shapes 128,1024 4096,4096 --dtype float16
    python scripts/bench_compare.py --op matmul --shapes 512,512 1024,1024 4096,4096 --gpu metax
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

import torch

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

WARMUP = 20
TRIALS = 100

RIDGE_POINTS = {
    "h100": 295,
    "a100": 156,
    "l4": 121,
    "metax": 100,
    "default": 150,
}

_BENCH_CASES = {
    "add": {
        "nt": ("examples.add", "add"),
        "torch": lambda a, b: torch.add(a, b),
        "args": lambda s, d: (
            torch.randn(*s, dtype=d, device="cuda"),
            torch.randn(*s, dtype=d, device="cuda"),
        ),
        "flops": lambda s: 0,
        "bytes": lambda s, d: sum(torch.randn(*s, dtype=d).numel() * torch.randn(*s, dtype=d).element_size() for _ in range(3)),
    },
    "silu": {
        "nt": ("examples.silu", "silu"),
        "torch": lambda x: torch.nn.functional.silu(x),
        "args": lambda s, d: (torch.randn(*s, dtype=d, device="cuda"),),
        "flops": lambda s: 0,
        "bytes": lambda s, d: torch.randn(*s, dtype=d).numel() * torch.randn(*s, dtype=d).element_size() * 2,
    },
    "softmax": {
        "nt": ("examples.softmax", "softmax"),
        "torch": lambda x: torch.softmax(x, dim=-1),
        "args": lambda s, d: (torch.randn(*s, dtype=d, device="cuda"),),
        "flops": lambda s: 0,
        "bytes": lambda s, d: torch.randn(*s, dtype=d).numel() * torch.randn(*s, dtype=d).element_size() * 2,
    },
    "matmul": {
        "nt": ("examples.matmul", "mm"),
        "torch": lambda a, b: torch.mm(a, b),
        "args": lambda s, d: (
            torch.randn(*s, dtype=d, device="cuda"),
            torch.randn(s[-1], s[-1], dtype=d, device="cuda"),
        ),
        "flops": lambda s: 2 * s[0] * s[-1] * s[-1],
        "bytes": lambda s, d: (s[0] * s[-1] + s[-1] * s[-1] + s[0] * s[-1]) * torch.randn(1, dtype=d).element_size(),
    },
}


def measure_gpu(fn, args, warmup=WARMUP, trials=TRIALS):
    """Measure GPU-only time using CUDA events."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(trials):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / trials


def measure_e2e(fn, args, warmup=WARMUP, trials=TRIALS):
    """Measure E2E time using wall clock."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(trials):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / trials


def classify_bottleneck(flops, bytes_moved, gpu_ms, ridge_point):
    """Classify as compute-bound or memory-bound using Roofline model."""
    if gpu_ms <= 0 or bytes_moved <= 0:
        return "unknown", 0, 0

    gpu_s = gpu_ms * 1e-3
    achieved_gflops = (flops / gpu_s) / 1e9 if flops > 0 else 0
    achieved_bw = (bytes_moved / gpu_s) / 1e9

    if flops > 0 and achieved_bw > 0:
        arithmetic_intensity = flops / bytes_moved
        roofline_threshold = ridge_point
        if arithmetic_intensity > roofline_threshold:
            return "compute-bound", achieved_gflops, achieved_bw
        else:
            return "memory-bound", achieved_gflops, achieved_bw
    elif bytes_moved > 0:
        return "memory-bound", 0, achieved_bw
    else:
        return "unknown", 0, 0


def parse_shape(shape_str):
    return tuple(int(x) for x in shape_str.split(","))


def run_benchmark(op, dtype, shapes, gpu_type):
    if op not in _BENCH_CASES:
        print(f"[ERROR] No benchmark case for '{op}'. Available: {list(_BENCH_CASES.keys())}")
        return

    case = _BENCH_CASES[op]
    dtype_obj = getattr(torch, dtype)
    ridge = RIDGE_POINTS.get(gpu_type, RIDGE_POINTS["default"])

    print(f"{'=' * 85}")
    print(f"  Roofline Benchmark: {op}")
    print(f"  dtype: {dtype}, GPU ridge point: {ridge} GFLOP/s per GB/s")
    print(f"{'=' * 85}")
    print(f"\n  {'shape':>20s} | {'nt GPU':>10s} | {'torch GPU':>10s} | {'speedup':>8s} | "
          f"{'BW GB/s':>8s} | {'bottleneck':>14s}")
    print(f"  {'-' * 20}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 14}")

    for shape in shapes:
        shape_tuple = shape if isinstance(shape, tuple) else (shape,)
        args = case["args"](shape_tuple, dtype_obj)

        # Compute data volume
        nbytes = sum(
            a.numel() * a.element_size() for a in args if isinstance(a, torch.Tensor)
        )
        # Add output tensor
        if len(args) >= 1 and isinstance(args[0], torch.Tensor):
            nbytes += args[0].numel() * args[0].element_size()

        flops = case["flops"](shape_tuple)

        # Build NT kernel
        nt_mod = importlib.import_module(case["nt"][0])
        nt_fn = getattr(nt_mod, case["nt"][1])
        torch_fn = case["torch"]

        try:
            nt_gpu = measure_gpu(nt_fn, args)
            torch_gpu = measure_gpu(torch_fn, args)
            speedup = torch_gpu / nt_gpu if nt_gpu > 0 else float("inf")

            bottleneck, gflops, bw = classify_bottleneck(flops, nbytes, nt_gpu, ridge)

            shape_label = "x".join(str(x) for x in shape_tuple)
            print(f"  {shape_label:>20s} | {nt_gpu:>9.4f}ms | {torch_gpu:>9.4f}ms | "
                  f"{speedup:>7.2f}x | {bw:>7.1f} | {bottleneck:>14s}")
        except Exception as e:
            shape_label = "x".join(str(x) for x in shape_tuple)
            print(f"  {shape_label:>20s} | {'FAIL':>10s} | {'---':>10s} | {'---':>8s} | "
                  f"{'---':>8s} | {e}")

    print(f"\n  Roofline interpretation:")
    print(f"    Ridge point = {ridge} GFLOP/s per GB/s")
    print(f"    Arithmetic intensity > ridge → compute-bound (optimize FLOPs)")
    print(f"    Arithmetic intensity < ridge → memory-bound (optimize bandwidth)")


def main():
    parser = argparse.ArgumentParser(description="Roofline-aware benchmark comparison")
    parser.add_argument("--op", type=str, required=True, help="Operator name")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument(
        "--shapes", nargs="*", default=None,
        help="Shapes as M or MxN (e.g. 1024 4096 or 512,512 1024,1024)"
    )
    parser.add_argument(
        "--gpu", type=str, default="metax",
        choices=list(RIDGE_POINTS.keys()),
        help="GPU type for ridge point calculation"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return

    if args.shapes:
        shapes = [parse_shape(s) for s in args.shapes]
    else:
        # Default shapes based on operator
        default_shapes = {
            "add": [1024, 4096, 65536, 1048576],
            "silu": [1024, 4096, 65536, 1048576],
            "softmax": [(128, 1024), (1024, 1024), (4096, 4096)],
            "matmul": [(512, 512), (1024, 1024), (4096, 4096)],
        }
        shapes = default_shapes.get(args.op, [1024, 4096, 65536])

    run_benchmark(args.op, args.dtype, shapes, args.gpu)


if __name__ == "__main__":
    main()
