#!/usr/bin/env python3
"""
轻量 Benchmark — 不依赖 triton.testing.Benchmark，直接用 CUDA Event 计时。

用法：
    python tests/bench_light.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from operators import (
    create_add_kernel,
    create_strided_add_kernel,
)

DTYPE = torch.float16
DEVICE = "cuda"
WARMUP = 20
REPEATS = 100


def bench_fn(name, fn, *args, warmup=WARMUP, repeats=REPEATS, **kwargs):
    """用 CUDA Event 计时，返回 (median_ms, min_ms, max_ms)。"""
    # warmup
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]

    for i in range(repeats):
        start_events[i].record()
        fn(*args, **kwargs)
        end_events[i].record()

    torch.cuda.synchronize()
    times = sorted(
        s.elapsed_time(e) for s, e in zip(start_events, end_events)
    )
    median = times[len(times) // 2]
    return median, times[0], times[-1]


def main():
    if not torch.cuda.is_available():
        print("需要 CUDA 环境")
        return 1

    print("=" * 60)
    print("轻量 Benchmark: Stride vs Contiguous")
    print(f"  dtype={DTYPE}, warmup={WARMUP}, repeats={REPEATS}")
    print("=" * 60)

    add_kernel = create_add_kernel()
    strided_kernel = create_strided_add_kernel()

    sizes = [1024, 8192, 65536, 262144, 1048576]

    print(f"\n{'size':>10s}  {'contiguous(ms)':>15s}  {'strided(ms)':>15s}  {'overhead':>10s}")
    print("-" * 56)

    for n in sizes:
        x = torch.randn((n,), dtype=DTYPE, device=DEVICE)
        y = torch.randn((n,), dtype=DTYPE, device=DEVICE)
        out_c = torch.empty_like(x)
        out_s = torch.zeros_like(x)

        # contiguous
        med_c, _, _ = bench_fn("contiguous", add_kernel, x, y, out_c)

        # strided (BLOCK_SIZE 用合理的分块值，不是全量)
        med_s, _, _ = bench_fn(
            "strided", strided_kernel, x, y, out_s,
            BLOCK_SIZE=min(n // 2, 2048)
        )

        ratio = med_s / med_c
        print(f"  {n:>10d}  {med_c:>15.4f}  {med_s:>15.4f}  {ratio:>9.2f}x")

    # 正确性验证
    print(f"\n{'─' * 56}")
    x = torch.randn((1024,), dtype=DTYPE, device=DEVICE)
    y = torch.randn((1024,), dtype=DTYPE, device=DEVICE)
    c = torch.zeros_like(x)
    c_expected = torch.zeros_like(x)
    c_expected[::2] = x[::2] + y[::2]
    strided_kernel(x, y, c, BLOCK_SIZE=1024)
    ok = torch.allclose(c, c_expected, atol=1e-5, rtol=1e-3)
    print(f"  Correctness: {'✓ PASS' if ok else '✗ FAIL'}")
    print(f"  {'─' * 56}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
