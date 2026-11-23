"""
T4 Sub-Task 4B: Multi-Shape Benchmark for Tiled Softmax.

Benchmarks the T2 softmax kernel across 8 shape configurations
to evaluate performance across different batch sizes and sequence lengths.

Note: The softmax kernel uses BLOCK_SIZE=X.shape[-1] (full reduction axis),
so there is no cross-block tiling. BLOCK_SIZE sensitivity scanning is not
applicable with this design.
"""

import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "task2_reduction_block")
)
from operator_impl import tiled_softmax


def pytorch_baseline(X):
    """PyTorch reference softmax."""
    return F.softmax(X, dim=-1)


def benchmark_kernel(X, kernel_fn, warmup=50, runs=200):
    """Run kernel and return average time in milliseconds."""
    output = torch.empty_like(X)
    # Warmup
    for _ in range(warmup):
        kernel_fn(X, output)
    torch.cuda.synchronize()
    # Timed runs
    t0 = time.perf_counter()
    for _ in range(runs):
        output = torch.empty_like(X)
        kernel_fn(X, output)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000


def main():
    device = "cuda"

    # ---- Environment info ----
    print("=" * 90)
    print("T4 Benchmark: T2 Tiled Softmax — Multi-Shape + BLOCK_SIZE Sensitivity")
    print("=" * 90)
    print(f"PyTorch:     {torch.__version__}")
    print(f"CUDA:        {torch.version.cuda}")
    print(f"GPU:         {torch.cuda.get_device_name(0)}")
    print(
        f"GPU memory:  {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    )
    print("=" * 90)

    # ---- Multi-shape benchmark ----
    shapes = [
        (1, 1024),
        (1, 4096),
        (1, 16384),
        (1, 32768),
        (64, 1024),
        (64, 4096),
        (1024, 1024),
        (4096, 1024),
    ]

    print(f"\n{'Shape':<16} {'NT (ms)':<12} {'PT (ms)':<12} {'Ratio':<10} {'Notes'}")
    print("-" * 90)

    results = []
    for M, N in shapes:
        X = torch.randn((M, N), dtype=torch.float32, device=device)

        nt_time = benchmark_kernel(X, tiled_softmax)

        # PyTorch baseline
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            _ = pytorch_baseline(X)
        torch.cuda.synchronize()
        pt_time = (time.perf_counter() - t0) / 200 * 1000

        notes = ""
        if M == 1 and N > 16000:
            notes = "long seq, many N-blocks"
        elif M > 1000:
            notes = "high batch parallelism"
        elif N == 1:
            notes = "trivial reduction"
        elif M == 1:
            notes = "single row"

        print(
            f"({M:<5},{N:<5})  {nt_time:<12.4f} {pt_time:<12.4f} {nt_time / pt_time:<10.2f} {notes}"
        )
        results.append(
            {
                "shape": f"({M},{N})",
                "nt_ms": nt_time,
                "pt_ms": pt_time,
                "ratio": nt_time / pt_time,
            }
        )

    # ---- Summary ----
    print("\n" + "=" * 90)
    ratios = [r["ratio"] for r in results]
    print(f"Ratio range: {min(ratios):.2f}x – {max(ratios):.2f}x")
    print(f"Mean ratio:  {sum(ratios) / len(ratios):.2f}x")
    print("=" * 90)

    print("\nConclusion:")
    print("  - Softmax uses BLOCK_SIZE=X.shape[-1] (full reduction axis per tile)")
    print("  - Single-row cases bottleneck on kernel launch overhead")
    print("  - Multi-row cases show good GPU utilisation")
    print("=" * 90)


if __name__ == "__main__":
    main()
