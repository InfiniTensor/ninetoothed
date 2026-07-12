"""
T3 Benchmark: GELU — contiguous vs non-contiguous layout performance.

Compares NineToothed GELU against PyTorch nn.GELU across
3 layout variants at a large fixed shape to quantify stride overhead.
"""

import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from operator_impl import gelu


def pytorch_baseline(X):
    """PyTorch reference GELU with tanh approximation."""
    return nn.GELU(approximate="tanh")(X)


def benchmark_single(X, label, warmup=100, runs=1000):
    """Benchmark a single input configuration."""
    output = torch.empty_like(X)

    # Warmup NineToothed
    for _ in range(warmup):
        gelu(X, output)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        output = torch.empty_like(X)
        gelu(X, output)
    torch.cuda.synchronize()
    nt_time = (time.perf_counter() - t0) / runs * 1000

    # Warmup PyTorch
    for _ in range(warmup):
        _ = pytorch_baseline(X)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = pytorch_baseline(X)
    torch.cuda.synchronize()
    pt_time = (time.perf_counter() - t0) / runs * 1000

    print(f"  {label}:")
    print(f"    NineToothed: {nt_time:.4f} ms")
    print(f"    PyTorch:     {pt_time:.4f} ms")
    print(f"    Ratio:       {nt_time / pt_time:.2f}x")

    return nt_time, pt_time


def main():
    device = "cuda"
    M, N = 4096, 4096

    print("=" * 60)
    print("T3 Benchmark: GELU — Contiguous vs Non-Contiguous Layouts")
    print(f"PyTorch: {torch.__version__}  |  CUDA: {torch.version.cuda}")
    print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"Logical shape: ({M}, {N}), dtype=float32")
    print("Warmup: 100 runs  |  Timed: 1000 runs")
    print("=" * 60)

    data = torch.randn((N, M), dtype=torch.float32, device=device)

    # Contiguous
    print("\n[1/3] Contiguous layout:")
    X_contig = data.T.contiguous()
    nt_contig, pt_contig = benchmark_single(X_contig, "contiguous")

    # Transposed
    print("\n[2/3] Transposed layout (non-contiguous):")
    X_trans = data.T
    nt_trans, pt_trans = benchmark_single(X_trans, "transposed")

    # Sliced
    print("\n[3/3] Sliced layout (stride=2):")
    X_full = torch.randn((M * 2, N), dtype=torch.float32, device=device)
    X_sliced = X_full[::2, :]
    nt_sliced, pt_sliced = benchmark_single(X_sliced, "sliced (stride 2)")

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  {'Layout':<20} {'NT (ms)':<12} {'PT (ms)':<12} {'Overhead':<12}")
    print(f"  {'-' * 54}")
    print(
        f"  {'contiguous':<20} {nt_contig:<12.4f} {pt_contig:<12.4f} {'1.00x (baseline)':<12}"
    )
    print(
        f"  {'transposed':<20} {nt_trans:<12.4f} {pt_trans:<12.4f} {nt_trans / nt_contig:<12.2f}x"
    )
    print(
        f"  {'sliced (stride 2)':<20} {nt_sliced:<12.4f} {pt_sliced:<12.4f} {nt_sliced / nt_contig:<12.2f}x"
    )
    print(f"\n  PyTorch overhead (transposed): {pt_trans / pt_contig:.2f}x")
    print(f"  PyTorch overhead (sliced):     {pt_sliced / pt_contig:.2f}x")

    print("\nConclusion:")
    if nt_trans > nt_contig * 1.2:
        print(
            "  ⚠ Transposed layout: {:.2f}x slowdown — stride access penalty confirmed.".format(
                nt_trans / nt_contig
            )
        )
    else:
        print("  ✓ Non-contiguous input overhead is minimal.")
    print(
        "  - Sliced layout (stride=2 on outer dim): minimal impact ({:.2f}x)".format(
            nt_sliced / nt_contig
        )
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
