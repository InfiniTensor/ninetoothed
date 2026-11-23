"""Benchmark NineToothed add against torch.add."""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from add import add  # noqa: E402


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def _benchmark(fn, device, warmup=10, repeats=50):
    for _ in range(warmup):
        fn()

    _sync(device)

    start = time.perf_counter()

    for _ in range(repeats):
        fn()

    _sync(device)

    return (time.perf_counter() - start) / repeats


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark elementwise / broadcast add"
    )
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--broadcast", choices=("none", "1d", "row"), default="1d")
    parser.add_argument("--dtype", default="float32")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; skipping benchmark.")
        return

    device = "cuda"
    dtype = getattr(torch, args.dtype)
    m, n = args.m, args.n

    lhs = torch.rand((m, n), dtype=dtype, device=device)

    if args.broadcast == "none":
        rhs = torch.rand((m, n), dtype=dtype, device=device)
    elif args.broadcast == "1d":
        rhs = torch.rand(n, dtype=dtype, device=device)
    else:
        rhs = torch.rand((1, n), dtype=dtype, device=device)

    output_shape = torch.broadcast_shapes(lhs.shape, rhs.shape)
    output = torch.empty(output_shape, dtype=dtype, device=device)

    def run_torch():
        torch.add(lhs, rhs, out=output)

    def run_ninetoothed():
        add(lhs, rhs)

    torch_ms = _benchmark(run_torch, device) * 1000
    nt_ms = _benchmark(run_ninetoothed, device) * 1000

    print(f"shape lhs={tuple(lhs.shape)}, rhs={tuple(rhs.shape)}, dtype={args.dtype}")
    print(f"torch.add:     {torch_ms:.4f} ms/iter")
    print(f"ninetoothed:   {nt_ms:.4f} ms/iter")
    print(f"ratio (nt/torch): {nt_ms / torch_ms:.3f}x")


if __name__ == "__main__":
    main()
