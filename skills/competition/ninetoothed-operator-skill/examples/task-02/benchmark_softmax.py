"""Benchmark NineToothed row-wise softmax against torch.softmax."""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from softmax import softmax  # noqa: E402


def _sync():
    torch.cuda.synchronize()


def _benchmark(fn, warmup=10, repeats=50):
    for _ in range(warmup):
        fn()

    _sync()
    start = time.perf_counter()

    for _ in range(repeats):
        fn()

    _sync()
    return (time.perf_counter() - start) / repeats


def main():
    parser = argparse.ArgumentParser(description="Benchmark row-wise softmax")
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; skipping benchmark.")
        return

    dtype = getattr(torch, args.dtype)
    input = torch.randn((args.m, args.n), dtype=dtype, device="cuda")
    output = torch.empty_like(input)

    def run_torch():
        torch.softmax(input, dim=-1, out=output)

    def run_ninetoothed():
        softmax(input)

    torch_ms = _benchmark(run_torch, args.warmup, args.repeats) * 1000
    nt_ms = _benchmark(run_ninetoothed, args.warmup, args.repeats) * 1000

    print(f"shape input={tuple(input.shape)}, dtype={args.dtype}, device=cuda")
    print(f"torch.softmax: {torch_ms:.4f} ms/iter")
    print(f"ninetoothed:   {nt_ms:.4f} ms/iter")
    print(f"ratio (nt/torch): {nt_ms / torch_ms:.3f}x")


if __name__ == "__main__":
    main()
