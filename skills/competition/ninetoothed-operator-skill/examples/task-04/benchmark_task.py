"""Benchmark and sanity-check the task-01 add operator."""

import argparse
import sys
import time
from pathlib import Path

import torch

TASK_01_DIR = Path(__file__).resolve().parents[1] / "task-01"
sys.path.insert(0, str(TASK_01_DIR))

from add import add  # noqa: E402


def _sync():
    torch.cuda.synchronize()


def _benchmark(fn, warmup, repeats):
    for _ in range(warmup):
        fn()

    _sync()
    start = time.perf_counter()

    for _ in range(repeats):
        fn()

    _sync()
    return (time.perf_counter() - start) / repeats


def _make_inputs(m, n, case, dtype):
    lhs = torch.rand((m, n), dtype=dtype, device="cuda")

    if case == "same":
        rhs = torch.rand((m, n), dtype=dtype, device="cuda")
    elif case == "vector":
        rhs = torch.rand(n, dtype=dtype, device="cuda")
    elif case == "row":
        rhs = torch.rand((1, n), dtype=dtype, device="cuda")
    else:
        raise ValueError(f"Unknown case: {case}")

    return lhs, rhs


def main():
    parser = argparse.ArgumentParser(
        description="Task 04 benchmark/debug harness for task-01 add"
    )
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--case", choices=("same", "vector", "row"), default="vector")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; skipping benchmark.")
        return

    dtype = getattr(torch, args.dtype)
    lhs, rhs = _make_inputs(args.m, args.n, args.case, dtype)
    output = torch.empty(
        torch.broadcast_shapes(lhs.shape, rhs.shape), dtype=dtype, device="cuda"
    )

    expected = lhs + rhs
    actual = add(lhs, rhs)
    max_error = (actual - expected).abs().max().item()
    print(
        f"correctness: allclose={torch.allclose(actual, expected)},"
        f" max_error={max_error:.6g}"
    )

    def run_torch():
        torch.add(lhs, rhs, out=output)

    def run_ninetoothed():
        add(lhs, rhs)

    torch_ms = _benchmark(run_torch, args.warmup, args.repeats) * 1000
    nt_ms = _benchmark(run_ninetoothed, args.warmup, args.repeats) * 1000
    numel = output.numel()

    print(
        f"shape lhs={tuple(lhs.shape)}, rhs={tuple(rhs.shape)},"
        f" dtype={args.dtype}, case={args.case}"
    )
    print(f"elements:      {numel}")
    print(f"torch.add:     {torch_ms:.4f} ms/iter")
    print(f"ninetoothed:   {nt_ms:.4f} ms/iter")
    print(f"ratio (nt/torch): {nt_ms / torch_ms:.3f}x")


if __name__ == "__main__":
    main()
