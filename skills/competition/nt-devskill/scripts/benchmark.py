"""Performance benchmark script for nt-devskill examples.

Compares kernel latency across ninetoothed and torch backends
using CUDA events for accurate GPU timing.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --op matmul --dtype float32
"""

import argparse
import importlib
import math
import sys
from pathlib import Path

import torch

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))


WARMUP = 10
TRIALS = 50


def _bench(fn, args, warmup=WARMUP, trials=TRIALS):
    for _ in range(warmup):
        fn(*args)

    torch.cuda.synchronize()

    import time as _time

    # E2E time (wall clock, includes Python dispatch + GPU)
    t0 = _time.perf_counter()
    for _ in range(trials):
        fn(*args)
    torch.cuda.synchronize()
    e2e_ms = (_time.perf_counter() - t0) * 1000 / trials

    # GPU-only time (CUDA events, excludes Python dispatch)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(trials):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    gpu_ms = start.elapsed_time(end) / trials

    return {"e2e": e2e_ms, "gpu": gpu_ms}


def _report(name, results):
    print(f"\n--- {name} ---")
    print(f"  {'backend':>20s} | {'E2E (ms)':>10s} | {'GPU (ms)':>10s} | {'host %':>7s}")
    print(f"  {'-' * 20}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 7}")
    for backend, times in results.items():
        e2e = times["e2e"]
        gpu = times["gpu"]
        host_pct = (e2e - gpu) / e2e * 100 if e2e > 0 else 0
        print(f"  {backend:>20s} | {e2e:>10.4f} | {gpu:>10.4f} | {host_pct:>6.1f}%")

    # Compute speedup if both backends present
    backends = list(results.keys())
    if "ninetoothed" in backends and "torch" in backends:
        nt_gpu = results["ninetoothed"]["gpu"]
        torch_gpu = results["torch"]["gpu"]
        speedup = torch_gpu / nt_gpu if nt_gpu > 0 else float("inf")
        print(f"\n  GPU speedup (nt/torch): {speedup:.2f}x")


_BENCH_CASES = [
    {
        "name": "add",
        "shapes": [(98432,)],
        "backends": {
            "ninetoothed": ("examples.add", "add"),
            "torch": (None, lambda a, b: torch.add(a, b)),
        },
        "args_fn": lambda shapes: (
            torch.randn(*shapes[0], dtype=torch.float16, device="cuda"),
            torch.randn(*shapes[0], dtype=torch.float16, device="cuda"),
        ),
    },
    {
        "name": "matmul",
        "shapes": [(1024, 1024)],
        "backends": {
            "ninetoothed": ("examples.matmul", "mm"),
            "torch": (None, lambda a, b: torch.mm(a, b)),
        },
        "args_fn": lambda shapes: (
            torch.randn(*shapes[0], dtype=torch.float16, device="cuda"),
            torch.randn(*shapes[0], dtype=torch.float16, device="cuda"),
        ),
    },
    {
        "name": "softmax",
        "shapes": [(4096, 4096)],
        "backends": {
            "ninetoothed": ("examples.softmax", "softmax"),
            "torch": (None, lambda x: torch.softmax(x, dim=-1)),
        },
        "args_fn": lambda shapes: (
            torch.randn(*shapes[0], dtype=torch.float16, device="cuda"),
        ),
    },
    {
        "name": "fused_rms_norm",
        "shapes": [(4096, 4096)],
        "backends": {
            "ninetoothed": ("examples.fused_rms_norm", "fused_rms_norm"),
            "torch": (
                None,
                lambda x, w, eps: torch.nn.functional.rms_norm(
                    x, x.shape[-1:], w, eps
                ),
            ),
        },
        "args_fn": lambda shapes: (
            torch.randn(*shapes[0], dtype=torch.float16, device="cuda"),
            torch.randn(shapes[0][1], dtype=torch.float16, device="cuda"),
            1e-5,
        ),
    },
    {
        "name": "silu",
        "shapes": [(4096, 4096)],
        "backends": {
            "ninetoothed": ("examples.silu", "silu"),
            "torch": (None, lambda x: torch.nn.functional.silu(x)),
        },
        "args_fn": lambda shapes: (
            torch.randn(*shapes[0], dtype=torch.float16, device="cuda"),
        ),
    },
    {
        "name": "bmm",
        "shapes": [(4, 1024, 1024)],
        "backends": {
            "ninetoothed": ("examples.bmm", "bmm"),
            "torch": (None, lambda a, b: torch.bmm(a, b)),
        },
        "args_fn": lambda shapes: (
            torch.randn(*shapes[0], dtype=torch.float16, device="cuda"),
            torch.randn(*shapes[0], dtype=torch.float16, device="cuda"),
        ),
    },
    {
        "name": "addmm",
        "shapes": [(1024, 1024)],
        "backends": {
            "ninetoothed": ("examples.addmm", "addmm"),
            "torch": (None, lambda c, a, b: torch.addmm(c, a, b)),
        },
        "args_fn": lambda shapes: (
            torch.randn(*shapes[0], dtype=torch.float16, device="cuda"),
            torch.randn(*shapes[0], dtype=torch.float16, device="cuda"),
            torch.randn(*shapes[0], dtype=torch.float16, device="cuda"),
        ),
    },
    {
        "name": "scaled_dot_product_attention",
        "shapes": [(2, 8, 1024, 64)],
        "backends": {
            "ninetoothed": ("examples.scaled_dot_product_attention", "scaled_dot_product_attention"),
            "torch": (
                None,
                lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(q, k, v),
            ),
        },
        "args_fn": lambda shapes: (
            torch.randn(*shapes[0], dtype=torch.float16, device="cuda"),
            torch.randn(*shapes[0], dtype=torch.float16, device="cuda"),
            torch.randn(*shapes[0], dtype=torch.float16, device="cuda"),
        ),
    },
    {
        "name": "swiglu",
        "shapes": [(4096, 4096)],
        "backends": {
            "ninetoothed": ("examples.swiglu", "swiglu"),
            "torch": (
                None,
                lambda a, b: a * (b * torch.sigmoid(b.float()).half()),
            ),
        },
        "args_fn": lambda shapes: (
            torch.randn(*shapes[0], dtype=torch.float16, device="cuda"),
            torch.randn(*shapes[0], dtype=torch.float16, device="cuda"),
        ),
    },
    {
        "name": "conv2d",
        "shapes": [(4, 64, 32, 32)],
        "backends": {
            "ninetoothed": ("examples.conv2d", "conv2d"),
            "torch": (None, lambda i, f: torch.nn.functional.conv2d(i, f)),
        },
        "args_fn": lambda shapes: (
            torch.randn(*shapes[0], dtype=torch.float16, device="cuda"),
            torch.randn(128, 64, 3, 3, dtype=torch.float16, device="cuda"),
        ),
    },
    {
        "name": "max_pool2d",
        "shapes": [(4, 64, 32, 32)],
        "backends": {
            "ninetoothed": ("examples.max_pool2d", "max_pool2d"),
            "torch": (
                None,
                lambda x: torch.nn.functional.max_pool2d(x, (2, 2), stride=(2, 2)),
            ),
        },
        "args_fn": lambda shapes: (
            torch.randn(*shapes[0], dtype=torch.float16, device="cuda"),
            (2, 2),
        ),
    },
    {
        "name": "rotary_position_embedding",
        "shapes": [(2, 128, 8, 64)],
        "backends": {
            "ninetoothed": ("examples.rotary_position_embedding", "rotary_position_embedding"),
        },
        "args_fn": lambda shapes: _make_rope_args(shapes[0]),
    },
]


def _make_rope_args(shape):
    """Generate RoPE test inputs."""
    batch, seq_len, num_heads, head_dim = shape
    input = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.float16, device="cuda")
    positions = torch.arange(seq_len, dtype=torch.float32, device="cuda")
    freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim // 2, dtype=torch.float32, device="cuda") / (head_dim // 2)))
    angles = positions[:, None] * freqs[None, :]
    sin_table = torch.sin(angles).to(torch.float16)
    cos_table = torch.cos(angles).to(torch.float16)
    return (input, sin_table, cos_table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return

    dtype = getattr(torch, args.dtype)

    cases = _BENCH_CASES if args.op is None else [c for c in _BENCH_CASES if c["name"] == args.op]

    if not cases:
        print(f"[ERROR] Unknown operator: {args.op}")
        sys.exit(1)

    print(f"Benchmarking {len(cases)} operator(s) ...")

    for case in cases:
        torch_args = case["args_fn"](case["shapes"])
        # Recreate with requested dtype
        torch_args = tuple(
            a.to(dtype=dtype) if isinstance(a, torch.Tensor) else a
            for a in torch_args
        )

        results = {}
        for backend, (mod_path, fn_spec) in case["backends"].items():
            if callable(fn_spec):
                fn = fn_spec
            else:
                mod = importlib.import_module(mod_path)
                fn = getattr(mod, fn_spec)

            ms = _bench(fn, torch_args)
            results[backend] = ms

        _report(case["name"], results)


if __name__ == "__main__":
    main()
