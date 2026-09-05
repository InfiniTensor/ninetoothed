"""Overhead breakdown diagnostic for a single operator.

Measures E2E (wall clock) vs GPU-only (CUDA events) timing,
computes bandwidth utilization, and classifies the bottleneck.

Usage:
    python scripts/diag_overhead.py --op add
    python scripts/diag_overhead.py --op matmul --dtype float16
    python scripts/diag_overhead.py --op softmax --shapes 1024x1024 4096x4096
"""

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

_DEFAULT_SHAPES = {
    "add": [(98432,)],
    "silu": [(4096, 4096)],
    "swiglu": [(4096, 4096)],
    "softmax": [(4096, 4096)],
    "fused_rms_norm": [(4096, 4096)],
    "matmul": [(1024, 1024)],
    "bmm": [(4, 1024, 1024)],
    "addmm": [(1024, 1024)],
    "scaled_dot_product_attention": [(2, 8, 1024, 64)],
    "conv2d": [(4, 64, 32, 32)],
    "max_pool2d": [(4, 64, 32, 32)],
    "rotary_position_embedding": [(2, 128, 8, 64)],
}

_BENCH_CASES = {
    "add": {
        "nt": ("examples.add", "add"),
        "torch": lambda a, b: torch.add(a, b),
        "args": lambda s, d: (
            torch.randn(*s[0], dtype=d, device="cuda"),
            torch.randn(*s[0], dtype=d, device="cuda"),
        ),
    },
    "silu": {
        "nt": ("examples.silu", "silu"),
        "torch": lambda x: torch.nn.functional.silu(x),
        "args": lambda s, d: (torch.randn(*s[0], dtype=d, device="cuda"),),
    },
    "swiglu": {
        "nt": ("examples.swiglu", "swiglu"),
        "torch": lambda a, b: a * (b * torch.sigmoid(b.float()).half()),
        "args": lambda s, d: (
            torch.randn(*s[0], dtype=d, device="cuda"),
            torch.randn(*s[0], dtype=d, device="cuda"),
        ),
    },
    "softmax": {
        "nt": ("examples.softmax", "softmax"),
        "torch": lambda x: torch.softmax(x, dim=-1),
        "args": lambda s, d: (torch.randn(*s[0], dtype=d, device="cuda"),),
    },
    "fused_rms_norm": {
        "nt": ("examples.fused_rms_norm", "fused_rms_norm"),
        "torch": lambda x, w, eps: torch.nn.functional.rms_norm(x, x.shape[-1:], w, eps),
        "args": lambda s, d: (
            torch.randn(*s[0], dtype=d, device="cuda"),
            torch.randn(s[0][1], dtype=d, device="cuda"),
            1e-5,
        ),
    },
    "matmul": {
        "nt": ("examples.matmul", "mm"),
        "torch": lambda a, b: torch.mm(a, b),
        "args": lambda s, d: (
            torch.randn(*s[0], dtype=d, device="cuda"),
            torch.randn(*s[0], dtype=d, device="cuda"),
        ),
    },
    "bmm": {
        "nt": ("examples.bmm", "bmm"),
        "torch": lambda a, b: torch.bmm(a, b),
        "args": lambda s, d: (
            torch.randn(*s[0], dtype=d, device="cuda"),
            torch.randn(*s[0], dtype=d, device="cuda"),
        ),
    },
    "addmm": {
        "nt": ("examples.addmm", "addmm"),
        "torch": lambda c, a, b: torch.addmm(c, a, b),
        "args": lambda s, d: (
            torch.randn(*s[0], dtype=d, device="cuda"),
            torch.randn(*s[0], dtype=d, device="cuda"),
            torch.randn(*s[0], dtype=d, device="cuda"),
        ),
    },
    "max_pool2d": {
        "nt": ("examples.max_pool2d", "max_pool2d"),
        "torch": lambda x: torch.nn.functional.max_pool2d(x, (2, 2), stride=(2, 2)),
        "args": lambda s, d: (
            torch.randn(*s[0], dtype=d, device="cuda"),
            (2, 2),
        ),
    },
}


def measure(fn, args, warmup, trials):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(trials):
        fn(*args)
    torch.cuda.synchronize()
    e2e_ms = (time.perf_counter() - t0) * 1000 / trials

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(trials):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    gpu_ms = start.elapsed_time(end) / trials

    return e2e_ms, gpu_ms


def compute_bytes(args, dtype):
    nbytes = 0
    for a in args:
        if isinstance(a, torch.Tensor):
            nbytes += a.numel() * a.element_size()
    return nbytes


def parse_shape(shape_str):
    return tuple(int(x) for x in shape_str.split("x"))


def run_diagnostic(op, dtype, shapes):
    if op not in _BENCH_CASES:
        print(f"[ERROR] No diagnostic case for '{op}'. Available: {list(_BENCH_CASES.keys())}")
        return

    case = _BENCH_CASES[op]
    dtype_obj = getattr(torch, dtype)

    if shapes is None:
        shapes = _DEFAULT_SHAPES.get(op, [(1024, 1024)])

    print(f"{'=' * 60}")
    print(f"  Overhead Breakdown: {op}")
    print(f"  dtype: {dtype}")
    print(f"{'=' * 60}")

    for shape in shapes:
        shape = shape if isinstance(shape, list) else [shape]
        print(f"\n--- shape: {shape} ---")

        args = case["args"](shape, dtype_obj)
        nbytes = compute_bytes(args, dtype_obj)

        nt_mod = importlib.import_module(case["nt"][0])
        nt_fn = getattr(nt_mod, case["nt"][1])
        torch_fn = case["torch"]

        print(f"\n  [ninetoothed]")
        nt_e2e, nt_gpu = measure(nt_fn, args, WARMUP, TRIALS)
        nt_host = nt_e2e - nt_gpu
        nt_host_pct = nt_host / nt_e2e * 100 if nt_e2e > 0 else 0
        nt_bw_gpu = nbytes / (nt_gpu * 1e-3) / 1e9 if nt_gpu > 0 else 0
        nt_bw_e2e = nbytes / (nt_e2e * 1e-3) / 1e9 if nt_e2e > 0 else 0

        print(f"    E2E:           {nt_e2e:.4f} ms")
        print(f"    GPU-only:      {nt_gpu:.4f} ms")
        print(f"    Host overhead: {nt_host:.4f} ms ({nt_host_pct:.1f}%)")
        print(f"    BW (GPU):      {nt_bw_gpu:.2f} GB/s")
        print(f"    BW (E2E):      {nt_bw_e2e:.2f} GB/s")

        print(f"\n  [torch reference]")
        torch_e2e, torch_gpu = measure(torch_fn, args, WARMUP, TRIALS)
        torch_host = torch_e2e - torch_gpu
        torch_host_pct = torch_host / torch_e2e * 100 if torch_e2e > 0 else 0
        torch_bw_gpu = nbytes / (torch_gpu * 1e-3) / 1e9 if torch_gpu > 0 else 0

        print(f"    E2E:           {torch_e2e:.4f} ms")
        print(f"    GPU-only:      {torch_gpu:.4f} ms")
        print(f"    Host overhead: {torch_host:.4f} ms ({torch_host_pct:.1f}%)")
        print(f"    BW (GPU):      {torch_bw_gpu:.2f} GB/s")

        gpu_speedup = torch_gpu / nt_gpu if nt_gpu > 0 else float("inf")
        e2e_speedup = torch_e2e / nt_e2e if nt_e2e > 0 else float("inf")

        print(f"\n  [comparison]")
        print(f"    GPU speedup:   {gpu_speedup:.2f}x")
        print(f"    E2E speedup:   {e2e_speedup:.2f}x")

        print(f"\n  [diagnosis]")
        if nt_host_pct > 50:
            print(f"    BOTTLENECK: LAUNCH-BOUND (host {nt_host_pct:.0f}%)")
            print(f"    → Kernel launch overhead dominates.")
            print(f"    → Actions: reduce dispatch overhead, cache compiled kernel, fuse kernels.")
        elif nt_host_pct > 20:
            print(f"    BOTTLENECK: MIXED (host {nt_host_pct:.0f}%)")
            print(f"    → Both host dispatch and GPU compute contribute.")
            print(f"    → Actions: optimize tile sizes AND reduce Python overhead.")
        else:
            print(f"    BOTTLENECK: COMPUTE-BOUND (host {nt_host_pct:.0f}%)")
            print(f"    → GPU computation dominates.")
            print(f"    → Actions: optimize tile sizes, enable autotuning, improve memory access.")

        if nt_bw_gpu < 50 and nt_host_pct < 30:
            print(f"    LOW BANDWIDTH: {nt_bw_gpu:.1f} GB/s (MetaX C500 peak ~800 GB/s)")
            print(f"    → Memory access pattern may be suboptimal.")
            print(f"    → Actions: check coalescing, vectorize loads, adjust tile alignment.")
        elif nt_bw_gpu > 200:
            print(f"    GOOD BANDWIDTH: {nt_bw_gpu:.1f} GB/s")

        if gpu_speedup < 0.9:
            print(f"    SLOWER THAN TORCH: {gpu_speedup:.2f}x")
            print(f"    → Tile size sweep recommended. Run: python scripts/diag_tile_sweep.py --op {op}")


def main():
    parser = argparse.ArgumentParser(description="Overhead breakdown diagnostic")
    parser.add_argument("--op", type=str, required=True, help="Operator name")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument(
        "--shapes", nargs="*", default=None,
        help="Shapes as MxN or BxHxMxN (e.g. 1024x1024 4096x4096)"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return

    shapes = None
    if args.shapes:
        shapes = [parse_shape(s) for s in args.shapes]

    run_diagnostic(args.op, args.dtype, shapes)


if __name__ == "__main__":
    main()
