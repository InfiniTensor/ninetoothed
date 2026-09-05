"""Tile size sweep diagnostic for a single operator.

Tests multiple tile sizes and reports GPU time, bandwidth, and the optimal choice.
For operators that accept tile size as a parameter (e.g., via premake or Symbol).

Usage:
    python scripts/diag_tile_sweep.py --op add
    python scripts/diag_tile_sweep.py --op matmul --tiles 32x32x16 64x64x32 128x128x32
    python scripts/diag_tile_sweep.py --op softmax --tiles 128 256 512 1024 2048
"""

import argparse
import importlib
import sys
import time
from pathlib import Path

import torch

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

WARMUP = 10
TRIALS = 50

_SWEEP_CONFIGS = {
    "add": {
        "param": "block_size",
        "tiles": [256, 512, 1024, 2048, 4096],
        "shape": (98432,),
    },
    "silu": {
        "param": "block_size",
        "tiles": [256, 512, 1024, 2048, 4096],
        "shape": (4096, 4096),
    },
    "swiglu": {
        "param": "block_size",
        "tiles": [256, 512, 1024, 2048, 4096],
        "shape": (4096, 4096),
    },
    "softmax": {
        "param": "block_size",
        "tiles": [128, 256, 512, 1024, 2048, 4096],
        "shape": (4096, 4096),
    },
    "matmul": {
        "param": "block_m_n_k",
        "tiles": [
            (32, 32, 16), (32, 32, 32), (64, 64, 16), (64, 64, 32),
            (128, 64, 32), (64, 128, 32), (128, 128, 32), (128, 128, 64),
        ],
        "shape": (1024, 1024),
    },
}


def measure_gpu(fn, args, warmup=WARMUP, trials=TRIALS):
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


def compute_bytes(args):
    nbytes = 0
    for a in args:
        if isinstance(a, torch.Tensor):
            nbytes += a.numel() * a.element_size()
    return nbytes


def build_kernel_1d(op_name, block_size, dtype):
    mod = importlib.import_module(f"examples.{op_name}.kernel")
    tensors = mod.tensors
    arrangement = mod.arrangement
    application = mod.application

    import functools
    import ninetoothed

    arr = functools.partial(arrangement, BLOCK_SIZE=block_size)
    return ninetoothed.make(arr, application, tensors)


def build_kernel_matmul(block_m, block_n, block_k, dtype):
    mod = importlib.import_module("examples.matmul.kernel")
    import functools
    import ninetoothed

    arr = functools.partial(
        mod.arrangement,
        BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k,
    )
    return ninetoothed.make(arr, mod.application, mod.tensors)


def parse_tile(tile_str):
    parts = tile_str.split("x")
    if len(parts) == 1:
        return int(parts[0])
    return tuple(int(x) for x in parts)


def run_sweep(op, dtype, tiles):
    dtype_obj = getattr(torch, dtype)

    if op not in _SWEEP_CONFIGS:
        print(f"[ERROR] No sweep config for '{op}'. Available: {list(_SWEEP_CONFIGS.keys())}")
        print(f"[HINT] Use --tiles to specify manually: --tiles 256 512 1024 2048")
        return

    config = _SWEEP_CONFIGS[op]
    if tiles is None:
        tiles = config["tiles"]

    shape = config["shape"]
    param_type = config["param"]

    args_tensors = tuple(
        torch.randn(*shape, dtype=dtype_obj, device="cuda")
        for _ in range(2 if op in ("add", "matmul", "swiglu") else 1)
    )
    nbytes = compute_bytes(args_tensors)

    print(f"{'=' * 70}")
    print(f"  Tile Size Sweep: {op}")
    print(f"  shape: {shape}, dtype: {dtype}")
    print(f"  data volume: {nbytes / 1024 / 1024:.1f} MB")
    print(f"{'=' * 70}")
    print(f"\n  {'tile':>20s} | {'GPU ms':>10s} | {'BW GB/s':>10s} | {'vs best':>10s} | status")
    print(f"  {'-' * 20}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}")

    results = []
    best_ms = float("inf")
    best_tile = None

    for tile in tiles:
        tile_label = str(tile) if isinstance(tile, int) else "x".join(str(x) for x in tile)
        try:
            if param_type == "block_size":
                kernel = build_kernel_1d(op, tile, dtype_obj)
                fn_args = args_tensors
            elif param_type == "block_m_n_k":
                bm, bn, bk = tile
                kernel = build_kernel_matmul(bm, bn, bk, dtype_obj)
                fn_args = args_tensors

            ms = measure_gpu(kernel, fn_args)
            bw = nbytes / (ms * 1e-3) / 1e9 if ms > 0 else 0

            if ms < best_ms:
                best_ms = ms
                best_tile = tile

            results.append((tile, ms, bw))
            ratio = ms / best_ms
            status = "BEST" if ms == best_ms else f"{ratio:.2f}x"
            print(f"  {tile_label:>20s} | {ms:>10.4f} | {bw:>10.2f} | {ratio:>9.2f}x | {status}")

        except Exception as e:
            print(f"  {tile_label:>20s} | {'FAIL':>10s} | {'---':>10s} | {'---':>10s} | {e}")

    print(f"\n  [result]")
    if best_tile is not None:
        best_label = str(best_tile) if isinstance(best_tile, int) else "x".join(str(x) for x in best_tile)
        print(f"    Optimal tile: {best_label}")
        print(f"    GPU time:     {best_ms:.4f} ms")

        worst_ms = max(ms for _, ms, _ in results)
        print(f"    Range:        {best_ms:.4f} - {worst_ms:.4f} ms ({worst_ms / best_ms:.1f}x spread)")
        print(f"    Recommendation: Use tile={best_label} for production.")
    else:
        print(f"    All tiles FAILED. Check kernel implementation.")


def main():
    parser = argparse.ArgumentParser(description="Tile size sweep diagnostic")
    parser.add_argument("--op", type=str, required=True, help="Operator name")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument(
        "--tiles", nargs="*", default=None,
        help="Tile sizes: single (256 512 1024) or multi-dim (32x32x16 64x64x32)"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return

    tiles = None
    if args.tiles:
        tiles = [parse_tile(t) for t in args.tiles]

    run_sweep(args.op, args.dtype, tiles)


if __name__ == "__main__":
    main()
