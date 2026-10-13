#!/usr/bin/env python3
"""Deterministic optimization strategy selector for NineToothed operators.

Given an operator's context (family, dtype, bottleneck type), recommends
the highest-priority optimization strategy with supporting rationale.

This is a pure, deterministic, offline function — no inference, no network.
Falls back to family defaults when no specific rule matches.

NOTE: This is a STATIC tool — it currently always uses family-level defaults
because no real evolution data has been collected yet. It does NOT have a
learned rules.json. a real strategy selector would be populated by evaluation-driven evolution.

Usage:
    python scripts/strategy_selector.py --family elementwise --dtype float16
    python scripts/strategy_selector.py --family reduction --bottleneck memory_bound
    python scripts/strategy_selector.py --family matmul --dtype float16 --bottleneck compute_bound
"""
from __future__ import annotations

import argparse

# Family-level default strategies
_FAMILY_DEFAULTS = {
    "elementwise": {
        "strategy": "tile_size_sweep",
        "rationale": "Elementwise ops are memory-bound; sweep tile sizes to find bandwidth-optimal config",
        "actions": [
            "Run diag_overhead.py to confirm memory-bound",
            "Run diag_tile_sweep.py with candidates [256, 512, 1024, 2048, 4096]",
            "Apply optimal tile from sweep results",
            "Re-validate correctness + benchmark",
        ],
    },
    "reduction": {
        "strategy": "fp32_accumulate + tile_fit",
        "rationale": "Reductions need fp32 accumulation for precision; tile must fit one row",
        "actions": [
            "Ensure ntl.cast(x, ntl.float32) before accumulation",
            "Set BLOCK_SIZE = input.shape[-1] for full-row tile",
            "If row too large, use blocked reduction with partial sums",
            "Benchmark with bench_compare.py --op <name> --gpu metax",
        ],
    },
    "matmul": {
        "strategy": "autotune_block_mnk",
        "rationale": "MatMul is compute-bound; autotuning BLOCK_M/N/K finds optimal tile",
        "actions": [
            "Use block_size() for BLOCK_M, BLOCK_N, BLOCK_K",
            "Or sweep: BLOCK_M=N ∈ [32, 64, 128], BLOCK_K ∈ [16, 32, 64]",
            "Ensure fp32 accumulator: acc = ntl.zeros(..., dtype=ntl.float32)",
            "Benchmark large shapes (4096x4096) for Roofline classification",
        ],
    },
    "attention": {
        "strategy": "online_softmax + head_dim_constraint",
        "rationale": "Attention is compute-bound with head_dim constraint; online softmax avoids O(N²) memory",
        "actions": [
            "Use shape_options={'constexpr': True, 'upper_bound': 128} for head_dim",
            "Implement online softmax with m_i tracking",
            "Use ntl.exp2(x * 1.44269504089) instead of ntl.exp for speed",
            "4D tiling requires dtype.dtype.squeeze() double-level access",
        ],
    },
    "convolution": {
        "strategy": "im2col + matmul_reuse",
        "rationale": "Convolution via im2col reuses optimized matmul arrangement/application",
        "actions": [
            "Use tile(strides=, dilation=) for window extraction",
            "ravel() + flatten() to collapse window dims",
            "permute() to align for matmul",
            "Import from examples.matmul.kernel import arrangement, application",
        ],
    },
    "composition": {
        "strategy": "reuse_arrangement_application",
        "rationale": "Composition operators reuse existing kernel's arrangement/application",
        "actions": [
            "Identify base kernel (e.g., matmul for bmm/addmm/conv2d)",
            "Import arrangement/application from base kernel",
            "Modify only the application function for fused operations",
            "Use from-import pattern (not import-as-module)",
        ],
    },
}

# Bottleneck-specific overrides
_BOTTLENECK_OVERRIDES = {
    ("elementwise", "memory_bound"): {
        "strategy": "maximize_tile_bandwidth",
        "rationale": "Memory-bound elementwise: larger tiles reduce launch overhead",
        "extra_actions": [
            "Try BLOCK_SIZE up to 8192 for 1D elementwise",
            "Check bandwidth utilization: target >200 GB/s on MetaX C500",
        ],
    },
    ("elementwise", "compute_bound"): {
        "strategy": "check_libdevice",
        "rationale": "Compute-bound elementwise: check if libdevice function is the bottleneck",
        "extra_actions": [
            "Check if using libdevice function (lgamma, copysign, etc.)",
            "Try alternative approximations (Stirling, Lanczos, polynomial)",
            "Profile with inspect_generated.py to check tl.math calls",
        ],
    },
    ("reduction", "memory_bound"): {
        "strategy": "blocked_reduction",
        "rationale": "Memory-bound reduction with large rows: use blocked approach",
        "extra_actions": [
            "Split row into blocks, compute partial reductions",
            "Combine partial results in a second pass",
            "Ensure fp32 intermediate accumulation",
        ],
    },
    ("matmul", "compute_bound"): {
        "strategy": "optimize_block_k + num_warps",
        "rationale": "Compute-bound matmul: optimize BLOCK_K and num_warps for throughput",
        "extra_actions": [
            "Sweep BLOCK_K ∈ [16, 32, 64] — smaller K reduces register pressure",
            "Try num_warps ∈ [2, 4, 8] — more warps hide latency",
            "Check with inspect_generated.py for tl.dot count and config",
        ],
    },
    ("matmul", "memory_bound"): {
        "strategy": "increase_block_mn",
        "rationale": "Memory-bound matmul (small shapes): increase BLOCK_M/N to improve reuse",
        "extra_actions": [
            "Try BLOCK_M=BLOCK_N=128 or 256",
            "Check if autotuning is selecting suboptimal small blocks",
            "Consider fixed tile size instead of autotuning for known shapes",
        ],
    },
}


def recommend(family: str, dtype: str = "float16", bottleneck: str = "unknown") -> dict:
    """Return optimization strategy recommendation. Never raises."""
    key = (family, bottleneck)
    rec = _BOTTLENECK_OVERRIDES.get(key)
    if rec:
        actions = list(_FAMILY_DEFAULTS.get(family, {}).get("actions", []))
        actions.extend(rec.get("extra_actions", []))
        return {
            "strategy": rec["strategy"],
            "rationale": rec["rationale"],
            "actions": actions,
            "source": "bottleneck_override",
        }

    base = _FAMILY_DEFAULTS.get(family)
    if base:
        return {
            "strategy": base["strategy"],
            "rationale": base["rationale"],
            "actions": base["actions"],
            "source": "family_default",
        }

    return {
        "strategy": "tile_size_sweep",
        "rationale": "Unknown family; start with tile size sweep and Roofline classification",
        "actions": [
            "Run diag_overhead.py to classify bottleneck",
            "Run diag_tile_sweep.py to find optimal tile",
            "Run bench_compare.py for Roofline analysis",
        ],
        "source": "generic_fallback",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--family", required=True,
        choices=["elementwise", "reduction", "matmul", "attention", "convolution", "composition"],
    )
    p.add_argument("--dtype", default="float16")
    p.add_argument(
        "--bottleneck", default="unknown",
        choices=["memory_bound", "compute_bound", "unknown"],
    )
    args = p.parse_args(argv)

    rec = recommend(args.family, args.dtype, args.bottleneck)

    print(f"Family      : {args.family}")
    print(f"Dtype       : {args.dtype}")
    print(f"Bottleneck  : {args.bottleneck}")
    print(f"")
    print(f"Strategy    : {rec['strategy']}")
    print(f"Rationale   : {rec['rationale']}")
    print(f"Source      : {rec['source']}")
    print(f"")
    print(f"Actions:")
    for i, action in enumerate(rec["actions"], 1):
        print(f"  {i}. {action}")


if __name__ == "__main__":
    main()
