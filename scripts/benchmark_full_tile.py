#!/usr/bin/env python3

import argparse
import json
import statistics
from pathlib import Path

import torch

from ninetoothed.aot import _load_launch_func


def _make_case(case_id, size, dtype):
    if case_id == "hit_small":
        size = 256
        input = torch.randn(size, device="cuda", dtype=dtype)
        other = torch.randn_like(input)
        output = torch.empty_like(input)
        specialization_hit = True
    elif case_id == "hit_large":
        input = torch.randn(size, device="cuda", dtype=dtype)
        other = torch.randn_like(input)
        output = torch.empty_like(input)
        specialization_hit = True
    elif case_id == "fallback_nondivisible":
        size += 1
        input = torch.randn(size, device="cuda", dtype=dtype)
        other = torch.randn_like(input)
        output = torch.empty_like(input)
        specialization_hit = False
    elif case_id == "fallback_noncontiguous":
        input = torch.randn(size * 2, device="cuda", dtype=dtype)[::2]
        other = torch.randn(size, device="cuda", dtype=dtype)
        output = torch.empty_like(other)
        specialization_hit = False
    else:
        raise ValueError(f"Unknown case: {case_id}.")

    return (input, other, output), specialization_hit


def _measure(kernel, values, warmup, iterations):
    for _ in range(warmup):
        kernel(*values)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    for _ in range(iterations):
        kernel(*values)

    end.record()
    end.synchronize()

    return start.elapsed_time(end) / iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--submitted-dir", type=Path, required=True)
    parser.add_argument("--comparison-id", default="baseline_vs_submitted")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--submitted-label", default="submitted")
    parser.add_argument(
        "--full-tile-variant",
        choices=("baseline", "submitted", "both", "neither"),
        default="submitted",
    )
    parser.add_argument("--kernel-name", required=True)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--size", type=int, default=1 << 20)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    kernels = {
        "baseline": _load_launch_func(args.kernel_name, args.baseline_dir),
        "submitted": _load_launch_func(args.kernel_name, args.submitted_dir),
    }
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(20260712)
    results = []

    for case_id in (
        "hit_small",
        "hit_large",
        "fallback_nondivisible",
        "fallback_noncontiguous",
    ):
        values, specialization_hit = _make_case(case_id, args.size, dtype)
        timings = {"baseline": [], "submitted": []}
        measurement_order = []

        for round_id in range(1, args.rounds + 1):
            order = (
                ("baseline", "submitted") if round_id % 2 else ("submitted", "baseline")
            )
            measurement_order.append(list(order))

            for variant in order:
                timings[variant].append(
                    _measure(kernels[variant], values, args.warmup, args.iterations)
                )
                torch.testing.assert_close(values[2], values[0] + values[1])

        baseline_runtime = statistics.median(timings["baseline"])
        submitted_runtime = statistics.median(timings["submitted"])
        results.append(
            {
                "comparison_id": args.comparison_id,
                "baseline_label": args.baseline_label,
                "submitted_label": args.submitted_label,
                "full_tile_variant": args.full_tile_variant,
                "case_id": case_id,
                "dtype": args.dtype,
                "numel": values[0].numel(),
                "baseline_runtime_ms": baseline_runtime,
                "submitted_runtime_ms": submitted_runtime,
                "speedup": baseline_runtime / submitted_runtime,
                "specialization_hit": specialization_hit,
                "workload_expected_to_hit_full_tile": specialization_hit,
                "correctness_passed": True,
                "warmup": args.warmup,
                "rounds": args.rounds,
                "iterations": args.iterations,
                "measurement_order": measurement_order,
                "baseline_round_runtime_ms": timings["baseline"],
                "submitted_round_runtime_ms": timings["submitted"],
            }
        )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
