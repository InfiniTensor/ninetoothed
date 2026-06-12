"""Benchmark suite for T1-2-1 NineToothed code generation specialization.

Measures runtime and generated code metrics for:
- Specialization-hit scenarios (divisible tiles, contiguous inputs)
- Fallback scenarios (non-divisible, non-contiguous)

Output: JSON file with per-scenario metrics.
"""

import json
import pathlib
import re
import time

import torch
import triton

import ninetoothed
from ninetoothed import Symbol, Tensor
from ninetoothed.generation import CodeGenerator, TilingHint


# ---------------------------------------------------------------------------
# Utility: count generated code metrics
# ---------------------------------------------------------------------------

def count_metrics(source_text):
    """Count code quality metrics in generated Triton source."""
    return {
        "mask_expr_count": len(re.findall(r"mask=", source_text)),
        "stride_expr_count": len(re.findall(r"_stride_\d+", source_text)),
        "pointer_expr_count": len(re.findall(r"_pointer\s*\+", source_text)),
        "source_line_count": len(source_text.splitlines()),
    }


# ---------------------------------------------------------------------------
# Benchmark kernel: 1D vector add
# ---------------------------------------------------------------------------

BENCH_BLOCK_SIZE = 256


def bench_add_arrangement(x, output):
    return x.tile((BENCH_BLOCK_SIZE,)), output.tile((BENCH_BLOCK_SIZE,))


def bench_add_application(x, output):
    output = x  # noqa: F841


def run_add_kernel(arrangement, application, tensors, input_data, output_data,
                   device, kernel_name, tiling_hint=None, warmup=5, iters=100):
    """Run a kernel and return (runtime_ms, source_metrics)."""
    if tiling_hint is not None and tiling_hint.is_active():
        code_gen = CodeGenerator(tiling_hint=tiling_hint)
        source_file = code_gen(
            application,
            caller="torch",
            kernel_name=kernel_name,
            num_warps=4,
            num_stages=3,
            max_num_configs=1,
            prettify=False,
        )
    else:
        kernel = ninetoothed.make(arrangement, application, tensors)
        source_file = kernel._source

    source_text = pathlib.Path(source_file).read_text()
    metrics = count_metrics(source_text)

    # Load and run
    import importlib
    import sys

    module_name = f"bench_{kernel_name}"
    spec = importlib.util.spec_from_file_location(module_name, source_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    launch_func = getattr(module, f"launch_{kernel_name}")

    # Warmup
    for _ in range(warmup):
        launch_func(input_data, output_data)

    # Timed runs
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        launch_func(input_data, output_data)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    runtime_ms = (elapsed / iters) * 1000.0

    return runtime_ms, metrics


# ---------------------------------------------------------------------------
# Benchmark scenarios
# ---------------------------------------------------------------------------

def benchmark_scenario(scenario_name, size, arrangement_fn, application_fn,
                       tensors, device, tiling_hint, specialization_hit):
    """Run one benchmark scenario and return results dict."""
    input_data = torch.randn(size, device=device) if isinstance(size, tuple) is False \
        else torch.randn(size, device=device)
    output_data = torch.empty_like(input_data)

    kernel_name = f"bench_{scenario_name}"

    # Run with specialization hints (submitted)
    submitted_runtime, submitted_metrics = run_add_kernel(
        arrangement_fn, application_fn, tensors, input_data, output_data,
        device, kernel_name, tiling_hint=tiling_hint,
    )

    # Run without hints (baseline)
    baseline_runtime, baseline_metrics = run_add_kernel(
        arrangement_fn, application_fn, tensors, input_data, output_data,
        device, f"{kernel_name}_baseline", tiling_hint=None,
    )

    speedup = baseline_runtime / submitted_runtime if submitted_runtime > 0 else 0.0

    return {
        "scenario": scenario_name,
        "size": size,
        "baseline_runtime_ms": round(baseline_runtime, 4),
        "submitted_runtime_ms": round(submitted_runtime, 4),
        "speedup": round(speedup, 4),
        "specialization_hit": specialization_hit,
        "baseline_metrics": baseline_metrics,
        "submitted_metrics": submitted_metrics,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: CUDA not available, running on CPU (timings not meaningful)")
        return

    results = []

    # Scenario 1: Contiguous + divisible (should hit both specializations)
    hint_combined = TilingHint(
        has_divisible_tiles=True,
        contiguous_dims={("tensor_0", 0)},
        known_strides={("tensor_0", 0): 1},
        exact_innermost_sizes=True,
    )
    result = benchmark_scenario(
        "contiguous_divisible_hit",
        2048,  # 2048/256 = 8, perfectly divisible
        bench_add_arrangement,
        bench_add_application,
        (Tensor(1), Tensor(1)),
        device,
        hint_combined,
        specialization_hit=True,
    )
    results.append(result)

    # Scenario 2: Contiguous only, non-divisible
    hint_contiguous = TilingHint(
        has_divisible_tiles=False,
        contiguous_dims={("tensor_0", 0)},
        known_strides={("tensor_0", 0): 1},
        exact_innermost_sizes=False,
    )
    result = benchmark_scenario(
        "contiguous_only_hit",
        1027,  # 1027/256 = 4.01, not divisible
        bench_add_arrangement,
        bench_add_application,
        (Tensor(1), Tensor(1)),
        device,
        hint_contiguous,
        specialization_hit=True,
    )
    results.append(result)

    # Scenario 3: Divisible only, non-contiguous
    hint_divisible = TilingHint(
        has_divisible_tiles=True,
        exact_innermost_sizes=True,
    )
    result = benchmark_scenario(
        "divisible_only_hit",
        2048,
        bench_add_arrangement,
        bench_add_application,
        (Tensor(1), Tensor(1)),
        device,
        hint_divisible,
        specialization_hit=True,
    )
    results.append(result)

    # Scenario 4: Pure fallback — non-divisible, non-contiguous
    hint_fallback = TilingHint()  # All defaults = no specialization
    result = benchmark_scenario(
        "pure_fallback",
        1027,
        bench_add_arrangement,
        bench_add_application,
        (Tensor(1), Tensor(1)),
        device,
        hint_fallback,
        specialization_hit=False,
    )
    results.append(result)

    # Scenario 5: 2D divisible
    def bench_2d_arrangement(x, output):
        return x.tile((64, 64)), output.tile((64, 64))

    hint_2d = TilingHint(
        has_divisible_tiles=True,
        exact_innermost_sizes=True,
    )
    input_2d = torch.randn((512, 512), device=device)
    output_2d = torch.empty_like(input_2d)

    # Baseline 2D
    bl_rt_2d, bl_met_2d = run_add_kernel(
        bench_2d_arrangement, bench_add_application,
        (Tensor(2), Tensor(2)), input_2d, output_2d,
        device, "bench_2d_div_baseline", tiling_hint=None,
    )
    sub_rt_2d, sub_met_2d = run_add_kernel(
        bench_2d_arrangement, bench_add_application,
        (Tensor(2), Tensor(2)), input_2d, output_2d,
        device, "bench_2d_div", tiling_hint=hint_2d,
    )
    results.append({
        "scenario": "2d_divisible_hit",
        "size": "(512, 512)",
        "baseline_runtime_ms": round(bl_rt_2d, 4),
        "submitted_runtime_ms": round(sub_rt_2d, 4),
        "speedup": round(bl_rt_2d / sub_rt_2d if sub_rt_2d > 0 else 0, 4),
        "specialization_hit": True,
        "baseline_metrics": bl_met_2d,
        "submitted_metrics": sub_met_2d,
    })

    # Scenario 6: 2D non-divisible fallback
    input_2d_nd = torch.randn((519, 519), device=device)
    output_2d_nd = torch.empty_like(input_2d_nd)

    hint_fallback_2d = TilingHint()
    bl_rt_2d_nd, bl_met_2d_nd = run_add_kernel(
        bench_2d_arrangement, bench_add_application,
        (Tensor(2), Tensor(2)), input_2d_nd, output_2d_nd,
        device, "bench_2d_nd_baseline", tiling_hint=None,
    )
    sub_rt_2d_nd, sub_met_2d_nd = run_add_kernel(
        bench_2d_arrangement, bench_add_application,
        (Tensor(2), Tensor(2)), input_2d_nd, output_2d_nd,
        device, "bench_2d_nd", tiling_hint=hint_fallback_2d,
    )
    results.append({
        "scenario": "2d_non_divisible_fallback",
        "size": "(519, 519)",
        "baseline_runtime_ms": round(bl_rt_2d_nd, 4),
        "submitted_runtime_ms": round(sub_rt_2d_nd, 4),
        "speedup": round(bl_rt_2d_nd / sub_rt_2d_nd if sub_rt_2d_nd > 0 else 0, 4),
        "specialization_hit": False,
        "baseline_metrics": bl_met_2d_nd,
        "submitted_metrics": sub_met_2d_nd,
    })

    # Write results
    output_path = pathlib.Path(__file__).parent / "bench_specialization_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "benchmark_name": "T1-2-1 Specialization",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": device,
            "results": results,
            "summary": {
                "total_scenarios": len(results),
                "specialization_hit_scenarios": sum(
                    1 for r in results if r["specialization_hit"]
                ),
                "fallback_scenarios": sum(
                    1 for r in results if not r["specialization_hit"]
                ),
            },
        }, f, indent=2)

    print(f"Benchmark results written to: {output_path}")
    return results


if __name__ == "__main__":
    main()
