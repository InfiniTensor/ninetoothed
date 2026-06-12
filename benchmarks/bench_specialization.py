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

import ninetoothed, ninetoothed.naming as naming
from ninetoothed import Symbol, Tensor
from ninetoothed.generation import CodeGenerator, TilingHint


# ---------------------------------------------------------------------------
# Utility: count generated code metrics
# ---------------------------------------------------------------------------

def count_metrics(source_text):
    """Count code quality metrics in generated Triton source.

    mask_complexity counts boundary condition conjuncts (&) in mask
    expressions. mask=True has 0 complexity; compound masks have N.
    Stride/pointer counts only look in the kernel body (after the
    function signature's first line) to avoid counting parameter decls.
    """
    lines = source_text.splitlines()
    # Find where kernel body starts (first non-decorator, non-def line after def)
    body_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            body_start = i + 1
            break
    body_text = "\n".join(lines[body_start:]) if body_start < len(lines) else source_text

    # Count & inside mask= expressions as a proxy for mask complexity
    mask_parts = re.findall(r"mask=[^,)]+", body_text)
    mask_complexity = sum(
        part.count(" & ") for part in mask_parts
    )
    return {
        "mask_complexity": mask_complexity,
        "mask_expr_count": len(re.findall(r"mask=", body_text)),
        "stride_expr_count": len(re.findall(r"_stride_\d+", body_text)),
        "pointer_expr_count": len(re.findall(r"_pointers\s*\+", body_text)),
        "source_line_count": len(lines),
    }


# ---------------------------------------------------------------------------
# Benchmark kernel: 1D vector add
# ---------------------------------------------------------------------------

BENCH_BLOCK_SIZE = 256


def bench_add_arrangement(x, output):
    return x.tile((BENCH_BLOCK_SIZE,)), output.tile((BENCH_BLOCK_SIZE,))


def bench_add_application(x, output):
    output = x  # noqa: F841


def _prepare_app(arrangement, application, tensors):
    """Set up annotations on application so CodeGenerator can be called directly."""
    import inspect as _inspect
    params = _inspect.signature(application).parameters
    types = arrangement(*tensors)
    types = types if isinstance(types, tuple) else (types,)
    application.__annotations__ = {param: typ for param, typ in zip(params, types)}


def _auto_hint(tensors, has_divisible, use_contiguous):
    """Build a TilingHint using actual tensor source names from the list.

    Only marks innermost dimension as contiguous (stride=1). Outer dims
    have stride=N_cols etc., which is NOT 1 even for contiguous tensors.
    """
    contiguous_dims = set()
    known_strides = {}
    if use_contiguous:
        for t in tensors:
            if t.source.ndim == 0:
                continue
            bare = naming.remove_prefixes(t.source.name)
            innermost = t.source.ndim - 1
            contiguous_dims.add((bare, innermost))
            known_strides[(bare, innermost)] = 1
    return TilingHint(
        has_divisible_tiles=has_divisible,
        contiguous_dims=contiguous_dims,
        known_strides=known_strides,
        exact_innermost_sizes=has_divisible,
    )


def run_add_kernel(arrangement, application, tensors, input_data, output_data,
                   device, kernel_name, tiling_hint=None, warmup=5, iters=100):
    """Run a kernel and return (runtime_ms, source_metrics)."""
    if tiling_hint is not None and tiling_hint.is_active():
        _prepare_app(arrangement, application, tensors)
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
        kernel = ninetoothed.make(
            arrangement, application, tensors, kernel_name=kernel_name,
        )
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
        "variant_name": (
            "contiguous_divisible_fast" if specialization_hit and tiling_hint.has_divisible_tiles and bool(tiling_hint.contiguous_dims)
            else "divisible_fast" if specialization_hit and tiling_hint.has_divisible_tiles
            else "contiguous_fast" if specialization_hit and bool(tiling_hint.contiguous_dims)
            else "general_fallback"
        ),
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
    t1d_a = (Tensor(1), Tensor(1))
    result = benchmark_scenario(
        "contiguous_divisible_hit", 2048,
        bench_add_arrangement, bench_add_application, t1d_a,
        device, _auto_hint(t1d_a, True, True), specialization_hit=True,
    )
    results.append(result)

    # Scenario 2: Contiguous only, non-divisible
    t1d_b = (Tensor(1), Tensor(1))
    result = benchmark_scenario(
        "contiguous_only_hit", 1027,
        bench_add_arrangement, bench_add_application, t1d_b,
        device, _auto_hint(t1d_b, False, True), specialization_hit=True,
    )
    results.append(result)

    # Scenario 3: Divisible only, non-contiguous
    t1d_c = (Tensor(1), Tensor(1))
    result = benchmark_scenario(
        "divisible_only_hit", 2048,
        bench_add_arrangement, bench_add_application, t1d_c,
        device, _auto_hint(t1d_c, True, False), specialization_hit=True,
    )
    results.append(result)

    # Scenario 4: Pure fallback
    result = benchmark_scenario(
        "pure_fallback", 1027,
        bench_add_arrangement, bench_add_application, (Tensor(1), Tensor(1)),
        device, TilingHint(), specialization_hit=False,
    )
    results.append(result)

    # Scenario 5: 2D divisible
    def bench_2d_arrangement(x, output):
        return x.tile((64, 64)), output.tile((64, 64))

    t2d_a = (Tensor(2), Tensor(2))
    hint_2d = _auto_hint(t2d_a, True, False)
    input_2d = torch.randn((512, 512), device=device)
    output_2d = torch.empty_like(input_2d)

    bl_rt_2d, bl_met_2d = run_add_kernel(
        bench_2d_arrangement, bench_add_application, t2d_a,
        input_2d, output_2d, device, "bench_2d_div_baseline", tiling_hint=None,
    )
    sub_rt_2d, sub_met_2d = run_add_kernel(
        bench_2d_arrangement, bench_add_application, t2d_a,
        input_2d, output_2d, device, "bench_2d_div", tiling_hint=hint_2d,
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
    t2d_b = (Tensor(2), Tensor(2))
    hint_2d_fb = TilingHint()
    input_2d_nd = torch.randn((519, 519), device=device)
    output_2d_nd = torch.empty_like(input_2d_nd)

    bl_rt_2d_nd, bl_met_2d_nd = run_add_kernel(
        bench_2d_arrangement, bench_add_application, t2d_b,
        input_2d_nd, output_2d_nd, device, "bench_2d_nd_baseline", tiling_hint=None,
    )
    sub_rt_2d_nd, sub_met_2d_nd = run_add_kernel(
        bench_2d_arrangement, bench_add_application, t2d_b,
        input_2d_nd, output_2d_nd, device, "bench_2d_nd", tiling_hint=hint_2d_fb,
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
