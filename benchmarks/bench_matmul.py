"""Real matmul benchmark — compute-heavy kernel where mask/stride savings matter.

Compares baseline vs hinted code generation on matmul with divisible
dimensions (1024x1024x1024, tile 128x128). Each kernel call does ~64 tiles
of dot-product accumulation — enough compute that mask/stride overhead
is measurable.
"""

import json, pathlib, re, time
import torch, ninetoothed
import ninetoothed.language as ntl
import ninetoothed.naming as naming
from ninetoothed import Symbol, Tensor
from ninetoothed.generation import CodeGenerator, TilingHint

torch.manual_seed(42)

BLOCK_M = Symbol("BM", meta=True, lower_bound=64, upper_bound=128)
BLOCK_N = Symbol("BN", meta=True, lower_bound=64, upper_bound=128)
BLOCK_K = Symbol("BK", meta=True, lower_bound=64, upper_bound=128)


def matmul_arrangement(lhs, rhs, output):
    output_tiled = output.tile((BLOCK_M, BLOCK_N))
    lhs_tiled = lhs.tile((BLOCK_M, BLOCK_K)).tile((1, -1)).expand((-1, output_tiled.shape[1]))
    lhs_tiled.dtype = lhs_tiled.dtype.squeeze(0)
    rhs_tiled = rhs.tile((BLOCK_K, BLOCK_N)).tile((-1, 1)).expand((output_tiled.shape[0], -1))
    rhs_tiled.dtype = rhs_tiled.dtype.squeeze(1)
    return lhs_tiled, rhs_tiled, output_tiled


def matmul_application(lhs, rhs, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)
    for k in range(lhs.shape[0]):
        accumulator += ntl.dot(lhs[k], rhs[k])
    output = accumulator.to(ntl.float16)


def _prepare_app(arrangement, application, tensors):
    import inspect
    params = inspect.signature(application).parameters
    types = arrangement(*tensors)
    types = types if isinstance(types, tuple) else (types,)
    application.__annotations__ = {p: t for p, t in zip(params, types)}


def count_metrics(source_text):
    lines = source_text.splitlines()
    body_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            body_start = i + 1
            break
    body_text = "\n".join(lines[body_start:]) if body_start < len(lines) else source_text
    mask_parts = re.findall(r"mask=[^,)]+", body_text)
    mask_complexity = sum(p.count(" & ") for p in mask_parts)
    return {
        "mask_complexity": mask_complexity,
        "mask_expr_count": len(re.findall(r"mask=", body_text)),
        "stride_expr_count": len(re.findall(r"_stride_\d+", body_text)),
        "source_line_count": len(lines),
    }


def run_matmul(application, tensors, device, kernel_name, tiling_hint=None,
               M=1024, N=1024, K=1024, warmup=5, iters=100):
    """Run matmul and return (runtime_ms, metrics, source_text, correct)."""
    lhs = torch.randn((M, K), dtype=torch.float16, device=device)
    rhs = torch.randn((K, N), dtype=torch.float16, device=device)
    output = torch.empty((M, N), dtype=torch.float16, device=device)

    if tiling_hint is not None and tiling_hint.is_active():
        _prepare_app(matmul_arrangement, application, tensors)
        gen = CodeGenerator(tiling_hint=tiling_hint)
        sf = gen(application, caller="torch", kernel_name=kernel_name,
                 num_warps=4, num_stages=3, max_num_configs=None, prettify=False)
    else:
        k = ninetoothed.make(matmul_arrangement, application, tensors,
                             kernel_name=kernel_name, num_warps=4, num_stages=3)
        sf = k._source

    source_text = pathlib.Path(sf).read_text()
    metrics = count_metrics(source_text)

    import importlib, sys
    mod = importlib.util.module_from_spec(
        importlib.util.spec_from_file_location(f"mm_{kernel_name}", sf))
    sys.modules[f"mm_{kernel_name}"] = mod
    mod_spec = importlib.util.spec_from_file_location(f"mm_{kernel_name}", sf)
    mod = importlib.util.module_from_spec(mod_spec)
    sys.modules[f"mm_{kernel_name}"] = mod
    mod_spec.loader.exec_module(mod)
    launch = getattr(mod, f"launch_{kernel_name}")

    for _ in range(warmup):
        launch(lhs, rhs, output)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        launch(lhs, rhs, output)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    expected = torch.matmul(lhs.float(), rhs.float()).to(torch.float16)
    correct = torch.allclose(output, expected, atol=0.5)
    runtime_ms = (elapsed / iters) * 1000.0
    return runtime_ms, metrics, source_text, correct


def main():
    device = "cuda"
    if not torch.cuda.is_available():
        print("No CUDA!"); return

    results = []
    tensors = (Tensor(2, dtype=ninetoothed.float16),
               Tensor(2, dtype=ninetoothed.float16),
               Tensor(2, dtype=ninetoothed.float16))

    # Use a single fixed set of tensors so names are consistent
    bare_names = tuple(naming.remove_prefixes(t.source.name) for t in tensors)

    # Only mark innermost dim (dim 1 for 2D) as contiguous stride=1.
    # Outer dim (dim 0) has stride=N (number of columns), NOT 1.
    contig_dims = {(bare_names[i], 1) for i in range(3)}
    contig_strides = {(bare_names[i], 1): 1 for i in range(3)}

    scenarios = [
        ("matmul_stride_hit", 1024, 1024, 1024,
         TilingHint(has_divisible_tiles=False, exact_innermost_sizes=False,
                     contiguous_dims=contig_dims,
                     known_strides=contig_strides),
         True, "contiguous_fast"),
        ("matmul_fallback", 1027, 1023, 1025,
         TilingHint(), False, "general_fallback"),
    ]

    for name, M, N, K, hint, spec_hit, vname in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {name}  M={M} N={N} K={K}")
        print(f"{'='*60}")

        # Baseline
        bl_rt, bl_met, bl_src, bl_ok = run_matmul(
            matmul_application, tensors, device, f"mm_{name}_bl",
            tiling_hint=None, M=M, N=N, K=K,
        )
        print(f"Baseline:  {bl_rt:.3f}ms  mask_cmplx={bl_met['mask_complexity']}  "
              f"stride={bl_met['stride_expr_count']}  lines={bl_met['source_line_count']}  ok={bl_ok}")

        # Submitted
        sub_rt, sub_met, sub_src, sub_ok = run_matmul(
            matmul_application, tensors, device, f"mm_{name}_sub",
            tiling_hint=hint, M=M, N=N, K=K,
        )
        print(f"Submitted: {sub_rt:.3f}ms  mask_cmplx={sub_met['mask_complexity']}  "
              f"stride={sub_met['stride_expr_count']}  lines={sub_met['source_line_count']}  ok={sub_ok}")

        sp = bl_rt / sub_rt if sub_rt > 0 else 0
        print(f"Speedup: {sp:.4f}  hit={spec_hit}")

        # Print diff for first scenario
        if name == "matmul_divisible_hit":
            print(f"\n--- Source diff (first 3 changes) ---")
            bl_lines = bl_src.splitlines()
            sub_lines = sub_src.splitlines()
            diffs = 0
            for i, (bl, sl) in enumerate(zip(bl_lines, sub_lines)):
                if bl != sl and diffs < 3:
                    print(f"Line {i+1}:")
                    print(f"  - {bl[:120]}{'...' if len(bl)>120 else ''}")
                    print(f"  + {sl[:120]}{'...' if len(sl)>120 else ''}")
                    diffs += 1

        results.append({
            "scenario": name,
            "size": f"M={M},N={N},K={K}",
            "variant_name": vname,
            "baseline_runtime_ms": round(bl_rt, 4),
            "submitted_runtime_ms": round(sub_rt, 4),
            "speedup": round(sp, 4),
            "specialization_hit": spec_hit,
            "correctness_ok": bl_ok and sub_ok,
            "baseline_metrics": bl_met,
            "submitted_metrics": sub_met,
        })

    out = pathlib.Path(__file__).parent / "matmul_bench_results.json"
    with open(out, "w") as f:
        json.dump({"benchmark_name": "T1-2-1 Matmul", "device": device,
                    "results": results,
                    "summary": {"total": len(results),
                                "hit": sum(1 for r in results if r["specialization_hit"]),
                                "all_correct": all(r["correctness_ok"] for r in results)}},
                  f, indent=2)
    print(f"\nResults: {out}")
    return results


if __name__ == "__main__":
    main()
