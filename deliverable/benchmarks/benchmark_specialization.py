"""
Benchmark for NineToothed code generation specialization (T1-2-1).

Measures per-scenario:
  - baseline_runtime_ms, submitted_runtime_ms, speedup
  - specialization_hit (boolean)
  - Generated code metrics: mask_expr_count, stride_expr_count, source_line_count

Output: JSON with full comparison table.

Usage:
  python non-deliverable/benchmarks/benchmark_specialization.py [--output results.json]
"""

import argparse
import json
import os
import re
import time

import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor


# ── Known baseline metrics (collected from commit a1b0694) ─────────────────
# For scenarios where baseline crashed, baseline is None (comparison fields omitted).

BASELINE = {
    "vec_add_hit":              {"runtime_ms": 0.023, "mask_expr_count": 4, "stride_expr_count": 3, "source_line_count": 5},
    "vec_add_store_hit":        {"runtime_ms": 0.025, "mask_expr_count": 4, "stride_expr_count": 6, "source_line_count": 7},
    "vec_add_divisible_hit":    {"runtime_ms": 0.023, "mask_expr_count": 4, "stride_expr_count": 3, "source_line_count": 5},
    "unsqueeze_fallback":       None,  # baseline crash (Symbol.upper_bound AttributeError in _generate_autotune)
    "slice_fallback":           None,  # baseline crash (same bug)
    "expand_hit":               {"runtime_ms": 0.019, "mask_expr_count": 4, "stride_expr_count": 3, "source_line_count": 5},
    "matmul_small_hit":         {"runtime_ms": 0.237, "mask_expr_count": 27, "stride_expr_count": None, "source_line_count": None},
    "matmul_medium_hit":        {"runtime_ms": None,  "mask_expr_count": None,  "stride_expr_count": None,  "source_line_count": None},
    "matmul_large_hit":         {"runtime_ms": 0.092, "mask_expr_count": 27, "stride_expr_count": None,  "source_line_count": None},
    "softmax_hit":              {"runtime_ms": None,  "mask_expr_count": None,  "stride_expr_count": None,  "source_line_count": None},
    "ntops_add_hit":            {"runtime_ms": None,  "mask_expr_count": None,  "stride_expr_count": None,  "source_line_count": None},
    "ntops_relu_hit":           {"runtime_ms": None,  "mask_expr_count": None,  "stride_expr_count": None,  "source_line_count": None},
    "ntops_gelu_hit":           {"runtime_ms": None,  "mask_expr_count": None,  "stride_expr_count": None,  "source_line_count": None},
    "ntops_softmax_hit":        {"runtime_ms": None,  "mask_expr_count": None,  "stride_expr_count": None,  "source_line_count": None},
    "ntops_layer_norm_hit":     {"runtime_ms": None,  "mask_expr_count": None,  "stride_expr_count": None,  "source_line_count": None},
    "ntops_rms_norm_hit":       {"runtime_ms": None,  "mask_expr_count": None,  "stride_expr_count": None,  "source_line_count": None},
    "ntops_mm_divisible_hit":   {"runtime_ms": None,  "mask_expr_count": None,  "stride_expr_count": None,  "source_line_count": None},
    "ntops_unsqueeze_fallback": {"runtime_ms": None,  "mask_expr_count": None,  "stride_expr_count": None,  "source_line_count": None},
}


# ── Metric helpers ─────────────────────────────────────────────────────────

def _get_source(kernel):
    path = getattr(kernel, "_source", None)
    if path is None:
        return ""
    if isinstance(path, str) and os.path.exists(path):
        with open(path) as f:
            return f.read()
    return str(path)


def _count_mask_exprs(source):
    """Count mask sub-conditions. Mask ends before ', other=' or ', boundary_check=' etc."""
    idx = source.find("mask=")
    if idx < 0:
        return 0
    after = source[idx + 5:]
    m = re.search(
        r",[\s]*(?:other|boundary_check|cache_modifier|eviction_policy|padding_option)[\s]*=",
        after,
    )
    mask_text = after[: m.start()] if m else after
    return mask_text.count("&") + 1


def _count_source_lines(source):
    in_func = False
    count = 0
    for line in source.splitlines():
        s = line.strip()
        if "@triton.jit" in s:
            in_func = True
            continue
        if in_func:
            if s.startswith("def "):
                continue
            if s and not s.startswith("#") and not s.startswith("import"):
                count += 1
    return count


def _count_stride_exprs(source):
    idx = source.find("@triton.jit")
    if idx < 0:
        idx = 0
    return source[idx:].count("stride")


def _has_lower_bound(source):
    return " >= 0" in source


def _has_max_contiguous(source):
    return "max_contiguous" in source


def _detect_specialization(name, source):
    """Per-scenario specialization check.

    For hit-type scenarios: True if specialisation was triggered (lower-bound
    removed, broadcast simplified, contiguous path taken).

    For fallback-type scenarios: True if the generic path was correctly preserved
    (lower bound still present).
    """
    has_lb = _has_lower_bound(source)
    has_tlmc = _has_max_contiguous(source)
    mask_n = _count_mask_exprs(source)

    if name in ("vec_add_hit", "vec_add_store_hit", "vec_add_divisible_hit"):
        return not has_lb
    if name in ("unsqueeze_fallback", "slice_fallback"):
        return has_lb
    if name == "expand_hit":
        return has_tlmc or not has_lb or mask_n <= 3
    if name.startswith("matmul"):
        return not has_lb or has_tlmc or mask_n <= 5
    if name == "softmax_hit":
        return not has_lb
    if name.startswith("ntops_"):
        if name.endswith("_fallback"):
            return has_lb
        return not has_lb or has_tlmc
    return False


# ── Scenarios ──────────────────────────────────────────────────────────────

SCENARIOS = []


def _vec_add_hit(size=4096 * 1024, device="cuda"):
    BLOCK = 1024

    def arrangement(x):
        return x.tile((BLOCK,))

    def application(x):
        x

    kernel = ninetoothed.make(arrangement, application, (Tensor(1),))
    x = torch.randn((size,), device=device)
    return kernel, (x,)


SCENARIOS.append({
    "name": "vec_add_hit",
    "type": "hit",
    "input_shape": "(4096*1024,) tile=1024",
    "description": "1D vector add — lower bound + arange bounds removed",
    "run": _vec_add_hit,
})


def _vec_add_store_hit(size=4096 * 1024, device="cuda"):
    BLOCK = 1024

    def arrangement(x, output):
        return x.tile((BLOCK,)), output.tile((BLOCK,))

    def application(x, output):
        output = x

    kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1)))
    x = torch.randn((size,), device=device)
    out = torch.empty_like(x)
    return kernel, (x, out)


SCENARIOS.append({
    "name": "vec_add_store_hit",
    "type": "hit",
    "input_shape": "(4096*1024,) tile=1024",
    "description": "1D vector add with store — lower bound removed",
    "run": _vec_add_store_hit,
})


def _vec_add_divisible_hit(size=131072, device="cuda"):
    BLOCK = 1024

    def arrangement(x):
        return x.tile((BLOCK,))

    def application(x):
        x

    kernel = ninetoothed.make(arrangement, application, (Tensor(1),))
    x = torch.randn((size,), device=device)
    return kernel, (x,)


SCENARIOS.append({
    "name": "vec_add_divisible_hit",
    "type": "hit",
    "input_shape": "(131072,) tile=1024 [size%tile==0]",
    "description": "1D vector add divisible tile — source upper bound removed",
    "run": _vec_add_divisible_hit,
})


def _unsqueeze_fallback(size=4096 * 1024, device="cuda"):
    def arrangement(x, output):
        return x.unsqueeze(0), output.unsqueeze(0)

    def application(x, output):
        output = x

    kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1)))
    x = torch.randn((size,), device=device)
    out = torch.empty((1, size), device=device)
    return kernel, (x, out)


SCENARIOS.append({
    "name": "unsqueeze_fallback",
    "type": "fallback",
    "input_shape": "(4096*1024,) -> (1, N)",
    "description": "1D unsqueeze(0) — unary level, fallback preserves >=0",
    "run": _unsqueeze_fallback,
})


def _slice_fallback(size=4096 * 1024, device="cuda"):
    def arrangement(input, output, input_slices, output_slices):
        return input[input_slices], output[output_slices]

    def application(input, output):
        output = input

    tensors = (Tensor(1), Tensor(1), (slice(-100, None),), (slice(100, None),))
    kernel = ninetoothed.make(arrangement, application, tensors)
    x = torch.randn((size,), device=device)
    out = torch.empty((size - 100,), device=device)
    return kernel, (x, out)


SCENARIOS.append({
    "name": "slice_fallback",
    "type": "fallback",
    "input_shape": "(4096*1024,) -> (N-100,)",
    "description": "Slice negative start — pad-like fallback, preserves bounds",
    "run": _slice_fallback,
})


def _expand_hit(size=4096 * 1024, device="cuda"):
    BLOCK = 512

    def arrangement(x, BLOCK=BLOCK):
        return (x.expand((BLOCK,)).tile((BLOCK,)),)

    def application(x):
        x

    kernel = ninetoothed.make(arrangement, application, (Tensor(1),))
    x = torch.randn((size,), device=device)
    return kernel, (x,)


SCENARIOS.append({
    "name": "expand_hit",
    "type": "hit",
    "input_shape": f"(4096*1024,) expand+{512}",
    "description": "Expand + tile — broadcast 0>=0 and 0<expr simplified",
    "run": _expand_hit,
})


# ── Compute-bound: matmul (uses meta=True + max_num_configs) ───────────────

BM = Symbol("BM", meta=True)
BN = Symbol("BN", meta=True)
BK = Symbol("BK", meta=True)


def _arrangement_matmul(lhs, rhs, output, BM=BM, BN=BN, BK=BK):
    ot = output.tile((BM, BN))
    lt = lhs.tile((BM, BK)).tile((1, -1)).expand((-1, ot.shape[1]))
    lt.dtype = lt.dtype.squeeze(0)
    rt = rhs.tile((BK, BN)).tile((-1, 1)).expand((ot.shape[0], -1))
    rt.dtype = rt.dtype.squeeze(1)
    return lt, rt, ot


def _application_matmul(lhs, rhs, output):
    acc = ntl.zeros(output.shape, dtype=ntl.float32)
    for kd in range(lhs.shape[0]):
        acc += ntl.dot(lhs[kd], rhs[kd])
    output = acc.to(ntl.float16)


def _matmul_small_hit(device="cuda"):
    m = n = k = 512
    kernel = ninetoothed.make(
        _arrangement_matmul, _application_matmul,
        (Tensor(2), Tensor(2), Tensor(2)),
        max_num_configs=10,
    )
    lhs = torch.randn((m, k), device=device, dtype=torch.float16)
    rhs = torch.randn((k, n), device=device, dtype=torch.float16)
    out = torch.empty((m, n), device=device, dtype=torch.float16)
    return kernel, (lhs, rhs, out)


SCENARIOS.append({
    "name": "matmul_small_hit",
    "type": "hit",
    "input_shape": "(512,512)x(512,512) fp16",
    "description": "Matmul 512x512x512 — multi-level tile specialization",
    "run": _matmul_small_hit,
})


def _matmul_medium_hit(device="cuda"):
    m = n = k = 1024
    kernel = ninetoothed.make(
        _arrangement_matmul, _application_matmul,
        (Tensor(2), Tensor(2), Tensor(2)),
        max_num_configs=10,
    )
    lhs = torch.randn((m, k), device=device, dtype=torch.float16)
    rhs = torch.randn((k, n), device=device, dtype=torch.float16)
    out = torch.empty((m, n), device=device, dtype=torch.float16)
    return kernel, (lhs, rhs, out)


SCENARIOS.append({
    "name": "matmul_medium_hit",
    "type": "hit",
    "input_shape": "(1024,1024)x(1024,1024) fp16",
    "description": "Matmul 1024x1024x1024 — medium compute-bound kernel",
    "run": _matmul_medium_hit,
})


def _matmul_large_hit(device="cuda"):
    m = n = k = 2048
    kernel = ninetoothed.make(
        _arrangement_matmul, _application_matmul,
        (Tensor(2), Tensor(2), Tensor(2)),
        max_num_configs=10,
    )
    lhs = torch.randn((m, k), device=device, dtype=torch.float16)
    rhs = torch.randn((k, n), device=device, dtype=torch.float16)
    out = torch.empty((m, n), device=device, dtype=torch.float16)
    return kernel, (lhs, rhs, out)


SCENARIOS.append({
    "name": "matmul_large_hit",
    "type": "hit",
    "input_shape": "(2048,2048)x(2048,2048) fp16",
    "description": "Matmul 2048x2048x2048 — large compute-bound kernel",
    "run": _matmul_large_hit,
})


# ── Compute-bound: softmax ─────────────────────────────────────────────────

def _softmax_hit(device="cuda"):
    m, n = 2048, 2048
    BLOCK = Symbol("BLOCK", constexpr=True)

    @ninetoothed.jit
    def softmax_kernel(
        input_row: Tensor(2, other=float("-inf")).tile((1, BLOCK)),
        output_row: Tensor(2).tile((1, BLOCK)),
    ):
        row_minus_max = input_row - ntl.max(input_row)
        numerator = ntl.exp(row_minus_max)
        denominator = ntl.sum(numerator)
        output_row = numerator / denominator

    inp = torch.rand((m, n), device=device, dtype=torch.float32)
    out = torch.empty_like(inp)
    return softmax_kernel, (inp, out, n)


# ═══════════════════════════════════════════════════════════════════════════
# NTOps-based scenarios — reuse upstream operator library kernels
# ═══════════════════════════════════════════════════════════════════════════

def _ntops_add_hit(device="cuda"):
    import ntops
    x = torch.randn(4096 * 1024, device=device)
    y = torch.randn(4096 * 1024, device=device)
    # invoke once to compile, then grab source from cache
    _ = ntops.torch.add(x, y)
    from ntops.torch.utils import _cached_make
    kernel = _cached_make(ntops.kernels.add.premake, 1, dtype=x.dtype)
    return kernel, (x, y, 1.0, torch.empty_like(x))

SCENARIOS.append({
    "name": "ntops_add_hit",
    "type": "hit",
    "input_shape": "(4M,) element-wise",
    "description": "NTOps add — element_wise arrangement, contiguous flatten+tile",
    "run": _ntops_add_hit,
})

def _ntops_relu_hit(device="cuda"):
    import ntops
    x = torch.randn(4096 * 1024, device=device)
    _ = ntops.torch.relu(x)
    from ntops.torch.utils import _cached_make
    kernel = _cached_make(ntops.kernels.relu.premake, 1, dtype=x.dtype)
    return kernel, (x, torch.empty_like(x))

SCENARIOS.append({
    "name": "ntops_relu_hit",
    "type": "hit",
    "input_shape": "(4M,) element-wise",
    "description": "NTOps relu — unary activation, same path as add",
    "run": _ntops_relu_hit,
})

def _ntops_gelu_hit(device="cuda"):
    import ntops
    x = torch.randn(4096 * 1024, device=device)
    _ = ntops.torch.gelu(x)
    from ntops.torch.utils import _cached_make
    kernel = _cached_make(ntops.kernels.gelu.premake, 1, False, dtype=x.dtype)
    return kernel, (x, torch.empty_like(x))

SCENARIOS.append({
    "name": "ntops_gelu_hit",
    "type": "hit",
    "input_shape": "(4M,) element-wise",
    "description": "NTOps gelu — composite activation (erf), multiple ops in body",
    "run": _ntops_gelu_hit,
})

def _ntops_softmax_hit(device="cuda"):
    import ntops
    x = torch.randn(1024, 256, device=device)
    _ = ntops.torch.softmax(x, dim=-1)
    from ntops.torch.utils import _cached_make
    kernel = _cached_make(ntops.kernels.softmax.premake, 2, dim=-1, dtype=x.dtype)
    return kernel, (x, torch.empty_like(x))

SCENARIOS.append({
    "name": "ntops_softmax_hit",
    "type": "hit",
    "input_shape": "(1024,256) dim=-1",
    "description": "NTOps softmax — reduction arrangement, online algorithm",
    "run": _ntops_softmax_hit,
})

def _ntops_layer_norm_hit(device="cuda"):
    import ntops
    x = torch.randn(32, 256, device=device)
    w = torch.randn(256, device=device)
    b = torch.randn(256, device=device)
    _ = ntops.torch.layer_norm(x, (256,), weight=w, bias=b)
    from ntops.torch.utils import _cached_make
    kernel = _cached_make(ntops.kernels.layer_norm.premake, 2, (256,), dtype=x.dtype)
    wexp = w.unsqueeze(0).expand(32, -1)
    bexp = b.unsqueeze(0).expand(32, -1)
    import math
    return kernel, (x, wexp, bexp, 1e-5, torch.empty_like(x), math.prod((256,)))

SCENARIOS.append({
    "name": "ntops_layer_norm_hit",
    "type": "hit",
    "input_shape": "(32,256) norm(256)",
    "description": "NTOps layer_norm — reduction + broadcast weight/bias",
    "run": _ntops_layer_norm_hit,
})

def _ntops_rms_norm_hit(device="cuda"):
    import ntops, math
    x = torch.randn(32, 512, device=device)
    w = torch.randn(512, device=device)
    _ = ntops.torch.rms_norm(x, (512,), weight=w)
    from ntops.torch.utils import _cached_make
    kernel = _cached_make(ntops.kernels.rms_norm.premake, 2, 1, input_dtype=x.dtype)
    wexp = w.unsqueeze(0).expand(32, -1)
    return kernel, (x, wexp, 1e-5, torch.empty_like(x), math.prod((512,)))

SCENARIOS.append({
    "name": "ntops_rms_norm_hit",
    "type": "hit",
    "input_shape": "(32,512) norm(512)",
    "description": "NTOps rms_norm — reduction + broadcast weight",
    "run": _ntops_rms_norm_hit,
})

def _ntops_mm_hit(device="cuda"):
    import ntops
    x = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    y = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    _ = ntops.torch.mm(x, y)
    from ntops.torch.utils import _cached_make, _get_matmul_input_precision
    kernel = _cached_make(ntops.kernels.mm.premake, input_precision=_get_matmul_input_precision())
    p = _get_matmul_input_precision()
    return kernel, (x, y, torch.empty(1024, 1024, device=device, dtype=torch.float16), p)

SCENARIOS.append({
    "name": "ntops_mm_divisible_hit",
    "type": "hit",
    "input_shape": "(1024,1024)x(1024,1024) fp16",
    "description": "NTOps mm — matmul with triple tile, divisible dims",
    "run": _ntops_mm_hit,
})

# ── NTOps fallback scenarios ────────────────────────────────────────────────

def _ntops_unsqueeze_fallback(device="cuda"):
    """unsqueeze-like: using a unary op that adds a dim"""
    def arrangement(x, output):
        return x.unsqueeze(0), output.unsqueeze(0)
    def application(x, output):
        output = x
    kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1)))
    x = torch.randn(4096 * 1024, device=device)
    out = torch.empty((1, 4096 * 1024), device=device)
    return kernel, (x, out)

SCENARIOS.append({
    "name": "ntops_unsqueeze_fallback",
    "type": "fallback",
    "input_shape": "(4M,) -> (1, 4M)",
    "description": "Unsqueeze — unary level present, must preserve >=0",
    "run": _ntops_unsqueeze_fallback,
})

def bench_one(scenario, num_warmup=5, num_iter=20):
    kernel, args = scenario["run"]()
    source = _get_source(kernel)

    mask_count = _count_mask_exprs(source) if source else -1
    line_count = _count_source_lines(source) if source else -1
    stride_count = _count_stride_exprs(source) if source else -1
    has_lb = _has_lower_bound(source) if source else None
    has_tlmc = _has_max_contiguous(source) if source else None
    spec_hit = _detect_specialization(scenario["name"], source) if source else False

    # Warmup (triggers autotuning / JIT compilation)
    for _ in range(num_warmup):
        kernel(*args)
    torch.cuda.synchronize()

    # Timed iterations
    start = time.perf_counter()
    for _ in range(num_iter):
        kernel(*args)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) / num_iter * 1000

    return {
        "mask_expr_count": mask_count,
        "source_line_count": line_count,
        "stride_expr_count": stride_count,
        "has_lower_bound_in_mask": has_lb,
        "has_tl_max_contiguous": has_tlmc,
        "submitted_runtime_ms": round(elapsed_ms, 4),
        "specialization_hit": spec_hit,
    }


def run_all(output_path=None):
    results = []
    for s in SCENARIOS:
        name = s["name"]
        print(f"[bench] {name} ({s['type']}) ... ", end="", flush=True)
        try:
            m = bench_one(s)
            m["name"] = name
            m["type"] = s["type"]
            m["input_shape"] = s["input_shape"]
            m["description"] = s["description"]

            bl = BASELINE.get(name)
            if bl is not None:
                m["baseline_runtime_ms"] = bl["runtime_ms"]
                m["baseline_mask_expr_count"] = bl["mask_expr_count"]
                m["baseline_stride_expr_count"] = bl["stride_expr_count"]
                m["baseline_source_line_count"] = bl["source_line_count"]
                if m["submitted_runtime_ms"] > 0 and bl["runtime_ms"] is not None:
                    m["speedup"] = round(bl["runtime_ms"] / m["submitted_runtime_ms"], 4)
                else:
                    m["speedup"] = None
                if bl["mask_expr_count"] is not None and bl["mask_expr_count"] > 0:
                    m["mask_reduction"] = round(
                        (bl["mask_expr_count"] - m["mask_expr_count"]) / bl["mask_expr_count"], 4
                    )
                else:
                    m["mask_reduction"] = None
            else:
                m["baseline_runtime_ms"] = None
                m["baseline_mask_expr_count"] = None
                m["baseline_stride_expr_count"] = None
                m["baseline_source_line_count"] = None
                m["speedup"] = None
                m["mask_reduction"] = None

            results.append(m)
            msg = (f"OK  rt={m['submitted_runtime_ms']:.3f}ms"
                   f"  mask={m['mask_expr_count']}"
                   f"  spec_hit={m['specialization_hit']}"
                   f"  speedup={m.get('speedup')}")
            print(msg)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"FAIL  {e}")
            results.append({
                "name": name, "type": s["type"],
                "input_shape": s["input_shape"], "description": s["description"],
                "error": str(e),
            })

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[bench] Results written to {output_path}")

    _print_summary(results)
    return results


def _print_summary(results):
    hdr = (f"{'Scenario':<26} {'Type':<9} {'Base_ms':>8} {'Sub_ms':>8}"
           f" {'Speedup':>9} {'MaskRdx':>8} {'SpecHit':>8} {'Regr':>6}")
    sep = "-" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for r in results:
        if "error" in r:
            print(f"{r['name']:<26} {'FAIL':<9} {'—':>8} {'—':>8} {'—':>9} {'—':>8} {'—':>8} {'—':>6}")
            continue
        bl = f"{r['baseline_runtime_ms']:.3f}" if r.get("baseline_runtime_ms") is not None else "—"
        su = f"{r['speedup']:.3f}x" if r.get("speedup") is not None else "—"
        mr = f"{r['mask_reduction']:.1%}" if r.get("mask_reduction") is not None else "—"
        regr = "YES" if (r.get("speedup") is not None and r["speedup"] < 0.95) else "NO"
        print(f"{r['name']:<26} {r['type']:<9} {bl:>8} {r['submitted_runtime_ms']:.3f}"
              f"  {su:>8} {mr:>8} {str(r['specialization_hit']):>8} {regr:>6}")
    print(sep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()
    run_all(output_path=args.output)
