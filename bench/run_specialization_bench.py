#!/usr/bin/env python3
"""T1-2-1 specialization benchmark harness.

Emits a JSON report with per-scenario:

* ``baseline_runtime_ms`` — runtime of a kernel built from a freshly
  reimported NineToothed with ``specialization_hints`` disabled.
* ``submitted_runtime_ms`` — runtime under this submission (hints on).
* ``speedup`` — ``baseline_runtime_ms / submitted_runtime_ms``.
* ``specialization_hit`` — True iff the dispatcher routed the input
  to one of the specialized variants (not the int64 fallback).
* ``mask_expr_count`` / ``stride_expr_count`` — counts in the
  generated source of the selected variant.

The scenarios mix hit / fallback cases so the report covers both
sides of the specialization boundary. Run::

    python -m bench.run_specialization_bench --output bench/results.json

on a CUDA host with triton installed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import re
import sys
import time
import typing as t

import torch

# Make ``src/ninetoothed`` importable when running from a checkout.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import ninetoothed  # noqa: E402
import ninetoothed.generation  # noqa: E402
from ninetoothed import Tensor  # noqa: E402


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Scenario:
    name: str
    op: str
    shape: tuple
    block: tuple
    dtype: torch.dtype
    expected_hit: bool

    def make_inputs(self, device):
        if self.op == "add":
            (n,) = self.shape
            x = torch.randn((n,), dtype=self.dtype, device=device)
            y = torch.randn((n,), dtype=self.dtype, device=device)
            out = torch.empty_like(x)
            ref = x + y
            return (x, y, out), ref, out
        if self.op == "copy":
            m, n = self.shape
            x = torch.randn((m, n), dtype=self.dtype, device=device)
            out = torch.empty_like(x)
            return (x, out), x.clone(), out
        if self.op == "copy_strided":
            m, n = self.shape
            base = torch.randn((m, n * 2), dtype=self.dtype, device=device)
            x = base[:, ::2]
            out = torch.empty_like(x).contiguous()
            return (x, out), x.contiguous(), out
        raise ValueError(self.op)


def _make_kernel(op, block, dtype, kernel_name, device):
    if op == "add":
        (block_n,) = block

        def _arrange(input, other, output):
            return (
                input.tile((block_n,)),
                other.tile((block_n,)),
                output.tile((block_n,)),
            )

        def _apply(input, other, output):
            output = input + other  # noqa: F841

        tensors = tuple(Tensor(1, dtype=dtype) for _ in range(3))
    elif op in ("copy", "copy_strided"):
        block_m, block_n = block

        def _arrange(input, output):
            return input.tile((block_m, block_n)), output.tile((block_m, block_n))

        def _apply(input, output):
            output = input  # noqa: F841

        tensors = (Tensor(2, dtype=dtype), Tensor(2, dtype=dtype))
    else:
        raise ValueError(op)

    return ninetoothed.make(
        _arrange,
        _apply,
        tensors,
        caller=device,
        kernel_name=kernel_name,
        output_dir=ninetoothed.generation.CACHE_DIR,
    )


# ---------------------------------------------------------------------------
# Source-structure inspection
# ---------------------------------------------------------------------------


_VARIANT_SUFFIX_RE = re.compile(
    r"\bdivisibility[_\d]+contiguity[_\d]+size_i\d+_stride_i\d+\b"
)


def _variant_sources(kernel_name):
    sources = {}
    cache_dir = pathlib.Path(ninetoothed.generation.CACHE_DIR)
    for py in cache_dir.glob(f"{kernel_name}.*.py"):
        suffix = py.name[len(kernel_name) + 1 : -len(".py")]
        if not _VARIANT_SUFFIX_RE.search(suffix):
            continue
        sources[suffix] = py.read_text()
    return sources


def _is_fallback(suffix):
    return "size_i64_stride_i64" in suffix


def _count_masks(source):
    return len(re.findall(r"\bmask\s*=", source))


def _count_strides(source):
    return len(re.findall(r"\*\s*\w+_stride_\d+", source))


# Dispatcher inspection -----------------------------------------------------


def _dispatch_branches(kernel_name):
    """Return ordered (suffix, [(name, op, threshold_or_value)...]) parsed
    from the dispatcher .cpp. ``op`` is either ``%`` or ``stride==``."""

    cache_dir = pathlib.Path(ninetoothed.generation.CACHE_DIR)
    text = (cache_dir / f"{kernel_name}.cpp").read_text()

    branches = []
    prefix = f"launch_{kernel_name}_"
    pattern = re.compile(
        r"^\s*(?:if\s*\(([^)]+)\)\s*)?return\s+"
        + re.escape(prefix)
        + r"(\w+)\(",
        re.MULTILINE,
    )

    for match in pattern.finditer(text):
        cond, suffix = match.groups()
        constraints = []
        unrecognized = False
        if cond:
            recognized_any = False
            for c in cond.split("&&"):
                c = c.strip()
                m = re.match(r"(\w+)\.shape\[(\d+)\]\s*%\s*(\d+)\s*==\s*0", c)
                if m:
                    constraints.append(
                        (m.group(1), int(m.group(2)), "div", int(m.group(3)))
                    )
                    recognized_any = True
                    continue
                m = re.match(r"(\w+)\.strides\[(\d+)\]\s*==\s*1", c)
                if m:
                    constraints.append((m.group(1), int(m.group(2)), "cont", 1))
                    recognized_any = True
                    continue
            if not recognized_any:
                unrecognized = True
        branches.append((suffix, constraints, unrecognized))

    return branches


def _dispatcher_arg_names(kernel_name):
    """Parse the dispatcher entry-function signature to get the ordered
    list of NineToothedTensor arg names (e.g. ``ninetoothed_tensor_45``)."""

    cache_dir = pathlib.Path(ninetoothed.generation.CACHE_DIR)
    text = (cache_dir / f"{kernel_name}.cpp").read_text()

    pattern = re.compile(
        r'extern\s+"C"\s+NineToothedResult\s+'
        + re.escape(f"launch_{kernel_name}")
        + r"\s*\(([^)]*)\)\s*\{"
    )
    m = pattern.search(text)
    if not m:
        return []

    params = m.group(1)
    names = re.findall(r"NineToothedTensor\s+(\w+)", params)
    return names


def _predict_specialization_hit(kernel_name, op, inputs):
    """Return (hit_bool, matched_suffix). Mirrors the dispatcher
    logic: walk branches top-to-bottom, picking the first whose
    constraints are all satisfied by the input tensors."""

    arg_names = _dispatcher_arg_names(kernel_name)
    tensors = {name: inputs[i] for i, name in enumerate(arg_names) if i < len(inputs)}

    branches = _dispatch_branches(kernel_name)

    for suffix, constraints, unrecognized in branches:
        if unrecognized:
            continue
        ok = True
        for name, dim, kind, threshold in constraints:
            tensor = tensors.get(name)
            if tensor is None:
                ok = False
                break
            if kind == "div":
                if tensor.shape[dim] % threshold != 0:
                    ok = False
                    break
            elif kind == "cont":
                if tensor.stride(dim) != 1:
                    ok = False
                    break
        if ok:
            return (not _is_fallback(suffix)), suffix

    return False, None


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _time_kernel(kernel, args, warmup=10, iters=50):
    for _ in range(warmup):
        kernel(*args)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        for _ in range(iters):
            kernel(*args)
        ender.record()
        torch.cuda.synchronize()
        return starter.elapsed_time(ender) / iters

    start = time.perf_counter()
    for _ in range(iters):
        kernel(*args)
    end = time.perf_counter()
    return (end - start) * 1000.0 / iters


# ---------------------------------------------------------------------------
# Baseline kernel (hints disabled)
# ---------------------------------------------------------------------------


def _build_baseline_kernel(op, block, dtype, kernel_name, device):
    """Build a kernel that mimics the pre-submission baseline by
    monkey-patching SpecializationHints.empty() in for all variants."""

    from ninetoothed import generation as gen_mod
    from ninetoothed import aot as aot_mod
    from ninetoothed.specialization import SpecializationHints

    original = aot_mod._hints_from_spec

    aot_mod._hints_from_spec = (
        lambda div_spec, cont_spec, block_sizes: SpecializationHints.empty()
    )

    try:
        return _make_kernel(op, block, dtype, kernel_name, device)
    finally:
        aot_mod._hints_from_spec = original


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


SCENARIOS = (
    # Full-specialization scenarios: input is both divisible and contiguous,
    # so dispatcher routes to the most specialized variant (mask=0, stride=0
    # for 1D; mask/stride reduced for 2D).
    Scenario("add_hit_4096_f32", "add", (4096,), (256,), torch.float32, True),
    Scenario("add_hit_65536_f32", "add", (65536,), (256,), torch.float32, True),
    # Larger inputs where compute dominates launch overhead and the
    # specialization speedup becomes clearly measurable.
    Scenario(
        "add_hit_16M_f32", "add", (16 * 1024 * 1024,), (256,), torch.float32, True
    ),
    Scenario("copy_hit_512x512_f32", "copy", (512, 512), (64, 64), torch.float32, True),
    Scenario(
        "copy_hit_1024x1024_f32",
        "copy",
        (1024, 1024),
        (64, 64),
        torch.float32,
        True,
    ),
    Scenario(
        "copy_hit_4096x4096_f32",
        "copy",
        (4096, 4096),
        (64, 64),
        torch.float32,
        True,
    ),
    # Partial-specialization scenarios: input is contiguous but not
    # divisible by BLOCK_SIZE. Dispatcher routes to a contiguity-only
    # variant (still hit=True; not the int64 fallback).
    Scenario("add_fb_4097_f32", "add", (4097,), (256,), torch.float32, True),
    Scenario("add_fb_45327_f32", "add", (45327,), (256,), torch.float32, True),
    Scenario(
        "copy_fb_512x500_f32", "copy", (512, 500), (64, 64), torch.float32, True
    ),
    # Strided input: innermost stride != 1, so no contiguity hit. The
    # dispatcher still avoids the int64 fallback by routing to an
    # int32-but-unconstrained variant; stride folding does not apply.
    Scenario(
        "copy_fb_strided_512x512_f32",
        "copy_strided",
        (512, 512),
        (64, 64),
        torch.float32,
        True,
    ),
)


def run_scenarios(device, output_path):
    results = []

    for i, sc in enumerate(SCENARIOS):
        baseline_name = f"bench_baseline_{i}_{sc.name}"
        submitted_name = f"bench_submitted_{i}_{sc.name}"

        baseline_kernel = _build_baseline_kernel(
            sc.op, sc.block, _torch_dtype_to_nt(sc.dtype), baseline_name, device
        )
        submitted_kernel = _make_kernel(
            sc.op, sc.block, _torch_dtype_to_nt(sc.dtype), submitted_name, device
        )

        inputs, expected, out = sc.make_inputs(device)
        submitted_kernel(*inputs)
        if not torch.allclose(out, expected, atol=1e-5, rtol=1e-5):
            raise RuntimeError(f"correctness failure on {sc.name}")

        hit, suffix = _predict_specialization_hit(submitted_name, sc.op, inputs)

        sources = _variant_sources(submitted_name)
        sel_source = sources.get(suffix) if suffix else None
        mask_count = _count_masks(sel_source) if sel_source else None
        stride_count = _count_strides(sel_source) if sel_source else None

        # Re-run for fresh inputs to avoid stale GPU caching effects.
        inputs_b, _, _ = sc.make_inputs(device)
        inputs_s, _, _ = sc.make_inputs(device)

        baseline_ms = _time_kernel(baseline_kernel, inputs_b)
        submitted_ms = _time_kernel(submitted_kernel, inputs_s)

        results.append(
            {
                "scenario": sc.name,
                "op": sc.op,
                "shape": list(sc.shape),
                "block": list(sc.block),
                "dtype": str(sc.dtype),
                "expected_specialization_hit": sc.expected_hit,
                "specialization_hit": hit,
                "selected_variant": suffix,
                "baseline_runtime_ms": baseline_ms,
                "submitted_runtime_ms": submitted_ms,
                "speedup": baseline_ms / submitted_ms if submitted_ms > 0 else 0.0,
                "mask_expr_count": mask_count,
                "stride_expr_count": stride_count,
            }
        )

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"results": results}, indent=2))

    return results


def _torch_dtype_to_nt(dtype):
    if dtype == torch.float32:
        return ninetoothed.float32
    if dtype == torch.float16:
        return ninetoothed.float16
    if dtype == torch.bfloat16:
        return ninetoothed.bfloat16
    raise ValueError(dtype)


def _pretty_print(results):
    header = (
        "scenario", "hit", "expect", "baseline_ms", "submitted_ms",
        "speedup", "mask", "stride",
    )
    rows = [header]
    for r in results:
        rows.append((
            r["scenario"],
            str(r["specialization_hit"]),
            str(r["expected_specialization_hit"]),
            f"{r['baseline_runtime_ms']:.4f}",
            f"{r['submitted_runtime_ms']:.4f}",
            f"{r['speedup']:.3f}",
            str(r["mask_expr_count"]),
            str(r["stride_expr_count"]),
        ))
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    for row in rows:
        print("  ".join(f"{cell:<{w}}" for cell, w in zip(row, widths)))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device to run on (default: cuda if available)",
    )
    parser.add_argument(
        "--output",
        default=str(_REPO_ROOT / "bench" / "results.json"),
        help="output JSON path",
    )
    args = parser.parse_args(argv)

    if args.device == "cpu":
        raise SystemExit(
            "specialization benchmark requires a GPU; pass --device cuda"
        )

    results = run_scenarios(args.device, args.output)
    _pretty_print(results)
    print(f"\nwrote {len(results)} scenarios to {args.output}")


if __name__ == "__main__":
    main()
