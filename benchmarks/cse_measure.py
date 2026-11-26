"""Reproduce the CSE pass benchmark without ntops.lab.

The benchmark compiles the same ``log_softmax`` application twice: once with
the default pipeline (which now includes ``ssa.eliminate_common_subexpressions``)
and once with that pass removed.  Both kernels must produce the same values;
the baseline kernel emits nested reduction loops and is much slower.

Usage::

    python benchmarks/cse_measure.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.compiler.benchmark import benchmark_kernel
from ninetoothed.compiler.driver import CompileRequest, compile_kernel
from ninetoothed.compiler.passes import default_spec

CSE_PASS = "ssa.eliminate_common_subexpressions"


def arrangement(x, out):
    return x.tile((1, -1)), out.tile((1, -1))


def application(x, out):
    value = x.to(ntl.float32)
    m = ntl.max(value, axis=1)
    e = ntl.exp(value - m[:, None])
    # The expression below computes ``value - m`` a second time, *after* the
    # sum reduction. Without CSE, the emitter re-emits reduce.max inside the
    # sum loop, yielding an O(dim^2) kernel.
    out = value - m[:, None] - ntl.log(ntl.sum(e, axis=1))[:, None]


def make_kernel(*, cse: bool, num_warps=None, num_stages=None):
    pipeline = None

    if not cse:
        spec = default_spec("triton")
        passes = tuple(name for name in spec.passes if name != CSE_PASS)
        pipeline = replace(
            spec, passes=passes, mode="no-cse", reason="CSE disabled for benchmark"
        )

    return ninetoothed.make(
        arrangement,
        application,
        (Tensor(2, other=float("-inf")), Tensor(2)),
        kernel_name="cse_benchmark_log_softmax_" + ("on" if cse else "off"),
        num_warps=num_warps,
        num_stages=num_stages,
        pipeline=pipeline,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[4096, 32768, 131072])
    parser.add_argument("--dim", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--min-total-ms", type=float, default=20.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = []
    for dim in args.dim:
        kernels = {cse: make_kernel(cse=cse) for cse in (False, True)}

        for rows in args.rows:
            x = torch.randn((rows, dim), device="cuda", dtype=torch.float32)
            out = torch.empty_like(x)
            expected = torch.log_softmax(x, dim=1)

            timings = {}
            for cse, kernel in kernels.items():
                kernel(x, out)
                torch.cuda.synchronize()
                err = (out - expected).abs().max().item()

                ms = benchmark_kernel(
                    kernel, (x, out), min_total_ms=args.min_total_ms
                )
                timings["cse" if cse else "baseline"] = {"ms": ms, "err": err}

            baseline = timings["baseline"]["ms"]
            cse_ms = timings["cse"]["ms"]
            speedup = baseline / cse_ms
            row = {
                "rows": rows,
                "dim": dim,
                "baseline_ms": baseline,
                "cse_ms": cse_ms,
                "speedup": speedup,
                "baseline_err": timings["baseline"]["err"],
                "cse_err": timings["cse"]["err"],
            }
            results.append(row)
            print(
                f"rows={rows:6d} dim={dim:3d}  "
                f"baseline={baseline:.4f}ms cse={cse_ms:.4f}ms "
                f"speedup={speedup:.2f}x  "
                f"err={timings['cse']['err']:.2e}"
            )

    if args.output:
        with open(args.output, "w") as fh:
            json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
