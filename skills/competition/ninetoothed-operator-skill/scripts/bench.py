#!/usr/bin/env python3
"""Benchmark a NineToothed operator against a baseline.

Reports the five required facts: baseline, input sizes, command, result, conclusion.

Wraps ``triton.testing.do_bench`` when available (falls back to CUDA events).
NineToothed has no built-in benchmark harness, so this provides a consistent one.

This is a *library + example*. Import ``compare`` from your task record, or run
the built-in demo (add: NineToothed vs torch) as a template:

    python bench.py --demo add --size 1048576

Off-GPU there is nothing meaningful to time; the script says so and exits 0 rather
than fabricating numbers.
"""

import argparse
import statistics

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _do_bench(fn, warmup=25, rep=100):
    """Time ``fn`` (ms). Prefer triton.testing.do_bench, else CUDA events."""
    try:
        from triton.testing import do_bench

        return float(do_bench(fn, warmup=warmup, rep=rep))
    except Exception:
        # Fallback: manual CUDA-event timing.
        times = []
        for _ in range(warmup):
            fn()

        torch.cuda.synchronize()

        for _ in range(rep):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        return statistics.median(times)


def compare(
    candidate, baseline, *, label="operator", input_desc="", warmup=25, rep=100
):
    """Benchmark ``candidate`` against ``baseline`` (both zero-arg callables).

    Returns a dict with the five reporting facts and prints a summary. Both
    callables should execute the op on already-allocated inputs.
    """
    if torch is None or not torch.cuda.is_available():
        print(
            f"[{label}] No CUDA device available — skipping benchmark (not fabricating numbers)."
        )

        return {
            "baseline": "n/a (no GPU)",
            "input_sizes": input_desc,
            "result": "skipped",
            "conclusion": "Benchmark requires a GPU; run on an accelerator to obtain timings.",
        }

    cand_ms = _do_bench(candidate, warmup=warmup, rep=rep)
    base_ms = _do_bench(baseline, warmup=warmup, rep=rep)
    speedup = base_ms / cand_ms if cand_ms else float("nan")

    conclusion = (
        f"NineToothed is {speedup:.2f}x vs baseline"
        if speedup >= 1
        else f"NineToothed is {1 / speedup:.2f}x SLOWER than baseline (regression — investigate)"
    )

    print(f"[{label}] input: {input_desc}")
    print(f"  baseline : {base_ms:.4f} ms")
    print(f"  candidate: {cand_ms:.4f} ms")
    print(f"  {conclusion}")

    return {
        "baseline": f"{base_ms:.4f} ms",
        "input_sizes": input_desc,
        "result": f"candidate {cand_ms:.4f} ms, speedup {speedup:.2f}x",
        "conclusion": conclusion,
    }


def _demo_add(size: int):
    """Template: benchmark elementwise add, NineToothed vs torch."""
    import ninetoothed
    from ninetoothed import Symbol, Tensor

    def add(lhs, rhs):
        BLOCK_SIZE = Symbol("BLOCK_SIZE", meta=True)

        @ninetoothed.jit
        def add_kernel(
            lhs: Tensor(1).tile((BLOCK_SIZE,)),
            rhs: Tensor(1).tile((BLOCK_SIZE,)),
            output: Tensor(1).tile((BLOCK_SIZE,)),
        ):
            output = lhs + rhs  # noqa: F841

        output = torch.empty_like(lhs)
        add_kernel(lhs, rhs, output)

        return output

    device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
    lhs = torch.rand(size, dtype=torch.float32, device=device)
    rhs = torch.rand(size, dtype=torch.float32, device=device)

    return compare(
        candidate=lambda: add(lhs, rhs),
        baseline=lambda: lhs + rhs,
        label="add",
        input_desc=f"shape=({size},), dtype=float32",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo", choices=("add",), help="Run a built-in demo benchmark"
    )
    parser.add_argument(
        "--size", type=int, default=1 << 20, help="1-D size for the demo"
    )
    args = parser.parse_args()

    if args.demo == "add":
        _demo_add(args.size)
    else:
        print(
            "Import `compare` in your task record, or run with --demo add. See module docstring."
        )


if __name__ == "__main__":
    main()
