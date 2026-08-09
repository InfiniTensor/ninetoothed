#!/usr/bin/env python3
"""Fork-runnable verify for example 01 (correctness-first; preallocated-out bench)."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

_EX = Path(__file__).resolve().parent
sys.path.insert(0, str(_EX / "solution"))

from broadcast_add import broadcast_add  # noqa: E402

BENCH_SHAPES = ((128, 256), (512, 512), (2048, 2048))
CORR_SHAPES = ((8, 16), (32, 64), (128, 256))
QUICK_SHAPES = ((64, 128), (128, 256))

WARMUP = 30
REPEATS = 30
INNER_ITERATIONS = 100
QUICK_WARMUP = 3
QUICK_REPEATS = 3
QUICK_INNER = 8


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def _stats(samples: list[float]) -> dict[str, float | str]:
    ordered = sorted(samples)
    med = float(statistics.median(ordered))
    p10 = float(_percentile(ordered, 10))
    p90 = float(_percentile(ordered, 90))
    spread = float(p90 / med) if med > 0 else float("inf")
    if spread <= 2:
        label = "stable"
    elif spread <= 5:
        label = "noisy"
    else:
        label = "highly_noisy"
    return {
        "median_ms": med,
        "p10_ms": p10,
        "p90_ms": p90,
        "spread_ratio": spread,
        "stability": label,
    }


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.cwd(),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _require_fresh_path(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"STOP: output already exists (refusing overwrite): {path}")


def _validate_samples(label: str, samples: list[float], expected: int) -> None:
    if len(samples) != expected:
        raise SystemExit(
            f"STOP: incomplete batches for {label}: {len(samples)} != {expected}"
        )
    for i, v in enumerate(samples):
        if v != v:
            raise SystemExit(f"STOP: NaN timing for {label} batch={i}")
        if v < 0:
            raise SystemExit(f"STOP: negative timing for {label} batch={i}: {v}")
        if v == 0.0:
            raise SystemExit(f"STOP: zero timing for {label} batch={i}")


def _run_correctness(shapes: list[tuple[int, int]], *, dtype, device: str) -> None:
    for m, n in shapes:
        a = torch.rand(m, 1, dtype=dtype, device=device)
        b = torch.rand(1, n, dtype=dtype, device=device)
        out = broadcast_add(a, b)
        ref = torch.add(a, b)
        assert out.shape == ref.shape
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5), (
            f"correctness fail M={m} N={n}"
        )
        buf = torch.empty((m, n), dtype=dtype, device=device)
        ret = broadcast_add(a, b, out=buf)
        assert ret is buf
        assert torch.allclose(buf, ref, atol=1e-5, rtol=1e-5)
    print(f"correctness: PASS shapes={shapes}")


def _measure_alternating(
    nt_fn,
    torch_fn,
    *,
    warmup: int,
    repeats: int,
    inner: int,
) -> tuple[list[float], list[float]]:
    for _ in range(warmup):
        nt_fn()
        torch_fn()
    torch.cuda.synchronize()

    nt_batch_ms: list[float] = []
    torch_batch_ms: list[float] = []

    def _one_batch(fn) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end))

    for i in range(repeats):
        if i % 2 == 0:
            nt_batch_ms.append(_one_batch(nt_fn))
            torch_batch_ms.append(_one_batch(torch_fn))
        else:
            torch_batch_ms.append(_one_batch(torch_fn))
            nt_batch_ms.append(_one_batch(nt_fn))

    return nt_batch_ms, torch_batch_ms


def _per_iter_ms(batch_ms: list[float], inner: int) -> list[float]:
    return [b / float(inner) for b in batch_ms]


def _size_inversion_warnings(size_rows: list[dict]) -> list[dict]:
    warnings: list[dict] = []
    for side in ("nt", "torch"):
        prev_med = None
        prev_shape = None
        for row in size_rows:
            med = float(row[side]["median_ms"])
            shape = (row["M"], row["N"])
            if prev_med is not None and med < 0.5 * prev_med:
                warnings.append(
                    {
                        "impl": side,
                        "prev_shape": prev_shape,
                        "prev_median_ms": prev_med,
                        "shape": shape,
                        "median_ms": med,
                        "size_inversion_warning": True,
                    }
                )
            prev_med = med
            prev_shape = shape
    return warnings


def _write_markdown(path: Path, payload: dict) -> None:
    lines: list[str] = [
        "# Broadcast-add benchmark (example 01, Phase 2B)",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- device: `{payload['device']}`",
        f"- torch_version: `{payload['torch_version']}`",
        f"- ninetoothed_commit: `{payload['ninetoothed_commit']}`",
        f"- measurement_scope: `{payload['measurement_scope']}`",
        f"- allocation_inside_timed_region: `{payload['allocation_inside_timed_region']}`",
        f"- warmup/repeats/inner: `{payload['warmup']}/{payload['repeats']}/{payload['inner_iterations']}`",
        "",
        "Bounds: preallocated-output steady-state CUDA Event; single-GPU; "
        "quantitative claims only for stable/noisy rows; highly_noisy is descriptive only.",
        "",
        "| M | N | NT median | Torch median | ratio | NT stab | Torch stab |",
        "|---|---|----------:|-------------:|------:|---------|------------|",
    ]
    for row in payload["sizes"]:
        lines.append(
            f"| {row['M']} | {row['N']} | {row['nt']['median_ms']:.6f} | "
            f"{row['torch']['median_ms']:.6f} | {row['ratio_nt_over_torch']:.4f} | "
            f"{row['nt']['stability']} | {row['torch']['stability']} |"
        )
    lines.append("")
    if payload["size_inversion_warnings"]:
        lines.append("Size inversion warnings:")
        for w in payload["size_inversion_warnings"]:
            lines.append(
                f"- {w['impl']}: {w['prev_shape']} median={w['prev_median_ms']:.6f} "
                f"-> {w['shape']} median={w['median_ms']:.6f}"
            )
    else:
        lines.append("Size inversion warnings: none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_benchmark(
    *,
    shapes: list[tuple[int, int]],
    warmup: int,
    repeats: int,
    inner: int,
    mode: str,
    json_output: Path | None,
    markdown_output: Path | None,
) -> int:
    if not torch.cuda.is_available():
        print("STOP: CUDA required for --benchmark/--quick")
        return 1

    if json_output is not None:
        _require_fresh_path(json_output)
    if markdown_output is not None:
        _require_fresh_path(markdown_output)

    device = "cuda"
    dtype = torch.float32
    torch.manual_seed(42)

    _run_correctness(shapes, dtype=dtype, device=device)

    size_rows: list[dict] = []
    raw_batch_ms: list[dict] = []
    all_nt_med: list[float] = []
    all_torch_med: list[float] = []
    all_ratios: list[float] = []

    print(
        f"benchmark: mode={mode} dtype={dtype} device={device} "
        f"warmup={warmup} repeats={repeats} inner={inner} "
        f"timer=cuda.Event order=alternating "
        f"measurement_scope=preallocated-output"
    )

    for m, n in shapes:
        a = torch.rand(m, 1, dtype=dtype, device=device)
        b = torch.rand(1, n, dtype=dtype, device=device)
        nt_out = torch.empty((m, n), dtype=dtype, device=device)
        torch_out = torch.empty((m, n), dtype=dtype, device=device)

        # Correctness on this exact input before timing.
        broadcast_add(a, b, out=nt_out)
        torch.add(a, b, out=torch_out)
        assert torch.allclose(nt_out, torch_out, atol=1e-5, rtol=1e-5)

        def nt_once(a=a, b=b, nt_out=nt_out):
            broadcast_add(a, b, out=nt_out)

        def torch_once(a=a, b=b, torch_out=torch_out):
            torch.add(a, b, out=torch_out)

        nt_batch, torch_batch = _measure_alternating(
            nt_once,
            torch_once,
            warmup=warmup,
            repeats=repeats,
            inner=inner,
        )
        _validate_samples(f"nt M={m} N={n}", nt_batch, repeats)
        _validate_samples(f"torch M={m} N={n}", torch_batch, repeats)

        nt_iter = _per_iter_ms(nt_batch, inner)
        torch_iter = _per_iter_ms(torch_batch, inner)
        nt_s = _stats(nt_iter)
        torch_s = _stats(torch_iter)
        if not (nt_s["p10_ms"] <= nt_s["median_ms"] <= nt_s["p90_ms"]):
            raise SystemExit(f"STOP: NT percentile order broken M={m} N={n}")
        if not (torch_s["p10_ms"] <= torch_s["median_ms"] <= torch_s["p90_ms"]):
            raise SystemExit(f"STOP: Torch percentile order broken M={m} N={n}")
        ratio = float(nt_s["median_ms"]) / float(torch_s["median_ms"])

        print(
            f"shape M={m} N={n}: NT median={nt_s['median_ms']:.6f} "
            f"({nt_s['stability']}) Torch median={torch_s['median_ms']:.6f} "
            f"({torch_s['stability']}) ratio={ratio:.4f}"
        )

        size_rows.append(
            {
                "M": m,
                "N": n,
                "inner_iterations": inner,
                "nt": nt_s,
                "torch": torch_s,
                "ratio_nt_over_torch": ratio,
            }
        )
        raw_batch_ms.append(
            {
                "M": m,
                "N": n,
                "inner_iterations": inner,
                "nt_batch_ms": nt_batch,
                "torch_batch_ms": torch_batch,
                "nt_per_iter_ms": nt_iter,
                "torch_per_iter_ms": torch_iter,
            }
        )
        all_nt_med.append(float(nt_s["median_ms"]))
        all_torch_med.append(float(torch_s["median_ms"]))
        all_ratios.append(ratio)

    inversions = _size_inversion_warnings(size_rows)
    for w in inversions:
        print(
            f"WARNING size_inversion: {w['impl']} {w['prev_shape']}->"
            f"{w['shape']} medians {w['prev_median_ms']:.6f}->{w['median_ms']:.6f}"
        )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "ninetoothed_commit": _git_commit(),
        "dtype": str(dtype).replace("torch.", ""),
        "measurement_scope": "preallocated-output steady-state CUDA execution",
        "allocation_inside_timed_region": False,
        "inner_iterations_constant_across_sizes": True,
        "warmup": warmup,
        "repeats": repeats,
        "inner_iterations": inner,
        "mode": mode,
        "sizes": size_rows,
        "raw_batch_ms": raw_batch_ms,
        "median_ms": {"nt": all_nt_med, "torch": all_torch_med},
        "p10_ms": {
            "nt": [r["nt"]["p10_ms"] for r in size_rows],
            "torch": [r["torch"]["p10_ms"] for r in size_rows],
        },
        "p90_ms": {
            "nt": [r["nt"]["p90_ms"] for r in size_rows],
            "torch": [r["torch"]["p90_ms"] for r in size_rows],
        },
        "ratio": all_ratios,
        "spread_ratios": {
            "nt": [r["nt"]["spread_ratio"] for r in size_rows],
            "torch": [r["torch"]["spread_ratio"] for r in size_rows],
        },
        "stability_labels": {
            "nt": [r["nt"]["stability"] for r in size_rows],
            "torch": [r["torch"]["stability"] for r in size_rows],
        },
        "size_inversion_warnings": inversions,
        "bounds": (
            "single-GPU; preallocated out; quantitative claims only for "
            "stable/noisy; highly_noisy descriptive only; no cross-device claim"
        ),
    }

    # Persist only after measurement completes (never inside timed region).
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote JSON: {json_output}")
    if markdown_output is not None:
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(markdown_output, payload)
        print(f"wrote Markdown: {markdown_output}")

    print(
        "bounds: preallocated-output; single-GPU; quantitative claims only for stable/noisy"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    args = parser.parse_args()

    if args.benchmark and args.quick:
        print("STOP: pass only one of --benchmark or --quick")
        return 1

    if not torch.cuda.is_available():
        if args.benchmark or args.quick:
            print("STOP: CUDA required for --benchmark/--quick")
            return 1
        print("SKIP: CUDA required for this example kernel")
        return 0

    dtype = torch.float32
    device = "cuda"
    torch.manual_seed(42)

    if not args.benchmark and not args.quick:
        _run_correctness(list(CORR_SHAPES), dtype=dtype, device=device)
        return 0

    if args.quick:
        return _run_benchmark(
            shapes=list(QUICK_SHAPES),
            warmup=QUICK_WARMUP,
            repeats=QUICK_REPEATS,
            inner=QUICK_INNER,
            mode="quick",
            json_output=args.json_output,
            markdown_output=args.markdown_output,
        )

    return _run_benchmark(
        shapes=list(BENCH_SHAPES),
        warmup=WARMUP,
        repeats=REPEATS,
        inner=INNER_ITERATIONS,
        mode="benchmark",
        json_output=args.json_output,
        markdown_output=args.markdown_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
