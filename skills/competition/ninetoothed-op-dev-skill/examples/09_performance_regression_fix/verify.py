#!/usr/bin/env python3
"""Fork-runnable verify for example 09 (correctness-first; preallocated-out bench)."""

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

from add_tunable import add  # noqa: E402

BENCH_SIZES = (98_432, 1_048_576, 4_194_304)
CORR_SIZES = (16_384, 65_536, 98_432)
QUICK_SIZES = (16_384, 65_536)
BLOCK_SIZES = (32, 256, 1024)
CORR_BLOCKS = (32, 1024)

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


def _run_correctness(
    sizes: list[int],
    blocks: list[int],
    *,
    dtype,
    device: str,
) -> None:
    for size in sizes:
        lhs = torch.rand(size, dtype=dtype, device=device)
        rhs = torch.rand(size, dtype=dtype, device=device)
        ref = lhs + rhs
        for bs in blocks:
            out = add(lhs, rhs, block_size=bs)
            assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5), (
                f"correctness fail size={size} BLOCK_SIZE={bs}"
            )
            buf = torch.empty_like(lhs)
            ret = add(lhs, rhs, block_size=bs, out=buf)
            assert ret is buf
            assert torch.allclose(buf, ref, atol=1e-5, rtol=1e-5)
    print(f"correctness: PASS sizes={sizes} blocks={blocks}")


def _measure_rotated_blocks(
    make_fn,
    blocks: list[int],
    *,
    warmup: int,
    repeats: int,
    inner: int,
) -> dict[int, list[float]]:
    for bs in blocks:
        fn = make_fn(bs)
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()

    batch_ms: dict[int, list[float]] = {bs: [] for bs in blocks}

    def _one_batch(fn) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end))

    n_blocks = len(blocks)
    for i in range(repeats):
        order = blocks[i % n_blocks :] + blocks[: i % n_blocks]
        for bs in order:
            batch_ms[bs].append(_one_batch(make_fn(bs)))

    return batch_ms


def _per_iter_ms(batch_ms: list[float], inner: int) -> list[float]:
    return [b / float(inner) for b in batch_ms]


def _size_inversion_warnings(size_rows: list[dict], blocks: list[int]) -> list[dict]:
    warnings: list[dict] = []
    for bs in blocks:
        key = str(bs)
        prev_med = None
        prev_n = None
        for row in size_rows:
            med = float(row["blocks"][key]["median_ms"])
            n = int(row["N"])
            if prev_med is not None and med < 0.5 * prev_med:
                warnings.append(
                    {
                        "block_size": bs,
                        "prev_N": prev_n,
                        "prev_median_ms": prev_med,
                        "N": n,
                        "median_ms": med,
                        "size_inversion_warning": True,
                    }
                )
            prev_med = med
            prev_n = n
    return warnings


def _write_markdown(path: Path, payload: dict) -> None:
    lines: list[str] = [
        "# BLOCK_SIZE benchmark (example 09, Phase 2B)",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- device: `{payload['device']}`",
        f"- torch_version: `{payload['torch_version']}`",
        f"- ninetoothed_commit: `{payload['ninetoothed_commit']}`",
        f"- measurement_scope: `{payload['measurement_scope']}`",
        f"- allocation_inside_timed_region: `{payload['allocation_inside_timed_region']}`",
        f"- warmup/repeats/inner: `{payload['warmup']}/{payload['repeats']}/{payload['inner_iterations']}`",
        f"- size_dependence: `{payload['size_dependence']}`",
        "",
        "Bounds: constexpr BLOCK_SIZE; preallocated out; no meta=True; "
        "quantitative claims only for stable/noisy; size dependence disclosed.",
        "",
        "| N | BLOCK_SIZE | median ms | p10 | p90 | spread | stability |",
        "|---|------------|----------:|----:|----:|-------:|-----------|",
    ]
    for row in payload["sizes"]:
        for bs, st in row["blocks"].items():
            lines.append(
                f"| {row['N']} | {bs} | {st['median_ms']:.6f} | {st['p10_ms']:.6f} | "
                f"{st['p90_ms']:.6f} | {st['spread_ratio']:.3f} | {st['stability']} |"
            )
    lines.append("")
    lines.append("Fastest BLOCK_SIZE by size (median; all labels retained):")
    for row in payload["sizes"]:
        lines.append(
            f"- N={row['N']}: BLOCK_SIZE={row['fastest_block_size']} "
            f"(median={row['fastest_median_ms']:.6f}, "
            f"stability={row['fastest_stability']})"
        )
    lines.append("")
    if payload["size_inversion_warnings"]:
        lines.append("Size inversion warnings:")
        for w in payload["size_inversion_warnings"]:
            lines.append(
                f"- BLOCK_SIZE={w['block_size']}: N={w['prev_N']}->"
                f"{w['N']} medians {w['prev_median_ms']:.6f}->{w['median_ms']:.6f}"
            )
    else:
        lines.append("Size inversion warnings: none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_benchmark(
    *,
    sizes: list[int],
    blocks: list[int],
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

    _run_correctness(sizes, blocks, dtype=dtype, device=device)

    size_rows: list[dict] = []
    raw_batch_ms: list[dict] = []
    fastest_by_size: list[int] = []
    all_medians: list[dict] = []

    print(
        f"benchmark: mode={mode} dtype={dtype} device={device} "
        f"warmup={warmup} repeats={repeats} inner={inner} "
        f"timer=cuda.Event blocks={blocks} order=rotated "
        f"measurement_scope=preallocated-output"
    )

    for size in sizes:
        lhs = torch.rand(size, dtype=dtype, device=device)
        rhs = torch.rand(size, dtype=dtype, device=device)
        out_by_block = {bs: torch.empty_like(lhs) for bs in blocks}
        ref = lhs + rhs
        for bs in blocks:
            add(lhs, rhs, block_size=bs, out=out_by_block[bs])
            assert torch.allclose(out_by_block[bs], ref, atol=1e-5, rtol=1e-5)

        def make_fn(bs: int, lhs=lhs, rhs=rhs, out_by_block=out_by_block):
            out = out_by_block[bs]
            return lambda: add(lhs, rhs, block_size=bs, out=out)

        batches = _measure_rotated_blocks(
            make_fn,
            list(blocks),
            warmup=warmup,
            repeats=repeats,
            inner=inner,
        )

        block_stats: dict[str, dict] = {}
        block_raw: dict[str, dict] = {}
        best_bs = None
        best_med = float("inf")
        best_stab = None

        print(f"size N={size}:")
        for bs in blocks:
            raw = batches[bs]
            _validate_samples(f"N={size} BLOCK_SIZE={bs}", raw, repeats)
            per_iter = _per_iter_ms(raw, inner)
            st = _stats(per_iter)
            if not (st["p10_ms"] <= st["median_ms"] <= st["p90_ms"]):
                raise SystemExit(
                    f"STOP: percentile order broken N={size} BLOCK_SIZE={bs}"
                )
            print(
                f"  BLOCK_SIZE={bs}: median={st['median_ms']:.6f} "
                f"p10={st['p10_ms']:.6f} p90={st['p90_ms']:.6f} "
                f"spread={st['spread_ratio']:.3f} ({st['stability']})"
            )
            block_stats[str(bs)] = st
            block_raw[str(bs)] = {"batch_ms": raw, "per_iter_ms": per_iter}
            if float(st["median_ms"]) < best_med:
                best_med = float(st["median_ms"])
                best_bs = bs
                best_stab = st["stability"]

        assert best_bs is not None
        fastest_by_size.append(int(best_bs))
        size_rows.append(
            {
                "N": size,
                "inner_iterations": inner,
                "blocks": block_stats,
                "fastest_block_size": int(best_bs),
                "fastest_median_ms": float(best_med),
                "fastest_stability": best_stab,
            }
        )
        raw_batch_ms.append({"N": size, "inner_iterations": inner, "blocks": block_raw})
        all_medians.append(
            {str(bs): block_stats[str(bs)]["median_ms"] for bs in blocks}
        )

    unique_fastest = sorted(set(fastest_by_size))
    size_dependence = "yes" if len(unique_fastest) > 1 else "no"
    if size_dependence == "yes":
        conclusion = (
            "规模依赖: fastest BLOCK_SIZE differs across N; "
            "do not claim a single block is always optimal."
        )
    else:
        conclusion = (
            f"On this device/run, BLOCK_SIZE={unique_fastest[0]} had the best "
            f"median at every measured N; still not a universal claim."
        )

    inversions = _size_inversion_warnings(size_rows, list(blocks))
    for w in inversions:
        print(
            f"WARNING size_inversion: BLOCK_SIZE={w['block_size']} "
            f"N={w['prev_N']}->{w['N']} "
            f"medians {w['prev_median_ms']:.6f}->{w['median_ms']:.6f}"
        )

    print(f"size_dependence={size_dependence}; fastest_by_size={fastest_by_size}")
    print(f"conclusion: {conclusion}")

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
        "block_sizes": list(blocks),
        "sizes": size_rows,
        "raw_batch_ms": raw_batch_ms,
        "median_ms": all_medians,
        "p10_ms": [
            {bs: row["blocks"][bs]["p10_ms"] for bs in row["blocks"]}
            for row in size_rows
        ],
        "p90_ms": [
            {bs: row["blocks"][bs]["p90_ms"] for bs in row["blocks"]}
            for row in size_rows
        ],
        "ratio": [
            {
                f"{a}_over_{b}": (
                    float(row["blocks"][str(a)]["median_ms"])
                    / float(row["blocks"][str(b)]["median_ms"])
                )
                for a in blocks
                for b in blocks
                if a != b
            }
            for row in size_rows
        ],
        "spread_ratios": [
            {bs: row["blocks"][bs]["spread_ratio"] for bs in row["blocks"]}
            for row in size_rows
        ],
        "stability_labels": [
            {bs: row["blocks"][bs]["stability"] for bs in row["blocks"]}
            for row in size_rows
        ],
        "fastest_block_by_size": fastest_by_size,
        "size_dependence": size_dependence,
        "size_inversion_warnings": inversions,
        "conclusion": conclusion,
        "bounds": (
            "constexpr BLOCK_SIZE; preallocated out; no meta=True; "
            "quantitative claims only for stable/noisy; disclose size dependence"
        ),
    }

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
        "bounds: preallocated-output; constexpr BLOCK_SIZE; quantitative claims only for stable/noisy"
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
        _run_correctness(
            list(CORR_SIZES), list(CORR_BLOCKS), dtype=dtype, device=device
        )
        return 0

    if args.quick:
        return _run_benchmark(
            sizes=list(QUICK_SIZES),
            blocks=list(BLOCK_SIZES),
            warmup=QUICK_WARMUP,
            repeats=QUICK_REPEATS,
            inner=QUICK_INNER,
            mode="quick",
            json_output=args.json_output,
            markdown_output=args.markdown_output,
        )

    return _run_benchmark(
        sizes=list(BENCH_SIZES),
        blocks=list(BLOCK_SIZES),
        warmup=WARMUP,
        repeats=REPEATS,
        inner=INNER_ITERATIONS,
        mode="benchmark",
        json_output=args.json_output,
        markdown_output=args.markdown_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
