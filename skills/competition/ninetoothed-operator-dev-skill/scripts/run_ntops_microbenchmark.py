#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

CSV_FIELDS = [
    "timestamp",
    "task_id",
    "operator",
    "shape",
    "dtype",
    "device",
    "torch_version",
    "torch_cuda",
    "baseline",
    "candidate",
    "warmup",
    "repeat",
    "baseline_mean_ms",
    "baseline_median_ms",
    "baseline_min_ms",
    "baseline_samples_ms",
    "candidate_mean_ms",
    "candidate_median_ms",
    "candidate_min_ms",
    "candidate_samples_ms",
    "speedup_median",
    "correctness_status",
    "notes",
]

CAVEATS = [
    "single GPU",
    "single-server short benchmark",
    "not official hidden benchmark",
    "no long benchmark",
    "no AOT conclusion",
]

NO_FAKE_TIMING_NOTE = (
    "no fake timing; timing fields stay NA when CUDA or correctness preflight fails"
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _parse_shape(value: str) -> tuple[int, ...]:
    normalized = value.lower().replace(",", "x").replace(" ", "")
    parts = [item for item in normalized.split("x") if item]
    if not parts:
        raise argparse.ArgumentTypeError("shape must contain at least one dimension")
    try:
        shape = tuple(int(item) for item in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape dimensions must be integers") from exc
    if any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return shape


def _format_float(value: float | str) -> str:
    if isinstance(value, str):
        return value
    return f"{value:.6f}"


def _write_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow({key: row.get(key, "NA") for key in CSV_FIELDS})


def _write_md(
    path: Path, args: argparse.Namespace, row: dict[str, object], command: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ntops Microbenchmark",
        "",
        "## Environment",
        f"- timestamp: `{row['timestamp']}`",
        f"- task_id: `{args.task_id}`",
        f"- operator: `{args.operator}`",
        f"- shape: `{args.shape_text}`",
        f"- dtype: `{args.dtype}`",
        f"- device: `{row['device']}`",
        f"- torch_version: `{row['torch_version']}`",
        f"- torch_cuda: `{row['torch_cuda']}`",
        "",
        "## Command",
        "",
        "```text",
        command,
        "```",
        "",
        "## Correctness",
        f"- correctness_status: `{row['correctness_status']}`",
        "",
        "## Timing Summary",
        "",
        "| Metric | Baseline | Candidate |",
        "| --- | --- | --- |",
        f"| mean ms | `{row['baseline_mean_ms']}` | `{row['candidate_mean_ms']}` |",
        f"| median ms | `{row['baseline_median_ms']}` | `{row['candidate_median_ms']}` |",
        f"| min ms | `{row['baseline_min_ms']}` | `{row['candidate_min_ms']}` |",
        "",
        f"- baseline_samples_ms: `{row['baseline_samples_ms']}`",
        f"- candidate_samples_ms: `{row['candidate_samples_ms']}`",
        "",
        f"- speedup_median: `{row['speedup_median']}`",
        "",
        "## Caveats",
    ]
    lines.extend(f"- {item}" for item in CAVEATS)
    lines.append(f"- {NO_FAKE_TIMING_NOTE}")
    lines.append(
        "- Interpret this as a controlled short benchmark, not a broad performance conclusion."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _base_row(args: argparse.Namespace, torch_module=None) -> dict[str, object]:
    device = "NA"
    torch_version = "NA"
    torch_cuda = "NA"
    if torch_module is not None:
        torch_version = getattr(torch_module, "__version__", "NA")
        torch_cuda = getattr(torch_module.version, "cuda", None) or "NA"
        if torch_module.cuda.is_available():
            device = torch_module.cuda.get_device_name(0)
    return {
        "timestamp": _timestamp(),
        "task_id": args.task_id,
        "operator": args.operator,
        "shape": args.shape_text,
        "dtype": args.dtype,
        "device": device,
        "torch_version": torch_version,
        "torch_cuda": torch_cuda,
        "baseline": args.baseline,
        "candidate": args.candidate,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "baseline_mean_ms": "NA",
        "baseline_median_ms": "NA",
        "baseline_min_ms": "NA",
        "baseline_samples_ms": "NA",
        "candidate_mean_ms": "NA",
        "candidate_median_ms": "NA",
        "candidate_min_ms": "NA",
        "candidate_samples_ms": "NA",
        "speedup_median": "NA",
        "correctness_status": "BLOCKED",
        "notes": NO_FAKE_TIMING_NOTE,
    }


def _make_inputs(torch, args: argparse.Namespace, shape: tuple[int, ...]):
    dtype = torch.float32
    x = torch.randn(shape, device="cuda", dtype=dtype)
    if args.operator == "add":
        y = torch.randn(shape, device="cuda", dtype=dtype)
        return (x, y)
    return (x,)


def _call_baseline(torch, args: argparse.Namespace, inputs):
    x = inputs[0]
    if args.operator == "add":
        y = inputs[1]
        return torch.add(x, y)
    if args.operator == "relu":
        return torch.nn.functional.relu(x)
    if args.operator == "softmax":
        return torch.nn.functional.softmax(x, dim=-1)
    if args.operator == "layer_norm":
        normalized_shape = (x.shape[-1],)
        return torch.nn.functional.layer_norm(x, normalized_shape)
    raise ValueError(f"unsupported operator: {args.operator}")


def _call_candidate(ntops, args: argparse.Namespace, inputs):
    x = inputs[0]
    if args.operator == "add":
        return ntops.torch.add(x, inputs[1])
    if args.operator == "relu":
        return ntops.torch.relu(x)
    if args.operator == "softmax":
        return ntops.torch.softmax(x, -1)
    if args.operator == "layer_norm":
        return ntops.torch.layer_norm(x, (x.shape[-1],))
    raise ValueError(f"unsupported operator: {args.operator}")


def _time_ms(torch, fn, warmup: int, repeat: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    timings: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        timings.append(float(start.elapsed_time(end)))
    return timings


def _summarize(values: list[float]) -> tuple[float, float, float]:
    return statistics.mean(values), statistics.median(values), min(values)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a controlled short CUDA ntops microbenchmark."
    )
    parser.add_argument(
        "--operator", required=True, choices=["add", "relu", "softmax", "layer_norm"]
    )
    parser.add_argument(
        "--shape", required=True, help="Shape such as 1024x1024 or 64x1024."
    )
    parser.add_argument("--dtype", required=True, choices=["float32"])
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--baseline", default="pytorch", choices=["pytorch"])
    parser.add_argument("--candidate", default="ntops", choices=["ntops"])
    args = parser.parse_args()
    args.shape_text = args.shape

    if args.warmup < 0 or args.repeat <= 0:
        parser.error("--warmup must be >= 0 and --repeat must be > 0")

    command = " ".join(sys.argv)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)

    try:
        import ntops
        import torch
    except Exception as exc:
        row = _base_row(args)
        row["correctness_status"] = "BLOCKED_IMPORT"
        row["notes"] = f"{NO_FAKE_TIMING_NOTE}; import failed: {type(exc).__name__}"
        _write_csv(output_csv, row)
        _write_md(output_md, args, row, command)
        print(
            f"BLOCKED run_ntops_microbenchmark.py: import failed: {type(exc).__name__}: {exc}"
        )
        return 2

    row = _base_row(args, torch)
    shape = _parse_shape(args.shape)

    if not torch.cuda.is_available():
        row["correctness_status"] = "BLOCKED_CUDA_UNAVAILABLE"
        row["notes"] = (
            f"{NO_FAKE_TIMING_NOTE}; CUDA is required and CPU timing is forbidden"
        )
        _write_csv(output_csv, row)
        _write_md(output_md, args, row, command)
        print(
            "BLOCKED run_ntops_microbenchmark.py: CUDA unavailable; CPU benchmark is forbidden"
        )
        return 2

    inputs = _make_inputs(torch, args, shape)
    baseline_once = _call_baseline(torch, args, inputs)
    candidate_once = _call_candidate(ntops, args, inputs)
    torch.cuda.synchronize()
    if not torch.allclose(candidate_once, baseline_once, rtol=1e-4, atol=1e-4):
        max_abs = float((candidate_once - baseline_once).abs().max().item())
        row["correctness_status"] = "FAIL"
        row["notes"] = (
            f"{NO_FAKE_TIMING_NOTE}; correctness guard failed; max_abs={max_abs:.6g}"
        )
        _write_csv(output_csv, row)
        _write_md(output_md, args, row, command)
        print("FAIL run_ntops_microbenchmark.py: correctness guard failed")
        return 1

    row["correctness_status"] = "PASS"

    def baseline_fn():
        return _call_baseline(torch, args, inputs)

    def candidate_fn():
        return _call_candidate(ntops, args, inputs)

    baseline_times = _time_ms(torch, baseline_fn, args.warmup, args.repeat)
    candidate_times = _time_ms(torch, candidate_fn, args.warmup, args.repeat)
    baseline_mean, baseline_median, baseline_min = _summarize(baseline_times)
    candidate_mean, candidate_median, candidate_min = _summarize(candidate_times)
    speedup = (
        baseline_median / candidate_median if candidate_median > 0 else float("nan")
    )

    row.update(
        {
            "baseline_mean_ms": _format_float(baseline_mean),
            "baseline_median_ms": _format_float(baseline_median),
            "baseline_min_ms": _format_float(baseline_min),
            "baseline_samples_ms": ";".join(
                _format_float(value) for value in baseline_times
            ),
            "candidate_mean_ms": _format_float(candidate_mean),
            "candidate_median_ms": _format_float(candidate_median),
            "candidate_min_ms": _format_float(candidate_min),
            "candidate_samples_ms": ";".join(
                _format_float(value) for value in candidate_times
            ),
            "speedup_median": _format_float(speedup),
            "notes": "; ".join(CAVEATS),
        }
    )
    _write_csv(output_csv, row)
    _write_md(output_md, args, row, command)
    print("PASS run_ntops_microbenchmark.py")
    print(f"csv={output_csv}")
    print(f"markdown={output_md}")
    print(f"speedup_median={row['speedup_median']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
