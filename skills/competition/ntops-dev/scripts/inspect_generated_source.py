#!/usr/bin/env python3

import argparse
import os
import re
from pathlib import Path


CACHE_DIR = Path(os.environ.get("NINETOOTHED_CACHE_DIR", Path.home() / ".ninetoothed"))


def _latest_sources():
    if not CACHE_DIR.exists():
        return []
    candidates = [
        path
        for path in CACHE_DIR.rglob("*.py")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


def _trigger_compile(operator, dtype_name, shape):
    import torch

    import ntops

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to generate and inspect ntops source.")

    dtype = getattr(torch, dtype_name)

    if operator == "gelu":
        x = torch.randn(shape, device="cuda", dtype=dtype)
        ntops.torch.gelu(x)
    elif operator == "softmax":
        x = torch.randn(shape, device="cuda", dtype=dtype)
        ntops.torch.softmax(x, dim=-1)
    elif operator == "addmm":
        if len(shape) != 3:
            raise SystemExit("addmm shape must be m,n,k, for example 512,512,512")
        m, n, k = shape
        input_tensor = torch.randn((m, n), device="cuda", dtype=dtype)
        mat1 = torch.randn((m, k), device="cuda", dtype=dtype)
        mat2 = torch.randn((k, n), device="cuda", dtype=dtype)
        ntops.torch.addmm(input_tensor, mat1, mat2)
    else:
        raise SystemExit(f"Unsupported operator: {operator}")

    torch.cuda.synchronize()


def _count(pattern, text):
    return len(re.findall(pattern, text))


def _summarize(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    lang = r"(?:tl|triton\.language)"
    summary = {
        "path": str(path),
        "lines": len(lines),
        "bytes": path.stat().st_size,
        "language_load": _count(rf"\b{lang}\.load\b", text),
        "language_store": _count(rf"\b{lang}\.store\b", text),
        "language_dot": _count(rf"\b{lang}\.dot\b", text),
        "language_exp": _count(rf"\b{lang}\.exp\b", text),
        "language_erf": _count(rf"\b{lang}\.erf\b", text),
        "language_where": _count(rf"\b{lang}\.where\b", text),
        "language_arange": _count(rf"\b{lang}\.arange\b", text),
        "language_sum": _count(rf"\b{lang}\.sum\b", text),
        "language_max": _count(rf"\b{lang}\.max\b", text),
        "float32": _count(r"\bfloat32\b", text),
        "float16": _count(r"\bfloat16\b", text),
        "autotune": "triton.autotune" in text,
        "heuristics": "triton.heuristics" in text,
    }
    block_names = sorted(set(re.findall(r"\b[A-Z][A-Z0-9_]*BLOCK[A-Z0-9_]*\b", text)))
    constexpr_names = sorted(set(re.findall(r"\b[A-Z][A-Z0-9_]*\b", text)))
    summary["block_names"] = ", ".join(block_names[:20]) if block_names else "(none)"
    summary["constexpr_sample"] = ", ".join(constexpr_names[:20])
    return summary


def _operator_score(operator, summary):
    if operator == "softmax":
        return (
            summary["language_exp"] * 4
            + summary["language_sum"] * 3
            + summary["language_max"] * 3
            - summary["language_dot"] * 5
        )
    if operator == "addmm":
        return summary["language_dot"] * 5 + summary["language_load"]
    if operator == "gelu":
        return summary["language_erf"] * 5 + summary["language_exp"] * 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize NineToothed generated source. Known operators can be triggered; "
            "any operator can be inspected with --source or --no-trigger."
        )
    )
    parser.add_argument("operator", help="Operator label used for reporting/scoring.")
    parser.add_argument("--shape", default=None)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--latest", type=int, default=3)
    parser.add_argument(
        "--source",
        type=Path,
        help="Inspect one generated .py file directly without triggering compilation.",
    )
    parser.add_argument(
        "--no-trigger",
        action="store_true",
        help="Inspect latest cache files without invoking the operator.",
    )
    args = parser.parse_args()

    default_shapes = {
        "gelu": (1048576,),
        "softmax": (1024, 1024),
        "addmm": (512, 512, 512),
    }
    shape = (
        tuple(int(part) for part in args.shape.split(",") if part)
        if args.shape
        else default_shapes.get(args.operator)
    )

    if args.source:
        source = args.source.expanduser().resolve()
        if not source.is_file():
            raise SystemExit(f"Generated source not found: {source}")
        candidates = [source]
    elif args.no_trigger:
        candidates = _latest_sources()[: max(args.latest, 10)]
    else:
        if args.operator not in default_shapes:
            raise SystemExit(
                f"Cannot trigger {args.operator!r}; use --source PATH or --no-trigger."
            )
        before = {path: path.stat().st_mtime for path in _latest_sources()}
        _trigger_compile(args.operator, args.dtype, shape)
        after = _latest_sources()
        changed = [
            path
            for path in after
            if path not in before or path.stat().st_mtime != before[path]
        ]
        candidates = changed if changed else after[: max(args.latest, 10)]
    summarized = [(_operator_score(args.operator, _summarize(path)), path) for path in candidates]
    summarized.sort(key=lambda item: item[0], reverse=True)
    selected = [path for _, path in summarized[: args.latest]]

    print(f"operator={args.operator}")
    print(f"shape={shape if shape is not None else '(not triggered)'}")
    print(f"dtype={args.dtype}")
    print(f"cache_dir={CACHE_DIR}")
    print(f"generated_files_considered={len(selected)}")

    for index, path in enumerate(selected, start=1):
        summary = _summarize(path)
        print(f"\n[source {index}]")
        print(f"operator_match_score={_operator_score(args.operator, summary)}")
        for key, value in summary.items():
            print(f"{key}={value}")

    if not selected:
        print("No generated source files found.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
