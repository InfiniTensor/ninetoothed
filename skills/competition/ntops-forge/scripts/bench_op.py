#!/usr/bin/env python3
"""Lightweight GPU timing benchmark for ntops operators (finals supplement)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark ntops op vs PyTorch reference")
    ap.add_argument("--op", default="silu")
    ap.add_argument("--ntops-root", type=Path, required=True)
    ap.add_argument("--shape", default="4096,4096", help="comma-separated dims")
    ap.add_argument("--dtype", default="float16", choices=("float16", "float32"))
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=50)
    ap.add_argument("--output", type=Path, default=ROOT / "docs" / "bench_silu.json")
    args = ap.parse_args()

    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("FAIL: torch not installed")
        return 1

    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return 0

    sys.path.insert(0, str(args.ntops_root / "src"))
    try:
        import ntops.torch as nt
    except ImportError:
        print("FAIL: ntops not installed (pip install -e <ntops-root>)")
        return 1

    shape = tuple(int(x) for x in args.shape.split(",") if x.strip())
    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda")

    x = torch.randn(*shape, device=device, dtype=dtype)

    def time_fn(fn) -> float:
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.repeat):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / args.repeat * 1000

    if args.op == "silu":
        ref_ms = time_fn(lambda: F.silu(x))
        nt_ms = time_fn(lambda: nt.silu(x))
    else:
        print(f"FAIL: unsupported op {args.op!r} (silu only for now)")
        return 1

    ratio = nt_ms / ref_ms if ref_ms > 0 else 0
    result = {
        "op": args.op,
        "shape": list(shape),
        "dtype": args.dtype,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "pytorch_ms": round(ref_ms, 4),
        "ntops_ms": round(nt_ms, 4),
        "ratio_ntops_over_pytorch": round(ratio, 4),
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"OK: wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
