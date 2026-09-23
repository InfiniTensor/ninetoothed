"""Benchmark row-sum reduction vs torch.sum. Run on a CUDA host."""

import csv
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "scripts"))

import reduction as R  # noqa: E402
from bench_compare import benchmark, roofline, throughput  # noqa: E402


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA unavailable; this benchmark needs a GPU.")

        return 1

    rows = []

    for n in (1024, 4096, 4095):
        x = torch.randn(4096, n, dtype=torch.float16, device="cuda")
        bytes_moved = x.numel() * x.element_size()  # Read-dominated.

        for name, fn in (
            ("torch", lambda x=x: x.sum(dim=-1)),
            ("nt_sum", lambda x=x: R.row_reduce(x, "sum")),
        ):
            r = benchmark(fn)
            tp = throughput(r["mean_ms"], bytes_moved=bytes_moved)
            rows.append(
                {
                    "N": n,
                    "impl": name,
                    "mean_ms": round(r["mean_ms"], 4),
                    "GB_s": round(tp.get("GB_s", 0.0), 1),
                }
            )
            print(
                f"N={n:<5} {name:<7} {r['mean_ms']:.4f} ms  {tp.get('GB_s', 0):.1f} GB/s"
            )

    # Reduction has ~1 FLOP/element and reads N elements -> memory-bound
    # gpu auto-detected; on an unknown/CPU host verdict is reported as `unknown`.
    print(roofline(flops=4096 * 4096, bytes_moved=4096 * 4096 * 2))
    out = HERE / "bench.csv"

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["N", "impl", "mean_ms", "GB_s"])
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
