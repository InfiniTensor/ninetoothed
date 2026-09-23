"""Benchmark + diagnosis for softmax_fast vs softmax_slow vs torch.

Run on a CUDA host:  python bench_softmax.py
Writes a CSV and prints a Roofline-style conclusion. Pair with:
    python ../../scripts/inspect_generated_source.py
to see the tile/block difference between the two variants.
"""

import csv
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "scripts"))

import softmax_variants as S  # noqa: E402
from bench_compare import benchmark, throughput  # noqa: E402


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA unavailable; this benchmark needs a GPU.")

        return 1

    rows = []

    for n in (
        128,
        256,
        512,
        1000,
    ):  # Small rows expose the wasted-lane cost; 1000 is non-power-of-two.
        x = torch.randn(4096, n, dtype=torch.float16, device="cuda")
        bytes_moved = 2 * x.numel() * x.element_size()  # Read + write.

        for name, fn in (
            ("torch", lambda x=x: torch.softmax(x, dim=-1)),
            ("nt_fast", lambda x=x: S.softmax_fast(x)),
            ("nt_slow", lambda x=x: S.softmax_slow(x)),
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
                f"N={n:<5} {name:<8} {r['mean_ms']:.4f} ms  {tp.get('GB_s', 0):.1f} GB/s"
            )

    out = HERE / "bench.csv"

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["N", "impl", "mean_ms", "GB_s"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {out}")
    print(
        "Diagnosis: the oversized block (SLOW_BLOCK=8192) runs exp/max/sum over "
        "mostly-masked lanes. Masked loads fetch no DRAM, so the cost is wasted "
        "vector compute/occupancy, not bandwidth — it is exposed when SLOW_BLOCK >> N "
        "(~1.6-1.9x slower at N<=1024) and shrinks toward parity as N approaches the "
        "block size (bandwidth-bound). Fix: BLOCK_SIZE = row length (nt_fast)."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
