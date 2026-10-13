#!/usr/bin/env python3
"""Generate a repo-style pytest scaffold for a NineToothed operator.

The generated test follows the conventions observed in the NineToothed ``tests/``
directory: it parametrizes ``device`` from ``tests.utils.get_available_devices``
(so it SKIPS cleanly off-GPU instead of failing), parametrizes ``dtype`` and
shape, builds random inputs, calls the operator, and asserts ``torch.allclose``
against a PyTorch reference.

This only writes a *scaffold*. You must fill in the operator import, the reference
expression, and any layout/boundary specifics. It never fabricates a passing test.

Usage:
    python gen_test_scaffold.py --op softmax --rank 2 --lane reduction
    python gen_test_scaffold.py --op add --rank 1 --lane elementwise --out tests/test_my_add.py

Lanes: elementwise | reduction | layout | perf
"""

import argparse

_REFERENCE_HINT = {
    "elementwise": "expected = input + other  # TODO: replace with the real elementwise/broadcast reference",
    "reduction": "expected = torch.softmax(input, dim=-1)  # TODO: replace with the real reduction reference",
    "layout": "expected = torch.OP(input)  # TODO: reference on the SAME non-contiguous input",
    "perf": "expected = torch.OP(input)  # TODO: correctness reference; benchmark separately via scripts/bench.py",
}

_LAYOUT_NOTE = {
    "elementwise": [],
    "reduction": [
        "    # Reductions: ensure out-of-bounds fill is handled (other=-inf for max, 0 for sum).",
    ],
    "layout": [
        "    # Layout-sensitive: also test a NON-CONTIGUOUS input and compare.",
        "    # e.g. input = input.T  /  input = input[::2]  /  input = input.narrow(0, 1, n - 1)",
    ],
    "perf": [
        "    # Performance task: keep this test for correctness; measure speed with scripts/bench.py.",
    ],
}


def _build(op: str, rank: int, lane: str) -> str:
    names = [chr(ord("a") + i) for i in range(rank)]
    dims = ", ".join(names)
    reference = _REFERENCE_HINT.get(lane, _REFERENCE_HINT["elementwise"])
    layout_note = _LAYOUT_NOTE.get(lane, [])
    shape_tuple = f"({dims},)" if rank > 1 else f"({dims})"

    lines = [
        "import pytest",
        "import torch",
        "",
        "# TODO: import your operator, e.g.",
        f"# from ntops... import {op}  OR define {op}(...) following tests/test_add.py.",
        "from tests.utils import get_available_devices",
        "",
        "",
        '@pytest.mark.parametrize("device", get_available_devices())',
        '@pytest.mark.parametrize("dtype", (torch.float32,))',
    ]
    lines += [f'@pytest.mark.parametrize("{name}", (256,))' for name in names]
    lines.append(f"def test({dims}, dtype, device):")
    lines += layout_note
    lines.append(f"    input = torch.rand({shape_tuple}, dtype=dtype, device=device)")
    lines.append("")
    lines.append(
        f"    output = {op}(input)  # TODO: adapt the call signature to your operator"
    )
    lines.append(f"    {reference}")
    lines.append("")
    lines.append(
        "    # fp16/bf16 need a tolerance, e.g. torch.allclose(output, expected, atol=1e-2)"
    )
    lines.append("    assert torch.allclose(output, expected)")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", required=True, help="Operator name, e.g. softmax")
    parser.add_argument("--rank", type=int, default=2, help="Number of dimensions")
    parser.add_argument(
        "--lane",
        choices=("elementwise", "reduction", "layout", "perf"),
        default="elementwise",
    )
    parser.add_argument("--out", help="Write to this path instead of stdout")
    args = parser.parse_args()

    scaffold = _build(args.op, args.rank, args.lane)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(scaffold)

        print(f"Wrote scaffold to {args.out}")
        print("Fill in the TODOs, then run: bash scripts/run_ci_checks.sh " + args.out)
    else:
        print(scaffold)


if __name__ == "__main__":
    main()
