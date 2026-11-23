#!/usr/bin/env python3
"""Generate NineToothed kernel skeleton for ntops or examples style."""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

# ntops 仓库真实写法（premake + element_wise）
NTOPS_UNARY = dedent(
    """\
    import functools

    import ninetoothed.language as ntl
    from ninetoothed import Tensor

    from ntops.kernels.element_wise import arrangement


    def application(input, output):
        x = input
        output = x  # TODO: replace with operator formula


    def premake(ndim, dtype=None, block_size=None):
        arrangement_ = functools.partial(arrangement, block_size=block_size)
        tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))
        return arrangement_, application, tensors
    """
)

NTOPS_BINARY = dedent(
    """\
    import functools

    import ninetoothed
    from ninetoothed import Tensor

    from ntops.kernels.element_wise import arrangement


    def application(input, other, alpha, output):
        output = input + alpha * other  # TODO: replace formula


    def premake(ndim, dtype=None, block_size=None):
        arrangement_ = functools.partial(arrangement, block_size=block_size)
        tensors = (
            Tensor(ndim, dtype=dtype),
            Tensor(ndim, dtype=dtype),
            Tensor(0, dtype=ninetoothed.float64),
            Tensor(ndim, dtype=dtype),
        )
        return arrangement_, application, tensors
    """
)

# ninetoothed-examples 风格（教学/对照）
EXAMPLES_UNARY = dedent(
    """\
    import ninetoothed
    import ninetoothed.language as ntl
    from ninetoothed import Symbol, Tensor

    BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

    def arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
        return input.tile((BLOCK_SIZE,)), output.tile((BLOCK_SIZE,))

    def application(input, output):
        x = input
        output = x  # TODO: replace with operator formula

    tensors = (Tensor(1), Tensor(1))
    kernel = ninetoothed.make(arrangement, application, tensors)
    """
)

EXAMPLES_BINARY = dedent(
    """\
    import ninetoothed
    from ninetoothed import Symbol, Tensor

    BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

    def arrangement(input, other, output, BLOCK_SIZE=BLOCK_SIZE):
        return (
            input.tile((BLOCK_SIZE,)),
            other.tile((BLOCK_SIZE,)),
            output.tile((BLOCK_SIZE,)),
        )

    def application(input, other, output):
        output = input + other  # TODO: replace; noqa: F841

    tensors = tuple(Tensor(1) for _ in range(3))
    kernel = ninetoothed.make(arrangement, application, tensors)
    """
)

TEMPLATES = {
    ("ntops", "unary"): NTOPS_UNARY,
    ("ntops", "binary"): NTOPS_BINARY,
    ("examples", "unary"): EXAMPLES_UNARY,
    ("examples", "binary"): EXAMPLES_BINARY,
}


def inject_formula(body: str, pattern: str, formula: str) -> str:
    """Replace TODO placeholder in application() with task-card formula."""
    formula = formula.strip()
    if not formula:
        return body
    if not formula.startswith("output"):
        formula = f"output = {formula}"
    if "# noqa" not in formula:
        formula = f"{formula}  # noqa: F841"
    if pattern == "unary":
        old = "    output = x  # TODO: replace with operator formula"
        new = f"    {formula}"
    else:
        old = "    output = input + alpha * other  # TODO: replace formula"
        new = f"    {formula}"
    if old not in body:
        return body
    body = body.replace(old, new, 1)
    if pattern == "unary":
        body = body.replace("    x = input\n", "")
    return body


def main() -> None:
    p = argparse.ArgumentParser(description="Scaffold NineToothed kernel")
    p.add_argument("--name", required=True, help="Operator name, e.g. gelu")
    p.add_argument("--pattern", choices=("unary", "binary"), required=True)
    p.add_argument(
        "--style",
        choices=("ntops", "examples"),
        default="ntops",
        help="ntops=premake+element_wise (default); examples=legacy make",
    )
    p.add_argument("--out", required=True, type=Path, help="Output .py path")
    p.add_argument(
        "--formula",
        default="",
        help="Inject application formula from task card (e.g. 'output = input + other')",
    )
    args = p.parse_args()

    body = TEMPLATES[(args.style, args.pattern)]
    if args.formula and args.style == "ntops":
        body = inject_formula(body, args.pattern, args.formula)
    header = f'"""NineToothed kernel: {args.name} ({args.style}/{args.pattern} scaffold)."""\n\n'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(header + body, encoding="utf-8")
    tag = f"[{args.style}/{args.pattern}]"
    if args.formula:
        tag += " +formula"
    print(f"Wrote {args.out} {tag}")


if __name__ == "__main__":
    main()
