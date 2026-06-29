#!/usr/bin/env python3
"""Generate ntops torch wrapper for an operator with premake."""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

UNARY = dedent(
    """\
    import torch

    import ntops
    from ntops.torch.utils import _cached_make


    def {name}(input, inplace=False):
        if inplace:
            output = input
        else:
            output = torch.empty_like(input)

        kernel = _cached_make(ntops.kernels.{name}.premake, input.ndim)
        kernel(input, output)
        return output
    """
)

BINARY_ADDLIKE = dedent(
    """\
    import torch

    import ntops
    from ntops.torch.utils import _cached_make


    def {name}(input, other, *, alpha=1, out=None):
        if out is None:
            out = torch.empty_like(input)

        kernel = _cached_make(ntops.kernels.{name}.premake, input.ndim)
        kernel(input, other, alpha, out)
        return out
    """
)


def main() -> None:
    p = argparse.ArgumentParser(description="Scaffold ntops torch wrapper")
    p.add_argument("--name", required=True)
    p.add_argument("--pattern", choices=("unary", "binary"), default="unary")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    tpl = UNARY if args.pattern == "unary" else BINARY_ADDLIKE
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(tpl.format(name=args.name), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
