"""NineToothed operator code generator.

Usage:
    python generate_op.py --name softmax --num-inputs 1 --num-outputs 1 --dim 2

Generates:
    examples/<name>/kernel.py
    examples/<name>/torch_impl.py
    examples/<name>/__init__.py
    tests/test_<name>.py (appended)
"""

import argparse
import os
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parent.parent


def _template_kernel(op_name, dim):
    has_loop = dim > 1
    extra_imports = ""
    if has_loop:
        extra_imports = "import ninetoothed.language as ntl\n"

    return f'''\
import ninetoothed
{extra_imports}from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement({"input, " if dim >= 1 else ""}output, BLOCK_SIZE=BLOCK_SIZE):
    {"input_arranged = " if dim >= 1 else ""}input{"." if dim >= 1 else ""}tile((-1, BLOCK_SIZE){" if dim >= 1 else ""})
    output_arranged = output.tile((-1, BLOCK_SIZE))

    return {"input_arranged, " if dim >= 1 else ""}output_arranged


def application({"input, " if dim >= 1 else ""}output):
    output = {"input + 1  # <-- TODO: replace with actual computation" if dim >= 1 else "0  # <-- TODO: replace with actual computation"}


tensors = ({"Tensor(2), " if dim >= 2 else ""}{"Tensor(1), " if dim == 1 else ""}Tensor(0), Tensor(2))

kernel = ninetoothed.make(arrangement, application, tensors)
'''


def _template_torch(op_name):
    return f'''\
import torch

from examples.{op_name}.kernel import kernel


def {op_name}(input):
    output = torch.empty_like(input)

    kernel(input, output, BLOCK_SIZE=input.shape[-1])

    return output
'''


def _template_init(op_name):
    return f'''\
from examples.{op_name}.kernel import kernel
from examples.{op_name}.torch_impl import {op_name}


__all__ = ["{op_name}", "kernel"]
'''


def generate(op_name, num_inputs, dim):
    op_dir = SKILL_ROOT / "examples" / op_name
    op_dir.mkdir(parents=True, exist_ok=True)

    (op_dir / "kernel.py").write_text(_template_kernel(op_name, dim))
    (op_dir / "torch_impl.py").write_text(_template_torch(op_name))
    (op_dir / "__init__.py").write_text(_template_init(op_name))

    print(f"[OK] Generated examples/{op_name}/")

    (SKILL_ROOT / "examples" / "__init__.py").write_text("")
    print("[OK] examples/__init__.py ensured")


def main():
    parser = argparse.ArgumentParser(description="Generate a NineToothed operator template")
    parser.add_argument("--name", required=True, help="Operator name (e.g. softmax)")
    parser.add_argument("--num-inputs", type=int, default=1, help="Number of input tensors")
    parser.add_argument("--num-outputs", type=int, default=1, help="Number of output tensors")
    parser.add_argument("--dim", type=int, default=2, help="Tensor dimensionality")
    args = parser.parse_args()

    generate(args.name, args.num_inputs, args.dim)


if __name__ == "__main__":
    main()
