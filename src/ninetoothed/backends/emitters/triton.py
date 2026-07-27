"""Triton syntax hooks for the common SSA emitter."""

import math
import re
from dataclasses import dataclass
from typing import Any

from ninetoothed.backends.core import Target
from ninetoothed.backends.emitters import ssa as common
from ninetoothed.backends.emitters.base import EmitterTarget, ModuleRenderContext
from ninetoothed.ir import Kernel, ssa


@dataclass(frozen=True, kw_only=True)
class TritonTarget(EmitterTarget):
    backend: Target = Target.TRITON
    language: str = "python/triton"
    suffix: str = "triton.py"
    source_route: str = "ssa-unified-triton-emitter"
    vector_value_semantics: bool = True
    max_vector_numel: int | None = 1 << 20

    def program_id(self, axis: int = 0) -> str:
        return f"tl.program_id({axis})"

    def vector_reduce(self, operator: str, operand: str, axis: int) -> str:
        return f"tl.{operator}({operand}, axis={axis})"

    def vector_splat(self, shape: str, value: str, dtype: str) -> str:
        match = re.fullmatch(
            r"([A-Za-z_][A-Za-z0-9_]*)(?:\.source)?\.dtype", dtype.strip()
        )
        dtype_expr = (
            f"{match.group(1)}.dtype.element_ty"
            if match is not None
            else f"tl.{common.normalize_dtype(dtype)}"
        )

        return f"tl.full({shape}, {value}, {dtype_expr})"

    def literal(self, value: Any) -> str:
        if isinstance(value, float) and math.isinf(value):
            return "float('inf')" if value > 0 else "-float('inf')"

        if value == "inf":
            return "float('inf')"

        if value == "-inf":
            return "-float('inf')"
        return repr(value)

    def load(self, tensor, index, *, mask=None, other=0.0):
        mask_text = (
            "" if mask is None else f", mask={mask}, other={self.literal(other)}"
        )

        return f"tl.load({self.tensor_ref(tensor)} + {index}{mask_text})"

    def store(self, tensor, index, value, *, mask=None):
        mask_text = "" if mask is None else f", mask={mask}"

        return f"tl.store({self.tensor_ref(tensor)} + {index}, {value}{mask_text})"

    def cast(self, dtype, value):
        return f"{value}.to(tl.{common.normalize_dtype(dtype)})"

    def where(self, cond, yes, no):
        return f"tl.where({cond}, {yes}, {no})"

    def call(self, name, args):
        if name == "where":
            return self.where(args[0], args[1], args[2])

        if name == "atomic_add":
            return f"tl.atomic_add({args[0]}, {args[1]})"

        if name == "load" and args:
            return f"tl.load({args[0]})"

        if name == "block_dot" and len(args) == 2:
            return f"tl.dot({args[0]}, {args[1]})"

        if name == "expm1":
            return f"({self.call('exp', args)} - 1.0)"

        functions = {
            "abs": "tl.abs",
            "acos": "tl.acos",
            "asin": "tl.asin",
            "atan": "tl.atan",
            "atan2": "tl.atan2",
            "_atan2_approx": "tl.atan2",
            "ceil": "tl.ceil",
            "cos": "tl.cos",
            "cosh": "tl.cosh",
            "erf": "tl.erf",
            "exp": "tl.exp",
            "exp2": "tl.exp2",
            "floor": "tl.floor",
            "log": "tl.log",
            "log1p": "tl.log",
            "log2": "tl.log2",
            "log10": "tl.log",
            "maximum": "tl.maximum",
            "max": "tl.maximum",
            "minimum": "tl.minimum",
            "min": "tl.minimum",
            "pow": "tl.pow",
            "rand": "tl.rand",
            "rsqrt": "tl.rsqrt",
            "sin": "tl.sin",
            "sinh": "tl.sinh",
            "sqrt": "tl.sqrt",
            "tan": "tl.tan",
            "tanh": "tl.tanh",
        }

        if name == "log1p":
            return f"tl.log(1.0 + {args[0]})"

        if name == "log10":
            return f"(tl.log({args[0]}) / 2.302585092994046)"

        if name == "dot" and len(args) == 2:
            return f"(({args[0]}) * ({args[1]}))"

        function = functions.get(name, name)

        return f"{function}({', '.join(args)})"

    def local_decl(self, type_: ssa.Type, name: str, expr: str) -> str:
        del type_

        return f"{name} = {expr}"

    def loop_header(self, var, lower, upper, step):
        return f"for {var} in range({lower}, {upper}, {step}):"

    def reduce_update(self, operator, acc, term):
        if operator == "sum":
            return f"{acc} + {term}"

        function = "tl.maximum" if operator == "max" else "tl.minimum"

        return f"{function}({acc}, {term})"

    def block_coords(self, axes: tuple[str, ...]) -> tuple[str, ...]:
        if not axes:
            return ()

        if len(axes) == 1:
            return (f"tl.arange(0, {axes[0]})",)

        rank = len(axes)
        coords = []

        for dim, axis in enumerate(axes):
            slices = ["None"] * rank
            slices[dim] = ":"
            coords.append(f"tl.arange(0, {axis})[{', '.join(slices)}]")
        return tuple(coords)

    def block_shape(self, axes: tuple[str, ...]) -> str:
        values = ", ".join(axes)

        if len(axes) == 1:
            values += ","
        return f"({values})"

    def render_view(self, operation, context) -> str:
        value = common.emit_value(operation.operands[0], context)
        subscript = str(operation.attrs.get("subscript", "")).strip()

        if "None" not in subscript:
            return value

        if subscript.startswith("(") and subscript.endswith(")"):
            subscript = subscript[1:-1]
        return f"{value}[{subscript}]"

    def needs_block_init(self, name: str, value: ssa.Value, context) -> bool:
        if name.startswith("%"):
            operation = context.operations.get(name)

            if (
                operation is not None
                and operation.results
                and operation.results[0].type.kind == "scalar"
            ):
                return True

        if value.type.kind != "tensor" or not name.startswith("%"):
            return False

        operation = context.operations.get(name)

        return operation is not None and operation.opcode in {
            "arith.constant",
            "tensor.zeros",
            "tensor.full",
        }

    def render_module(self, context: ModuleRenderContext) -> str:
        kernel = context.kernel
        body = common.rewrite_index_math(context.body, c_style=False)
        runtime_params = set(kernel.metadata.get("runtime_shape_params", ()))
        params = ",\n    ".join(
            (
                *context.variables,
                *context.outputs,
                *[
                    axis if axis in runtime_params else f"{axis}: tl.constexpr"
                    for axis in context.shape_params
                ],
                "BLOCK: tl.constexpr",
            )
        )
        public_launch_params = (
            *context.variables,
            *context.outputs,
            *context.shape_params,
        )
        kernel_args = ",\n        ".join(
            (*context.variables, *context.outputs, *context.shape_params)
        )
        offsets = (
            "0"
            if context.block_program
            else "tl.program_id(0)"
            if context.scalar_program
            else "tl.arange(0, BLOCK)"
            if context.vector_program
            else "tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)"
        )
        block = (
            "1"
            if context.block_program or context.scalar_program
            else f"triton.next_power_of_2({context.total})"
            if context.vector_program
            else "256"
        )
        schedule = kernel.ssa.metadata.get("schedule", {}) if kernel.ssa else {}
        num_warps_value = kernel.compiler_options.get("num_warps") or schedule.get(
            "num_warps"
        )
        num_stages_value = kernel.compiler_options.get("num_stages") or schedule.get(
            "num_stages"
        )
        num_warps = (
            num_warps_value[0]
            if isinstance(num_warps_value, tuple)
            else num_warps_value or 4
        )
        num_stages = (
            num_stages_value[0]
            if isinstance(num_stages_value, tuple)
            else num_stages_value or 3
        )
        launch_params = ", ".join(
            (
                *public_launch_params,
                f"_ninetoothed_num_warps={num_warps}",
                f"_ninetoothed_num_stages={num_stages}",
            )
        )
        active_grid = bool(
            context.vector_program or context.block_program or context.scalar_program
        )
        mask = (
            "True"
            if context.block_program or context.scalar_program
            else f"offsets < ({context.total})"
        )
        result = context.outputs[0] if context.outputs else "None"

        return f'''"""Triton lowering generated by NineToothed from ssa.Program.

Kernel: {kernel.kernel_name}
Lowering IR: ssa.Program
"""

import triton
import triton.language as tl
from math import floor
from triton.language.extra import libdevice


@triton.jit
def {kernel.kernel_name}_kernel(
    {params},
):
    offsets = {offsets}
    {self.index_name} = offsets
    mask = {mask}
{common.indent_block(body, "    ")}

def launch_{kernel.kernel_name}({launch_params}):
    block = {block}
    grid = ({context.grid_total},) if {active_grid!r} else (triton.cdiv({context.grid_total}, block),)
    {kernel.kernel_name}_kernel[grid](
        {kernel_args},
        BLOCK=block,
        num_warps=_ninetoothed_num_warps,
        num_stages=_ninetoothed_num_stages,
    )
    return {result}
'''


TARGET = TritonTarget()


def emit(kernel: Kernel):
    return common.emit(kernel, TARGET)


__all__ = ["TARGET", "TritonTarget", "emit"]
