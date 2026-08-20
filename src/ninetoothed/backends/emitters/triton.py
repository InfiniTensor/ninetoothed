"""Triton syntax hooks for the common SSA emitter."""

import keyword
import math
import re
from dataclasses import dataclass, replace
from typing import Any

from ninetoothed.backends.core import Target
from ninetoothed.backends.emitters import ssa as common
from ninetoothed.backends.emitters.base import EmitterTarget, ModuleRenderContext
from ninetoothed.backends.emitters.expressions import replace_symbols
from ninetoothed.compiler.layout import LayoutTransfer, serialize_layout_transfer
from ninetoothed.ir import Kernel, ssa
from ninetoothed.naming import is_meta


def _legal_pre_tiled_block(shape, *, max_numel):
    static_numel = 1

    for extent in shape:
        if extent.op == "constant":
            value = int(extent.value)

            if value <= 0 or value & (value - 1):
                return False

            static_numel *= value
            continue

        symbols = _index_symbols(extent)

        if not symbols or any(not is_meta(symbol) for symbol in symbols):
            return False

    return max_numel is None or static_numel <= max_numel


def _index_symbols(expression):
    if expression.op == "symbol":
        return frozenset({str(expression.value)})

    return frozenset().union(
        *(_index_symbols(operand) for operand in expression.operands)
    )


def _target_metadata(kernel: Kernel) -> dict[str, Any]:
    target = dict(kernel.compiler_options.get("target", {}))
    profile = dict(target.get("profile", {}))

    return dict(profile.get("metadata", {}))


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

    def coerce_dot_args(self, operation, args, context):
        metadata = _target_metadata(context.kernel)
        coercions = {}

        for source, destination in dict(
            metadata.get("triton_dot_operand_coercions", {})
        ).items():
            if (
                not isinstance(source, str)
                or not source.strip()
                or not isinstance(destination, str)
                or not destination.strip()
            ):
                raise ValueError(
                    "Triton dot operand coercion metadata requires dtype names."
                )

            source = common.normalize_dtype(source)
            destination = common.normalize_dtype(destination)

            if any(
                not dtype.isidentifier() or keyword.iskeyword(dtype)
                for dtype in (source, destination)
            ):
                raise ValueError(
                    "Triton dot operand coercion metadata requires valid dtype names."
                )

            existing = coercions.get(source)

            if existing is not None and existing != destination:
                raise ValueError(f"Conflicting Triton dot coercion for `{source}`.")

            coercions[source] = destination

        if not coercions:
            return args

        result = []

        for index, value in enumerate(args):
            operand = (
                operation.operands[index] if index < len(operation.operands) else None
            )
            type_ = context.value_types.get(operand) if operand is not None else None
            source_dtype = (
                common.normalize_dtype(type_.dtype)
                if type_ is not None and type_.dtype is not None
                else None
            )
            target_dtype = coercions.get(source_dtype)

            if target_dtype is not None:
                result.append(self.cast(target_dtype, value))
                continue

            if source_dtype is not None:
                result.append(value)
                continue

            if context.temp_counter is None:
                context.temp_counter = [0]

            occupied = {self.symbol(name) for name in context.reserved_symbols}

            while True:
                temporary = f"ninetoothed_dot_arg_{context.temp_counter[0]}"
                context.temp_counter[0] += 1

                if temporary not in occupied:
                    break

            context.lines.append(f"{temporary} = {value}")
            runtime_value = temporary

            for runtime_source, runtime_target in reversed(sorted(coercions.items())):
                runtime_value = (
                    f"({temporary}.to(tl.{runtime_target}) "
                    f"if {temporary}.dtype == tl.{runtime_source} else {runtime_value})"
                )

            result.append(runtime_value)

        return tuple(result)

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

    def schedule_context(self, context: ModuleRenderContext) -> ModuleRenderContext:
        program = context.kernel.ssa
        schedule = program.metadata.get("schedule", {}) if program is not None else {}
        transfer = schedule.get("layout_transfer")

        if (
            schedule.get("granularity") != "layout-transfer"
            or not isinstance(transfer, LayoutTransfer)
            or not transfer.schedulable
        ):
            return context

        uses_grid_stride_layout = (
            _target_metadata(context.kernel).get("triton_grid_limit") is not None
        )
        source_row_axis = "[:, None]" if uses_grid_stride_layout else "[None, :]"
        source_column_axis = "[None, :]" if uses_grid_stride_layout else "[:, None]"
        value_shape = (
            transfer.source.layout.application_shape
            if transfer.requires_tiling
            else transfer.destination.layout.application_shape
        )

        if len(value_shape) != 2:
            return context

        if not transfer.requires_tiling and not _legal_pre_tiled_block(
            value_shape,
            max_numel=self.max_vector_numel,
        ):
            return context

        rows, columns = (_render_index_expr(value) for value in value_shape)
        program_shape = tuple(
            _render_index_expr(value)
            for value in transfer.destination.layout.view_shape
        )
        private_meta: tuple[tuple[str, int], ...] = ()
        lines = ["transfer_program = tl.program_id(0)"]

        if transfer.requires_tiling:
            tile = dict(schedule.get("tile", {}))
            tile_m = int(tile.get("block_m", 16))
            tile_n = int(tile.get("block_n", 16))
            private_meta = (("TILE_M", tile_m), ("TILE_N", tile_n))
            lines.extend(
                (
                    f"transfer_tile_columns = ({columns} + TILE_N - 1) // TILE_N",
                    "transfer_tile_row = transfer_program // transfer_tile_columns",
                    "transfer_tile_column = transfer_program % transfer_tile_columns",
                    "destination_value_0 = transfer_tile_row * TILE_M + "
                    "tl.arange(0, TILE_M)[:, None]",
                    "destination_value_1 = transfer_tile_column * TILE_N + "
                    "tl.arange(0, TILE_N)[None, :]",
                    f"source_value_0 = transfer_tile_row * TILE_M + "
                    f"tl.arange(0, TILE_M){source_row_axis}",
                    f"source_value_1 = transfer_tile_column * TILE_N + "
                    f"tl.arange(0, TILE_N){source_column_axis}",
                )
            )
            grid_total = (
                f"(({rows} + _ninetoothed_tile_m - 1) // "
                f"_ninetoothed_tile_m) * (({columns} + "
                "_ninetoothed_tile_n - 1) // _ninetoothed_tile_n)"
            )
        else:
            lines.extend(
                (
                    f"destination_value_0 = tl.arange(0, {rows})[:, None]",
                    f"destination_value_1 = tl.arange(0, {columns})[None, :]",
                    f"source_value_0 = tl.arange(0, {rows}){source_row_axis}",
                    f"source_value_1 = tl.arange(0, {columns}){source_column_axis}",
                )
            )
            grid_total = common.product(program_shape)

        destination_flat = f"(destination_value_0 * ({columns}) + destination_value_1)"
        source_flat = f"(source_value_0 * ({columns}) + source_value_1)"
        source_replacements = {
            "outer_index": "transfer_program",
            "index": source_flat,
            "value_0": "source_value_0",
            "value_1": "source_value_1",
        }
        destination_replacements = {
            "outer_index": "transfer_program",
            "index": destination_flat,
            "value_0": "destination_value_0",
            "value_1": "destination_value_1",
        }
        source_index = _render_access_expression(
            transfer.source.access_map.linear_index, source_replacements
        )
        destination_index = _render_access_expression(
            transfer.destination.access_map.linear_index,
            destination_replacements,
        )
        source_predicate = _render_access_expression(
            transfer.source.access_map.predicate, source_replacements
        )
        destination_predicate = _render_access_expression(
            transfer.destination.access_map.predicate, destination_replacements
        )
        source_mask = (
            f"({source_predicate}) & (source_value_0 < ({rows})) & "
            f"(source_value_1 < ({columns}))"
        )
        destination_mask = (
            f"({destination_predicate}) & (destination_value_0 < ({rows})) & "
            f"(destination_value_1 < ({columns}))"
        )
        lines.append(
            f"transfer_value = {self.load(transfer.source_binding, source_index, mask=source_mask)}"
        )

        if not uses_grid_stride_layout:
            lines.append("transfer_value = tl.trans(transfer_value)")

        lines.append(
            self.store(
                transfer.destination_binding,
                destination_index,
                "transfer_value",
                mask=destination_mask,
            )
        )
        metadata = {
            "layout_transfer": serialize_layout_transfer(transfer)
            | {
                "block_shape": (
                    ("TILE_M", "TILE_N")
                    if transfer.requires_tiling
                    else tuple(value.render() for value in value_shape)
                ),
                "private_meta_parameters": tuple(name for name, _value in private_meta),
            }
        }

        return replace(
            context,
            total=common.product((rows, columns)),
            body="\n".join(lines),
            outer_axes=program_shape,
            grid_total=grid_total,
            axes=(rows, columns),
            vector_program=False,
            block_program=True,
            scalar_program=False,
            private_meta_parameters=private_meta,
            scheduled_metadata=metadata,
        )

    def render_module(self, context: ModuleRenderContext) -> str:
        kernel = context.kernel
        body = common.rewrite_index_math(context.body, c_style=False)
        target_metadata = _target_metadata(kernel)
        grid_limit = target_metadata.get("triton_grid_limit")
        block_size = int(target_metadata.get("triton_block_size", 256))

        if block_size < 1 or block_size & (block_size - 1):
            raise ValueError("Triton block size must be a positive power of two.")

        if grid_limit is not None:
            grid_limit = int(grid_limit)

            if grid_limit < 1:
                raise ValueError("Triton grid limit must be positive.")

        runtime_params = set(kernel.metadata.get("runtime_shape_params", ()))
        private_meta = tuple(context.private_meta_parameters)
        params = ",\n    ".join(
            (
                *context.variables,
                *context.outputs,
                *[
                    axis if axis in runtime_params else f"{axis}: tl.constexpr"
                    for axis in context.shape_params
                ],
                *[f"{name}: tl.constexpr" for name, _value in private_meta],
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
        private_kernel_args = "\n        ".join(
            f"{name}=_ninetoothed_{name.lower()}," for name, _value in private_meta
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
        has_explicit_program_grid = bool(
            context.vector_program or context.block_program or context.scalar_program
        )
        uses_grid_stride_loop = grid_limit is not None and (
            not has_explicit_program_grid
            or schedule.get("granularity") == "layout-transfer"
        )
        program_id = (
            "ninetoothed_program_id" if uses_grid_stride_loop else "tl.program_id(0)"
        )

        if context.block_program:
            block = "1"
            offsets = "0"
        elif context.scalar_program:
            block = "1"
            offsets = program_id
        elif context.vector_program:
            block = f"triton.next_power_of_2({context.total})"
            offsets = "tl.arange(0, BLOCK)"
        else:
            block = str(block_size)
            offsets = f"{program_id} * BLOCK + tl.arange(0, BLOCK)"

        launch_params = ", ".join(
            (
                *public_launch_params,
                *[
                    f"_ninetoothed_{name.lower()}={value}"
                    for name, value in private_meta
                ],
                f"_ninetoothed_num_warps={num_warps}",
                f"_ninetoothed_num_stages={num_stages}",
            )
        )
        launch_grid = f"({context.grid_total},)"
        launch_guard = ""

        mask = (
            "True"
            if context.block_program or context.scalar_program
            else f"offsets < ({context.total})"
        )

        if uses_grid_stride_loop:
            logical_total = (
                context.grid_total
                if has_explicit_program_grid
                else f"triton.cdiv({context.grid_total}, block)"
            )
            kernel_grid_total = replace_symbols(
                context.grid_total,
                {f"_ninetoothed_{name.lower()}": name for name, _value in private_meta},
            )
            kernel_logical_total = (
                kernel_grid_total
                if has_explicit_program_grid
                else f"tl.cdiv({kernel_grid_total}, BLOCK)"
            )
            body = body.replace("tl.program_id(0)", program_id)

            kernel_body = (
                f"for {program_id} in tl.range(tl.program_id(0), "
                f"{kernel_logical_total}, tl.num_programs(0)):\n"
                f"    offsets = {offsets}\n"
                f"    {self.index_name} = offsets\n"
                f"    mask = {mask}\n"
                f"{common.indent_block(body, '    ')}"
            )
            launch_guard = (
                f"    if _ninetoothed_num_warps > {grid_limit}:\n"
                '        raise ValueError("Triton num_warps cannot exceed the grid limit.")\n'
            )
            launch_grid = (
                f"(min({logical_total}, max(1, {grid_limit} // "
                "_ninetoothed_num_warps)),)"
            )
        elif not has_explicit_program_grid:
            program_count = f"triton.cdiv({context.grid_total}, block)"
            launch_grid = f"({program_count},)"

        if not uses_grid_stride_loop:
            kernel_body = (
                f"offsets = {offsets}\n"
                f"{self.index_name} = offsets\n"
                f"mask = {mask}\n"
                f"{body}"
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
{common.indent_block(kernel_body, "    ")}

def launch_{kernel.kernel_name}({launch_params}):
    block = {block}
{launch_guard}    grid = {launch_grid}
    {kernel.kernel_name}_kernel[grid](
        {kernel_args},
        {private_kernel_args}
        BLOCK=block,
        num_warps=_ninetoothed_num_warps,
        num_stages=_ninetoothed_num_stages,
    )
    return {result}
'''


def _render_index_expr(expression) -> str:
    return common.rewrite_index_math(expression.render(), c_style=False)


def _render_access_expression(expression, replacements) -> str:
    return common.rewrite_index_math(
        replace_symbols(expression.render(), replacements),
        c_style=False,
    )


TARGET = TritonTarget()


def emit(kernel: Kernel):
    return common.emit(kernel, TARGET)


__all__ = ["TARGET", "TritonTarget", "emit"]
