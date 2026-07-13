"""TileLang syntax hooks for the common SSA emitter."""

import math
from dataclasses import dataclass
from typing import Any, Mapping

from ninetoothed.backends.core import Target
from ninetoothed.backends.emitters import ssa as common
from ninetoothed.backends.emitters.base import EmitterTarget, ModuleRenderContext
from ninetoothed.backends.emitters.context import EmitContext as _EmitContext
from ninetoothed.backends.emitters.context import TensorInfo as _TensorInfo
from ninetoothed.ir import Kernel, ssa

_Target = EmitterTarget
_buffer_storage_extent = common.buffer_storage_extent
_cooperative_dot_plan = common.cooperative_dot_plan
_dot_accumulator_dtype = common.dot_accumulator_dtype
_emit_element = common.emit_element
_emit_loop_bound = common.emit_loop_bound
_emit_operation = common.emit_operation
_indent_block = common.indent_block
_linearized_index = common.linearized_index
_logical_ssa_audit = common.logical_ssa_audit
_product = common.product
_resolved_dot_operand_dtype = common.resolved_dot_operand_dtype
_rewrite_index_math = common.rewrite_index_math
_value_axes_from_types = common.value_axes_from_types


_TIR_FUNCTIONS = {
    "abs": "T.abs",
    "acos": "T.acos",
    "asin": "T.asin",
    "atan": "T.atan",
    "atan2": "T.atan2",
    "_atan2_approx": "T.atan2",
    "ceil": "T.ceil",
    "cos": "T.cos",
    "cosh": "T.cosh",
    "dot": "T.dot",
    "erf": "T.erf",
    "exp": "T.exp",
    "exp2": "T.exp2",
    "floor": "T.floor",
    "log": "T.log",
    "log1p": "T.log1p",
    "log2": "T.log2",
    "log10": "T.log10",
    "maximum": "T.max",
    "max": "T.max",
    "minimum": "T.min",
    "min": "T.min",
    "pow": "T.pow",
    "rsqrt": "T.rsqrt",
    "sin": "T.sin",
    "sinh": "T.sinh",
    "sqrt": "T.sqrt",
    "tan": "T.tan",
    "tanh": "T.tanh",
}


@dataclass(frozen=True, kw_only=True)
class TileLangTarget(EmitterTarget):
    backend: Target = Target.TILELANG
    language: str = "python/tilelang"
    suffix: str = "tilelang.py"
    source_route: str = "ssa-unified-tilelang-emitter"
    buffer_suffix: str = "_buf"
    entrypoint_prefix: str = "build_"
    tir_value_semantics: bool = True

    def uses_mutable_scalar_slots(self) -> bool:
        return True

    def mutable_scalar_decl(self, type_: ssa.Type, name: str, init: str) -> list[str]:
        dtype = common.normalize_dtype(type_.dtype)

        return [f'{name} = T.alloc_var("{dtype}", {init})']

    def literal(self, value: Any) -> str:
        if isinstance(value, float) and math.isinf(value):
            return "float('inf')" if value > 0 else "-float('inf')"

        if value == "inf":
            return "float('inf')"

        if value == "-inf":
            return "-float('inf')"
        return repr(value)

    def load(self, tensor, index, *, mask=None, other=0.0):
        del mask, other

        return f"{self.tensor_ref(tensor)}[{index}]"

    def store(self, tensor, index, value, *, mask=None):
        assignment = f"{self.tensor_ref(tensor)}[{index}] = {value}"

        return assignment if mask is None else f"if {mask}:\n    {assignment}"

    def cast(self, dtype, value):
        return f'T.Cast("{common.normalize_dtype(dtype)}", {value})'

    def where(self, cond, yes, no):
        return f"T.if_then_else({cond}, {yes}, {no})"

    def call(self, name, args):
        if name == "where":
            return self.where(args[0], args[1], args[2])

        if name == "atomic_add":
            return f"T.atomic_add({args[0]}[0], {args[1]})"

        if name == "load" and args:
            return f"{args[0]}[0]"

        if name == "block_dot" and len(args) == 2:
            return f"T.dot({args[0]}, {args[1]})"

        if name == "rand":
            mixed = (
                f'T.Cast("uint32", T.bitwise_xor({args[0]}, {args[1]})) '
                "* 1664525 + 1013904223"
            )

            return f'T.Cast("float32", T.bitwise_and({mixed}, 16777215)) / 16777216.0'

        if name == "expm1":
            return f"({self.call('exp', args)} - 1.0)"

        function = _TIR_FUNCTIONS.get(name, f"T.{name}")

        return f"{function}({', '.join(args)})"

    def local_decl(self, type_: ssa.Type, name: str, expr: str) -> str:
        del type_

        return f"{name} = {expr}"

    def loop_header(self, var, lower, upper, step):
        serial = (
            f"T.serial({upper})"
            if lower == "0" and step == "1"
            else f"T.serial({lower}, {upper})"
            if step == "1"
            else f"T.serial({lower}, {upper}, {step})"
        )

        return f"for {var} in {serial}:"

    def reduce_update(self, operator, acc, term):
        if operator == "sum":
            return f"{acc} + {term}"

        function = "T.max" if operator == "max" else "T.min"

        return f"{function}({acc}, {term})"

    def render_module(self, context: ModuleRenderContext) -> str:
        cooperative = _render_tilelang_cooperative_dot_module(
            context.kernel,
            self,
            context.variables,
            context.outputs,
            context.shape_params,
            context.tensors,
            context.value_types,
            context.operations,
            context.stores,
            context.outer_axes,
        )

        if cooperative is not None:
            return cooperative
        return _render_tilelang_module(
            context.kernel,
            self,
            context.variables,
            context.outputs,
            context.shape_params,
            context.total,
            context.body,
            context.tensors,
            context.value_types,
        )


def _render_tilelang_cooperative_dot_module(
    kernel: Kernel,
    target: _Target,
    variables: tuple[str, ...],
    outputs: tuple[str, ...],
    shape_params: tuple[str, ...],
    tensors: Mapping[str, _TensorInfo],
    value_types: Mapping[str, ssa.Type],
    operations: Mapping[str, ssa.Operation],
    stores: tuple[ssa.Operation, ...],
    outer_axes: tuple[str, ...],
) -> str | None:
    plan = _cooperative_dot_plan(operations, stores, value_types)

    if plan is None or kernel.ssa is None:
        return None

    threads = common.schedule_int(kernel, "threads", 128)
    num_stages = common.schedule_int(kernel, "num_stages", 2)
    dot = plan.dot
    loop = plan.loop
    lhs, rhs = dot.operands[:2]
    lhs_axes = _value_axes_from_types(lhs, value_types)
    result_axes = tuple(str(dim) for dim in dot.results[0].type.shape)
    bm, bn = result_axes
    bk = lhs_axes[-1]
    output = plan.store.operands[1]
    parameter_names = (*variables, *outputs)
    buffer_names = tuple(name for name in parameter_names if tensors[name].ndim != 0)
    handle_args = ", ".join(
        [
            f"{name}: T.handle"
            if tensors[name].ndim != 0
            else f"{name}: {_tile_scalar_abi_dtype(tensors[name].dtype)}"
            for name in parameter_names
        ]
        + [f"{axis}: {_tile_param_dtype(axis, value_types)}" for axis in shape_params]
    )
    buffer_extents = {
        name: _rewrite_index_math(
            _buffer_storage_extent(tensors[name], fallback=_product(outer_axes)),
            c_style=False,
        )
        for name in buffer_names
    }
    buffers = "\n".join(
        f"        {name}_buf = T.match_buffer({name}, ({buffer_extents[name]},), {_tile_dtype(tensors[name].dtype)})"
        for name in buffer_names
    )

    def context(
        lines: list[str],
        *,
        coordinates: tuple[str, ...],
        bindings: Mapping[str, str] | None = None,
    ) -> _EmitContext:
        inner_index = _linearized_index(coordinates, result_axes)

        return _EmitContext(
            target=target,
            kernel=kernel,
            program=kernel.ssa,  # type: ignore[arg-type]
            operations=operations,
            value_types=value_types,
            lines=lines,
            memo={},
            tensor_infos=tensors,
            output=output,
            output_axes=result_axes,
            index_expr=inner_index,
            outer_index_expr="outer_index",
            inner_index_expr=inner_index,
            mask_expr=None,
            row_expr=coordinates[0],
            col_expr=coordinates[1],
            coordinate_exprs=coordinates,
            bindings={} if bindings is None else bindings,
            temp_counter=[0],
            materialized={},
            indent="",
        )

    bound_lines: list[str] = []
    bound_ctx = context(bound_lines, coordinates=("0", "0"))
    lower = _emit_loop_bound(loop.operands[0], bound_ctx)
    upper = _emit_loop_bound(loop.operands[1], bound_ctx)
    step = _emit_loop_bound(loop.operands[2], bound_ctx)

    if lower != "0" or step != "1":
        return None

    induction = str(loop.attrs.get("induction", "%iv"))
    loop_bindings = {induction: "ko"}
    lhs_lines: list[str] = []
    lhs_ctx = context(
        lhs_lines,
        coordinates=("mi", "ki"),
        bindings=loop_bindings,
    )
    lhs_value = _emit_element(lhs, ("mi", "ki"), lhs_ctx)
    lhs_lines.append(f"a_shared[mi, ki] = {lhs_value}")
    rhs_lines: list[str] = []
    rhs_ctx = context(
        rhs_lines,
        coordinates=("ki", "ni"),
        bindings=loop_bindings,
    )
    rhs_value = _emit_element(rhs, ("ki", "ni"), rhs_ctx)
    rhs_lines.append(f"b_shared[ki, ni] = {rhs_value}")

    output_lines: list[str] = []
    output_ctx = context(
        output_lines,
        coordinates=("mi", "ni"),
        bindings={loop.results[0].name: "c_local[mi, ni]"},
    )
    _emit_operation(plan.store, output_ctx)

    if not output_lines:
        return None

    lhs_storage_dtype = _resolved_dot_operand_dtype(lhs, output_ctx)
    rhs_storage_dtype = _resolved_dot_operand_dtype(rhs, output_ctx)
    supported_storage_dtypes = {
        "float16",
        "bfloat16",
        "float8_e4m3fn",
        "float8_e5m2",
    }

    if (
        lhs_storage_dtype not in supported_storage_dtypes
        or rhs_storage_dtype not in supported_storage_dtypes
    ):
        return None

    lhs_dtype = _tile_dtype(lhs_storage_dtype)
    rhs_dtype = _tile_dtype(rhs_storage_dtype)
    accumulator_dtype = _tile_dtype(_dot_accumulator_dtype(dot, output_ctx))
    grid_total = _rewrite_index_math(_product(outer_axes), c_style=False)
    bound_source = (
        "\n" + _indent_block("\n".join(bound_lines), "            ")
        if bound_lines
        else ""
    )
    lhs_source = _indent_block("\n".join(lhs_lines), "                    ")
    rhs_source = _indent_block("\n".join(rhs_lines), "                    ")
    output_source = _indent_block("\n".join(output_lines), "                ")

    return f'''"""TileLang lowering generated by NineToothed from ssa.Program.

Kernel: {kernel.kernel_name}
Schedule: cooperative linalg.dot -> T.gemm
"""

{_logical_ssa_audit(kernel, target)}

from math import floor

try:
    import tilelang
    import tilelang.language as T
except ImportError:
    tilelang = None
    T = None


def build_{kernel.kernel_name}():
    if tilelang is None:
        raise ImportError("TileLang is required to build this backend artifact.")

    @T.prim_func
    def {kernel.kernel_name}({handle_args}):
{buffers}
        with T.Kernel({grid_total}, threads={threads}) as outer_index:{bound_source}
            a_shared = T.alloc_shared(({bm}, {bk}), {lhs_dtype})
            b_shared = T.alloc_shared(({bk}, {bn}), {rhs_dtype})
            c_local = T.alloc_fragment(({bm}, {bn}), {accumulator_dtype})
            T.clear(c_local)
            for ko in T.Pipelined({upper}, num_stages={num_stages}):
                for mi, ki in T.Parallel({bm}, {bk}):
{lhs_source}
                for ki, ni in T.Parallel({bk}, {bn}):
{rhs_source}
                T.gemm(a_shared, b_shared, c_local)
            for mi, ni in T.Parallel({bm}, {bn}):
{output_source}

    return {kernel.kernel_name}
'''


def _render_tilelang_module(
    kernel: Kernel,
    target: _Target,
    variables: tuple[str, ...],
    outputs: tuple[str, ...],
    shape_params: tuple[str, ...],
    total: str,
    body: str,
    tensors: Mapping[str, _TensorInfo],
    value_types: Mapping[str, ssa.Type],
) -> str:
    threads = common.schedule_int(kernel, "threads", 256)
    total = _rewrite_index_math(total, c_style=False)
    body = _rewrite_index_math(body, c_style=False)
    parameter_names = (*variables, *outputs)
    buffer_names = tuple(name for name in parameter_names if tensors[name].ndim != 0)
    handle_args = ", ".join(
        [
            f"{name}: T.handle"
            if tensors[name].ndim != 0
            else f"{name}: {_tile_scalar_abi_dtype(tensors[name].dtype)}"
            for name in parameter_names
        ]
        + [f"{axis}: {_tile_param_dtype(axis, value_types)}" for axis in shape_params]
    )
    buffer_extents = {
        name: _rewrite_index_math(
            _buffer_storage_extent(tensors[name], fallback=total), c_style=False
        )
        for name in buffer_names
    }
    buffers = "\n".join(
        f"        {name}_buf = T.match_buffer({name}, ({buffer_extents[name]},), {_tile_dtype(tensors[name].dtype)})"
        for name in buffer_names
    )

    return f'''"""TileLang lowering generated by NineToothed from ssa.Program.

Kernel: {kernel.kernel_name}
"""

{_logical_ssa_audit(kernel, target)}

from math import floor

try:
    import tilelang
    import tilelang.language as T
except ImportError:
    tilelang = None
    T = None


def build_{kernel.kernel_name}():
    if tilelang is None:
        raise ImportError("TileLang is required to build this backend artifact.")

    @T.prim_func
    def {kernel.kernel_name}({handle_args}):
{buffers}
        for block_id in T.thread_binding(({total} + {threads - 1}) // {threads}, thread="blockIdx.x"):
            for tx in T.thread_binding({threads}, thread="threadIdx.x"):
                {target.index_name} = block_id * {threads} + tx
                if {target.index_name} < {total}:
{_indent_block(body, "                    ")}

    return {kernel.kernel_name}
'''


TARGET = TileLangTarget()


def emit(kernel: Kernel):
    return common.emit(kernel, TARGET)


__all__ = ["TARGET", "TileLangTarget", "emit"]


def _tile_dtype(dtype: str | None) -> str:
    dtype = common.normalize_dtype(dtype)
    types = {
        "float32": "T.float32",
        "float16": "T.float16",
        "bfloat16": "T.bfloat16",
        "float8_e4m3fn": "T.float8_e4m3fn",
        "float8_e5m2": "T.float8_e5m2",
        "float64": "T.float64",
        "int8": "T.int8",
        "uint8": "T.uint8",
        "int16": "T.int16",
        "uint16": "T.uint16",
        "int32": "T.int32",
        "uint32": "T.uint32",
        "int64": "T.int64",
        "uint64": "T.uint64",
        "bool": "T.bool",
    }

    if dtype not in types:
        raise ValueError(f"Unsupported TileLang SSA dtype: {dtype!r}.")
    return types[dtype]


def _tile_param_dtype(name: str, value_types: Mapping[str, ssa.Type]) -> str:
    type_ = value_types.get(name)

    if type_ is not None and type_.kind == "scalar" and type_.dtype:
        if common.normalize_dtype(type_.dtype) == "bool":
            return "T.int64"
        return _tile_scalar_abi_dtype(type_.dtype)
    return "T.int64"


def _tile_scalar_abi_dtype(dtype: str | None) -> str:
    dtype = common.normalize_dtype(dtype)

    if dtype in {"float16", "bfloat16", "float8_e4m3fn", "float8_e5m2"}:
        return "T.float32"
    return _tile_dtype(dtype)
