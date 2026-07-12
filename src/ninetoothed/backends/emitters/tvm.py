"""TVM syntax hooks for the common SSA emitter."""

import math
from dataclasses import dataclass
from typing import Any, Mapping

from ninetoothed.backends.core import Target
from ninetoothed.backends.emitters import ssa as common
from ninetoothed.backends.emitters.base import EmitterTarget, ModuleRenderContext
from ninetoothed.backends.emitters.context import TensorInfo as _TensorInfo
from ninetoothed.ir import Kernel, ssa

_Target = EmitterTarget
_buffer_storage_extent = common.buffer_storage_extent
_cooperative_dot_plan = common.cooperative_dot_plan
_default_strides = common.default_strides
_indent_block = common.indent_block
_logical_ssa_audit = common.logical_ssa_audit
_normalize_dtype = common.normalize_dtype
_rewrite_index_math = common.rewrite_index_math
_target_index_expr = common.target_index_expr


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
class TvmTarget(EmitterTarget):
    backend: Target = Target.TVM
    language: str = "python/tvm-script"
    suffix: str = "tvm.py"
    source_route: str = "ssa-unified-tvm-emitter"
    buffer_suffix: str = "_buf"
    entrypoint_prefix: str = "build_"
    tir_value_semantics: bool = True
    typed_index_literals: bool = True
    external_atomic_add: bool = True
    mutable_scalar_kind: str = "buffer"

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
            return f"T.atomic_add({args[0]}, {args[1]})"

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
        raw = f"{var}_raw"

        return (
            f"# for {var} in {serial}:\n"
            f'for {raw} in {serial}:\n    {var} = T.Cast("int64", {raw})'
        )

    def reduce_update(self, operator, acc, term):
        if operator == "sum":
            return f"{acc} + {term}"

        function = "T.max" if operator == "max" else "T.min"

        return f"{function}({acc}, {term})"

    def render_module(self, context: ModuleRenderContext) -> str:
        cooperative = _render_tvm_cooperative_dot_module(
            context.kernel,
            self,
            context.variables,
            context.outputs,
            context.shape_params,
            context.tensors,
            context.value_types,
            context.operations,
            context.stores,
        )

        if cooperative is not None:
            return cooperative
        return _render_tvm_module(
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


def _render_tvm_cooperative_dot_module(
    kernel: Kernel,
    target: _Target,
    variables: tuple[str, ...],
    outputs: tuple[str, ...],
    shape_params: tuple[str, ...],
    tensors: Mapping[str, _TensorInfo],
    value_types: Mapping[str, ssa.Type],
    operations: Mapping[str, ssa.Operation],
    stores: tuple[ssa.Operation, ...],
) -> str | None:
    plan = _cooperative_dot_plan(operations, stores, value_types)

    if plan is None or shape_params or len(outputs) != 1:
        return None

    if not _is_cast_only_store(
        plan.store.operands[0], plan.loop.results[0].name, operations
    ):
        return None

    lhs_value, rhs_value = plan.dot.operands[:2]
    lhs = _root_tensor_name(lhs_value, operations, tensors)
    rhs = _root_tensor_name(rhs_value, operations, tensors)
    output = plan.store.operands[1]

    if lhs is None or rhs is None or output not in tensors:
        return None

    if set(variables) != {lhs, rhs}:
        return None

    lhs_info = tensors[lhs]
    rhs_info = tensors[rhs]
    output_info = tensors[output]
    lhs_shape = lhs_info.source_shape or lhs_info.shape
    rhs_shape = rhs_info.source_shape or rhs_info.shape
    output_shape = output_info.source_shape or output_info.shape

    if not (
        len(lhs_shape) == len(rhs_shape) == len(output_shape) == 2
        and lhs_shape[1] == rhs_shape[0]
        and output_shape == (lhs_shape[0], rhs_shape[1])
    ):
        return None

    try:
        m, k = (int(dim) for dim in lhs_shape)
        rhs_k, n = (int(dim) for dim in rhs_shape)
    except ValueError:
        return None

    if rhs_k != k or any(dim <= 0 or dim % 16 for dim in (m, n, k)):
        return None

    if not all(
        _is_contiguous_matrix(info, shape)
        for info, shape in (
            (lhs_info, lhs_shape),
            (rhs_info, rhs_shape),
            (output_info, output_shape),
        )
    ):
        return None

    lhs_dtype = _normalize_dtype(lhs_info.dtype)
    rhs_dtype = _normalize_dtype(rhs_info.dtype)
    output_dtype = _normalize_dtype(output_info.dtype)

    if lhs_dtype != "float16" or rhs_dtype != "float16" or output_dtype != "float16":
        return None

    compute_name = f"{kernel.kernel_name}_compute"
    cast_name = f"{kernel.kernel_name}_cast"
    runtime_config = {
        "mode": "device_pipeline",
        "compute": compute_name,
        "compute_args": {
            "lhs_buf": lhs,
            "rhs_buf": rhs,
            "workspace_buf": "$workspace",
        },
        "cast": cast_name,
        "cast_args": {
            "workspace_buf": "$workspace",
            "output_buf": output,
        },
        "workspace_dtype": "float32",
        "workspace_numel": m * n,
        "output": output,
    }

    return f'''"""TVMScript lowering generated by NineToothed from ssa.Program.

Kernel: {kernel.kernel_name}
Schedule: affine linalg.dot -> DLight GPU Matmul
"""

{_logical_ssa_audit(kernel, target)}

try:
    import tvm
except ImportError:
    import tilelang  # noqa: F401
    import tvm

from tvm.s_tir import dlight as dl

try:
    from tvm.script import tirx as T
except ImportError:
    from tvm.script import tir as T


NINETOOTHED_TVM_RUNTIME = {runtime_config!r}


def build_{kernel.kernel_name}():
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def {compute_name}({lhs}: T.handle, {rhs}: T.handle, workspace: T.handle):
            T.func_attr({{
                'global_symbol': '{compute_name}',
                'tir.noalias': True,
                'tirx.is_global_func': True,
            }})
            lhs_buf = T.match_buffer({lhs}, ({m}, {k}), "{lhs_dtype}")
            rhs_buf = T.match_buffer({rhs}, ({k}, {n}), "{rhs_dtype}")
            workspace_buf = T.match_buffer(workspace, ({m}, {n}), "float32")
            for i, j, reduce_index in T.grid({m}, {n}, {k}):
                with T.sblock("dot"):
                    row, col, reduction = T.axis.remap("SSR", [i, j, reduce_index])
                    with T.init():
                        workspace_buf[row, col] = T.float32(0)
                    workspace_buf[row, col] = (
                        workspace_buf[row, col]
                        + T.Cast("float32", lhs_buf[row, reduction])
                        * T.Cast("float32", rhs_buf[reduction, col])
                    )

        @T.prim_func
        def {cast_name}(workspace: T.handle, {output}: T.handle):
            T.func_attr({{
                'global_symbol': '{cast_name}',
                'tir.noalias': True,
                'tirx.is_global_func': True,
            }})
            workspace_buf = T.match_buffer(workspace, ({m}, {n}), "float32")
            output_buf = T.match_buffer({output}, ({m}, {n}), "{output_dtype}")
            for i, j in T.grid({m}, {n}):
                with T.sblock("cast"):
                    row, col = T.axis.remap("SS", [i, j])
                    output_buf[row, col] = T.Cast(
                        "{output_dtype}", workspace_buf[row, col]
                    )

    target = tvm.target.Target("cuda")
    with target:
        scheduled = dl.ApplyDefaultSchedule(
            dl.gpu.Matmul(),
            dl.gpu.GEMV(),
            dl.gpu.Reduction(),
            dl.gpu.GeneralReduction(),
            dl.gpu.Fallback(),
        )(Module)
    for function_name in ("{compute_name}", "{cast_name}"):
        scheduled[function_name] = scheduled[function_name].with_attr(
            "global_symbol", function_name
        ).with_attr("tirx.is_global_func", True)
    return scheduled
'''


def _is_cast_only_store(
    value: str,
    source: str,
    operations: Mapping[str, ssa.Operation],
) -> bool:
    if value == source:
        return True

    producer = operations.get(value)

    return bool(
        producer is not None
        and producer.opcode == "tensor.cast"
        and len(producer.operands) == 1
        and _is_cast_only_store(producer.operands[0], source, operations)
    )


def _root_tensor_name(
    value: str,
    operations: Mapping[str, ssa.Operation],
    tensors: Mapping[str, _TensorInfo],
) -> str | None:
    if value in tensors:
        return value

    producer = operations.get(value)

    if producer is None or not producer.operands:
        return None

    if producer.opcode == "tensor.extract":
        return _root_tensor_name(producer.operands[0], operations, tensors)

    if producer.opcode not in {"tensor.cast", "tensor.view"}:
        return None

    if len(producer.operands) != 1:
        return None
    return _root_tensor_name(producer.operands[0], operations, tensors)


def _is_contiguous_matrix(info: _TensorInfo, shape: tuple[str, ...]) -> bool:
    strides = info.source_strides or _default_strides(shape)

    return strides == (shape[1], "1")


def _render_tvm_module(
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
    total = _rewrite_index_math(total, c_style=False)
    body = _rewrite_index_math(body, c_style=False)
    guard_total = _target_index_expr(target, total)
    block_extent = f"(({guard_total} + T.int64(255)) // T.int64(256))"
    linear_index = (
        f'T.Cast("int64", block_id) * T.int64({target.block_size}) '
        '+ T.Cast("int64", tx)'
    )
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
    dtypes = tuple(dict.fromkeys(tensors[name].dtype for name in buffer_names))
    dtype_vars = {dtype: f"_{dtype.replace('.', '_')}_dtype" for dtype in dtypes}
    dtype_declarations = "\n".join(
        f'{variable} = tvm.DataType("{dtype}")'
        for dtype, variable in dtype_vars.items()
    )
    buffer_extents = {
        name: _rewrite_index_math(
            _buffer_storage_extent(tensors[name], fallback=total), c_style=False
        )
        for name in buffer_names
    }
    buffers = "\n".join(
        f"            {name}_buf = T.match_buffer({name}, ({buffer_extents[name]},), {dtype_vars[tensors[name].dtype]})"
        for name in buffer_names
    )

    return f'''"""TVMScript lowering generated by NineToothed from ssa.Program.

Kernel: {kernel.kernel_name}
"""

{_logical_ssa_audit(kernel, target)}

from math import floor

try:
    import tvm
except ImportError:
    import tilelang
    import tvm

try:
    from tvm.script import tirx as T
except ImportError:
    from tvm.script import tir as T

{dtype_declarations}


def build_{kernel.kernel_name}():
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def {kernel.kernel_name}({handle_args}):
            T.func_attr({{
                'global_symbol': '{kernel.kernel_name}',
                'tir.noalias': True,
                'tirx.is_global_func': True,
            }})
{buffers}
            for block_id in T.thread_binding({block_extent}, thread='blockIdx.x'):
                for tx in T.thread_binding(256, thread='threadIdx.x'):
                    {target.index_name} = {linear_index}
                    if {target.index_name} < {guard_total}:
{_indent_block(body, "                        ")}

    return Module
'''


TARGET = TvmTarget()


def emit(kernel: Kernel, options=None):
    return common.emit(kernel, TARGET, options)


__all__ = ["TARGET", "TvmTarget", "emit"]


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
        raise ValueError(f"Unsupported TVM SSA dtype: {dtype!r}.")
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
