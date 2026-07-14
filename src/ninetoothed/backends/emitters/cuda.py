"""CUDA syntax hooks for the common SSA emitter."""

import math
import re
from dataclasses import dataclass, replace
from typing import Any

from ninetoothed.backends.core import Target
from ninetoothed.backends.emitters import ssa as common
from ninetoothed.backends.emitters.base import EmitterTarget, ModuleRenderContext
from ninetoothed.backends.emitters.context import EmitContext as _EmitContext
from ninetoothed.ir import Kernel, ssa

_access_axes = common.access_axes
_combined_mask = common.combined_mask
_current_coords = common.current_coords
_dtype_level = common.dtype_level
_emit_element = common.emit_element
_emit_loop_bound = common.emit_loop_bound
_emit_value = common.emit_value
_indent_lines = common.indent_lines
_linearized_index = common.linearized_index
_load_other = common.load_other
_local_symbol = common.local_symbol
_materialize_bool_expr = common.materialize_bool_expr
_materialize_index_expr = common.materialize_index_expr
_normalize_dtype = common.normalize_dtype
_source_index_for_value = common.source_index_for_value
_target_index_expr = common.target_index_expr
_value_axes = common.value_axes
_view_base_coords = common.view_base_coords


@dataclass(frozen=True, kw_only=True)
class CudaTarget(EmitterTarget):
    backend: Target = Target.CUDA
    language: str = "cuda/c++"
    suffix: str = "cu"
    source_route: str = "ssa-unified-cuda-emitter"
    c_style_syntax: bool = True
    native_block_matmul: bool = True

    def index_cast(self, value: str) -> str:
        return f"static_cast<int64_t>({value})"

    def assign_scalar(self, name: str, value: str, *, mutable: bool) -> str:
        del mutable

        return f"{name} = {value};"

    def literal(self, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"

        if isinstance(value, float) and math.isinf(value):
            return "INFINITY" if value > 0 else "-INFINITY"

        if value == "inf":
            return "INFINITY"

        if value == "-inf":
            return "-INFINITY"
        return repr(value)

    def load(self, tensor, index, *, mask=None, other=0.0):
        del mask, other
        rendered = f"({index})" if index.startswith("nt_idx_") else index

        return f"{self.tensor_ref(tensor)}[{rendered}]"

    def store(self, tensor, index, value, *, mask=None):
        rendered = f"({index})" if index.startswith("nt_idx_") else index
        assignment = f"{self.tensor_ref(tensor)}[{rendered}] = {value};"

        return assignment if mask is None else f"if ({mask}) {{\n    {assignment}\n}}"

    def cast(self, dtype, value):
        return f"static_cast<{self.type_name(dtype)}>({value})"

    def type_name(self, dtype, kind=None):
        return _cuda_type(dtype, kind)

    def where(self, cond, yes, no):
        return f"(({cond}) ? ({yes}) : ({no}))"

    def call(self, name, args):
        if name == "where":
            return self.where(args[0], args[1], args[2])

        if name == "atomic_add":
            return f"atomicAdd({args[0]}, {args[1]})"

        if name == "load" and args:
            return f"*({args[0]})"

        if name in {"block_dot", "dot"} and len(args) == 2:
            return f"(({args[0]}) * ({args[1]}))"

        if name == "rand" and len(args) >= 2:
            return f"ninetoothed_curand_uniform(({args[0]}), ({args[1]}))"

        functions = {
            "abs": "fabsf",
            "acos": "acosf",
            "asin": "asinf",
            "atan": "atanf",
            "atan2": "atan2f",
            "_atan2_approx": "atan2f",
            "ceil": "ceilf",
            "cos": "cosf",
            "cosh": "coshf",
            "dot": "dot",
            "erf": "erff",
            "exp": "expf",
            "exp2": "exp2f",
            "expm1": "expm1f",
            "floor": "floorf",
            "log": "logf",
            "log1p": "log1pf",
            "log2": "log2f",
            "log10": "log10f",
            "maximum": "fmaxf",
            "max": "fmaxf",
            "minimum": "fminf",
            "min": "fminf",
            "pow": "powf",
            "rsqrt": "rsqrtf",
            "sin": "sinf",
            "sinh": "sinhf",
            "sqrt": "sqrtf",
            "tan": "tanf",
            "tanh": "tanhf",
        }
        function = functions.get(name, name)

        return f"{function}({', '.join(args)})"

    def local_decl(self, type_: ssa.Type, name: str, expr: str) -> str:
        if type_.kind == "pointer":
            return f"auto {name} = {expr};"
        return f"{self.type_name(type_.dtype, type_.kind)} {name} = {expr};"

    def loop_header(self, var, lower, upper, step):
        return f"for (int64_t {var} = {lower}; {var} < {upper}; {var} += {step}) {{"

    def reduce_update(self, operator, acc, term):
        if operator == "sum":
            return f"{acc} + {term}"

        function = "fmaxf" if operator == "max" else "fminf"

        return f"{function}({acc}, {term})"

    def arithmetic_result_type(self, operation, context) -> ssa.Type:
        return _cuda_arithmetic_result_type(operation, context)

    def coerce_binary_args(self, operation, args, context):
        return _coerce_cuda_binary_args(operation, args, context)

    def emit_dot_operand(self, name, coords, context):
        return _emit_cuda_direct_dot_operand(name, coords, context)

    def emit_block_dot(self, operation, context, coords=None):
        return _emit_cuda_wmma_dot(operation, context, coords=coords)

    def emit_reduction_loop(self, local, operation, context):
        return _emit_cuda_wmma_reduction_loop(local, operation, context)

    def render_module(self, context: ModuleRenderContext) -> str:
        kernel = context.kernel
        threads = common.schedule_int(kernel, "threads", 256)
        total = _cuda_integer_expr(context.total)
        grid_total = _cuda_integer_expr(context.grid_total)
        body = _cuda_integer_expr(context.body)
        kernel_params = _render_signature_params(
            [
                *(
                    _signature_param(
                        name, context.tensors[name], readonly=True, restrict=True
                    )
                    for name in context.variables
                ),
                *(
                    _signature_param(
                        name, context.tensors[name], readonly=False, restrict=True
                    )
                    for name in context.outputs
                ),
                *(
                    _auxiliary_param(axis, context.tensors)
                    for axis in context.shape_params
                ),
            ]
        )
        launch_params = _render_signature_params(
            [
                *(
                    _signature_param(
                        name, context.tensors[name], readonly=True, restrict=False
                    )
                    for name in context.variables
                ),
                *(
                    _signature_param(
                        name, context.tensors[name], readonly=False, restrict=False
                    )
                    for name in context.outputs
                ),
                *(
                    _auxiliary_param(axis, context.tensors)
                    for axis in context.shape_params
                ),
                "cudaStream_t stream",
            ]
        )
        args = ", ".join((*context.variables, *context.outputs, *context.shape_params))

        if context.block_program:
            if threads != 256:
                raise ValueError(
                    "CUDA WMMA lowering currently requires exactly 256 threads."
                )

            schedule = kernel.ssa.metadata.get("schedule", {}) if kernel.ssa else {}
            mma_shape = schedule.get("mma_shape", {"m": 16, "n": 16, "k": 16})
            mma_m = int(mma_shape.get("m", 16))
            mma_n = int(mma_shape.get("n", 16))
            mma_k = int(mma_shape.get("k", 16))

            if (mma_m, mma_n, mma_k) != (16, 16, 16):
                raise ValueError(
                    "The current CUDA WMMA intrinsic lowering supports only "
                    "a 16x16x16 fragment; use a CTA tiling pass to compose fragments."
                )

            tile_rows = _cuda_integer_expr(
                f"(({context.axes[0]}) + {mma_m - 1}) / {mma_m}"
            )
            tile_cols = _cuda_integer_expr(
                f"(({context.axes[1]}) + {mma_n - 1}) / {mma_n}"
            )
            kernel_prelude = f"""    int64_t nt_tile_rows = {tile_rows};
    int64_t nt_tile_cols = {tile_cols};
    int64_t nt_tiles_per_outer = nt_tile_rows * nt_tile_cols;
    int64_t nt_outer_index = static_cast<int64_t>(blockIdx.x) / nt_tiles_per_outer;
    int64_t nt_tile_index = static_cast<int64_t>(blockIdx.x) % nt_tiles_per_outer;
    int64_t nt_tile_row = (nt_tile_index / nt_tile_cols) * {mma_m};
    int64_t nt_tile_col = (nt_tile_index % nt_tile_cols) * {mma_n};
    int64_t nt_matrix_row = nt_tile_row + static_cast<int64_t>(threadIdx.x) / {mma_n};
    int64_t nt_matrix_col = nt_tile_col + static_cast<int64_t>(threadIdx.x) % {mma_n};
    int64_t {self.index_name} = nt_matrix_row * ({context.axes[1]}) + nt_matrix_col;
    bool nt_matrix_active = nt_matrix_row < ({context.axes[0]}) && nt_matrix_col < ({context.axes[1]});
{common.indent_block(body, "    ")}"""
            blocks_expr = grid_total
        else:
            kernel_prelude = f"""    int64_t {self.index_name} = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if ({self.index_name} < {total}) {{
{common.indent_block(body, "        ")}
    }}"""
            blocks_expr = f"({total} + threads - 1) / threads"

        curand_header, curand_support = _curand_support(context)

        return f"""// Generated by NineToothed's CUDA SSA backend.
// Kernel: {kernel.kernel_name}
// Lowering IR: ssa.Program

#include <math.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <stdint.h>{curand_header}

using namespace nvcuda;

{curand_support}

extern "C" __global__ void {kernel.kernel_name}_kernel(
{kernel_params}
) {{
{kernel_prelude}
}}

extern "C" int launch_{kernel.kernel_name}(
{launch_params}
) {{
    constexpr int threads = {threads};
    int64_t blocks = {blocks_expr};
    if (blocks <= 0) {{
        return static_cast<int>(cudaSuccess);
    }}
    {kernel.kernel_name}_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        {args}
    );
    return static_cast<int>(cudaGetLastError());
}}
"""


def _curand_support(context: ModuleRenderContext) -> tuple[str, str]:
    operations = {id(operation): operation for operation in context.operations.values()}

    if not any(operation.opcode == "math.rand" for operation in operations.values()):
        return "", ""
    return (
        "\n#include <curand_kernel.h>",
        """__device__ __forceinline__ float ninetoothed_curand_uniform(
    uint64_t seed, uint64_t offset
) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, 0ULL, offset, &state);
    return 1.0f - curand_uniform(&state);
}""",
    )


def _render_signature_params(params: list[str]) -> str:
    return ",\n".join(f"    {param}" for param in params)


def _signature_param(name, info, *, readonly: bool, restrict: bool) -> str:
    dtype = _cuda_type(info.dtype)

    if info.ndim == 0:
        return f"{dtype} {name}"

    const = "const " if readonly else ""
    qualifier = " __restrict__" if restrict else ""

    return f"{const}{dtype}*{qualifier} {name}"


def _auxiliary_param(name, tensors) -> str:
    info = tensors.get(name)

    if info is not None and info.ndim == 0:
        return _signature_param(name, info, readonly=True, restrict=False)
    return f"int64_t {name}"


def _cuda_type(dtype: str | None, kind: str | None = None) -> str:
    dtype = common.normalize_dtype(dtype)

    if kind == "pointer":
        return f"{_cuda_type(dtype)}*"

    if kind == "index" or dtype in {"index", "int64"}:
        return "int64_t"

    types = {
        "float32": "float",
        "float16": "half",
        "bfloat16": "__nv_bfloat16",
        "float8_e4m3fn": "__nv_fp8_e4m3",
        "float8_e5m2": "__nv_fp8_e5m2",
        "float64": "double",
        "int8": "int8_t",
        "uint8": "uint8_t",
        "int16": "int16_t",
        "uint16": "uint16_t",
        "int32": "int32_t",
        "uint32": "uint32_t",
        "uint64": "uint64_t",
        "bool": "bool",
    }

    if dtype not in types:
        raise ValueError(f"Unsupported CUDA SSA dtype: {dtype!r}.")
    return types[dtype]


def _cuda_integer_expr(expr: str) -> str:
    previous = None
    current = common.rewrite_index_math(expr, c_style=True).replace("//", "/")
    current = re.sub(r"\bTrue\b", "true", current)
    current = re.sub(r"\bFalse\b", "false", current)
    pattern = re.compile(r"floor\(\(([^()]+)\)/([A-Za-z_][A-Za-z0-9_]*)\)")

    while current != previous:
        previous = current
        current = pattern.sub(r"((\1)/(\2))", current)
    return current


def _cuda_arithmetic_result_type(op: ssa.Operation, ctx: _EmitContext) -> ssa.Type:
    result_type = op.results[0].type

    if not op.opcode.startswith("arith."):
        return result_type

    if _normalize_dtype(result_type.dtype) not in {
        "float8_e4m3fn",
        "float8_e5m2",
        "float16",
        "bfloat16",
    }:
        return result_type
    return replace(result_type, dtype="float32")


def _emit_cuda_direct_dot_operand(
    name: str, coords: tuple[str, ...], ctx: _EmitContext
) -> tuple[str, str | None]:
    original_name = name
    original_coords = coords
    producer = ctx.operations.get(name)

    while (
        producer is not None and producer.opcode == "tensor.view" and producer.operands
    ):
        coords = _view_base_coords(producer, coords, ctx)
        name = producer.operands[0]
        producer = ctx.operations.get(name)

    info = ctx.tensor_infos.get(name)

    if info is not None and info.ndim == 0:
        return _emit_value(name, ctx), None

    if info is None or _load_other(info) != 0.0:
        return _emit_element(original_name, original_coords, ctx), None

    dtype_level = _dtype_level(name, ctx)
    axes = _access_axes(info, ctx, dtype_level, fallback=_value_axes(name, ctx))
    view_index = _linearized_index(coords, axes) if coords else "0"
    source_index = _target_index_expr(
        ctx.target,
        _source_index_for_value(
            info,
            view_index,
            ctx,
            level=dtype_level,
            value_coords=coords,
        ),
    )
    source_index = _materialize_index_expr(source_index, ctx)
    mask = _combined_mask(
        ctx.target,
        None,
        info,
        view_index,
        ctx=ctx,
        level=dtype_level,
        value_coords=coords,
    )

    if mask is not None:
        mask = _materialize_bool_expr(mask, ctx) or mask
    return ctx.target.load(name, source_index), mask


def _emit_cuda_wmma_dot(
    op: ssa.Operation, ctx: _EmitContext, *, coords: tuple[str, ...] | None
) -> str:
    lhs, rhs = op.operands[:2]
    lhs_axes = _value_axes(lhs, ctx)
    rhs_axes = _value_axes(rhs, ctx)
    result_axes = tuple(str(dim) for dim in op.results[0].type.shape)
    result_coords = (
        tuple(coords) if coords is not None else _current_coords(result_axes, ctx)
    )

    if len(result_coords) != 2:
        raise ValueError("CUDA WMMA dot lowering requires a rank-2 result domain.")

    local = f"{_local_symbol(op.results[0].name, ctx)}_wmma"
    row_base = f"(({result_coords[0]}) / 16) * 16"
    col_base = f"(({result_coords[1]}) / 16) * 16"
    k_extent = lhs_axes[-1]
    loop_var = f"{local}_k"
    lhs_row = f"({row_base}) + (static_cast<int64_t>(threadIdx.x) / 16)"
    lhs_col = f"{loop_var} + (static_cast<int64_t>(threadIdx.x) % 16)"
    rhs_row = f"{loop_var} + (static_cast<int64_t>(threadIdx.x) / 16)"
    rhs_col = f"({col_base}) + (static_cast<int64_t>(threadIdx.x) % 16)"

    ctx.lines.extend(
        (
            f"__shared__ half {local}_lhs[256];",
            f"__shared__ half {local}_rhs[256];",
            f"__shared__ float {local}_out[256];",
            (f"wmma::fragment<wmma::accumulator, 16, 16, 16, float> {local}_acc;"),
            f"if (threadIdx.x < 32) {{ wmma::fill_fragment({local}_acc, 0.0f); }}",
            "__syncthreads();",
            ctx.target.loop_header(loop_var, "0", k_extent, "16"),
        )
    )

    body_lines: list[str] = []
    body = ctx.child(lines=body_lines, memo=dict(ctx.memo))
    lhs_value = _emit_element(lhs, (lhs_row, lhs_col), body)
    rhs_value = _emit_element(rhs, (rhs_row, rhs_col), body)
    lhs_active = f"({lhs_row}) < ({lhs_axes[0]}) && ({lhs_col}) < ({lhs_axes[1]})"
    rhs_active = f"({rhs_row}) < ({rhs_axes[0]}) && ({rhs_col}) < ({rhs_axes[1]})"
    body_lines.extend(
        (
            (
                f"{local}_lhs[threadIdx.x] = ({lhs_active}) "
                f"? __float2half_rn(static_cast<float>({lhs_value})) "
                ": __float2half_rn(0.0f);"
            ),
            (
                f"{local}_rhs[threadIdx.x] = ({rhs_active}) "
                f"? __float2half_rn(static_cast<float>({rhs_value})) "
                ": __float2half_rn(0.0f);"
            ),
            "__syncthreads();",
            "if (threadIdx.x < 32) {",
            (
                "    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, "
                f"wmma::row_major> {local}_a;"
            ),
            (
                "    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, "
                f"wmma::row_major> {local}_b;"
            ),
            f"    wmma::load_matrix_sync({local}_a, {local}_lhs, 16);",
            f"    wmma::load_matrix_sync({local}_b, {local}_rhs, 16);",
            f"    wmma::mma_sync({local}_acc, {local}_a, {local}_b, {local}_acc);",
            "}",
            "__syncthreads();",
        )
    )
    ctx.lines.extend(_indent_lines(body_lines, ctx.target))
    ctx.lines.extend(
        (
            "}",
            "if (threadIdx.x < 32) {",
            (
                f"    wmma::store_matrix_sync({local}_out, {local}_acc, 16, "
                "wmma::mem_row_major);"
            ),
            "}",
            "__syncthreads();",
        )
    )

    return f"{local}_out[threadIdx.x]"


def _coerce_cuda_binary_args(
    op: ssa.Operation, args: tuple[str, ...], ctx: _EmitContext
) -> tuple[str, ...]:
    if len(args) != len(op.operands):
        return args

    result_type = op.results[0].type if op.results else None

    if result_type is not None and result_type.kind == "pointer":
        return args

    if any(
        (type_ := ctx.value_types.get(operand)) is not None and type_.kind == "pointer"
        for operand in op.operands
    ):
        return args

    dtype = (
        _normalize_dtype(result_type.dtype)
        if result_type is not None and _normalize_dtype(result_type.dtype) != "bool"
        else _cuda_common_operand_dtype(op.operands, ctx)
    )

    if dtype is None:
        return args

    if dtype in {"float8_e4m3fn", "float8_e5m2", "float16", "bfloat16"}:
        dtype = "float32"

    coerced = []

    for operand, value in zip(op.operands, args):
        operand_type = ctx.value_types.get(operand)
        operand_dtype = _normalize_dtype(
            operand_type.dtype if operand_type is not None else None
        )
        coerced.append(
            ctx.target.cast(dtype, value) if operand_dtype != dtype else value
        )
    return tuple(coerced)


def _cuda_common_operand_dtype(
    operands: tuple[str, ...], ctx: _EmitContext
) -> str | None:
    ranks = {
        "bool": 0,
        "int32": 1,
        "int64": 2,
        "float8_e4m3fn": 3,
        "float8_e5m2": 3,
        "float16": 4,
        "bfloat16": 4,
        "float32": 5,
        "float64": 6,
    }
    dtypes = [
        _normalize_dtype(type_.dtype)
        for operand in operands
        if (type_ := ctx.value_types.get(operand)) is not None
    ]

    return max(dtypes, key=lambda dtype: ranks.get(dtype, -1)) if dtypes else None


def _emit_cuda_wmma_reduction_loop(
    local: str, op: ssa.Operation, ctx: _EmitContext
) -> str | None:
    if len(op.results) != 1 or len(op.regions) != 1:
        return None

    iter_attrs = tuple(op.attrs.get("iter_args", ()))

    if len(iter_attrs) != 1:
        return None

    region = op.regions[0]
    dot = next(
        (
            inner
            for inner in region.operations
            if inner.opcode in {"linalg.dot", "linalg.matmul"}
            and len(inner.operands) >= 2
            and len(inner.results) == 1
        ),
        None,
    )
    add = next(
        (
            inner
            for inner in region.operations
            if inner.opcode == "arith.add" and len(inner.results) == 1
        ),
        None,
    )
    yield_op = next(
        (inner for inner in region.operations if inner.opcode == "scf.yield"), None
    )

    if dot is None or add is None or yield_op is None:
        return None

    block_arg = str(iter_attrs[0]["block_arg"])
    dot_result = dot.results[0].name

    if set(add.operands) != {block_arg, dot_result}:
        return None

    if yield_op.operands != (add.results[0].name,):
        return None

    initial = str(iter_attrs[0]["initial"])
    initial_op = ctx.operations.get(initial)

    if initial_op is None or initial_op.opcode not in {
        "arith.constant",
        "tensor.zeros",
        "tensor.full",
    }:
        return None

    if initial_op.opcode != "tensor.zeros" and initial_op.attrs.get("value", 0) != 0:
        return None

    lhs, rhs = dot.operands[:2]
    lhs_axes = _value_axes(lhs, ctx)
    rhs_axes = _value_axes(rhs, ctx)
    result_axes = tuple(str(dim) for dim in dot.results[0].type.shape)
    result_coords = _current_coords(result_axes, ctx)

    if len(lhs_axes) != 2 or len(rhs_axes) != 2 or len(result_coords) != 2:
        return None

    lower = _emit_loop_bound(op.operands[0], ctx)
    upper = _emit_loop_bound(op.operands[1], ctx)
    step = _emit_loop_bound(op.operands[2], ctx)
    loop_var = f"{local}_i"
    induction = str(op.attrs.get("induction", "%iv"))
    bindings = dict(ctx.bindings or {})
    bindings[induction] = loop_var
    local_name = f"{_local_symbol(op.results[0].name, ctx)}_wmma_reduction"
    row_base = f"(({result_coords[0]}) / 16) * 16"
    col_base = f"(({result_coords[1]}) / 16) * 16"
    k_var = f"{local_name}_k"
    lhs_row = f"({row_base}) + (static_cast<int64_t>(threadIdx.x) / 16)"
    lhs_col = f"{k_var} + (static_cast<int64_t>(threadIdx.x) % 16)"
    rhs_row = f"{k_var} + (static_cast<int64_t>(threadIdx.x) / 16)"
    rhs_col = f"({col_base}) + (static_cast<int64_t>(threadIdx.x) % 16)"

    ctx.lines.extend(
        (
            f"__shared__ half {local_name}_lhs[256];",
            f"__shared__ half {local_name}_rhs[256];",
            f"__shared__ float {local_name}_out[256];",
            (f"wmma::fragment<wmma::accumulator, 16, 16, 16, float> {local_name}_acc;"),
            (
                f"if (threadIdx.x < 32) {{ wmma::fill_fragment({local_name}_acc, "
                "0.0f); }"
            ),
            "__syncthreads();",
            ctx.target.loop_header(loop_var, lower, upper, step),
            f"    {ctx.target.loop_header(k_var, '0', lhs_axes[-1], '16')}",
        )
    )

    body_lines: list[str] = []
    body = ctx.child(lines=body_lines, memo=dict(ctx.memo), bindings=bindings)
    lhs_value = _emit_element(lhs, (lhs_row, lhs_col), body)
    rhs_value = _emit_element(rhs, (rhs_row, rhs_col), body)
    lhs_active = f"({lhs_row}) < ({lhs_axes[0]}) && ({lhs_col}) < ({lhs_axes[1]})"
    rhs_active = f"({rhs_row}) < ({rhs_axes[0]}) && ({rhs_col}) < ({rhs_axes[1]})"
    body_lines.extend(
        (
            (
                f"{local_name}_lhs[threadIdx.x] = ({lhs_active}) "
                f"? __float2half_rn(static_cast<float>({lhs_value})) "
                ": __float2half_rn(0.0f);"
            ),
            (
                f"{local_name}_rhs[threadIdx.x] = ({rhs_active}) "
                f"? __float2half_rn(static_cast<float>({rhs_value})) "
                ": __float2half_rn(0.0f);"
            ),
            "__syncthreads();",
            "if (threadIdx.x < 32) {",
            (
                "    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, "
                f"wmma::row_major> {local_name}_a;"
            ),
            (
                "    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, "
                f"wmma::row_major> {local_name}_b;"
            ),
            f"    wmma::load_matrix_sync({local_name}_a, {local_name}_lhs, 16);",
            f"    wmma::load_matrix_sync({local_name}_b, {local_name}_rhs, 16);",
            (
                f"    wmma::mma_sync({local_name}_acc, {local_name}_a, "
                f"{local_name}_b, {local_name}_acc);"
            ),
            "}",
            "__syncthreads();",
        )
    )
    ctx.lines.extend(_indent_lines(_indent_lines(body_lines, ctx.target), ctx.target))
    ctx.lines.extend(
        (
            "    }",
            "}",
            "if (threadIdx.x < 32) {",
            (
                f"    wmma::store_matrix_sync({local_name}_out, {local_name}_acc, "
                "16, wmma::mem_row_major);"
            ),
            "}",
            "__syncthreads();",
        )
    )
    result = f"{local_name}_out[threadIdx.x]"
    ctx.memo[op.results[0].name] = result

    return result


TARGET = CudaTarget()


def emit(kernel: Kernel):
    return common.emit(kernel, TARGET)


__all__ = ["CudaTarget", "TARGET", "emit"]
