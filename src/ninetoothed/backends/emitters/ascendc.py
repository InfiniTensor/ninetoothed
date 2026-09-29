"""AscendC syntax hooks for the common SSA emitter.

The Huawei AscendC task model maps one block per ``GetBlockIdx()`` on the
AI cores of Ascend 910B devices.  Kernels are declared ``__global__
__aicore__`` with ``GM_ADDR`` device pointers, host launchers use the
``<<<blocks, workspace, stream>>>`` trigram against ``aclrtStream``
streams, and the whole unit compiles with the CANN ``ccec`` compiler into
a shared library exposing the same C-ABI launcher contract as the CUDA and
BangC backends.
"""

import math
import re
from dataclasses import dataclass, replace
from typing import Any, Mapping

from ninetoothed.backends.core import Target
from ninetoothed.backends.emitters import ssa as common
from ninetoothed.backends.emitters.analysis import walk_ops as _walk_ops
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
_fresh_temp = common.fresh_temp
_indent_lines = common.indent_lines
_linearized_index = common.linearized_index
_load_other = common.load_other
_local_symbol = common.local_symbol
_materialize_bool_expr = common.materialize_bool_expr
_materialize_index_expr = common.materialize_index_expr
_nested_local_suffix = common.nested_local_suffix
_normalize_dtype = common.normalize_dtype
_reduction_identity = common.reduction_identity
_source_index_for_value = common.source_index_for_value
_target_index_expr = common.target_index_expr
_value_axes = common.value_axes
_view_base_coords = common.view_base_coords


@dataclass(frozen=True, kw_only=True)
class AscendCTarget(EmitterTarget):
    backend: Target = Target.ASCENDC
    language: str = "ascendc/c++"
    suffix: str = "ascendc"
    source_route: str = "ssa-unified-ascendc-emitter"
    c_style_syntax: bool = True

    def index_cast(self, value: str) -> str:
        return f"(int64_t)({value})"

    def assign_scalar(self, name: str, value: str, *, mutable: bool) -> str:
        del mutable

        return f"{name} = {value};"

    def literal(self, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"

        if isinstance(value, float) and math.isinf(value):
            return "nt_inf()" if value > 0 else "-nt_inf()"

        if value == "inf":
            return "nt_inf()"

        if value == "-inf":
            return "-nt_inf()"

        if isinstance(value, float):
            # Ccec treats unsuffixed float literals as double, which is not
            # supported in aicore functions.
            return f"{value}f"

        if isinstance(value, int):
            return repr(value)

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
        normalized = _normalize_dtype(dtype)

        if normalized == "float16":
            return f"((half)(float)({value}))"

        return f"(({self.type_name(dtype)})({value}))"

    def type_name(self, dtype, kind=None):
        return _ascendc_type(dtype, kind)

    def where(self, cond, yes, no):
        no = _ascendc_float_suffix(no)

        return f"(({cond}) ? ({yes}) : ({no}))"

    def call(self, name, args):
        if name == "where":
            return self.where(args[0], args[1], args[2])

        if name == "load" and args:
            return f"*({args[0]})"

        if name in {"block_dot", "dot"} and len(args) == 2:
            return f"(({args[0]}) * ({args[1]}))"

        if name == "rand" and len(args) >= 2:
            return f"ninetoothed_rand_uniform(({args[0]}), ({args[1]}))"

        if name == "sqrt" and len(args) == 1:
            return f"sqrt(({args[0]}))"

        function = _ASCENDC_CALL_FUNCS.get(name, name)

        return f"{function}({', '.join(args)})"

    def supports_cooperative_reduction(self, schedule):
        return bool(schedule.get("ascendc_cooperative_reduction"))

    def thread_id(self):
        # One block is one serial program; the cooperative lowering uses a
        # single lane so thread-strided loops degenerate to full coverage.
        return "0"

    def thread_count(self):
        return "1"

    def emit_cooperative_reduction(self, local, operation, context):
        return _emit_ascendc_cooperative_reduction(local, operation, context)

    def program_id(self, axis=0):
        if axis != 0:
            raise ValueError("AscendC kernels use a one-dimensional block grid.")
        return "(int64_t)GetBlockIdx()"

    def emit_dot_operand(self, name, coords, context):
        return _emit_ascendc_direct_dot_operand(name, coords, context)

    def local_decl(self, type_: ssa.Type, name: str, expr: str) -> str:
        if type_.kind == "pointer":
            return f"auto {name} = {expr};"
        return f"{self.type_name(type_.dtype, type_.kind)} {name} = {expr};"

    def loop_header(self, var, lower, upper, step):
        return f"for (int64_t {var} = {lower}; {var} < {upper}; {var} += {step}) {{"

    def reduce_update(self, operator, acc, term):
        if operator == "sum":
            return f"{acc} + {term}"

        function = "max" if operator == "max" else "min"

        return f"{function}({acc}, {term})"

    def arithmetic_result_type(self, operation, context) -> ssa.Type:
        return _ascendc_arithmetic_result_type(operation, context)

    def coerce_binary_args(self, operation, args, context):
        return _coerce_ascendc_binary_args(operation, args, context)

    def render_module(self, context: ModuleRenderContext) -> str:
        if context.vector_program or context.block_program or context.scalar_program:
            raise ValueError(
                "The AscendC backend renders every kernel through the generic "
                "flat-index block domain; vector/block program modes are not "
                "supported."
            )

        _ascendc_validate_reduction_lowering(context)
        _ascendc_validate_atomics(context)

        kernel = context.kernel
        options = dict(kernel.compiler_options.get("backend_options", {}))
        default_chunk = int(options.get("task_chunk", 1024))
        chunk = common.schedule_int(kernel, "ascendc_task_chunk", default_chunk)
        total = _ascendc_integer_expr(context.total)
        grid_total = _ascendc_integer_expr(context.grid_total)
        body = _ascendc_integer_expr(context.body)

        kernel_params = _render_signature_params(
            [
                *(
                    _signature_param(name, context.tensors[name])
                    for name in context.variables
                ),
                *(
                    _signature_param(name, context.tensors[name])
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
                    _launch_param(name, context.tensors[name])
                    for name in context.variables
                ),
                *(
                    _launch_param(name, context.tensors[name])
                    for name in context.outputs
                ),
                *(
                    _auxiliary_param(axis, context.tensors)
                    for axis in context.shape_params
                ),
                "aclrtStream stream",
            ]
        )

        tensor_casts = []
        launch_args = []

        for name in (*context.variables, *context.outputs):
            info = context.tensors.get(name)

            if info is not None and info.ndim != 0:
                tensor_casts.append(
                    f"    __gm__ {self.type_name(info.dtype)}* {name} = "
                    f"(__gm__ {self.type_name(info.dtype)}*){name}_gm;"
                )
                launch_args.append(f"(GM_ADDR)(uintptr_t){name}")
            else:
                launch_args.append(name)

        launch_args.extend(context.shape_params)

        if context.cooperative_reduction_program:
            from ninetoothed.backends.emitters.ascendc_reduce import (
                match_reduce_broadcast,
                render_reduce_broadcast,
            )

            reduce_match = match_reduce_broadcast(context)

            if reduce_match is not None:
                kernel_prelude = render_reduce_broadcast(reduce_match)
                tasks_expr = reduce_match[3]
            else:
                # The runtime may execute entities whose GetBlockIdx() exceeds
                # the launched grid; without a bound guard they compute
                # out-of-domain coordinates and corrupt adjacent allocations.
                kernel_prelude = (
                    "\n".join(tensor_casts)
                    + f"""
    if ((int64_t)GetBlockIdx() < ({grid_total})) {{
{common.indent_block(body, "        ")}
    }}"""
                    if tensor_casts
                    else f"""
    if ((int64_t)GetBlockIdx() < ({grid_total})) {{
{common.indent_block(body, "        ")}
    }}"""
                )

            tasks_expr = grid_total
        else:
            from ninetoothed.backends.emitters.ascendc_reduce import (
                match_row_reduce as _row_match,
            )
            from ninetoothed.backends.emitters.ascendc_reduce import (
                render_row_reduce as _row_render,
            )
            from ninetoothed.backends.emitters.ascendc_vector import (
                _VECTOR_ELEM_CHUNK,
                match_vector_elementwise,
                render_vector_elementwise,
            )

            row_match = _row_match(context)
            vector_match = match_vector_elementwise(context, _normalize_dtype)
            total_param = _elementwise_total_param(context)

            if row_match is not None:
                kernel_prelude = _row_render(row_match)
                tasks_expr = f"(({row_match[3]}) + 7) / 8"
            elif vector_match is not None and total_param is not None:
                kernel_prelude = render_vector_elementwise(vector_match, total_param)
                tasks_expr = (
                    f"({total_param} + {_VECTOR_ELEM_CHUNK} - 1) / {_VECTOR_ELEM_CHUNK}"
                )
            else:
                kernel_prelude = (
                    "\n".join(tensor_casts)
                    + f"""
    const int64_t nt_chunk = {chunk};
    for (int64_t nt_lane = 0; nt_lane < nt_chunk; nt_lane++) {{
        int64_t {self.index_name} = (int64_t)GetBlockIdx() * nt_chunk + nt_lane;
        if ({self.index_name} < {total}) {{
{common.indent_block(body, "            ")}
        }}
    }}"""
                )
                tasks_expr = f"({total} + nt_chunk - 1) / nt_chunk"

        math_tail = any(
            f"nt_{name}(" in kernel_prelude
            for name in (
                "exp",
                "log",
                "tanh",
                "sigmoid",
                "sin",
                "cos",
                "tan",
                "atan",
                "erf",
                "floor",
                "ceil",
                "rsqrt",
                "fabs",
            )
        )

        support = _ascendc_support(context, force_math=math_tail)

        return f"""// Generated by NineToothed's AscendC SSA backend.
// Kernel: {kernel.kernel_name}
// Lowering IR: ssa.Program

#include "kernel_operator.h"
#include "acl/acl.h"

using namespace AscendC;

{support}

extern "C" __global__ __aicore__ void {kernel.kernel_name}_kernel(
{kernel_params}
) {{
{kernel_prelude}
}}

extern "C" int launch_{kernel.kernel_name}(
{launch_params}
) {{
    const int64_t nt_chunk = {chunk};
    int64_t nt_blocks = {tasks_expr};
    if (nt_blocks <= 0) {{
        return 0;
    }}
    if (nt_blocks > 2147483647LL) {{
        return 1;
    }}
    {kernel.kernel_name}_kernel<<<(uint32_t)(nt_blocks), nullptr, stream>>>(
        {", ".join(launch_args)}
    );
    return 0;
}}
"""


def _ascendc_float_suffix(value: str) -> str:
    """Add the float suffix to bare decimal literals for aicore.

    ccec treats unsuffixed decimal literals as double, which is not
    supported in aicore functions.
    """
    stripped = value.strip()

    if re.fullmatch(r"-?\d+\.\d+", stripped):
        return f"{stripped}f"

    return value


def _emit_ascendc_cooperative_reduction(
    local: str, operation: ssa.Operation, ctx: _EmitContext
) -> str:
    """Reduce the full extent inside one block; no cross-thread steps needed."""
    schedule = dict(ctx.reduction_schedule or {})
    axis = int(schedule["axis"]) if "axis" in schedule else None

    if axis is None or not operation.operands or not operation.results:
        raise ValueError("Malformed AscendC cooperative reduction operation.")

    operator = operation.opcode[len("reduce.") :]
    operand = operation.operands[0]
    operand_axes = _value_axes(operand, ctx)
    extent = str(schedule.get("extent", operand_axes[axis]))
    operand_type = ctx.value_types.get(operand)
    result_dtype = _normalize_dtype(operation.results[0].type.dtype or "float32")
    operand_dtype = _normalize_dtype(
        operand_type.dtype if operand_type is not None else result_dtype
    )
    accumulator_dtype = _ascendc_reduction_accumulator_dtype(
        operand_dtype, result_dtype
    )
    accumulator_type = ssa.Type(kind="scalar", dtype=accumulator_dtype)
    accumulator = _fresh_temp(ctx, "nt_reduce_acc")
    reduction_index = _fresh_temp(ctx, "nt_reduce_index")
    identity = _reduction_identity(operator, accumulator_type, ctx.target)
    coordinates = list(ctx.coordinate_exprs)
    coordinates[axis] = reduction_index
    coordinates = tuple(coordinates)
    linear = _target_index_expr(
        ctx.target, _linearized_index(coordinates, operand_axes)
    )
    body_lines: list[str] = []
    body = ctx.child(
        lines=body_lines,
        memo=dict(ctx.memo),
        coordinate_exprs=coordinates,
        index_expr=linear,
        inner_index_expr=linear,
        reduce_axis=axis,
        reduce_index=reduction_index,
        mask_expr=None,
        local_suffix=_nested_local_suffix(ctx, accumulator),
    )
    term = _emit_element(operand, coordinates, body)
    term = ctx.target.cast(accumulator_dtype, term)
    body_lines.append(
        f"{accumulator} = "
        f"{_ascendc_reduction_update(operator, accumulator, term, accumulator_dtype)};"
    )

    ctx.lines.append(ctx.target.local_decl(accumulator_type, accumulator, identity))
    ctx.lines.append(
        ctx.target.loop_header(
            reduction_index,
            ctx.target.thread_id(),
            extent,
            ctx.target.thread_count(),
        )
    )
    ctx.lines.extend(_indent_lines(body_lines, ctx.target))
    ctx.lines.append("}")
    result_type = ssa.Type(kind="scalar", dtype=result_dtype)
    ctx.lines.append(
        ctx.target.local_decl(
            result_type, local, ctx.target.cast(result_dtype, accumulator)
        )
    )

    return local


def _ascendc_reduction_accumulator_dtype(operand_dtype: str, result_dtype: str) -> str:
    if operand_dtype in {"float16"}:
        return "float32"

    if operand_dtype == "bool":
        return "int32"
    return result_dtype


def _ascendc_reduction_update(operator: str, lhs: str, rhs: str, dtype: str) -> str:
    if operator == "sum":
        return f"({lhs}) + ({rhs})"

    if dtype == "float32":
        function = "max" if operator == "max" else "min"

        return f"{function}({lhs}, {rhs})"

    comparison = ">" if operator == "max" else "<"

    return f"(({lhs}) {comparison} ({rhs}) ? ({lhs}) : ({rhs}))"


def _emit_ascendc_direct_dot_operand(
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


_ASCENDC_MATH_SUPPORT = """// Scalar transcendental functions are not hardware built-ins on AscendC,
// so the generated kernels carry compact software implementations.  Device
// functions cannot recurse, so every routine is range-reduced inline.
union NtFloatBits {
    float f;
    int32_t i;
};

__aicore__ inline float nt_inf() {
    NtFloatBits v;
    v.i = 0x7f800000;
    return v.f;
}

__aicore__ inline float nt_exp(float x) {
    if (x > 88.72f) { return nt_inf(); }
    if (x < -103.97f) { return 0.0f; }
    float t = x * 1.442695041f;
    int k = (int)(t + (t > 0.0f ? 0.5f : -0.5f));
    float r = (float)k;
    x = x - r * 0.693147181f;
    // 10th-order minimax polynomial for exp on [-ln2/2, ln2/2]
    float p = 1.0f + x * (1.0f + x * (0.5000000000f + x * (0.1666666667f
        + x * (0.04166666667f + x * (0.008333333333f + x * (0.001388888889f
        + x * (0.0001984126984f + x * (0.00002480158730f
        + x * (0.000002755731922f + x * 0.0000002755731922f)))))))));
    // Scale in two steps: a single 2^k exponent field overflows into the
    // sign bit for subnormal results (k + 127 <= 0) and produces a huge
    // negative value instead.  Halving the exponent keeps both scale
    // factors in range, and IEEE multiplication rounds gradually.
    int k1 = k / 2;
    int k2 = k - k1;
    NtFloatBits s1;
    NtFloatBits s2;
    s1.i = (k1 + 127) << 23;
    s2.i = (k2 + 127) << 23;
    return (p * s1.f) * s2.f;
}

__aicore__ inline float nt_log(float x) {
    if (x <= 0.0f) { return -nt_inf(); }
    NtFloatBits v;
    v.f = x;
    int biased = (v.i >> 23) & 0xff;
    float m;
    int e;
    if (biased == 0) {
        // Subnormal inputs carry no implicit bit: x = M * 2^-149.
        e = -126;
        m = (float)(v.i & 0x007fffff) * (1.0f / 8388608.0f);
    }
    else {
        e = biased - 127;
        v.i = (v.i & 0x007fffff) | 0x3f800000;
        m = v.f;
    }
    // Normalize m into [0.75, 1.5] so |u| <= 0.2 below.
    while (m > 1.5f) { m = m * 0.5f; e = e + 1; }
    while (m < 0.75f) { m = m * 2.0f; e = e - 1; }
    // log(m) = 2*artanh(u) with u = (m-1)/(m+1): truncation ~ u^11/11.
    float u = (m - 1.0f) / (m + 1.0f);
    float u2 = u * u;
    float p = u * (1.0f + u2 * (0.33333334f + u2 * (0.2f + u2 * (0.14285715f
        + u2 * 0.11111111f))));
    return e * 0.69314718f + 2.0f * p;
}

__aicore__ inline float nt_tanh(float x) {
    // exp(2x) saturates around x ~ 44; (inf - 1) / (inf + 1) would be NaN.
    if (x > 22.0f) { return 1.0f; }
    if (x < -22.0f) { return -1.0f; }
    float e2 = nt_exp(2.0f * x);
    return (e2 - 1.0f) / (e2 + 1.0f);
}

__aicore__ inline float nt_sigmoid(float x) {
    return 1.0f / (1.0f + nt_exp(-x));
}

__aicore__ inline float nt_sin(float x) {
    // Integer modular range reduction: a float32 argument carries a 24-bit
    // mantissa, so x * 2^32 is exact in int64 for |x| < 2^30, and the
    // Q32 remainder plus the k * delta correction keeps arguments as large
    // as 1e9 accurate to ~5e-8 radians.
    const int64_t NT_TWO_PI_Q32 = 26986075419LL;
    const float NT_DELTA = 0.044036865234375f;
    bool negative = x < 0.0f;
    float ax = negative ? -x : x;
    float s;

    // The Q32 reduction is exact while x * 2^32 stays inside the aicore's
    // reliable 48-bit integer range; beyond that the fallback keeps the
    // quadrant fold, whose accuracy degrades with |x|.
    if (ax < 16384.0f) {
        // Float-to-int64 conversions beyond 2^31 and int64-to-float
        // round-trips past 2^31 both miscompile on the aicore scalar path,
        // so the scaling goes through int32 halves.
        int64_t xi = (int64_t)(int32_t)(ax * 65536.0f);
        xi = xi << 16;
        int64_t k = xi / NT_TWO_PI_Q32;
        int64_t r = xi - k * NT_TWO_PI_Q32;
        float rf = (float)(int32_t)(r >> 16) * 65536.0f
            + (float)(int32_t)(r & 0xFFFF);
        s = (rf - (float)(int32_t)k * NT_DELTA) * (1.0f / 4294967296.0f);
    }
    else {
        s = ax - (float)((int)(ax * 0.15915494f)) * 6.2831853f;
    }
    // Nearest-quadrant fold leaves |t| <= pi/4 for both series.
    int q = (int)(s * 0.63661975f + 0.5f);
    float t = s - (float)q * 1.57079637f;
    float t2 = t * t;
    float sin_t = t * (1.0f - t2 * (0.16666667f - t2 * (0.0083333310f
        - t2 * (1.9841270e-4f - t2 * 2.7557314e-6f))));
    float cos_t = 1.0f - t2 * (0.50000000f - t2 * (0.041666638f
        - t2 * (0.0013888378f - t2 * 2.4433157e-5f)));
    float result = (q & 1) ? cos_t : sin_t;

    if (q & 2) { result = -result; }

    return negative ? -result : result;
}

__aicore__ inline float nt_cos(float x) {
    return nt_sin(x + 1.5707964f);
}

__aicore__ inline float nt_tan(float x) {
    return nt_sin(x) / nt_cos(x);
}

__aicore__ inline float nt_sinh(float x) {
    float e = nt_exp(x);
    return 0.5f * (e - 1.0f / e);
}

__aicore__ inline float nt_cosh(float x) {
    float e = nt_exp(x);
    return 0.5f * (e + 1.0f / e);
}

// atanh-free, recursion-free atan: fold x > 1 through the reciprocal
// identity atan(x) = pi/2 - atan(1/x), then arguments above 0.35 through
// the additive sqrt(3) identity, so the odd Taylor polynomial only sees
// |v| <= 2 - sqrt(3) ~ 0.268.
__aicore__ inline float nt_atan_poly(float x) {
    float v2 = x * x;
    return x * (1.0f - v2 * (0.33333334f - v2 * (0.2f - v2 * (0.14285715f
        - v2 * (0.11111111f - v2 * 0.09090909f)))));
}

__aicore__ inline float nt_atan_pos(float x) {
    const float NT_HALF_PI = 1.5707964f;
    const float NT_PI_6 = 0.5235988f;
    const float NT_SQRT3 = 1.7320508f;

    if (x > 1.0f) {
        float v = 1.0f / x;
        float inner = nt_atan_poly(v);

        if (v > 0.35f) {
            inner = nt_atan_poly((NT_SQRT3 * v - 1.0f) / (NT_SQRT3 + v))
                + NT_PI_6;
        }
        return NT_HALF_PI - inner;
    }

    if (x > 0.35f) {
        return nt_atan_poly((NT_SQRT3 * x - 1.0f) / (NT_SQRT3 + x)) + NT_PI_6;
    }

    return nt_atan_poly(x);
}

__aicore__ inline float nt_atan(float x) {
    if (x < 0.0f) { return -nt_atan_pos(-x); }
    return nt_atan_pos(x);
}

__aicore__ inline float nt_asin(float x) {
    float r = 1.0f - x * x;
    float root = r >= 0.0f ? sqrt(r) : 0.0f;
    float v = x >= 0.0f ? (root > 0.0f ? x / root : nt_inf()) : -1.0f;
    if (x < 0.0f) {
        v = root > 0.0f ? x / root : -nt_inf();
    }
    return nt_atan(v);
}

__aicore__ inline float nt_acos(float x) {
    return 1.5707964f - nt_asin(x);
}

__aicore__ inline float nt_atan2(float y, float x) {
    const float NT_PI_F = 3.14159265f;
    if (x > 0.0f) { return nt_atan(y / x); }
    if (x < 0.0f && y >= 0.0f) { return nt_atan(y / x) + NT_PI_F; }
    if (x < 0.0f) { return nt_atan(y / x) - NT_PI_F; }
    if (y > 0.0f) { return 1.5707964f; }
    if (y < 0.0f) { return -1.5707964f; }
    return 0.0f;
}

__aicore__ inline float nt_floor(float x) {
    float r = (float)(int)x;
    if (x < 0.0f && x != r) { r -= 1.0f; }
    return r;
}

__aicore__ inline float nt_ceil(float x) {
    float r = (float)(int)x;
    if (x > 0.0f && x != r) { r += 1.0f; }
    return r;
}

__aicore__ inline float nt_nan() {
    NtFloatBits v;
    v.i = 0x7fc00000;
    return v.f;
}

__aicore__ inline float nt_pow(float x, float y) {
    if (y == 0.0f) { return 1.0f; }
    if (x == 0.0f) { return (y > 0.0f) ? 0.0f : nt_inf(); }
    if (x < 0.0f && y != nt_floor(y)) {
        // Negative base with a non-integer exponent is undefined.
        return nt_nan();
    }
    NtFloatBits v;
    v.f = (x < 0.0f) ? -x : x;
    int biased = (v.i >> 23) & 0xff;
    float m;
    int e;
    if (biased == 0) {
        // Subnormal bases carry no implicit bit: x = M * 2^-149.
        e = -126;
        m = (float)(v.i & 0x007fffff) * (1.0f / 8388608.0f);
    }
    else {
        e = biased - 127;
        v.i = (v.i & 0x007fffff) | 0x3f800000;
        m = v.f;
    }
    float lg = e * 0.69314718f + nt_log(m);
    float r = nt_exp(((x < 0.0f) ? nt_floor(y) : y) * lg);
    if (x < 0.0f) {
        float yi = nt_floor(y);
        if (y != yi) { return 0.0f; }
        return (((int)yi) & 1) ? -r : r;
    }
    return r;
}

__aicore__ inline float nt_fabs(float x) {
    NtFloatBits v;
    v.f = x;
    v.i &= 0x7fffffff;
    return v.f;
}

__aicore__ inline float nt_rsqrt(float x) {
    return 1.0f / sqrt(x);
}

__aicore__ inline float nt_exp2(float x) {
    return nt_exp(x * 0.69314718f);
}

__aicore__ inline float nt_log2(float x) {
    return nt_log(x) * 1.44269504f;
}

__aicore__ inline float nt_log10(float x) {
    return nt_log(x) * 0.43429448f;
}

__aicore__ inline float nt_log1p(float x) {
    return nt_log(1.0f + x);
}

__aicore__ inline float nt_expm1(float x) {
    return nt_exp(x) - 1.0f;
}

__aicore__ inline float nt_erf(float x) {
    float s = x < 0.0f ? -1.0f : 1.0f;
    x = x * s;
    float t = 1.0f / (1.0f + 0.3275911f * x);
    float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f)
        * t - 0.284496736f) * t + 0.254829592f) * t * nt_exp(-x * x);
    return s * y;
}"""


# Bare function names accepted by `call()`; names mapping to `nt_*` helpers
# require the software math library below to be emitted.
_ASCENDC_CALL_FUNCS = {
    "abs": "nt_fabs",
    "acos": "nt_acos",
    "asin": "nt_asin",
    "atan": "nt_atan",
    "atan2": "nt_atan2",
    "ceil": "nt_ceil",
    "cos": "nt_cos",
    "cosh": "nt_cosh",
    "erf": "nt_erf",
    "exp": "nt_exp",
    "exp2": "nt_exp2",
    "expm1": "nt_expm1",
    "floor": "nt_floor",
    "log": "nt_log",
    "log1p": "nt_log1p",
    "log2": "nt_log2",
    "log10": "nt_log10",
    "max": "max",
    "maximum": "max",
    "min": "min",
    "minimum": "min",
    "pow": "nt_pow",
    "rsqrt": "nt_rsqrt",
    "sin": "nt_sin",
    "sinh": "nt_sinh",
    "tan": "nt_tan",
    "tanh": "nt_tanh",
}

_MATH_OPCODES = {
    "math.exp": "nt_exp",
    "math.log": "nt_log",
    "math.tanh": "nt_tanh",
    "math.sigmoid": "nt_sigmoid",
    "math.sin": "nt_sin",
    "math.cos": "nt_cos",
    "math.tan": "nt_tan",
    "math.sinh": "nt_sinh",
    "math.cosh": "nt_cosh",
    "math.asin": "nt_asin",
    "math.acos": "nt_acos",
    "math.atan": "nt_atan",
    "math.atan2": "nt_atan2",
    "math.pow": "nt_pow",
    "math.erf": "nt_erf",
    "math.abs": "nt_fabs",
    "math.floor": "nt_floor",
    "math.ceil": "nt_ceil",
    "math.rsqrt": "nt_rsqrt",
    "math.exp2": "nt_exp2",
    "math.log2": "nt_log2",
    "math.log10": "nt_log10",
    "math.log1p": "nt_log1p",
    "math.expm1": "nt_expm1",
}


def _ascendc_validate_reduction_lowering(context: ModuleRenderContext) -> None:
    """Reject reductions that cannot lower through the cooperative schedule.

    The generic flat-index path decomposes store addresses with the domain's
    value axes, so a reduction whose store target drops the reduction axis
    computes out-of-domain output addresses and silently loses updates
    across blocks.  Until the shared reduction domain supports collapsing
    stores, AscendC only admits reductions lowered through the cooperative
    one-block-per-slice schedule (which requires one tile spanning the full
    reduction axis).
    """
    program = context.kernel.ssa
    schedule = dict(program.metadata.get("schedule", {})) if program else {}
    reduction = schedule.get("reduction", {})

    if not isinstance(reduction, Mapping):
        return

    mode = reduction.get("mode")

    if mode == "none":
        return

    if context.cooperative_reduction_program:
        # A store target with fewer dimensions than the reduction domain
        # still navigates through collapsed addressing and computes
        # out-of-domain output addresses.
        value_shape = tuple(reduction.get("value_shape", ()))
        outputs = tuple(context.outputs)

        if value_shape and outputs:
            info = context.tensors.get(outputs[0])

            if info is not None and info.ndim != len(value_shape):
                raise ValueError(
                    "The AscendC backend requires reduction store targets "
                    "to keep the dimensionality of the reduction domain."
                )

        return

    program_ops = context.kernel.ssa.blocks if program else ()
    has_tensor_stores = any(
        operation.opcode == "mem.store" and len(operation.operands) == 2
        for block in program_ops
        for operation in _walk_ops(block.operations)
    )

    if not has_tensor_stores:
        return

    from ninetoothed.backends.emitters.ascendc_reduce import (
        match_row_reduce as _match_row,
    )

    if _match_row(context) is not None:
        # Collapsed (rows, 1) stores lower through the per-row kernel
        # instead of the generic flat-index path.
        return

    raise ValueError(
        "The AscendC backend only supports reductions lowered through the "
        "cooperative row-vector schedule; tile the reduction axis so one "
        "tile spans its full extent (e.g. BLOCK >= axis length)."
    )


def _elementwise_total_param(context: ModuleRenderContext) -> str | None:
    """Return the shared extent expression of a 1-D elementwise kernel."""
    extents = set()

    for name in (*context.variables, *context.outputs):
        info = context.tensors.get(name)

        if info is None or info.ndim != 1:
            continue

        attrs = info.attrs or {}
        shape = attrs.get("source_shape") or info.shape

        if not shape:
            return None

        extents.add(str(shape[0]))

    if not extents:
        return None

    if len(extents) == 1:
        return extents.pop()

    # Unsized specializations name each tensor's size parameter
    # separately, but a 1-D elementwise kernel launches with equal
    # lengths, so distinct size parameters are one shared extent.
    if len(extents) <= 3 and all(re.fullmatch(r"\w+", expr) for expr in extents):
        return sorted(extents)[0]

    return None


def _ascendc_validate_atomics(context: ModuleRenderContext) -> None:
    """Reject mem.atomic_add: scalar atomics are unavailable on AscendC.

    The CANN runtime may execute one logical block on several cores without
    distinct block indices, so neither a serialized single block nor a
    block-indexed partition can guarantee exactly-once read-modify-write
    updates.  Fail at compile time instead of silently corrupting results.
    """
    program = context.kernel.ssa

    if program is None:
        return

    uses_atomics = any(
        operation.opcode == "mem.atomic_add"
        for block in program.blocks
        for operation in _walk_ops(block.operations)
    )

    if uses_atomics:
        raise ValueError(
            "The AscendC backend does not support atomic updates: scalar "
            "atomics are unavailable and block execution is not guaranteed "
            "to be exclusive, so read-modify-write updates cannot be made "
            "exact."
        )


def _ascendc_support(context: ModuleRenderContext, *, force_math: bool = False) -> str:
    """Emit helper functions required by the kernel body."""
    operations = {id(operation): operation for operation in context.operations.values()}
    lines: list[str] = []

    needed_math = sorted(
        {
            _MATH_OPCODES[operation.opcode]
            for operation in operations.values()
            if operation.opcode in _MATH_OPCODES
        }
    )
    derived = set()

    for name in needed_math:
        if name == "nt_log":
            derived.add("nt_log")
        elif name in {"nt_pow"}:
            derived.update({"nt_exp", "nt_log"})
        elif name in {"nt_tanh", "nt_sigmoid"}:
            derived.add("nt_exp")
        elif name in {"nt_sin"}:
            pass
        elif name == "nt_cos":
            derived.add("nt_sin")
        elif name == "nt_tan":
            derived.update({"nt_sin", "nt_cos"})
        elif name in {"nt_sinh", "nt_cosh"}:
            derived.add("nt_exp")
        elif name in {"nt_asin", "nt_acos"}:
            derived.update({"nt_atan"})
        elif name == "nt_atan2":
            derived.add("nt_atan")
        elif name == "nt_erf":
            derived.add("nt_exp")

    uses_math = force_math or any(
        operation.opcode in _MATH_OPCODES for operation in operations.values()
    )

    # An `x ** y` expression lowers to `arith.pow` -> `nt_pow`, and
    # namespace calls emit `call.{name}` ops; both reference software math
    # helpers without any `math.*` opcode being present.
    nt_funcs = {
        value for value in _ASCENDC_CALL_FUNCS.values() if value.startswith("nt_")
    }
    needed_calls = {
        _ASCENDC_CALL_FUNCS.get(operation.opcode[len("call.") :])
        for operation in operations.values()
        if operation.opcode.startswith("call.")
    }
    needed_opcodes = {operation.opcode for operation in operations.values()} & {
        "arith.pow"
    }
    uses_math = uses_math or bool(needed_calls & nt_funcs) or bool(needed_opcodes)

    if uses_math:
        lines.append(_ASCENDC_MATH_SUPPORT)
    else:
        # Nt_inf() is referenced by tile `other` defaults and reduction
        # identities even in math-free kernels.
        lines.append(
            """union NtFloatBits {
    float f;
    int32_t i;
};

__aicore__ inline float nt_inf() {
    NtFloatBits v;
    v.i = 0x7f800000;
    return v.f;
}"""
        )

    if any(operation.opcode == "math.rand" for operation in operations.values()):
        lines.append(
            """\
// 64-bit immediates cannot be loaded on the scalar path, so the generator
// is kept entirely in uint32 arithmetic.
__aicore__ inline float ninetoothed_rand_uniform(
    uint64_t seed, uint64_t offset
) {
    uint32_t z = (uint32_t)(seed) ^ ((uint32_t)(offset) * 0x9E3779B1u);
    z = z + 0x6D2B79F5u;
    z = (z ^ (z >> 15)) * 0x85EBCA6Bu;
    z = (z ^ (z >> 13)) * 0xC2B2AE35u;
    z = z ^ (z >> 16);
    // aicore disallows direct uint-to-float casts; the shift guarantees
    // the value fits in a positive int32 so the signed cast is exact.
    return (float)(int32_t)(z >> 8) * (1.0f / 16777216.0f);
}"""
        )

    return "\n\n".join(lines)


def _render_signature_params(params: list[str]) -> str:
    return ",\n".join(f"    {param}" for param in params)


def _signature_param(name, info) -> str:
    dtype = _ascendc_type(info.dtype)

    if info.ndim == 0:
        return f"{dtype} {name}"

    return f"GM_ADDR {name}_gm"


def _launch_param(name, info) -> str:
    if info.ndim == 0:
        return f"{_ascendc_type(info.dtype)} {name}"

    return f"void* {name}"


def _auxiliary_param(name, tensors) -> str:
    info = tensors.get(name)

    if info is not None and info.ndim == 0:
        return _signature_param(name, info)
    return f"int64_t {name}"


def _ascendc_type(dtype: str | None, kind: str | None = None) -> str:
    dtype = common.normalize_dtype(dtype)

    if kind == "pointer":
        return f"{_ascendc_type(dtype)}*"

    if kind == "index" or dtype in {"index", "int64"}:
        return "int64_t"

    types = {
        "float32": "float",
        "float16": "half",
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

    if dtype in {"bfloat16", "float8_e4m3fn", "float8_e5m2"}:
        raise ValueError(
            f"The AscendC backend does not support `{dtype}` in the scalar "
            "lowering yet."
        )

    if dtype not in types:
        raise ValueError(f"Unsupported AscendC SSA dtype: {dtype!r}.")
    return types[dtype]


def _ascendc_integer_expr(expr: str) -> str:
    previous = None
    current = common.rewrite_index_math(expr, c_style=True).replace("//", "/")
    current = re.sub(r"\bTrue\b", "true", current)
    current = re.sub(r"\bFalse\b", "false", current)
    pattern = re.compile(r"floor\(\(([^()]+)\)/([A-Za-z_][A-Za-z0-9_]*)\)")

    while current != previous:
        previous = current
        current = pattern.sub(r"((\1)/(\2))", current)
    return current


def _ascendc_arithmetic_result_type(op: ssa.Operation, ctx: _EmitContext) -> ssa.Type:
    result_type = op.results[0].type

    if not op.opcode.startswith("arith."):
        return result_type

    if _normalize_dtype(result_type.dtype) not in {"float16"}:
        return result_type
    return replace(result_type, dtype="float32")


def _coerce_ascendc_binary_args(
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
        else _ascendc_common_operand_dtype(op.operands, ctx)
    )

    if dtype is None:
        return args

    if dtype == "float16":
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


def _ascendc_common_operand_dtype(
    operands: tuple[str, ...], ctx: _EmitContext
) -> str | None:
    ranks = {
        "bool": 0,
        "int32": 1,
        "int64": 2,
        "float16": 3,
        "float32": 4,
        "float64": 5,
    }
    dtypes = [
        _normalize_dtype(type_.dtype)
        for operand in operands
        if (type_ := ctx.value_types.get(operand)) is not None
    ]

    return max(dtypes, key=lambda dtype: ranks.get(dtype, -1)) if dtypes else None


TARGET = AscendCTarget()


def emit(kernel: Kernel):
    return common.emit(kernel, TARGET)


__all__ = ["AscendCTarget", "TARGET", "emit"]
