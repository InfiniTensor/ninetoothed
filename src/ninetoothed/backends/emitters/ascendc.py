"""AscendC scalar lowering through the shared typed SSA emitter."""

import math
import re
from dataclasses import dataclass, replace
from typing import Any

from ninetoothed.backends.core import Target
from ninetoothed.backends.emitters import ssa as common
from ninetoothed.backends.emitters.analysis import walk_ops as _walk_ops
from ninetoothed.backends.emitters.base import EmitterTarget, ModuleRenderContext
from ninetoothed.backends.emitters.c_types import (
    CScalarArithmeticMixin,
    render_c_type_support,
    render_int32_cast_support,
)
from ninetoothed.backends.emitters.context import EmitContext as _EmitContext
from ninetoothed.backends.emitters.expressions import c_integer_expr
from ninetoothed.backends.emitters.math_helpers import render_float32_trig_support
from ninetoothed.backends.emitters.task_launch import allocate_private_symbol
from ninetoothed.ir import Kernel, ssa

_normalize_dtype = common.normalize_dtype


@dataclass(frozen=True, kw_only=True)
class AscendCTarget(CScalarArithmeticMixin, EmitterTarget):
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

        if value == "nan" or (isinstance(value, float) and math.isnan(value)):
            return f"{self.global_helper('nt_nan')}()"

        if isinstance(value, float) and math.isinf(value):
            function = self.global_helper("nt_inf")

            return f"{function}()" if value > 0 else f"-{function}()"

        if value == "inf":
            return f"{self.global_helper('nt_inf')}()"

        if value == "-inf":
            return f"-{self.global_helper('nt_inf')}()"

        if isinstance(value, float):
            # Ccec treats unsuffixed float literals as double, which is not
            # supported in aicore functions.
            return f"{value}f"

        if isinstance(value, int):
            if value == -(1 << 63):
                # A bare decimal 2^63 has unsigned type before unary minus.
                # In a masked term it would promote signed comparisons.
                return "(-9223372036854775807LL - 1LL)"

            return repr(value)

        return repr(value)

    def cast(self, dtype, value):
        normalized = _normalize_dtype(dtype)

        if normalized == "int32":
            return f"{self.global_helper('nt_cast_int32')}({value})"

        if normalized == "float16":
            return f"((half)(float)({value}))"

        if normalized == "bfloat16":
            return f"NtBFloat16((float)({value}))"

        if normalized in {"uint32", "uint64"}:
            return f"(({self.type_name(dtype)})(int64_t)({value}))"

        if normalized == "float32":
            return f"{self.global_helper('nt_to_float')}({value})"

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
            return f"{self.global_helper('ninetoothed_rand_uniform')}(({args[0]}), ({args[1]}))"

        if name == "sqrt" and len(args) == 1:
            return f"sqrt(({args[0]}))"

        function = _ASCENDC_CALL_FUNCS.get(name, name)

        if name in _ASCENDC_CALL_FUNCS and function.startswith("nt_"):
            function = self.global_helper(function)

        return f"{function}({', '.join(args)})"

    def program_id(self, axis=0):
        if axis != 0:
            raise ValueError("AscendC kernels use a one-dimensional block grid.")
        return "(int64_t)GetBlockIdx()"

    def loop_header(self, var, lower, upper, step):
        return f"for (int64_t {var} = {lower}; {var} < {upper}; {var} += {step}) {{"

    def reduce_update(self, operator, acc, term):
        if operator == "sum":
            return f"{acc} + {term}"

        function = "max" if operator == "max" else "min"

        return f"{function}({acc}, {term})"

    def reduce_update_typed(self, operator, acc, term, dtype):
        if dtype is None:
            return super().reduce_update_typed(operator, acc, term, dtype)

        term = self.cast(dtype, term)

        return _ascendc_reduction_update(
            operator,
            acc,
            term,
            _ascendc_reduction_accumulator_dtype(dtype, dtype),
            target=self,
        )

    def reduction_compute_type(self, result_type):
        if result_type.dtype in {"float16", "bfloat16"}:
            return replace(result_type, dtype="float32")
        return result_type

    def reduction_result(self, value, result_type):
        if result_type.dtype in {"float16", "bfloat16"}:
            return self.cast(result_type.dtype, value)
        return value

    def arithmetic_result_type(self, operation, context) -> ssa.Type:
        if operation.opcode in {"call.isnan", "call.isfinite"}:
            return replace(operation.results[0].type, dtype="bool")
        return _ascendc_arithmetic_result_type(operation, context)

    def store(self, tensor, index, value, *, mask=None):
        statement = f"nt_scalar_store({self.tensor_ref(tensor)}, {index}, {value});"

        return statement if mask is None else f"if ({mask}) {{\n    {statement}\n}}"

    def load(self, tensor, index, *, mask=None, other=0.0):
        return f"nt_scalar_load({self.tensor_ref(tensor)}, {index})"

    def coerce_binary_args(self, operation, args, context):
        return _coerce_ascendc_binary_args(operation, args, context)

    def schedule_context(self, context):
        from ninetoothed.backends.emitters.memory_guard import (
            dense_staging_memory_guard,
        )
        from ninetoothed.backends.emitters.scalar_tiles import plan_scalar_tiles

        arena = []
        arena_bytes = []

        def storage(slots):
            declarations, size = _ascendc_scalar_tile_storage(slots)
            arena.extend(declarations)
            arena_bytes.append(size)

            return ()

        try:
            plan = plan_scalar_tiles(
                context,
                self,
                storage=storage,
                budget=184 * 1024,
                inline_elementwise=True,
            )
        except ValueError as error:
            context.lowering_plans["scalar_tile_rejection"] = str(error)
            plan = None

        if plan is not None:
            guard = c_integer_expr(
                dense_staging_memory_guard(
                    context, context.variables, context.outputs, dense_inputs=False
                )
            )

            if guard != "false":
                context.lowering_plans["scalar_tiles"] = (plan, guard, tuple(arena))
                context.scheduled_metadata["ascendc_scalar_tile_storage"] = {
                    "bytes": arena_bytes[0],
                    "estimated_storage_bytes": plan.scratch_bytes,
                    "logical_tasks": plan.tasks,
                    "budget": 184 * 1024,
                    "nonoverlap_guard": guard,
                }

        context.scheduled_metadata["ascendc_execution"] = {
            "physical_blocks": 1,
            "scalar_executor": "block-zero/subblock-zero",
            "parallel_scalar_stores": False,
        }

        return context

    def render_module(self, context: ModuleRenderContext) -> str:
        body = c_integer_expr(context.body)
        total = c_integer_expr(context.total)
        names = (*context.variables, *context.outputs, *context.shape_params)
        gm = {}

        for name in (*context.variables, *context.outputs):
            if context.tensors[name].ndim:
                gm[name] = allocate_private_symbol(
                    f"nt_gm_{name}", body, *names, *gm.values()
                )

        half_storage = {}

        for name in names:
            info = context.tensors.get(name)

            if (
                info is not None
                and info.ndim == 0
                and _normalize_dtype(info.dtype) == "float16"
            ):
                half_storage[name] = allocate_private_symbol(
                    f"nt_half_bits_{name}",
                    body,
                    *names,
                    *gm.values(),
                    *half_storage.values(),
                )

        parameters = [
            f"GM_ADDR {gm[name]}"
            if name in gm
            else _auxiliary_param(name, context.tensors)
            for name in names
        ]
        host_parameters = [
            _host_auxiliary_param(name, context.tensors, half_storage) for name in names
        ]
        stream = allocate_private_symbol(
            "nt_stream", body, *parameters, *host_parameters
        )
        host_parameters.append(f"aclrtStream {stream}")

        if half_storage:
            context.scheduled_metadata["ascendc_host_scalar_abi"] = {
                "version": "uint16-half-v1",
                "half_storage_bindings": tuple(half_storage),
            }

        decode = "\n".join(
            f"    half {name};\n    ::memcpy(&{name}, &{bits}, sizeof({name}));"
            for name, bits in half_storage.items()
        )
        casts = "\n".join(
            f"    __gm__ {self.type_name(context.tensors[name].dtype)}* {name} = (__gm__ {self.type_name(context.tensors[name].dtype)}*){pointer};"
            for name, pointer in gm.items()
        )
        fallback = f"for (int64_t {self.index_name}=0; {self.index_name}<({total}); ++{self.index_name}) {{\n{common.indent_block(body, '    ')}\n}}"
        selected = context.lowering_plans.get("scalar_tiles")
        task_body = fallback

        if selected is not None:
            plan, guard, arena = selected
            logical = allocate_private_symbol("nt_scalar_task", body, *names)
            tiled = plan.body.replace("GetBlockIdx()", logical)
            task_body = (
                f"if ({guard}) {{\n"
                + common.indent_block("\n".join(arena), "    ")
                + f"\n    for (int64_t {logical}=0; {logical}<({plan.tasks}); ++{logical}) {{\n"
                + common.indent_block(tiled, "        ")
                + "\n    }\n} else {\n"
                + common.indent_block(fallback, "    ")
                + "\n}"
            )

        support = "\n\n".join(
            (
                _ascendc_support(context),
                render_int32_cast_support("__aicore__ inline"),
                render_c_type_support(task_body, "__aicore__ inline"),
            )
        )
        arguments = [
            f"(GM_ADDR)(uintptr_t){name}" if name in gm else name for name in names
        ]
        kernel_name = context.kernel.kernel_name

        return f"""// Generated by NineToothed's AscendC SSA backend.
#include "kernel_operator.h"
#include "acl/acl.h"
#include <type_traits>
#include <limits>
#include <string.h>
using namespace AscendC;
{support}
extern "C" __global__ __aicore__ void {kernel_name}_kernel(
{_render_signature_params(parameters)}
) {{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#ifdef __DAV_C220_CUBE__
    return;
#endif
    if (GetBlockIdx()!=0 || GetSubBlockIdx()!=0) {{ return; }}
{casts}
{common.indent_block(task_body, "    ")}
}}
extern "C" int launch_{kernel_name}(
{_render_signature_params(host_parameters)}
) {{
{decode}
    {kernel_name}_kernel<<<1, nullptr, {stream}>>>({", ".join(arguments)});
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


def _ascendc_reduction_accumulator_dtype(operand_dtype: str, result_dtype: str) -> str:
    if operand_dtype in {"float16"}:
        return "float32"

    if operand_dtype == "bool":
        return "int32"
    return result_dtype


def _ascendc_reduction_update(
    operator: str, lhs: str, rhs: str, dtype: str, *, target=None
) -> str:
    del dtype

    if operator == "sum":
        return f"({lhs}) + ({rhs})"

    # Computed dtypes are C++ type text, so Python string tests cannot decide
    # floating versus integral. Explicitly keep the accumulator's value type
    # while the shared template propagates either NaN through the reduction.
    value_dtype = f"std::remove_cv<std::remove_reference<decltype({lhs})>::type>::type"

    function = f"_nt_reduce_{operator}"

    if target is not None:
        function = target.global_helper(function)
    return f"{function}<{value_dtype}>({lhs}, {rhs})"


def _ascendc_scalar_tile_storage(slots):
    offsets = []
    size = 0

    for dtype, count, name, width in slots:
        size = (size + 31) // 32 * 32
        offsets.append((dtype, name, size))
        size += count * width

    size = (size + 31) // 32 * 32

    if size > 192 * 1024 - 256:
        raise ValueError("Aligned scalar tile storage exceeds the UB budget.")
    return [
        "TPipe nt_tile_pipe;",
        "TBuf<QuePosition::VECCALC> nt_tile_arena;",
        f"nt_tile_pipe.InitBuffer(nt_tile_arena, {size});",
        "auto nt_tile_storage = nt_tile_arena.Get<uint8_t>();",
        "__ubuf__ uint8_t *nt_tile_base = (__ubuf__ uint8_t *)nt_tile_storage.GetPhyAddr();",
        *(
            f"__ubuf__ {dtype} *{name} = (__ubuf__ {dtype} *)(nt_tile_base + {offset});"
            for dtype, name, offset in offsets
        ),
    ], size


_ASCENDC_MATH_SUPPORT = (
    """// Scalar transcendental functions are not hardware built-ins on AscendC,
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

__aicore__ inline float nt_nan() {
    NtFloatBits v;
    v.i = 0x7fc00000;
    return v.f;
}

__aicore__ inline float nt_exp(float x) {
    if (x != x) { return x; }
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
    if (x != x || x < 0.0f) { return nt_nan(); }
    if (x == 0.0f) { return -nt_inf(); }
    if (x == nt_inf()) { return x; }
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
    if (x == 0.0f) { return x; }
    // exp(2x) saturates around x ~ 44; (inf - 1) / (inf + 1) would be NaN.
    if (x > 22.0f) { return 1.0f; }
    if (x < -22.0f) { return -1.0f; }
    float e2 = nt_exp(2.0f * x);
    return (e2 - 1.0f) / (e2 + 1.0f);
}

__aicore__ inline float nt_sigmoid(float x) {
    return 1.0f / (1.0f + nt_exp(-x));
}

"""
    + render_float32_trig_support("__aicore__")
    + """

__aicore__ inline float nt_tan(float x) {
    return nt_sin(x) / nt_cos(x);
}

__aicore__ inline float nt_sinh(float x) {
    if (x == 0.0f) { return x; }
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
    if (x != x || x > 1.0f || x < -1.0f) { return nt_nan(); }
    if (x == 0.0f) { return x; }
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
    const float NT_HALF_PI = 1.5707964f;
    if (x != x || y != y) { return nt_nan(); }
    NtFloatBits bx;
    NtFloatBits by;
    bx.f = x;
    by.f = y;
    bool negative_x = bx.i < 0;
    bool negative_y = by.i < 0;
    if (y == 0.0f) {
        return negative_x ? (negative_y ? -NT_PI_F : NT_PI_F) : y;
    }
    if (x == 0.0f) { return negative_y ? -NT_HALF_PI : NT_HALF_PI; }
    bx.i &= 0x7fffffff;
    by.i &= 0x7fffffff;
    float angle;
    if (by.f == nt_inf()) {
        angle = bx.f == nt_inf()
            ? (negative_x ? 2.35619449f : 0.785398163f) : NT_HALF_PI;
    }
    else if (bx.f == nt_inf()) {
        angle = negative_x ? NT_PI_F : 0.0f;
    }
    else {
        angle = nt_atan_pos(by.f / bx.f);
        if (negative_x) { angle = NT_PI_F - angle; }
    }
    return negative_y ? -angle : angle;
}

__aicore__ inline float nt_floor(float x) {
    NtFloatBits v;
    v.f = x;
    int exponent = ((v.i >> 23) & 0xff) - 127;
    // Large finite float32 values already have no fractional bits.
    if (exponent >= 23 || (v.i & 0x7fffffff) == 0) { return x; }
    if (exponent < 0) { return x < 0.0f ? -1.0f : 0.0f; }
    int32_t fraction = (1 << (23 - exponent)) - 1;
    if ((v.i & fraction) == 0) { return x; }
    v.i &= ~fraction;
    return x < 0.0f ? v.f - 1.0f : v.f;
}

__aicore__ inline float nt_ceil(float x) {
    return -nt_floor(-x);
}

__aicore__ inline float nt_pow(float x, float y) {
    if (y == 0.0f) { return 1.0f; }
    if (x == 1.0f) { return 1.0f; }
    if (x != x || y != y) { return nt_nan(); }
    NtFloatBits base_bits;
    NtFloatBits exponent_bits;
    base_bits.f = x;
    exponent_bits.f = y;
    int exponent = ((exponent_bits.i >> 23) & 0xff) - 127;
    // Above 2^24 float32 exponents are all even; inspect the significand
    // without a large float-to-int conversion.
    bool odd = exponent >= 0 && exponent <= 23 && y == nt_floor(y)
        && ((((exponent_bits.i & 0x007fffff) | 0x00800000)
            >> (23 - exponent)) & 1);
    float ax = x < 0.0f ? -x : x;
    if (y == nt_inf() || y == -nt_inf()) {
        if (ax == 1.0f) { return 1.0f; }
        return ((ax > 1.0f) == (y > 0.0f)) ? nt_inf() : 0.0f;
    }
    if (ax == 0.0f || ax == nt_inf()) {
        float special = ((ax == 0.0f) == (y > 0.0f)) ? 0.0f : nt_inf();
        return base_bits.i < 0 && odd ? -special : special;
    }
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
        return odd ? -r : r;
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
    if (x == 0.0f) { return x; }
    float s = x < 0.0f ? -1.0f : 1.0f;
    x = x * s;
    float t = 1.0f / (1.0f + 0.3275911f * x);
    float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f)
        * t - 0.284496736f) * t + 0.254829592f) * t * nt_exp(-x * x);
    return s * y;
}"""
)

_ASCENDC_CALL_FUNCS = {
    "abs": "nt_fabs",
    "acos": "nt_acos",
    "asin": "nt_asin",
    "atan": "nt_atan",
    "atan2": "nt_atan2",
    "ceil": "nt_ceil",
    "copysign": "nt_copysign",
    "fmod": "nt_fmod",
    "isfinite": "nt_isfinite",
    "isnan": "nt_isnan",
    "nearbyint": "nt_nearbyint",
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
    "sigmoid": "nt_sigmoid",
    "sin": "nt_sin",
    "sinh": "nt_sinh",
    "tan": "nt_tan",
    "tanh": "nt_tanh",
}


def _ascendc_validate_atomics(kernel: Kernel) -> None:
    """Reject updates without a validated cross-kernel atomic primitive.

    Serial scalar writes inside one launch do not establish atomicity with
    concurrently executing kernels on other streams.
    """
    program = kernel.ssa

    if program is None:
        return

    uses_atomics = any(
        operation.opcode == "mem.atomic_add"
        for block in program.blocks
        for operation in _walk_ops(block.operations)
    )

    if uses_atomics:
        raise ValueError(
            "The AscendC backend does not support atomic updates: a validated "
            "device atomic primitive is required for concurrent launches."
        )


def _ascendc_support(context: ModuleRenderContext) -> str:
    """Emit helper functions required by the kernel body."""
    operations = {id(operation): operation for operation in context.operations.values()}
    lines: list[str] = [
        """struct NtBFloat16 {
    uint16_t bits;
    NtBFloat16() = default;
    __aicore__ inline NtBFloat16(float value) {
        union { float f; int32_t i; } word;
        word.f = value;
        if ((word.i & 0x7fffffff) > 0x7f800000) { bits = (uint16_t)((word.i >> 16) | 64); }
        else { bits = (uint16_t)(((uint32_t)word.i + 0x7fffu + ((word.i >> 16) & 1)) >> 16); }
    }
    __aicore__ inline operator float() const {
        union { float f; int32_t i; } word;
        word.i = (int32_t)bits << 16;
        return word.f;
    }
};
template <class T> struct _nt_value_cast;
template <class T> __aicore__ inline T nt_scalar_load(const __gm__ T *base, int64_t index) {
    return base[index];
}
__aicore__ inline NtBFloat16 nt_scalar_load(const __gm__ NtBFloat16 *base, int64_t index) {
    NtBFloat16 result; result.bits = ((const __gm__ uint16_t *)base)[index]; return result;
}
template <class T, class V> __aicore__ inline void nt_scalar_store(__gm__ T *base, int64_t index, V value) {
    base[index] = _nt_value_cast<T>::apply(value);
}
template <class V> __aicore__ inline void nt_scalar_store(__gm__ NtBFloat16 *base, int64_t index, V value) {
    NtBFloat16 result((float)value); ((__gm__ uint16_t *)base)[index] = result.bits;
}
template <class T> __aicore__ inline float nt_to_float(T value) {
    return static_cast<float>(value);
}
__aicore__ inline float nt_to_float(bool value) { return value ? 1.0f : 0.0f; }
__aicore__ inline float nt_to_float(uint32_t value) {
    return (float)(int64_t)value;
}
__aicore__ inline float nt_to_float(uint64_t value) {
    return (float)(int64_t)(value >> 32) * 4294967296.0f
         + (float)(int64_t)(uint32_t)value;
}"""
    ]

    lines.append(_ASCENDC_MATH_SUPPORT)

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


def _launch_param(name, info, *, half_storage=None) -> str:
    if info.ndim == 0:
        if half_storage is not None and name in half_storage:
            return f"uint16_t {half_storage[name]}"

        return f"{_ascendc_type(info.dtype)} {name}"

    return f"void* {name}"


def _host_auxiliary_param(name, tensors, half_storage) -> str:
    info = tensors.get(name)

    if info is not None and info.ndim == 0:
        return _launch_param(name, info, half_storage=half_storage)
    return f"int64_t {name}"


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
        "bfloat16": "NtBFloat16",
        "int8": "int8_t",
        "uint8": "uint8_t",
        "int16": "int16_t",
        "uint16": "uint16_t",
        "int32": "int32_t",
        "uint32": "uint32_t",
        "uint64": "uint64_t",
        "bool": "bool",
    }

    if dtype == "float64":
        raise ValueError(
            "The AscendC backend does not support `float64` device arithmetic "
            "or conversions on the dav-c220 target."
        )

    if dtype in {"float8_e4m3fn", "float8_e5m2"}:
        raise ValueError(
            f"The AscendC backend does not support `{dtype}` in the scalar "
            "lowering yet."
        )

    if dtype not in types:
        raise ValueError(f"Unsupported AscendC SSA dtype: {dtype!r}.")
    return types[dtype]


def _ascendc_arithmetic_result_type(op: ssa.Operation, ctx: _EmitContext) -> ssa.Type:
    return op.results[0].type


def _coerce_ascendc_binary_args(
    op: ssa.Operation, args: tuple[str, ...], ctx: _EmitContext
) -> tuple[str, ...]:
    if len(args) != len(op.operands):
        return args

    result_type = op.results[0].type if op.results else None

    if result_type is not None and result_type.dtype is None:
        return args

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
    _ascendc_validate_atomics(kernel)

    return common.emit(kernel, TARGET)
