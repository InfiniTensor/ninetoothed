"""AscendC scalar lowering through the shared typed SSA emitter."""

import math
import re
from dataclasses import dataclass, replace
from typing import Any

from ninetoothed.backends.core import Target
from ninetoothed.backends.emitters import ssa as common
from ninetoothed.backends.emitters.analysis import walk_ops as _walk_ops
from ninetoothed.backends.emitters.ascendc_math import render_math_support
from ninetoothed.backends.emitters.base import EmitterTarget, ModuleRenderContext
from ninetoothed.backends.emitters.c_types import (
    CScalarArithmeticMixin,
    render_c_type_support,
    render_int32_cast_support,
)
from ninetoothed.backends.emitters.expressions import c_integer_expr
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

        return f"nt_scalar_where({cond}, {yes}, {no})"

    def call(self, name, args):
        if name == "where":
            return self.where(args[0], args[1], args[2])

        if name == "load" and args:
            return f"nt_scalar_load({args[0]}, 0)"

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
        return operation.results[0].type

    def check_numeric_result(self, operation, value, context):
        if operation.opcode == "select.where" and operation.results[0].type.dtype:
            return self.cast(operation.results[0].type.dtype, value)
        return super().check_numeric_result(operation, value, context)

    def store(self, tensor, index, value, *, mask=None):
        statement = f"nt_scalar_store({self.tensor_ref(tensor)}, {index}, {value});"

        return statement if mask is None else f"if ({mask}) {{\n    {statement}\n}}"

    def load(self, tensor, index, *, mask=None, other=0.0):
        return f"nt_scalar_load({self.tensor_ref(tensor)}, {index})"

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
            _auxiliary_param(name, context.tensors, half_storage) for name in names
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
                _ascendc_support(context, task_body),
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


def _ascendc_reduction_update(operator: str, lhs: str, rhs: str, *, target) -> str:
    if operator == "sum":
        return f"({lhs}) + ({rhs})"

    # Computed dtypes are C++ type text, so Python string tests cannot decide
    # floating versus integral. Explicitly keep the accumulator's value type
    # while the shared template propagates either NaN through the reduction.
    value_dtype = f"std::remove_cv<std::remove_reference<decltype({lhs})>::type>::type"

    function = f"_nt_reduce_{operator}"

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


def _ascendc_support(context: ModuleRenderContext, body: str) -> str:
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
// Reserve the last 128 bytes of UB for scalar DMA scratch. The tile arena
// is separately bounded at UB - 256 bytes; all GM accesses use DMA, not DCache.

template<class T> __aicore__ inline T nt_dma_load(const __gm__ T* base, int64_t index) {
 LocalTensor<uint8_t> scratch(TPosition::VECCALC, 192*1024-128, 32);
 GlobalTensor<uint8_t> gm; gm.SetGlobalBuffer((__gm__ uint8_t *)(base+index));
 DataCopyExtParams copy{1,(uint32_t)sizeof(T),0,0,0};
 DataCopyPadExtParams<uint8_t> pad{false,0,0,0};
 SetFlag<HardEvent::S_MTE2>(0); WaitFlag<HardEvent::S_MTE2>(0);
 DataCopyPad(scratch,gm,copy,pad);
 SetFlag<HardEvent::MTE2_S>(0); WaitFlag<HardEvent::MTE2_S>(0);
 return *((__ubuf__ T *)scratch.GetPhyAddr());
}
template<class T> __aicore__ inline void nt_dma_store(__gm__ T* base, int64_t index,T value) {
 LocalTensor<uint8_t> scratch(TPosition::VECCALC, 192*1024-128, 32);
 *((__ubuf__ T *)scratch.GetPhyAddr())=value;
 SetFlag<HardEvent::S_MTE3>(0); WaitFlag<HardEvent::S_MTE3>(0);
 GlobalTensor<uint8_t> gm; gm.SetGlobalBuffer((__gm__ uint8_t *)(base+index));
 DataCopyExtParams copy{1,(uint32_t)sizeof(T),0,0,0};
 DataCopyPad(gm,scratch,copy);
 SetFlag<HardEvent::MTE3_S>(0); WaitFlag<HardEvent::MTE3_S>(0);
}

template <class T> struct _nt_value_cast;
template <class T> __aicore__ inline T nt_scalar_load(const __gm__ T *base, int64_t index) {
    return nt_dma_load(base, index);
}
__aicore__ inline NtBFloat16 nt_scalar_load(const __gm__ NtBFloat16 *base, int64_t index) {
    NtBFloat16 result; result.bits = nt_dma_load((const __gm__ uint16_t *)base, index); return result;
}
template <class T, class V> __aicore__ inline void nt_scalar_store(__gm__ T *base, int64_t index, V value) {
    nt_dma_store(base, index, _nt_value_cast<T>::apply(value));
}
template <class V> __aicore__ inline void nt_scalar_store(__gm__ NtBFloat16 *base, int64_t index, V value) {
    NtBFloat16 result((float)value); nt_dma_store((__gm__ uint16_t *)base, index, result.bits);
}
template <class T> struct NtWhereValue {
    using type = typename std::conditional<std::is_same<T, NtBFloat16>::value
        || std::is_same<T, half>::value, float, T>::type;
};
template <class A, class B>
__aicore__ inline auto nt_scalar_where(bool condition, A yes, B no)
    -> typename std::common_type<typename NtWhereValue<A>::type, typename NtWhereValue<B>::type>::type {
    using R = typename std::common_type<typename NtWhereValue<A>::type, typename NtWhereValue<B>::type>::type;
    return condition ? static_cast<R>(yes) : static_cast<R>(no);
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

    lines.append(render_math_support(body))

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


def _auxiliary_param(name, tensors, half_storage=None) -> str:
    info = tensors.get(name)

    if info is not None and info.ndim == 0:
        if half_storage is not None and name in half_storage:
            return f"uint16_t {half_storage[name]}"
        return f"{_ascendc_type(info.dtype)} {name}"
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


TARGET = AscendCTarget()


def emit(kernel: Kernel):
    _ascendc_validate_atomics(kernel)

    return common.emit(kernel, TARGET)
