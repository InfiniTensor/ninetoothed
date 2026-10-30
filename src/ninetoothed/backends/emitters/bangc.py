"""BangC scalar task lowering through the shared typed SSA emitter."""

import math
import re
from dataclasses import dataclass
from typing import Any

from ninetoothed.backends.core import Target
from ninetoothed.backends.emitters import ssa as common
from ninetoothed.backends.emitters.bangc_launch import (
    MAX_BLOCK_TASKS,
    render_segmented_launch,
)
from ninetoothed.backends.emitters.base import EmitterTarget, ModuleRenderContext
from ninetoothed.backends.emitters.c_types import (
    CComputedTypesMixin,
    render_c_type_support,
    render_int32_cast_support,
)
from ninetoothed.backends.emitters.context import EmitContext as _EmitContext
from ninetoothed.backends.emitters.expressions import c_integer_expr
from ninetoothed.backends.emitters.math_helpers import render_float32_trig_support
from ninetoothed.backends.emitters.task_launch import (
    allocate_private_symbol,
    allocate_task_launch_symbols,
)
from ninetoothed.ir import Kernel, ssa

_normalize_dtype = common.normalize_dtype


@dataclass(frozen=True, kw_only=True)
class BangCTarget(CComputedTypesMixin, EmitterTarget):
    backend: Target = Target.BANGC
    language: str = "bangc/c++"
    suffix: str = "mlu"
    source_route: str = "ssa-unified-bangc-emitter"
    c_style_syntax: bool = True

    def index_cast(self, value: str) -> str:
        return f"(int64_t)({value})"

    def assign_scalar(self, name: str, value: str, *, mutable: bool) -> str:
        del mutable

        return f"{name} = {value};"

    def literal(self, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"

        if isinstance(value, int) and (value < -(1 << 31) or value >= 1 << 31):
            return f"NtInt64({value & 0xFFFFFFFF}u, {(value >> 32) & 0xFFFFFFFF}u)"

        if isinstance(value, float) and math.isinf(value):
            return "INFINITY" if value > 0 else "-INFINITY"

        if value == "nan" or (isinstance(value, float) and math.isnan(value)):
            return "NAN"

        if value == "inf":
            return "INFINITY"

        if value == "-inf":
            return "-INFINITY"
        return repr(value)

    def cast(self, dtype, value):
        normalized = _normalize_dtype(dtype)

        if normalized in {"int32", "int64", "uint32", "uint64"} and re.fullmatch(
            r"[-+]?\d+\.0*[fF]?", value.strip()
        ):
            # Cncc device code miscompiles double literals cast to
            # integers (the masked-load `other` of 0.0 turned into
            # garbage); integer targets get an integer literal instead.
            value = str(int(float(value)))

        if normalized == "bfloat16":
            # C-style casts to bfloat16_t are ambiguous on cncc; use the
            # explicit conversion intrinsic instead.
            return f"{self.global_helper('__float2bfloat16_rn')}((float)({value}))"

        if normalized == "float16":
            return f"{self.global_helper('__float2half_rn')}((float)({value}))"

        if normalized == "int32":
            return f"{self.global_helper('nt_cast_int32')}({value})"

        return f"(({self.type_name(dtype)})({value}))"

    def type_name(self, dtype, kind=None):
        return _bangc_type(dtype, kind)

    def store(self, tensor, index, value, *, mask=None):
        value = self.cast_like(value, f"{self.tensor_ref(tensor)}[0]")

        return super().store(tensor, index, value, mask=mask)

    def local_decl(self, type_, name, expr):
        if type_.kind == "index" or _normalize_dtype(type_.dtype) == "index":
            expr = self.index_cast(expr)
        return super().local_decl(type_, name, expr)

    def where(self, cond, yes, no):
        return f"(({cond}) ? ({yes}) : ({no}))"

    def call(self, name, args):
        if name == "where":
            return self.where(args[0], args[1], args[2])

        if name == "atomic_add":
            return (
                f"{self.global_helper('ninetoothed_atomic_add')}({args[0]}, {args[1]})"
            )

        if name == "load" and args:
            return f"*({args[0]})"

        if name in {"block_dot", "dot"} and len(args) == 2:
            return f"(({args[0]}) * ({args[1]}))"

        if name == "rand" and len(args) >= 2:
            return f"{self.global_helper('ninetoothed_rand_uniform')}(({args[0]}), ({args[1]}))"

        if name == "rsqrt" and len(args) == 1:
            return f"(1.0f / sqrtf({args[0]}))"

        if name == "reciprocal" and len(args) == 1:
            return f"(1.0f / ({args[0]}))"

        if name == "sigmoid" and len(args) == 1:
            return f"(1.0f / (1.0f + expf(-({args[0]}))))"

        if name in {"sin", "cos"} and len(args) == 1:
            return f"{self.global_helper('nt_' + name)}({args[0]})"

        functions = {
            "abs": "fabsf",
            "acos": "acosf",
            "asin": "asinf",
            "atan": "atanf",
            "atan2": "atan2f",
            "_atan2_approx": "atan2f",
            "ceil": "ceilf",
            "copysign": "nt_copysign",
            "fmod": "nt_fmod",
            "nearbyint": "nt_nearbyint",
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
            "maximum": "nt_max",
            "max": "nt_max",
            "minimum": "nt_min",
            "min": "nt_min",
            "pow": "powf",
            "round": "roundf",
            "sin": "sinf",
            "sinh": "sinhf",
            "sqrt": "sqrtf",
            "tan": "tanf",
            "tanh": "tanhf",
        }
        function = functions.get(name, name)

        return f"{function}({', '.join(args)})"

    def program_id(self, axis=0):
        if axis == 0:
            return "(int64_t)(taskIdX)"

        if axis == 1:
            return "(int64_t)(taskIdY)"

        if axis == 2:
            return "(int64_t)(taskIdZ)"

        raise ValueError("BangC supports at most three task-grid axes.")

    def atomic_add(self, operands: tuple[str, ...], dtype: str) -> str:
        suffix = _atomic_suffix(dtype)

        return (
            f"{self.global_helper('ninetoothed_atomic_add_' + suffix)}({operands[0]}, {operands[1]})"
            if suffix is not None
            else f"{self.global_helper('ninetoothed_atomic_add_f32')}({operands[0]}, {operands[1]})"
        )

    def loop_header(self, var, lower, upper, step):
        return f"for (int64_t {var} = {lower}; {var} < {upper}; {var} += {step}) {{"

    def reduce_update(self, operator, acc, term):
        if operator == "sum":
            return f"{acc} + {term}"

        function = "nt_max" if operator == "max" else "nt_min"

        return f"{function}({acc}, {term})"

    def arithmetic_result_type(self, operation, context) -> ssa.Type:
        return _bangc_arithmetic_result_type(operation, context)

    def coerce_binary_args(self, operation, args, context):
        return _coerce_bangc_binary_args(operation, args, context)

    def inline_constant(self, value, type_: ssa.Type) -> str:
        dtype = "int64" if type_.kind == "index" else type_.dtype

        return self.cast(dtype, self.literal(value))

    def cast_like(self, value: str, reference: str) -> str:
        # Cncc miscompiles casts of integral-valued double literals to integer
        # types. An integer spelling is exact for every destination type and
        # avoids making the computed destination dtype a Python-side guess.
        if re.fullmatch(r"[-+]?\d+\.0*[fF]?", value.strip()):
            value = value.strip().rstrip("fF").split(".", 1)[0]
        return f"{self.global_helper('_nt_cast_like')}<{self.value_dtype(reference)}>({value})"

    def reduction_identity(self, dtype: str, operator: str, shape: str) -> str:
        del shape

        if operator == "sum":
            accumulator = (
                f"std::conditional<std::is_integral<{dtype}>::value && "
                f"(sizeof({dtype}) < 4), int32_t, {dtype}>::type"
            )

            return f"static_cast<{accumulator}>(0)"

        if operator not in {"max", "min"}:
            raise ValueError(f"Unsupported C++ reduction operator: {operator!r}.")

        bound = "low" if operator == "max" else "high"

        return f"{self.global_helper('_nt_reduction_bounds')}<{dtype}>::{bound}()"

    def true_divide(self, lhs: str, rhs: str) -> str:
        return f"{self.global_helper('_nt_true_divide')}({lhs}, {rhs})"

    def modulo(self, lhs, rhs):
        return f"{self.global_helper('_nt_modulo')}({lhs}, {rhs})"

    def reduce_update_typed(self, operator, acc, term, dtype):
        if dtype is None:
            term = self.cast_like(term, acc)

            if operator == "sum":
                return f"({acc} + {term})"
            return f"{self.global_helper('_nt_reduce_' + operator)}({acc}, {term})"
        return self.reduce_update(operator, acc, term)

    def schedule_context(self, context: ModuleRenderContext) -> ModuleRenderContext:
        from ninetoothed.backends.emitters.memory_guard import (
            dense_staging_memory_guard,
        )
        from ninetoothed.backends.emitters.scalar_tiles import plan_scalar_tiles

        plan = plan_scalar_tiles(context, self, storage="__nram__", budget=384 * 1024)

        if plan is not None:
            guard = c_integer_expr(
                dense_staging_memory_guard(
                    context, context.variables, context.outputs, dense_inputs=False
                )
            )

            if guard != "false":
                context.lowering_plans["scalar_tiles"] = (plan, guard)
                context.scheduled_metadata["bangc_scalar_tile_storage"] = {
                    "bytes": plan.scratch_bytes,
                    "tasks": plan.tasks,
                    "budget": 384 * 1024,
                    "nonoverlap_guard": guard,
                }
        return context

    def render_module(self, context: ModuleRenderContext) -> str:
        kernel = context.kernel
        body = c_integer_expr(context.body)
        total = c_integer_expr(context.total)
        chunk = 1024
        atomic = any(
            op.opcode == "mem.atomic_add" for op in context.operations.values()
        )

        if atomic:
            tasks = f"(({total}) > 0 ? 1 : 0)"
            chunk_expr = total
        else:
            tasks = f"(({total}) / {chunk} + (({total}) % {chunk} != 0))"
            chunk_expr = str(chunk)

        queue = allocate_private_symbol(
            "nt_queue",
            body,
            *context.variables,
            *context.outputs,
            *context.shape_params,
        )

        def params(host=False):
            return [
                *(
                    _signature_param(
                        name,
                        context.tensors[name],
                        readonly=name not in context.outputs,
                        restrict=False,
                        host=host,
                    )
                    for name in (*context.variables, *context.outputs)
                ),
                *(
                    _auxiliary_param(name, context.tensors, host=host)
                    for name in context.shape_params
                ),
            ]

        kernel_params = params()
        launch_params = params(host=True)
        symbols = allocate_task_launch_symbols(body, *kernel_params, *launch_params)
        start = allocate_private_symbol(
            "nt_start", body, *kernel_params, *symbols.parameters
        )
        lane = allocate_private_symbol("nt_lane", body, *kernel_params, start)
        count = allocate_private_symbol("nt_count", body, *kernel_params, start, lane)
        task_body = f"""    const int64_t {start} = (int64_t)taskIdX * ({chunk_expr});
    int64_t {count} = ({total}) - {start};
    if ({count} > ({chunk_expr})) {{ {count} = ({chunk_expr}); }}
    for (int64_t {lane} = 0; {lane} < {count}; ++{lane}) {{
        int64_t {self.index_name} = {start} + {lane};
{common.indent_block(body, "        ")}
    }}"""
        tile_storage = context.lowering_plans.get("scalar_tiles")

        if tile_storage is not None:
            tile_plan, guard = tile_storage
            task_body = f"if ({guard}) {{\n{common.indent_block(tile_plan.body, '    ')}\n}} else {{\n{task_body}\n}}"
            tasks = f"({guard}) ? ({tile_plan.tasks}) : ({tasks})"

        task_body = symbols.kernel_body(
            task_body,
            task_pattern=r"\btaskIdX\b",
            physical_index="taskIdX",
            max_tasks=MAX_BLOCK_TASKS,
        )
        arguments = tuple(
            f"nt_host_wide<{str(_normalize_dtype(context.tensors[name].dtype) == 'int64').lower()}>((int64_t){name})"
            if name in context.tensors
            and context.tensors[name].ndim == 0
            and _normalize_dtype(context.tensors[name].dtype) in {"int64", "uint64"}
            else name
            for name in (*context.variables, *context.outputs, *context.shape_params)
        )
        launcher = render_segmented_launch(
            kernel.kernel_name, arguments, tasks, symbols, queue=queue
        )
        support = (
            render_c_type_support(
                task_body,
                "__mlu_func__",
                cast_conversions={
                    "half": "__float2half_rn((float)(value))",
                    "bfloat16_t": "__float2bfloat16_rn((float)(value))",
                },
            )
            + "\n"
            + _bangc_support(context, task_body)
        )
        kernel_decl = _render_signature_params([*kernel_params, *symbols.parameters])
        launch_decl = _render_signature_params([*launch_params, f"cnrtQueue_t {queue}"])

        return f"""// Generated by NineToothed's BangC SSA backend.
// Kernel: {kernel.kernel_name}
// Lowering IR: ssa.Program
#include <bang.h>
#include <cnrt.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <type_traits>
#include <limits>
{support}
__mlu_entry__ void {kernel.kernel_name}_kernel(
{kernel_decl}
) {{
{task_body}
}}
extern "C" int launch_{kernel.kernel_name}(
{launch_decl}
) {{
{launcher}
}}
"""


def _atomic_suffix(dtype: str | None) -> str | None:
    dtype = common.normalize_dtype(dtype)

    return {
        "float32": "f32",
        "float16": "f16",
        "bfloat16": "bf16",
        "int32": "i32",
        "int64": "i64",
        "uint32": "i32",
        "uint64": "i64",
    }.get(dtype)


def _bangc_support(context: ModuleRenderContext, rendered_body: str = "") -> str:
    """Emit helper functions required by the kernel body."""
    operations = {id(operation): operation for operation in context.operations.values()}
    from ninetoothed.backends.emitters.bangc_int64 import render_int64_support

    lines: list[str] = [render_int64_support()]

    if "_nt_reduction_bounds<" in rendered_body:
        lines.append(
            """template <bool Signed> struct _nt_reduction_bounds<NtWide<Signed>> {
    static __mlu_func__ NtWide<Signed> low() {
        return NtWide<Signed>(0u, Signed ? 0x80000000u : 0u);
    }
    static __mlu_func__ NtWide<Signed> high() {
        return NtWide<Signed>(0xffffffffu, Signed ? 0x7fffffffu : 0xffffffffu);
    }
};"""
        )

    if "nt_cast_int32(" in context.body:
        lines.append(render_int32_cast_support("__mlu_func__"))

    if any(operation.opcode == "arith.div" for operation in operations.values()):
        # MLU590's public header has vector division but no div_scalar.
        # Keep true division in NRAM rather than rounding a reciprocal
        # constant or emitting a nonexistent intrinsic.
        lines.append(
            """static __mlu_func__ void nt_bang_div_scalar(
    float *dst, const float *src, float scalar, uint32_t count
) {
    for (uint32_t i = 0; i < count; ++i) { dst[i] = src[i] / scalar; }
}"""
        )

    if any(
        operation.opcode in {"math.sin", "math.cos", "call.sin", "call.cos"}
        for operation in operations.values()
    ):
        lines.append(
            "union NtFloatBits { float f; int32_t i; };\n"
            "static __mlu_func__ float nt_inf() { return INFINITY; }\n"
            + render_float32_trig_support("__mlu_func__")
        )

    if any(operation.opcode == "math.rand" for operation in operations.values()):
        # NOTE: 64-bit immediate constants are truncated on the MLU590 scalar
        # path, so the generator is kept entirely in uint32 arithmetic.
        lines.append(
            """static __mlu_func__ float ninetoothed_rand_uniform(
    uint64_t seed, uint64_t offset
) {
    uint32_t z = (uint32_t)(seed) ^ ((uint32_t)(offset) * 0x9E3779B1u);
    z = z + 0x6D2B79F5u;
    z = (z ^ (z >> 15)) * 0x85EBCA6Bu;
    z = (z ^ (z >> 13)) * 0xC2B2AE35u;
    z = z ^ (z >> 16);
    return (float)(z >> 8) * (1.0f / 16777216.0f);
}"""
        )

    needs_extremum = bool(re.search(r"\bnt_(?:max|min)\s*\(", rendered_body)) or any(
        operation.opcode.startswith("reduce.")
        or operation.opcode in {"arith.maximum", "arith.minimum"}
        or operation.opcode in {"math.max", "math.min"}
        or operation.opcode
        in {
            "math.exp",
            "math.exp2",
            "math.sin",
            "math.cos",
            "math.tanh",
            "math.sigmoid",
            "call.exp",
            "call.exp2",
            "call.sin",
            "call.cos",
            "call.tanh",
            "call.sigmoid",
            "math.erf",
            "call.erf",
        }
        for operation in operations.values()
    )

    if needs_extremum:
        # Integer comparisons must not round-trip through fmaxf/fminf,
        # which loses precision beyond 2^24; float keeps NaN semantics.
        # cncc misresolves same-name __mlu_func__ overloads, so extremums
        # are type-generic macros: integer comparisons stay exact and the
        # ternary preserves a NaN operand for floats, matching vector
        # nan_maximum/nan_minimum and the language's extrema.
        lines.append(
            """#define nt_max(a, b) ((a) != (a) ? (a) : ((b) != (b) ? (b) : (((a) > (b)) ? (a) : (b))))
#define nt_min(a, b) ((a) != (a) ? (a) : ((b) != (b) ? (b) : (((a) < (b)) ? (a) : (b))))"""
        )

    atomic_dtypes = set()

    for operation in operations.values():
        if operation.opcode != "mem.atomic_add" or not operation.operands:
            continue

        operand_type = context.value_types.get(operation.operands[-1])
        atomic_dtypes.add(
            _normalize_dtype(operand_type.dtype if operand_type else None)
        )

    # NOTE: every `__bang_atomic_add` spelling (C intrinsics and raw
    # `atom.add.scalar` assembly, block and union task types) traps with
    # CN_INVOKE_ERROR on MLU590, so the update degrades to a plain GDRAM
    # read-modify-write.  Atomic kernels are therefore serialized onto a
    # single task by `render_module`, which keeps the accumulation exact.
    for dtype in sorted(atomic_dtypes):
        variant = _atomic_suffix(dtype)

        if variant is None:
            continue

        address_type = {
            "f32": "float",
            "f16": "half",
            "bf16": "bfloat16_t",
            "i32": "int32_t",
            "i64": "int64_t",
        }[variant]

        # Integer updates must stay in integer arithmetic: routing the
        # value through float32 loses every increment past 2^24.
        value_type = {
            "f32": "float",
            "f16": "float",
            "bf16": "float",
            "i32": "int32_t",
            "i64": "int64_t",
        }[variant]

        if variant in {"f16", "bf16"}:
            load = (
                "__bfloat162float(address[0])"
                if variant == "bf16"
                else "__half2float(address[0])"
            )
            store = "__float2bfloat16_rn" if variant == "bf16" else "__float2half_rn"
            body = f"""    float nt_prev = {load};
    address[0] = {store}(nt_prev + value);"""
            lines.append(
                f"""static __mlu_func__ void ninetoothed_atomic_add_{variant}(
    {address_type}* address, {value_type} value
) {{
{body}
}}"""
            )
        elif variant in {"i32", "i64"}:
            # The call site picks the variant from the update value's dtype,
            # which may differ from the buffer's; deduce the address type
            # and accumulate in the buffer's own integer width.
            lines.append(
                f"""template <typename T, typename V>
static __mlu_func__ void ninetoothed_atomic_add_{variant}(
    T* address, V value
) {{
    address[0] = (T)(address[0] + (T)value);
}}"""
            )
        else:
            body = """    address[0] = address[0] + value;"""
            lines.append(
                f"""static __mlu_func__ void ninetoothed_atomic_add_{variant}(
    {address_type}* address, {value_type} value
) {{
{body}
}}"""
            )

    return "\n\n".join(lines)


def _render_signature_params(params: list[str]) -> str:
    return ",\n".join(f"    {param}" for param in params)


def _signature_param(name, info, *, readonly: bool, restrict: bool, host=False) -> str:
    dtype = _bangc_type(info.dtype)

    if host and info.ndim == 0:
        dtype = {"NtInt64": "int64_t", "NtUInt64": "uint64_t"}.get(dtype, dtype)

    if info.ndim == 0:
        return f"{dtype} {name}"

    const = "const " if readonly else ""
    qualifier = " __restrict__" if restrict else ""

    return f"{const}{dtype}*{qualifier} {name}"


def _auxiliary_param(name, tensors, host=False) -> str:
    info = tensors.get(name)

    if info is not None and info.ndim == 0:
        return _signature_param(name, info, readonly=True, restrict=False, host=host)
    return f"int64_t {name}"


def _bangc_type(dtype: str | None, kind: str | None = None) -> str:
    dtype = common.normalize_dtype(dtype)

    if kind == "pointer":
        return f"{_bangc_type(dtype)}*"

    if kind == "index" or dtype == "index":
        return "int64_t"

    types = {
        "float32": "float",
        "float16": "half",
        "bfloat16": "bfloat16_t",
        "float64": "double",
        "int8": "int8_t",
        "uint8": "uint8_t",
        "int16": "int16_t",
        "uint16": "uint16_t",
        "int32": "int32_t",
        "uint32": "uint32_t",
        "int64": "NtInt64",
        "uint64": "NtUInt64",
        "bool": "bool",
    }

    if dtype in {"float8_e4m3fn", "float8_e5m2"}:
        raise ValueError(
            "The BangC backend does not support float8 dtypes on this toolchain."
        )

    if dtype not in types:
        raise ValueError(f"Unsupported BangC SSA dtype: {dtype!r}.")
    return types[dtype]


def _bangc_arithmetic_result_type(op: ssa.Operation, ctx: _EmitContext) -> ssa.Type:
    return op.results[0].type


def _coerce_bangc_binary_args(
    op: ssa.Operation, args: tuple[str, ...], ctx: _EmitContext
) -> tuple[str, ...]:
    if len(args) != len(op.operands):
        return args

    result_type = op.results[0].type if op.results else None

    if result_type is not None and result_type.dtype is None:
        # Let C++ infer the actual promotion. Choosing a known integer
        # constant's dtype while another operand is unknown narrows even a
        # floating input to integer. check_numeric_result validates the
        # resulting computed type, including inline tensor expressions.
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
        else _bangc_common_operand_dtype(op.operands, ctx)
    )

    if dtype is None:
        return args

    if dtype in {"float16", "bfloat16"}:
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


def _bangc_common_operand_dtype(
    operands: tuple[str, ...], ctx: _EmitContext
) -> str | None:
    ranks = {
        "bool": 0,
        "int32": 1,
        "int64": 2,
        "float16": 3,
        "bfloat16": 3,
        "float32": 4,
        "float64": 5,
    }
    dtypes = [
        _normalize_dtype(type_.dtype)
        for operand in operands
        if (type_ := ctx.value_types.get(operand)) is not None
    ]

    return max(dtypes, key=lambda dtype: ranks.get(dtype, -1)) if dtypes else None


TARGET = BangCTarget()


def emit(kernel: Kernel):
    return common.emit(kernel, TARGET)
