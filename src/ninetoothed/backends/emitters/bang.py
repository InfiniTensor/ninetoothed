"""BANG C++ syntax hooks for the common SSA emitter.

The BANG backend targets Cambricon MLUs through cncc: each kernel is
emitted as one BANG C++ translation unit that contains both the
``__mlu_global__`` device kernel and the CNRT host launcher, mirroring
how the CUDA backend ships kernel and launcher in a single ``.cu`` file.

The current lowering is a correctness-first flat mapping: one BANG block
task processes ``threads`` consecutive elements serially.  NRAM staging
and vector intrinsics are left for later schedule-driven work.
"""

import math
import re
from dataclasses import dataclass
from typing import Any

from ninetoothed.backends.core import Target
from ninetoothed.backends.emitters import ssa as common
from ninetoothed.backends.emitters.base import EmitterTarget, ModuleRenderContext
from ninetoothed.ir import ssa

_BANG_FUNCTIONS = {
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


@dataclass(frozen=True, kw_only=True)
class BangTarget(EmitterTarget):
    backend: Target = Target.BANG
    language: str = "bang/c++"
    suffix: str = "mlu"
    source_route: str = "ssa-unified-bang-emitter"
    c_style_syntax: bool = True

    def index_cast(self, value: str) -> str:
        return f"static_cast<int64_t>({value})"

    def assign_scalar(self, name: str, value: str, *, mutable: bool) -> str:
        del mutable

        return f"{name} = {value};"

    def literal(self, value: Any) -> str:
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value == -(1 << 63)
        ):
            return "(-9223372036854775807LL - 1LL)"

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

    def inline_constant(self, value, type_):
        return f"static_cast<{self.type_name(type_.dtype, type_.kind)}>({self.literal(value)})"

    def cast_like(self, value, reference):
        return f"static_cast<{self.value_dtype(reference)}>({value})"

    def value_dtype(self, value):
        return f"std::remove_cv<std::remove_reference<decltype({value})>::type>::type"

    def true_divide(self, lhs: str, rhs: str) -> str:
        return f"_nt_true_divide({lhs}, {rhs})"

    def reduction_identity(self, dtype, operator, shape):
        del shape

        if operator == "sum":
            dtype = (
                f"std::conditional<std::is_integral<{dtype}>::value && "
                f"(sizeof({dtype}) < 4), int64_t, {dtype}>::type"
            )

            return f"static_cast<{dtype}>(0)"

        bound = "low" if operator == "max" else "high"

        return f"_nt_limits<{dtype}>::{bound}"

    def type_name(self, dtype, kind=None):
        return _bang_type(dtype, kind)

    def where(self, cond, yes, no):
        return f"(({cond}) ? ({yes}) : ({no}))"

    def call(self, name, args):
        if name == "where":
            return self.where(args[0], args[1], args[2])

        if name == "load" and args:
            return f"*({args[0]})"

        if name in {"block_dot", "dot"} and len(args) == 2:
            return f"(({args[0]}) * ({args[1]}))"

        if name == "atomic_add":
            raise NotImplementedError(
                "The BANG backend does not lower atomic_add yet."
            )

        if name == "rand":
            raise NotImplementedError("The BANG backend does not lower rand yet.")

        function = _BANG_FUNCTIONS.get(name, name)

        return f"{function}({', '.join(args)})"

    def program_id(self, axis=0):
        if axis != 0:
            raise ValueError("The BANG backend uses a one-dimensional task grid.")
        return "static_cast<int64_t>(taskId)"

    def thread_id(self):
        return "static_cast<int64_t>(0)"

    def thread_count(self):
        return "static_cast<int64_t>(1)"

    def local_decl(self, type_: ssa.Type, name: str, expr: str) -> str:
        if type_.kind == "pointer" or (
            type_.dtype is None and type_.kind in {"scalar", "tensor"}
        ):
            return f"auto {name} = {expr};"
        return f"{self.type_name(type_.dtype, type_.kind)} {name} = {expr};"

    def loop_header(self, var, lower, upper, step):
        return f"for (int64_t {var} = {lower}; {var} < {upper}; {var} += {step}) {{"

    def reduce_update(self, operator, acc, term):
        if operator == "sum":
            return f"{acc} + {term}"

        function = "fmaxf" if operator == "max" else "fminf"

        return f"{function}({acc}, {term})"

    def reduce_update_typed(self, operator, acc, term, dtype):
        if dtype is None:
            term = self.cast_like(term, acc)

            if operator == "sum":
                return f"({acc} + {term})"
            return f"_nt_reduce_{operator}({acc}, {term})"
        return _bang_reduction_update(operator, acc, term, dtype)

    def render_module(self, context: ModuleRenderContext) -> str:
        if context.block_program or context.cooperative_reduction_program:
            raise ValueError(
                "The BANG backend currently lowers only flat elementwise "
                "programs; block/WMMA-style programs are future work."
            )

        kernel = context.kernel
        elements_per_task = common.schedule_int(kernel, "threads", 256)
        total = _bang_integer_expr(context.total)
        body = _bang_integer_expr(context.body)
        kernel_params = _render_signature_params(
            [
                *(
                    _signature_param(name, context.tensors[name], readonly=True)
                    for name in context.variables
                ),
                *(
                    _signature_param(name, context.tensors[name], readonly=False)
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
                    _signature_param(name, context.tensors[name], readonly=True)
                    for name in context.variables
                ),
                *(
                    _signature_param(name, context.tensors[name], readonly=False)
                    for name in context.outputs
                ),
                *(
                    _auxiliary_param(axis, context.tensors)
                    for axis in context.shape_params
                ),
                "cnrtQueue_t stream",
            ]
        )
        args = ", ".join((*context.variables, *context.outputs, *context.shape_params))
        kernel_prelude = f"""    int64_t nt_task_base = static_cast<int64_t>(taskId) * {elements_per_task};
    for (int64_t nt_lane = 0; nt_lane < {elements_per_task}; nt_lane++) {{
        int64_t {self.index_name} = nt_task_base + nt_lane;
        if ({self.index_name} < {total}) {{
{common.indent_block(body, "            ")}
        }}
    }}"""
        arithmetic_support = _ARITHMETIC_SUPPORT if "_nt_" in body else ""

        return f"""// Generated by NineToothed's BANG SSA backend.
// Kernel: {kernel.kernel_name}
// Lowering IR: ssa.Program

#include <bang.h>
#include <cnrt.h>
#include <limits>
#include <math.h>
#include <stdint.h>
#include <type_traits>

{arithmetic_support}

extern "C" __mlu_global__ void {kernel.kernel_name}_kernel(
{kernel_params}
) {{
{kernel_prelude}
}}

extern "C" int launch_{kernel.kernel_name}(
{launch_params}
) {{
    if ({total} <= 0) {{
        return 0;
    }}
    cnrtDim3_t nt_dim;
    nt_dim.x = static_cast<unsigned int>(({total} + {elements_per_task} - 1) / {elements_per_task});
    nt_dim.y = 1;
    nt_dim.z = 1;
    void* nt_args[] = {{{", ".join(f"&{name}" for name in (*context.variables, *context.outputs, *context.shape_params))}}};
    cnrtRet_t nt_ret = cnrtInvokeKernel(
        (const void*){kernel.kernel_name}_kernel,
        nt_dim,
        cnrtFuncTypeBlock,
        nt_args,
        0,
        stream
    );
    if (nt_ret != cnrtSuccess) {{
        return static_cast<int>(nt_ret);
    }}
    return static_cast<int>(cnrtQueueSync(stream));
}}
"""


def _render_signature_params(params: list[str]) -> str:
    return ",\n".join(f"    {param}" for param in params)


def _signature_param(name, info, *, readonly: bool) -> str:
    dtype = _bang_type(info.dtype)

    if info.ndim == 0:
        return f"{dtype} {name}"

    const = "const " if readonly else ""

    return f"{const}{dtype}* {name}"


def _auxiliary_param(name, tensors) -> str:
    info = tensors.get(name)

    if info is not None and info.ndim == 0:
        return _signature_param(name, info, readonly=True)
    return f"int64_t {name}"


def _bang_type(dtype: str | None, kind: str | None = None) -> str:
    dtype = common.normalize_dtype(dtype)

    if kind == "pointer":
        return f"{_bang_type(dtype)}*"

    if kind == "index" or dtype in {"index", "int64"}:
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
        "uint64": "uint64_t",
        "bool": "bool",
    }

    if dtype not in types:
        raise ValueError(f"Unsupported BANG SSA dtype: {dtype!r}.")
    return types[dtype]


def _bang_integer_expr(expr: str) -> str:
    previous = None
    current = common.rewrite_index_math(expr, c_style=True).replace("//", "/")
    current = re.sub(r"\bTrue\b", "true", current)
    current = re.sub(r"\bFalse\b", "false", current)
    pattern = re.compile(r"floor\(\(([^()]+)\)/([A-Za-z_][A-Za-z0-9_]*)\)")

    while current != previous:
        previous = current
        current = pattern.sub(r"((\1)/(\2))", current)
    return current


def _bang_reduction_update(operator: str, lhs: str, rhs: str, dtype: str) -> str:
    del dtype

    if operator == "sum":
        return f"({lhs} + {rhs})"
    return f"_nt_reduce_{operator}({lhs}, {rhs})"


TARGET = BangTarget()


def emit(kernel):
    return common.emit(kernel, TARGET)


__all__ = ["BangTarget", "TARGET", "emit"]


_ARITHMETIC_SUPPORT = """
template <class T> inline T _nt_reduce_max(T lhs, T rhs) {
    return lhs > rhs ? lhs : rhs;
}
template <class T> inline T _nt_reduce_min(T lhs, T rhs) {
    return lhs < rhs ? lhs : rhs;
}
inline float _nt_reduce_max(float lhs, float rhs) { return fmaxf(lhs, rhs); }
inline float _nt_reduce_min(float lhs, float rhs) { return fminf(lhs, rhs); }
inline double _nt_reduce_max(double lhs, double rhs) { return fmax(lhs, rhs); }
inline double _nt_reduce_min(double lhs, double rhs) { return fmin(lhs, rhs); }

template <class T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
_nt_arithmetic(T value) { return value; }

template <class T>
inline typename std::enable_if<!std::is_arithmetic<T>::value, float>::type
_nt_arithmetic(T value) { return static_cast<float>(value); }

template <class T> struct _nt_limits {
    using R = typename std::conditional<std::is_arithmetic<T>::value, T, float>::type;
    static constexpr R low = std::numeric_limits<R>::has_infinity
        ? -std::numeric_limits<R>::infinity() : std::numeric_limits<R>::lowest();
    static constexpr R high = std::numeric_limits<R>::has_infinity
        ? std::numeric_limits<R>::infinity() : std::numeric_limits<R>::max();
};

template <class A, class B>
inline auto _nt_true_divide(A lhs, B rhs)
    -> typename std::conditional<std::is_integral<decltype(_nt_arithmetic(lhs) + _nt_arithmetic(rhs))>::value,
                                 float, decltype(_nt_arithmetic(lhs) + _nt_arithmetic(rhs))>::type {
    using R = typename std::conditional<
        std::is_integral<decltype(_nt_arithmetic(lhs) + _nt_arithmetic(rhs))>::value, float, decltype(_nt_arithmetic(lhs) + _nt_arithmetic(rhs))>::type;
    return static_cast<R>(_nt_arithmetic(lhs)) / static_cast<R>(_nt_arithmetic(rhs));
}
"""
