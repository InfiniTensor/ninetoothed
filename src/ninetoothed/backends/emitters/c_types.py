"""Computed C++ scalar types shared by native emitter strategies."""

from collections.abc import Mapping
from typing import Any

from ninetoothed.dtype import normalize_dtype
from ninetoothed.ir import ssa


class CComputedTypesMixin:
    """Keep computed dtype witnesses and declarations in the same C++ type."""

    def load(self, tensor, index, *, mask=None, other=0.0):
        del mask, other
        rendered = f"({index})" if index.startswith("nt_idx_") else index

        return f"{self.tensor_ref(tensor)}[{rendered}]"

    def store(self, tensor, index, value, *, mask=None):
        rendered = f"({index})" if index.startswith("nt_idx_") else index
        assignment = f"{self.tensor_ref(tensor)}[{rendered}] = {value};"

        return assignment if mask is None else f"if ({mask}) {{\n    {assignment}\n}}"

    def value_dtype(self, value):
        return f"std::remove_cv<std::remove_reference<decltype({value})>::type>::type"

    def dtype_equal(self, lhs, rhs):
        return f"std::is_same<{self.value_dtype(lhs)}, {self.value_dtype(rhs)}>::value"

    def select_dtype_sample(self, condition, yes, no):
        dtype = (
            f"std::conditional<({condition}), {self.value_dtype(yes)}, "
            f"{self.value_dtype(no)}>::type"
        )

        return f"static_cast<{dtype}>(1)"

    def local_decl(self, type_: ssa.Type, name: str, expr: str) -> str:
        if type_.kind == "pointer" or (
            type_.dtype is None and type_.kind in {"scalar", "tensor"}
        ):
            return f"auto {name} = {expr};"
        return f"{self.type_name(type_.dtype, type_.kind)} {name} = {expr};"

    def check_numeric_result(self, operation, value, context):
        declared = operation.results[0].type if operation.results else None

        if (
            declared is not None
            and declared.kind in {"scalar", "tensor"}
            and operation.opcode.startswith("arith.")
            and declared.dtype in {"float16", "bfloat16"}
        ):
            rounded = self.cast(declared.dtype, value)
            computed = self.arithmetic_result_type(operation, context)

            return self.cast(computed.dtype, rounded)
        return value

    def global_helper(self, name: str) -> str:
        """Resolve generated helpers in the global C++ namespace."""
        return f"::{name}"


class CScalarArithmeticMixin(CComputedTypesMixin):
    """C++ scalar values, conversions and reduction identities for native targets."""

    def coerce_binary_args(
        self, op: ssa.Operation, args: tuple[str, ...], ctx: Any
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
            (type_ := ctx.value_types.get(operand)) is not None
            and type_.kind == "pointer"
            for operand in op.operands
        ):
            return args

        dtype = (
            normalize_dtype(result_type.dtype)
            if result_type is not None and normalize_dtype(result_type.dtype) != "bool"
            else _common_operand_dtype(op.operands, ctx)
        )

        if dtype is None:
            return args

        if dtype in {"float16", "bfloat16"}:
            dtype = "float32"

        coerced = []

        for operand, value in zip(op.operands, args):
            operand_type = ctx.value_types.get(operand)
            operand_dtype = normalize_dtype(
                operand_type.dtype if operand_type is not None else None
            )
            coerced.append(
                ctx.target.cast(dtype, value) if operand_dtype != dtype else value
            )
        return tuple(coerced)

    def cast_like(self, value, reference):
        return f"{self.global_helper('_nt_cast_like')}<{self.value_dtype(reference)}>({value})"

    def inline_constant(self, value, type_: ssa.Type) -> str:
        dtype = "int64" if type_.kind == "index" else type_.dtype

        return self.cast(dtype, self.literal(value))

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

    def reduce_update_typed(self, operator, acc, term, dtype):
        if dtype is None:
            term = self.cast_like(term, acc)

            if operator == "sum":
                return f"({acc} + {term})"
            return f"{self.global_helper('_nt_reduce_' + operator)}({acc}, {term})"
        return self.reduce_update(operator, acc, term)


def render_c_type_support(
    body: str,
    qualifier: str,
    *,
    cast_conversions: Mapping[str, str] | None = None,
) -> str:
    """Emit type-dependent arithmetic with a target's device qualifier."""
    support = []

    if any(
        name in body
        for name in (
            "nt_fmod(",
            "nt_copysign(",
            "nt_nearbyint(",
            "nt_isnan(",
            "nt_isfinite(",
            "_nt_modulo(",
        )
    ):
        from ninetoothed.backends.emitters.scalar_math import render_scalar_math_support

        support.append(render_scalar_math_support("NT_DEVICE"))

    if "_nt_cast_like<" in body or "nt_scalar_store(" in body:
        support.append(_C_CAST_SUPPORT)

        for dtype, conversion in (cast_conversions or {}).items():
            support.append(
                f"""template <> struct _nt_value_cast<{dtype}> {{
    template <class V> static NT_DEVICE {dtype} apply(V value) {{
        return {conversion};
    }}
}};"""
            )

        support.append(
            """template <class T, class V> NT_DEVICE T _nt_cast_like(V value) {
    return _nt_value_cast<T>::apply(value);
}"""
        )

    if any(
        marker in body
        for marker in (
            "_nt_limits<",
            "_nt_true_divide(",
            "_nt_reduce_",
            "_nt_reduction_bounds<",
        )
    ):
        support.append(_C_TYPE_SUPPORT)

    return "\n\n".join(support).replace("NT_DEVICE", qualifier)


_C_CAST_SUPPORT = """template <class T> struct _nt_value_cast {
    static NT_DEVICE T apply(bool value) {
        return static_cast<T>(value ? int32_t(1) : int32_t(0));
    }
    template <class V> static NT_DEVICE T apply(V value) {
        return static_cast<T>(value);
    }
};"""


def render_int32_cast_support(qualifier, *, name="nt_cast_int32"):
    """Preserve integral casts and define nonfinite/overflow float conversion.

    Vendor saturating conversions disagree with the language's Torch oracle:
    float values outside the signed int32 range (including NaN) yield INT_MIN.
    Integral values never round-trip through floating point. Half-like classes
    convert through float, while standard floating types retain their precision.
    """
    source = """template <class T, bool Integral = std::is_integral<T>::value>
struct NtInt32Cast {
    static NT_DEVICE int32_t apply(T value) {
        using F = typename std::conditional<std::is_floating_point<T>::value, T, float>::type;
        F f = static_cast<F>(value);
        return (f >= -2147483648.0f && f < 2147483648.0f)
            ? static_cast<int32_t>(f) : (-2147483647 - 1);
    }
};
template <class T> struct NtInt32Cast<T, true> {
    static NT_DEVICE int32_t apply(T value) {
        return static_cast<int32_t>(value);
    }
};
template <class T> static NT_DEVICE int32_t NT_CAST_INT32(T value) {
    return NtInt32Cast<T>::apply(value);
}"""

    return source.replace("NT_DEVICE", qualifier).replace("NT_CAST_INT32", name)


_C_TYPE_SUPPORT = """template <class T> struct _nt_limits {
    using R = typename std::conditional<std::is_arithmetic<T>::value, T, float>::type;
    static constexpr R low = std::numeric_limits<R>::has_infinity
        ? -std::numeric_limits<R>::infinity() : std::numeric_limits<R>::lowest();
    static constexpr R high = std::numeric_limits<R>::has_infinity
        ? std::numeric_limits<R>::infinity() : std::numeric_limits<R>::max();
};

template <class T> struct _nt_reduction_bounds {
    static NT_DEVICE auto low() { return _nt_limits<T>::low; }
    static NT_DEVICE auto high() { return _nt_limits<T>::high; }
};

template <class T> NT_DEVICE T _nt_reduce_max(T lhs, T rhs) {
    return lhs != lhs ? lhs : (rhs != rhs ? rhs : (lhs > rhs ? lhs : rhs));
}
template <class T> NT_DEVICE T _nt_reduce_min(T lhs, T rhs) {
    return lhs != lhs ? lhs : (rhs != rhs ? rhs : (lhs < rhs ? lhs : rhs));
}

template <class T>
NT_DEVICE typename std::enable_if<std::is_arithmetic<T>::value, T>::type
_nt_arithmetic(T value) { return value; }

template <class T>
NT_DEVICE typename std::enable_if<!std::is_arithmetic<T>::value, float>::type
_nt_arithmetic(T value) { return static_cast<float>(value); }

template <class A, class B>
NT_DEVICE auto _nt_true_divide(A lhs, B rhs)
    -> typename std::conditional<std::is_integral<decltype(_nt_arithmetic(lhs) + _nt_arithmetic(rhs))>::value,
                                 float, decltype(_nt_arithmetic(lhs) + _nt_arithmetic(rhs))>::type {
    using R = typename std::conditional<
        std::is_integral<decltype(_nt_arithmetic(lhs) + _nt_arithmetic(rhs))>::value,
        float, decltype(_nt_arithmetic(lhs) + _nt_arithmetic(rhs))>::type;
    return static_cast<R>(_nt_arithmetic(lhs)) / static_cast<R>(_nt_arithmetic(rhs));
}
"""


def _common_operand_dtype(operands: tuple[str, ...], ctx: Any) -> str | None:
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
        normalize_dtype(type_.dtype)
        for operand in operands
        if (type_ := ctx.value_types.get(operand)) is not None
    ]

    return max(dtypes, key=lambda dtype: ranks.get(dtype, -1)) if dtypes else None
