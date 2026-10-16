"""NumPy semantics for the SSA operations supported by the CPU interpreter.

Every rule is registered against an exact opcode or an opcode prefix that ends
with a dot. The registry is intentionally open, so a new operation can be added
locally without touching the executor:

.. code-block:: python

    from ninetoothed.interpreter.ops import register


    @register("my.op", summary="Do something new.")
    def _my_op(context):
        context.bind(context.value(context.operation.operands[0]))
"""

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ninetoothed.interpreter.dtypes import resolve_dtype
from ninetoothed.interpreter.errors import (
    InvalidProgramError,
    UnsupportedOperationError,
)

_FLOAT_DTYPES = "`float32`, `float16`"
_DEFAULT_DTYPES = "`float32`, `int32`, `bool`"
_ANY_DTYPE = "any supported dtype"

_BINARY_OPERATORS = {
    "add": np.add,
    "sub": np.subtract,
    "subtract": np.subtract,
    "mul": np.multiply,
    "multiply": np.multiply,
    "div": np.true_divide,
    "truediv": np.true_divide,
    "floordiv": np.floor_divide,
    "mod": np.mod,
    "pow": np.power,
    "maximum": np.maximum,
    "max": np.maximum,
    "minimum": np.minimum,
    "min": np.minimum,
    "and": np.bitwise_and,
    "or": np.bitwise_or,
    "bitand": np.bitwise_and,
    "bitor": np.bitwise_or,
    "bitxor": np.bitwise_xor,
    "bitwise_and": np.bitwise_and,
    "bitwise_or": np.bitwise_or,
    "bitwise_xor": np.bitwise_xor,
    "shift_left": np.left_shift,
    "shift_right": np.right_shift,
    "bitwise_left_shift": np.left_shift,
    "bitwise_right_shift": np.right_shift,
}

_UNARY_OPERATORS = {
    "neg": np.negative,
    "pos": np.positive,
    "not": np.logical_not,
    "invert": np.invert,
}

_COMPARISONS = {
    "eq": np.equal,
    "ne": np.not_equal,
    "lt": np.less,
    "le": np.less_equal,
    "gt": np.greater,
    "ge": np.greater_equal,
}

_MATH_FUNCTIONS = {
    "abs": np.abs,
    "fabs": np.fabs,
    "acos": np.arccos,
    "acosh": np.arccosh,
    "asin": np.arcsin,
    "asinh": np.arcsinh,
    "atan": np.arctan,
    "atan2": np.arctan2,
    "atanh": np.arctanh,
    "ceil": np.ceil,
    "cos": np.cos,
    "cosh": np.cosh,
    "erf": math.erf,
    "erfc": math.erfc,
    "exp": np.exp,
    "exp2": np.exp2,
    "expm1": np.expm1,
    "floor": np.floor,
    "fmod": np.fmod,
    "hypot": np.hypot,
    "log": np.log,
    "log1p": np.log1p,
    "log2": np.log2,
    "log10": np.log10,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "pow": np.power,
    "remainder": np.remainder,
    "rint": np.rint,
    "round": np.round,
    "rsqrt": lambda value: 1.0 / np.sqrt(value),
    "sin": np.sin,
    "sinh": np.sinh,
    "sqrt": np.sqrt,
    "tan": np.tan,
    "tanh": np.tanh,
    "trunc": np.trunc,
}

_REDUCTIONS = {
    "sum": np.sum,
    "max": np.max,
    "min": np.min,
    "prod": np.prod,
    "mean": np.mean,
    "any": np.any,
    "all": np.all,
}

_CALL_FUNCTIONS = {
    "where": np.where,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "tanh": np.tanh,
}


@dataclass(frozen=True, kw_only=True)
class OperationSupport:
    """One entry of the interpreter's operation support matrix.

    :param opcode: The SSA opcode, or an opcode prefix ending with a dot.
    :param summary: What the interpreter does for the operation.
    :param dtypes: The dtypes the operation supports.
    :param notes: Additional limitations worth documenting.
    """

    opcode: str
    summary: str
    dtypes: str = _DEFAULT_DTYPES
    notes: str = ""


@dataclass(frozen=True, kw_only=True)
class OperationContext:
    """The execution context handed to one operation rule.

    :param operation: The SSA operation being executed.
    :param location: The SSA location of the operation.
    :param suffix: The opcode remainder for prefix rules, otherwise an empty string.
    :param executor: The executor running the program.
    :param state: The runtime state of the current program instance.
    """

    operation: Any
    location: str
    suffix: str
    executor: Any
    state: Any

    @property
    def opcode(self) -> str:
        """Return the opcode of the operation."""
        return self.operation.opcode

    def value(self, name: str):
        """Return an SSA operand value."""
        return self.state.lookup(name)

    def operand_values(self) -> tuple:
        """Return every operand value, in order."""
        return tuple(self.value(name) for name in self.operation.operands)

    def result_type(self, index: int = 0):
        """Return the declared SSA type of one result."""
        if index >= len(self.operation.results):
            return None

        return self.operation.results[index].type

    def bind(self, value, index: int = 0) -> None:
        """Bind one result of the operation in the SSA environment."""
        self.state.bind(self.operation.results[index].name, value)

    def unsupported(self, detail: str):
        """Raise an unsupported-operation error for this operation."""
        raise UnsupportedOperationError(self.opcode, self.location, detail)


_RULES: dict[str, tuple[OperationSupport, Callable]] = {}


def register(
    opcode: str,
    *,
    summary: str,
    dtypes: str = _DEFAULT_DTYPES,
    notes: str = "",
) -> Callable:
    """Register an operation rule.

    :param opcode: The exact opcode or an opcode prefix ending with a dot.
    :param summary: What the rule does, for the support matrix.
    :param dtypes: The dtypes the rule supports.
    :param notes: Additional limitations worth documenting.
    :return: The decorator that registers the handler.
    """

    def decorator(handler: Callable) -> Callable:
        _RULES[opcode] = (
            OperationSupport(
                opcode=opcode, summary=summary, dtypes=dtypes, notes=notes
            ),
            handler,
        )

        return handler

    return decorator


def resolve_rule(opcode: str):
    """Return the rule registered for an opcode.

    :param opcode: The SSA opcode.
    :return: A pair of the support entry and the handler, or ``None``.
    """
    if opcode in _RULES:
        support, handler = _RULES[opcode]

        return support, handler, ""

    best = None

    for prefix, (support, handler) in _RULES.items():
        if not prefix.endswith(".") or not opcode.startswith(prefix):
            continue

        if best is None or len(prefix) > len(best[0]):
            best = (prefix, support, handler)

    if best is None:
        return None

    prefix, support, handler = best

    return support, handler, opcode[len(prefix) :]


def supported_operations() -> tuple[OperationSupport, ...]:
    """Return the support matrix of the interpreter's operations."""
    return tuple(
        support
        for support, _ in sorted(_RULES.values(), key=lambda item: item[0].opcode)
    )


_UNSUPPORTED_REASONS = {
    "call.rand": "random number generation is not interpreted",
    "index.offset": "offset vectors are not interpreted",
    "linalg.matmul_transpose": "transposed matrix multiplication is not interpreted",
    "math.rand": "random number generation is not interpreted",
    "mem.atomic_add": "atomics are outside the scope of the CPU interpreter",
    "mem.data_ptr": "indirect memory access through pointers is not interpreted",
    "mem.load": "indirect memory access through pointers is not interpreted",
}


def unsupported_operations() -> tuple[tuple[str, str], ...]:
    """Return the operations the interpreter rejects, with the reason."""
    return tuple(sorted(_UNSUPPORTED_REASONS.items()))


def execute(context: OperationContext) -> None:
    """Execute one SSA operation.

    :param context: The execution context of the operation.
    :raises UnsupportedOperationError: When no rule handles the opcode.
    """
    rule = resolve_rule(context.operation.opcode)

    if rule is None:
        raise UnsupportedOperationError(
            context.operation.opcode,
            context.location,
            _UNSUPPORTED_REASONS.get(
                context.operation.opcode,
                "the CPU interpreter has no rule for this operation",
            ),
        )

    support, handler, suffix = rule
    del support

    handler(
        OperationContext(
            operation=context.operation,
            location=context.location,
            suffix=suffix,
            executor=context.executor,
            state=context.state,
        )
    )


def backend_dtype(value, context: OperationContext) -> Any:
    """Return the result of an operation with the backend's runtime dtype.

    ``ninetoothed`` records a declared dtype on most SSA results, but the Triton
    emitter drops it and lets the generated expression take the dtype of its
    operands. The frontend also types an integer literal as ``int64``, so forcing
    the declared dtype would truncate a `float32` operand. The interpreter
    therefore keeps NumPy's promoted dtype and leaves the explicit dtype changes
    to `tensor.cast`, the tensor allocations, and the matrix products.

    :param value: The NumPy result of the operation.
    :param context: The execution context of the operation.
    :return: The value with the dtype the backend would produce.
    """
    del context

    return value


def cast_to_type(value, type_) -> Any:
    """Cast an interpreter value to the dtype declared by an SSA type.

    :param value: The interpreter value.
    :param type_: The declared SSA type, or ``None``.
    :return: The value with the declared dtype, when it has a dtype.
    """
    dtype = resolve_dtype(None if type_ is None else type_.dtype)

    if dtype is None:
        return value

    if isinstance(value, np.ndarray):
        if value.dtype == dtype:
            return value

        return value.astype(dtype)

    if isinstance(value, (bool, int, float, np.generic)):
        return dtype.type(value)

    return value


def _apply(function, values, context):
    try:
        return function(*values)
    except (TypeError, ValueError) as exc:
        raise InvalidProgramError(
            f"Cannot execute `{context.opcode}` at `{context.location}`: {exc}."
        ) from exc


@register("arith.constant", summary="Bind the recorded constant.", dtypes=_ANY_DTYPE)
def _lower_constant(context: OperationContext) -> None:
    context.bind(context.operation.attrs.get("value"))


@register("arith.", summary="Apply the NumPy arithmetic operator.")
def _lower_arith(context: OperationContext) -> None:
    values = context.operand_values()

    if context.suffix in _UNARY_OPERATORS:
        result = _apply(_UNARY_OPERATORS[context.suffix], values, context)
    elif context.suffix in _BINARY_OPERATORS:
        result = _apply(_BINARY_OPERATORS[context.suffix], values, context)
    else:
        context.unsupported(f"`arith.{context.suffix}` has no NumPy equivalent")

    context.bind(backend_dtype(result, context))


@register("cmp.", summary="Apply the NumPy comparison operator.", dtypes=_ANY_DTYPE)
def _lower_comparison(context: OperationContext) -> None:
    if context.suffix not in _COMPARISONS:
        context.unsupported(f"`cmp.{context.suffix}` has no NumPy equivalent")

    values = context.operand_values()
    context.bind(_apply(_COMPARISONS[context.suffix], values, context).astype(np.bool_))


@register("math.", summary="Apply the NumPy equivalent of the math intrinsic.")
def _lower_math(context: OperationContext) -> None:
    if context.suffix not in _MATH_FUNCTIONS:
        context.unsupported(
            f"`math.{context.suffix}` has no NumPy equivalent in the interpreter"
        )

    values = context.operand_values()
    result = _apply(_MATH_FUNCTIONS[context.suffix], values, context)

    context.bind(backend_dtype(result, context))


@register("call.", summary="Apply the NumPy equivalent of the intrinsic call.")
def _lower_call(context: OperationContext) -> None:
    if context.suffix not in _CALL_FUNCTIONS:
        context.unsupported(f"`call.{context.suffix}` is not supported")

    values = context.operand_values()
    result = _apply(_CALL_FUNCTIONS[context.suffix], values, context)

    context.bind(backend_dtype(result, context))


@register(
    "reduce.",
    summary="Reduce the operand with the NumPy reduction counterpart.",
    notes="The `axis` attribute may be omitted for a full reduction.",
)
def _lower_reduction(context: OperationContext) -> None:
    if context.suffix not in _REDUCTIONS:
        context.unsupported(f"`reduce.{context.suffix}` is not supported")

    value = context.value(context.operation.operands[0])
    axis = context.operation.attrs.get("axis")
    function = _REDUCTIONS[context.suffix]

    if axis is None:
        result = _apply(function, (value,), context)
    else:
        result = _apply(function, (value, int(axis)), context)

    context.bind(backend_dtype(result, context))


@register("select.where", summary="Select between two operands.", dtypes=_ANY_DTYPE)
def _lower_select(context: OperationContext) -> None:
    condition, yes, no = context.operand_values()

    context.bind(np.where(condition, yes, no))


@register("tensor.zeros", summary="Allocate a zero-filled value.")
def _lower_zeros(context: OperationContext) -> None:
    dtype = resolve_dtype(context.operation.attrs.get("dtype"))
    shape = context.state.evaluate_shape(context.operation.attrs.get("shape"))

    context.bind(np.zeros(shape, dtype=dtype if dtype is not None else np.float32))


@register(
    "tensor.empty",
    summary="Allocate an uninitialized value.",
    notes="Filled with zeros; the interpreter has no undefined values.",
)
def _lower_empty(context: OperationContext) -> None:
    _lower_zeros(context)


@register("tensor.full", summary="Allocate a constant-filled value.")
def _lower_full(context: OperationContext) -> None:
    dtype = resolve_dtype(context.operation.attrs.get("dtype"))
    shape = context.state.evaluate_shape(context.operation.attrs.get("shape"))
    values = context.operand_values()
    fill = values[0] if values else context.operation.attrs.get("value", 0.0)

    context.bind(np.full(shape, fill, dtype=dtype if dtype is not None else np.float32))


@register("tensor.extract", summary="Index a tensor value.", dtypes=_ANY_DTYPE)
def _lower_extract(context: OperationContext) -> None:
    if context.operation.attrs.get("source"):
        context.unsupported("indexing the raw source view of a tensor")

    value = context.value(context.operation.operands[0])
    indices = tuple(
        _resolve_index(context, name) for name in context.operation.operands[1:]
    )

    try:
        result = value[indices]
    except (IndexError, TypeError) as exc:
        raise InvalidProgramError(
            f"Cannot index an SSA value at `{context.location}`: {exc}."
        ) from exc

    context.bind(backend_dtype(result, context))


@register(
    "tensor.cast", summary="Cast a value to the recorded dtype.", dtypes=_ANY_DTYPE
)
def _lower_cast(context: OperationContext) -> None:
    value = context.value(context.operation.operands[0])
    dtype = resolve_dtype(context.operation.attrs.get("dtype"))

    if dtype is None:
        context.bind(value)

        return

    context.bind(np.asarray(value).astype(dtype))


@register(
    "tensor.view",
    summary="Reshape a tensor value to the recorded view.",
    dtypes=_ANY_DTYPE,
)
def _lower_view(context: OperationContext) -> None:
    value = context.value(context.operation.operands[0])
    shape = context.state.evaluate_shape(context.result_type().shape)

    try:
        context.bind(np.reshape(value, shape))
    except ValueError as exc:
        raise InvalidProgramError(
            f"Cannot view the value at `{context.location}` as {shape}: {exc}."
        ) from exc


@register("shape.dim", summary="Resolve one dimension of a shape.", dtypes=_ANY_DTYPE)
def _lower_shape_dim(context: OperationContext) -> None:
    name = context.operation.operands[0]
    dim = int(context.operation.attrs.get("dim", 0))

    if context.operation.attrs.get("source"):
        shape = context.state.source_shape(name)
    else:
        shape = np.shape(context.value(name))

    if dim >= len(shape) or dim < -len(shape):
        raise InvalidProgramError(
            f"Dimension {dim} is out of range for the value at `{context.location}`."
        )

    context.bind(int(shape[dim]))


@register("tensor.stride", summary="Resolve one stride of a shape.", dtypes=_ANY_DTYPE)
def _lower_stride(context: OperationContext) -> None:
    name = context.operation.operands[0]
    dim = int(context.operation.attrs.get("dim", 0))

    if context.operation.attrs.get("source"):
        shape = context.state.source_shape(name)
    else:
        shape = np.shape(context.value(name))

    strides = _default_strides(shape)

    if dim >= len(strides) or dim < -len(strides):
        raise InvalidProgramError(
            f"Stride {dim} is out of range for the value at `{context.location}`."
        )

    context.bind(int(strides[dim]))


@register(
    "symbol.attr", summary="Resolve a symbolic layout expression.", dtypes=_ANY_DTYPE
)
def _lower_symbol(context: OperationContext) -> None:
    context.bind(context.state.evaluate(context.operation.attrs.get("expr", "0")))


@register("tuple.construct", summary="Build a tuple value.", dtypes=_ANY_DTYPE)
def _lower_tuple(context: OperationContext) -> None:
    context.bind(tuple(context.operand_values()))


@register("mem.store", summary="Store a value through its arrangement mask.")
def _lower_store(context: OperationContext) -> None:
    if context.operation.attrs.get("subscript") is not None:
        context.unsupported("storing into a subscripted view of a tensor")

    value = context.value(context.operation.operands[0])
    target = context.operation.operands[1]

    context.executor.store(context, target, value)


@register("linalg.dot", summary="Multiply two 2-D tiles.", dtypes=_FLOAT_DTYPES)
def _lower_dot(context: OperationContext) -> None:
    left, right = context.operand_values()
    context.bind(_dot(left, right, context))


@register(
    "linalg.matmul",
    summary="Multiply two 2-D tiles.",
    dtypes=_FLOAT_DTYPES,
    notes="The result is accumulated in `float32` and cast like `linalg.dot`.",
)
def _lower_matmul(context: OperationContext) -> None:
    _lower_dot(context)


def _dot(left, right, context: OperationContext):
    left = np.asarray(left)
    right = np.asarray(right)

    if left.ndim != 2 or right.ndim != 2:
        context.unsupported(
            f"dot operands must be 2-D, got {left.ndim}-D and {right.ndim}-D"
        )

    narrow = min(left.dtype, right.dtype, key=lambda dtype: dtype.itemsize)

    if np.issubdtype(narrow, np.floating) and narrow.itemsize < 4:
        accumulated = np.matmul(left.astype(np.float32), right.astype(np.float32))
        result = accumulated.astype(narrow)
    else:
        result = np.matmul(left, right)

    return cast_to_type(result, context.result_type())


def _resolve_index(context: OperationContext, name: str):
    value = context.value(name)

    if isinstance(value, np.ndarray) and value.ndim > 0:
        return value

    return int(value)


def _default_strides(shape) -> tuple:
    shape = tuple(int(dim) for dim in shape)
    strides = [1] * len(shape)

    for dim in range(len(shape) - 2, -1, -1):
        strides[dim] = strides[dim + 1] * shape[dim + 1]

    return tuple(strides)


@register("scf.for", summary="Execute a counted loop.", dtypes=_ANY_DTYPE)
def _lower_for(context: OperationContext) -> None:
    from ninetoothed.interpreter.executor import run_for

    run_for(context.executor, context)


@register("scf.if", summary="Execute one conditional region.", dtypes=_ANY_DTYPE)
def _lower_if(context: OperationContext) -> None:
    from ninetoothed.interpreter.executor import run_if

    run_if(context.executor, context)


@register("scf.yield", summary="Yield the operands of a region.", dtypes=_ANY_DTYPE)
def _lower_yield(context: OperationContext) -> None:
    context.unsupported("`scf.yield` must be the last operation of an SSA region")


@register(
    "linalg.transpose",
    summary="Swap the last two dimensions of a tile.",
    dtypes=_FLOAT_DTYPES,
)
def _lower_transpose(context: OperationContext) -> None:
    value = np.asarray(context.value(context.operation.operands[0]))

    if value.ndim < 2:
        context.unsupported("a transpose needs a value with at least two dimensions")

    context.bind(np.swapaxes(value, -1, -2))


__all__ = [
    "OperationContext",
    "OperationSupport",
    "backend_dtype",
    "cast_to_type",
    "execute",
    "register",
    "resolve_rule",
    "supported_operations",
    "unsupported_operations",
]
