"""Numeric evaluation of the existing structured layout expression IR."""

import operator
from functools import lru_cache, partial

import numpy as np

from ninetoothed.ir import IndexExpr

BINARY = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": np.true_divide,
    "truediv": np.true_divide,
    "floordiv": np.floor_divide,
    "mod": np.remainder,
    "pow": np.power,
    "and": np.logical_and,
    "or": np.logical_or,
    "bitand": np.bitwise_and,
    "bitor": np.bitwise_or,
    "bitxor": np.bitwise_xor,
    "eq": operator.eq,
    "ne": operator.ne,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
}

UNARY = {
    "neg": operator.neg,
    "pos": operator.pos,
    "invert": np.invert,
    "not": np.logical_not,
}


def evaluate(expression, symbols):
    """Evaluate a trusted IndexExpr using explicit supported operations only."""
    expression = IndexExpr.parse(expression)

    if expression.op == "constant":
        return expression.value

    if expression.op == "symbol":
        return _symbol(expression.value, symbols)

    try:
        evaluator = _compiled_expression(_ExpressionIdentity(expression))
    except TypeError:
        # Public IndexExpr constants can be unhashable; retain their semantics.
        return _evaluate(expression, symbols)
    return evaluator(symbols)


class _ExpressionIdentity:
    __slots__ = ("expression",)

    def __init__(self, expression):
        self.expression = expression

    def __hash__(self):
        return id(self.expression)

    def __eq__(self, other):
        return (
            isinstance(other, _ExpressionIdentity)
            and self.expression is other.expression
        )


def _constant(value, symbols):
    return value


def _symbol(name, symbols):
    try:
        return symbols[name]
    except KeyError as exc:
        raise ValueError(f"Unbound layout symbol `{name}`.") from exc


def _binary(function, left, right, symbols):
    return function(left(symbols), right(symbols))


def _unary(function, operand, symbols):
    return function(operand(symbols))


def _next_power_of_2(value):
    """Round an integer-normalized extent up; nonpositive extents stay empty."""
    extent = int(value)

    return 0 if extent <= 0 else 1 << (extent - 1).bit_length()


@lru_cache(maxsize=256)
def _compiled_expression(identity):
    """Cache only a bounded evaluation plan, never symbol values or arrays."""
    expression = identity.expression
    # Identity preserves literal types: structural equality equates True, 1 and
    # 1.0. The hashability check rejects array/list constants before retention.
    hash(expression)

    return _build_expression(expression)


def _build_expression(expression):
    if expression.op == "constant":
        return partial(_constant, expression.value)

    if expression.op == "symbol":
        return partial(_symbol, expression.value)

    if not all(isinstance(value, IndexExpr) for value in expression.operands):
        return partial(_evaluate, expression)

    if expression.op in BINARY and len(expression.operands) == 2:
        left, right = map(_build_expression, expression.operands)

        return partial(_binary, BINARY[expression.op], left, right)

    if expression.op in UNARY and len(expression.operands) == 1:
        return partial(
            _unary, UNARY[expression.op], _build_expression(expression.operands[0])
        )
    return partial(_evaluate, expression)


def _evaluate(expression, symbols):
    expression = IndexExpr.parse(expression)
    op = expression.op

    if op == "constant":
        return expression.value

    if op == "symbol":
        try:
            return symbols[expression.value]
        except KeyError as exc:
            raise ValueError(f"Unbound layout symbol `{expression.value}`.") from exc

    if op == "attribute":
        # Dotted function names are stored in call.value, not evaluated as objects.
        name = expression.render()

        if name in symbols:
            return symbols[name]

        raise ValueError(f"Unsupported layout attribute `{name}`.")

    values = tuple(_evaluate(value, symbols) for value in expression.operands)

    if op in BINARY:
        return BINARY[op](*values)

    if op in UNARY:
        return UNARY[op](*values)

    if op == "tuple":
        return values

    if op == "subscript":
        return values[0][values[1]]

    if op == "call":
        name = str(expression.value).rsplit(".", 1)[-1]
        functions = {
            "ceil": np.ceil,
            "floor": np.floor,
            "abs": np.abs,
            "Min": np.minimum,
            "Max": np.maximum,
            "min": np.minimum,
            "max": np.maximum,
            "minimum": np.minimum,
            "maximum": np.maximum,
            "ceiling": np.ceil,
            "cdiv": lambda x, y: -(-x // y),
            "next_power_of_2": _next_power_of_2,
            "int": int,
        }

        if name in functions:
            return functions[name](*values)

    raise ValueError(f"Unsupported layout expression `{expression.render()}`.")


def shape_value(shape, symbols):
    """Resolve a symbolic shape, requiring nonnegative integer extents."""
    result = []

    for dimension in shape:
        value = evaluate(dimension, symbols)

        if np.ndim(value) != 0 or int(value) != value or int(value) < 0:
            raise ValueError(f"Invalid shape extent `{value}`.")

        result.append(int(value))
    return tuple(result)


def numpy_dtype(dtype, fallback=None):
    """Normalize NineToothed and backend dtype spellings."""
    if dtype is None:
        return None if fallback is None else np.dtype(fallback)

    name = str(dtype).rsplit(".", 1)[-1]

    if name in {"symbol", "none"}:
        return None

    try:
        result = _numpy_dtype_from_name(name)
    except TypeError as exc:
        raise ValueError(f"Unsupported interpreter dtype `{dtype}`.") from exc

    if result is None:
        raise ValueError(f"Unsupported interpreter dtype `{dtype}`.")
    return result


@lru_cache(maxsize=64)
def _numpy_dtype_from_name(name):
    """Reuse immutable dtype descriptors without retaining user objects."""
    names = {
        "fp16": "float16",
        "fp32": "float32",
        "fp64": "float64",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
        "u8": "uint8",
        "u16": "uint16",
        "u32": "uint32",
        "u64": "uint64",
        "index": "int64",
        "i1": "bool",
    }

    result = np.dtype(names.get(name, name))

    if result.name not in {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
    }:
        return None
    return result
