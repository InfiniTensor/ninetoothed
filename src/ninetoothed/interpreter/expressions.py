"""Numeric evaluation of the existing structured layout expression IR."""

import operator

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

    values = tuple(evaluate(value, symbols) for value in expression.operands)

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
            "next_power_of_2": lambda x: 1 << (int(x) - 1).bit_length(),
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

    if name in {"symbol", "none"}:
        return None

    try:
        result = np.dtype(names.get(name, name))
    except TypeError as exc:
        raise ValueError(f"Unsupported interpreter dtype `{dtype}`.") from exc

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
        raise ValueError(f"Unsupported interpreter dtype `{dtype}`.")
    return result
