"""Shared helpers for interpreter operation handlers."""

import numpy as np

from ..errors import UnsupportedOperationError
from ..memory import View
from ..values import (
    ARRAY,
    SCALAR,
    VIEW,
    Pointer,
    Value,
    array,
    materialize,
    pointer,
    scalar,
    tuple_,
    view,
)


def operand(state, operation, index):
    """Return one operand value of ``operation``.

    :param state: The interpreter state.
    :param operation: The SSA operation.
    :param index: The operand position.
    :return: The :class:`~ninetoothed.interpret.values.Value`.
    """
    try:
        name = operation.operands[index]
    except IndexError as exc:
        raise UnsupportedOperationError(
            f"Operation `{operation.opcode}` expects at least {index + 1} operand(s).",
            opcode=operation.opcode,
            location=state.location,
        ) from exc

    return state.value(name)


def operands(state, operation):
    """Return every operand value of ``operation``."""
    return tuple(state.value(name) for name in operation.operands)


def numbers(state, operation):
    """Materialize every operand of ``operation`` as a NumPy array."""
    return tuple(
        materialize(value, state.context) for value in operands(state, operation)
    )


def build_value(type_, data):
    """Wrap raw data into a runtime value matching ``type_``."""
    if isinstance(data, View):
        return view(type_, data)

    if isinstance(data, Pointer):
        return pointer(type_, data)

    if isinstance(data, tuple):
        return tuple_(type_, data)

    array_ = np.asarray(data)

    if array_.ndim == 0:
        return scalar(type_, array_[()])

    return array(type_, array_)


def bind(state, operation, *values):
    """Bind the results of ``operation``.

    :param state: The interpreter state.
    :param operation: The SSA operation.
    :param values: One runtime value per SSA result.
    """
    results = operation.results

    if len(values) != len(results):
        raise UnsupportedOperationError(
            f"Operation `{operation.opcode}` produces {len(results)} result(s) but "
            f"the handler produced {len(values)}.",
            opcode=operation.opcode,
            location=state.location,
        )

    for result, value in zip(results, values):
        state.values[result.name] = _coerce(result.type, value)


def bind_data(state, operation, *datas):
    """Bind the results of ``operation`` from raw data."""
    bind(
        state,
        operation,
        *(
            build_value(result.type, data)
            for result, data in zip(operation.results, datas)
        ),
    )


def _coerce(type_, value):
    if isinstance(value, Value):
        if value.type is type_:
            return value

        return Value(type_, value.kind, value.data)

    return build_value(type_, value)


def result_dtype(operation, index=0, fallback="float32"):
    """Return the NineToothed dtype name of a result."""
    try:
        return operation.results[index].type.dtype or fallback
    except IndexError:  # pragma: no cover - defensive
        return fallback


def is_scalar_value(value):
    """Return whether a runtime value is a scalar."""
    return value.kind == SCALAR


def as_index(value):
    """Return the integer payload of a scalar value."""
    if value.kind == ARRAY and value.data.ndim == 0:
        return int(value.data)

    if value.kind == SCALAR:
        return int(np.asarray(value.data))

    raise UnsupportedOperationError(f"Cannot use {value!r} as a scalar index.")


__all__ = [
    "ARRAY",
    "SCALAR",
    "VIEW",
    "as_index",
    "bind",
    "bind_data",
    "build_value",
    "is_scalar_value",
    "numbers",
    "operand",
    "operands",
    "result_dtype",
]
