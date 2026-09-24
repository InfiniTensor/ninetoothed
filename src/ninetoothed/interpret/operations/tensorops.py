"""Tensor, shape, index, and symbol operations."""

import numpy as np

from ninetoothed.dtype import normalize_dtype

from ..dtypes import cast_array, resolve_dtype
from ..errors import UnsupportedAccessError, UnsupportedOperationError
from ..memory import INTEGER, NEW_AXIS, SLICE, View, parse_subscript
from ..registry import register
from ..values import materialize
from .common import as_index, bind, bind_data, operand, result_dtype


def _resolve_shape(state, operation, fallback):
    """Return the concrete result shape of a tensor constructor."""
    shape = operation.attrs.get("shape")

    if shape:
        try:
            resolved = state.evaluate_attribute(shape)

            if resolved is not None:
                return tuple(int(dim) for dim in resolved)
        except Exception:  # noqa: BLE001 - fall back to the SSA result type
            pass

    return tuple(state.resolve_shape(operation.results[0].type.shape))


def _make_filled(state, operation, fill):
    shape = _resolve_shape(state, operation, ())
    dtype = result_dtype(operation, fallback="float32")
    bind_data(state, operation, np.full(shape, fill, dtype=state.numpy_dtype(dtype)))


@register(
    "tensor.zeros",
    category="tensor",
    summary="Create a zero-filled tensor of a compile-time shape.",
)
def _handle_zeros(state, operation):
    _make_filled(state, operation, 0)


@register(
    "tensor.empty",
    category="tensor",
    summary="Create an uninitialized tensor (modelled as zeros).",
)
def _handle_empty(state, operation):
    _make_filled(state, operation, 0)


@register(
    "tensor.full",
    category="tensor",
    summary="Create a tensor filled with a constant or a broadcast scalar.",
)
def _handle_full(state, operation):
    if operation.operands:
        fill = materialize(operand(state, operation, 0), state.context)

        if np.asarray(fill).size != 1:
            raise UnsupportedOperationError(
                "The `tensor.full` opcode expects a scalar fill value.",
                opcode=operation.opcode,
                location=state.location,
            )

        fill = np.asarray(fill).reshape(()).item()
    else:
        fill = operation.attrs.get("value", 0)

    if isinstance(fill, str):
        fill = {"inf": np.inf, "-inf": -np.inf}.get(fill)

    _make_filled(state, operation, fill)


def _subscript_kinds(operation, state):
    text = operation.attrs.get("subscript")

    if text is None:
        return ()

    return parse_subscript(text)


def _apply_subscript(state, operation, base, kinds, coords):
    if base.kind == "view":
        return base.data.index(
            kinds, coords, opcode=operation.opcode, location=state.location
        )

    if base.kind == "array":
        return base.data[_numpy_index(kinds, coords)]

    raise UnsupportedAccessError(
        f"Cannot subscript a `{base.kind}` value.",
        opcode=operation.opcode,
        location=state.location,
    )


def _numpy_index(kinds, coords):
    index = []
    cursor = 0

    for kind in kinds:
        if kind is NEW_AXIS:
            index.append(None)
        elif kind is SLICE:
            index.append(slice(None))
        else:
            index.append(int(coords[cursor]))
            cursor += 1

    return tuple(index)


@register(
    "tensor.extract",
    category="tensor",
    summary="Index a tensor view by one or more integer coordinates.",
)
def _handle_extract(state, operation):
    base = operand(state, operation, 0)
    coords = [
        as_index(operand(state, operation, position + 1))
        for position in range(len(operation.operands) - 1)
    ]
    result = _apply_subscript(
        state, operation, base, _subscript_kinds(operation, state), coords
    )

    if isinstance(result, View):
        bind(state, operation, result)
    else:
        bind_data(state, operation, result)


@register(
    "tensor.view",
    category="tensor",
    summary="Re-view a tensor (new axes and full slices).",
)
def _handle_view(state, operation):
    base = operand(state, operation, 0)
    kinds = _subscript_kinds(operation, state)

    for kind in kinds:
        if kind is INTEGER:
            raise UnsupportedAccessError(
                "The `tensor.view` opcode must not contain integer indices.",
                opcode=operation.opcode,
                location=state.location,
            )

    result = _apply_subscript(state, operation, base, kinds, ())

    if isinstance(result, View):
        bind(state, operation, result)
    else:
        bind_data(state, operation, result)


@register(
    "tensor.cast",
    category="tensor",
    summary="Cast a tensor to another dtype (C-style truncation, `!= 0` for bool).",
)
def _handle_cast(state, operation):
    value = operand(state, operation, 0)
    dtype = result_dtype(operation, fallback=None)

    if dtype in {None, "none", "symbol", "dtype"}:
        # The SSA result type did not resolve the cast target; fall back to the
        # textual `dtype` attribute recorded by the frontend.
        dtype = operation.attrs.get("dtype")

    if isinstance(dtype, str):
        dtype = normalize_dtype(dtype)

    if value.kind == "view":
        # Casting a view yields a materialized array; the CPU interpreter has no
        # typed views, so the cast is applied eagerly.
        source = state.context.read(value.data)

        bind_data(state, operation, cast_array(source, resolve_dtype(dtype)))

        return

    bind_data(
        state,
        operation,
        cast_array(materialize(value, state.context), resolve_dtype(dtype)),
    )


@register(
    "shape.dim",
    category="shape",
    summary="Static extent of one tensor dimension.",
)
def _handle_shape_dim(state, operation):
    value = operand(state, operation, 0)
    dim = operation.attrs.get("dim")
    source = bool(operation.attrs.get("source"))

    if source:
        if value.kind != "view":
            raise UnsupportedAccessError(
                "The `shape.dim` opcode with `source=True` requires a tensor view.",
                opcode=operation.opcode,
                location=state.location,
            )

        shape = value.data.tensor.source_shape
    elif value.kind == "view":
        shape = value.data.tensor.levels[value.data.level]
    else:
        shape = value.shape

    index = int(dim)

    if index < 0:
        index += len(shape)

    if index < 0 or index >= len(shape):
        raise UnsupportedAccessError(
            f"Dimension {dim} is out of range for a rank-{len(shape)} tensor.",
            opcode=operation.opcode,
            location=state.location,
        )

    bind_data(state, operation, np.asarray(int(shape[index]), dtype=np.int64))


@register(
    "tensor.stride",
    category="tensor",
    summary="Stride of one tensor dimension.",
)
def _handle_stride(state, operation):
    value = operand(state, operation, 0)
    dim = int(operation.attrs.get("dim", 0))
    source = bool(operation.attrs.get("source"))

    if value.kind == "view":
        tensor = value.data.tensor

        if source:
            strides = tensor.source_strides
        else:
            strides = _default_strides(tensor.levels[value.data.level])
    else:
        strides = _default_strides(value.shape)

    if dim < 0:
        dim += len(strides)

    if dim < 0 or dim >= len(strides):
        raise UnsupportedAccessError(
            f"Stride dimension {dim} is out of range.",
            opcode=operation.opcode,
            location=state.location,
        )

    bind_data(state, operation, np.asarray(int(strides[dim]), dtype=np.int64))


def _default_strides(shape):
    strides = [1]

    for size in reversed(tuple(shape)[1:]):
        strides.append(int(size) * strides[-1])

    return tuple(reversed(strides))


@register(
    "index.offset",
    category="index",
    summary="Source offset of a view along one source dimension.",
)
def _handle_offset(state, operation):
    value = operand(state, operation, 0)

    if value.kind != "view":
        raise UnsupportedAccessError(
            "The `index.offset` opcode requires a tensor view.",
            opcode=operation.opcode,
            location=state.location,
        )

    view = value.data
    template = view.tensor.template

    if template is None:
        raise UnsupportedAccessError(
            f"Tensor `{view.tensor.name}` has no access template.",
            opcode=operation.opcode,
            location=state.location,
        )

    dim = operation.attrs.get("dim")
    dim = 0 if dim is None else int(dim)
    offsets = template.offsets

    if dim < 0:
        dim += len(offsets)

    if dim < 0 or dim >= len(offsets):
        raise UnsupportedAccessError(
            f"Offset dimension {operation.attrs.get('dim')} is out of range for "
            f"tensor `{view.tensor.name}`.",
            opcode=operation.opcode,
            location=state.location,
        )

    namespace = state.context.namespace(view)
    result = np.asarray(offsets[dim].evaluate(namespace))
    expected = tuple(state.resolve_shape(operation.results[0].type.shape))

    if result.ndim == 0:
        bind_data(state, operation, np.asarray(result, dtype=np.int64))

        return

    if expected and result.size == int(np.prod(expected)):
        result = result.reshape(expected)

    bind_data(state, operation, result.astype(np.int64, copy=False))


@register(
    "symbol.attr",
    category="symbol",
    summary="Resolve a symbolic attribute reference against runtime symbols.",
)
def _handle_symbol(state, operation):
    text = operation.attrs.get("expr")

    if text is None:
        raise UnsupportedOperationError(
            "The `symbol.attr` opcode requires an `expr` attribute.",
            opcode=operation.opcode,
            location=state.location,
        )

    resolved = state.evaluate_attribute(text)

    if resolved is None:
        raise UnsupportedOperationError(
            f"Cannot resolve symbolic expression `{text}`.",
            opcode=operation.opcode,
            location=state.location,
        )

    bind_data(state, operation, np.asarray(resolved))


@register("tuple.construct", category="tuple", summary="Construct a tuple value.")
def _handle_tuple(state, operation):
    values = tuple(state.value(name) for name in operation.operands)
    bind(state, operation, tuple(values))


__all__ = []
