"""Memory operations: ``mem.data_ptr``, ``mem.load``, ``mem.store``."""

import numpy as np

from ..errors import UnsupportedAccessError, UnsupportedOperationError
from ..memory import View, parse_subscript
from ..registry import register
from ..values import Pointer, materialize
from .common import as_index, bind, bind_data, operand


@register(
    "mem.data_ptr",
    category="mem",
    summary="Base pointer of a tensor view.",
)
def _handle_data_ptr(state, operation):
    value = operand(state, operation, 0)

    if value.kind != "view":
        raise UnsupportedAccessError(
            "The `mem.data_ptr` opcode requires a tensor view.",
            opcode=operation.opcode,
            location=state.location,
        )

    bind(state, operation, Pointer(value.data.tensor, 0))


@register(
    "mem.load",
    category="mem",
    summary="Load consecutive elements through a pointer.",
)
def _handle_load(state, operation):
    value = operand(state, operation, 0)

    if value.kind != "pointer":
        raise UnsupportedAccessError(
            "The `mem.load` opcode requires a pointer operand.",
            opcode=operation.opcode,
            location=state.location,
        )

    tensor = value.data.tensor
    offset = int(value.data.offset)
    result_type = operation.results[0].type
    shape = tuple(state.resolve_shape(result_type.shape))
    size = tensor.buffer.size

    if result_type.kind == "tensor" and shape:
        count = int(np.prod(shape)) if shape else 1
        indices = offset + np.arange(count)
        mask = (indices >= 0) & (indices < size)
        gathered = np.where(mask, tensor.buffer[np.where(mask, indices, 0)], 0)
        other = tensor.other if tensor.other is not None else 0
        data = np.where(mask, gathered, other).reshape(shape)
    else:
        if offset < 0 or offset >= size:
            raise UnsupportedAccessError(
                f"The `mem.load` opcode reads offset {offset} outside a buffer "
                f"of {size} element(s) in tensor `{tensor.name}`.",
                opcode=operation.opcode,
                location=state.location,
            )

        data = tensor.buffer[offset]

    bind_data(state, operation, data)


@register(
    "mem.store",
    category="mem",
    summary="Store a value into a tensor view honouring the layout mask.",
)
def _handle_store(state, operation):
    value = materialize(operand(state, operation, 0), state.context)
    target = operand(state, operation, 1)

    if operation.attrs.get("source"):
        raise UnsupportedAccessError(
            "Storing through `.source[...]` is not supported by the CPU interpreter.",
            opcode=operation.opcode,
            location=state.location,
        )

    if target.kind != "view":
        raise UnsupportedAccessError(
            "The `mem.store` opcode requires a tensor view destination.",
            opcode=operation.opcode,
            location=state.location,
        )

    view = target.data
    indices = operation.attrs.get("indices") or ()

    if indices:
        kinds = parse_subscript(operation.attrs.get("subscript"))
        coords = [as_index(state.value(name)) for name in indices]
        view = view.index(
            kinds, coords, opcode=operation.opcode, location=state.location
        )

    if not isinstance(view, View):  # pragma: no cover - defensive
        raise UnsupportedOperationError(
            "The `mem.store` opcode resolved to a non-view destination.",
            opcode=operation.opcode,
            location=state.location,
        )

    state.context.write(view, value)


__all__ = []
