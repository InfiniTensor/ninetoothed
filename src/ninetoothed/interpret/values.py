"""Runtime values produced while interpreting an ``ssa.Program``.

The interpreter uses a single value class with a tagged payload instead of one
class per SSA type.  This keeps operation handlers uniform: they call
:func:`materialize` when they need numbers and :attr:`Value.data` when they need
to inspect the representation (for example to find the destination view of a
``mem.store``).
"""

import numpy as np

from .dtypes import resolve_dtype
from .errors import UnsupportedDTypeError
from .expr import Expression

#: Payload kinds carried by :class:`Value`.
SCALAR = "scalar"
ARRAY = "array"
VIEW = "view"
POINTER = "pointer"
TUPLE = "tuple"
SYMBOL = "symbol"


class Pointer:
    """A pointer into a CPU buffer.

    :param tensor: The tensor runtime owning the buffer.
    :param offset: The element offset from the start of the buffer.
    """

    __slots__ = ("offset", "tensor")

    def __init__(self, tensor, offset=0):
        self.tensor = tensor
        self.offset = offset

    def __repr__(self):
        return f"Pointer({self.tensor.name}, {self.offset})"


class Value:
    """A runtime SSA value.

    :param type_: The (specialized) :class:`ninetoothed.ir.ssa.Type` of the value.
    :param kind: One of the payload kinds declared in this module.
    :param data: The payload.
    """

    __slots__ = ("data", "kind", "type")

    def __init__(self, type_, kind, data):
        self.type = type_
        self.kind = kind
        self.data = data

    @property
    def dtype(self):
        """Return the NineToothed dtype name of the value."""
        return self.type.dtype

    @property
    def numpy_dtype(self):
        """Return the NumPy dtype used to materialize the value."""
        if self.kind == ARRAY:
            return self.data.dtype

        if self.kind == SCALAR:
            return np.asarray(self.data).dtype

        if self.kind == POINTER:
            return self.data.tensor.numpy_dtype

        return resolve_dtype(self.dtype)

    @property
    def shape(self):
        """Return the logical shape of the value."""
        if self.kind == ARRAY:
            return tuple(self.data.shape)

        if self.kind == SCALAR:
            return ()

        if self.kind == VIEW:
            return tuple(self.data.shape)

        if self.kind == POINTER:
            return ()

        if self.kind == TUPLE:
            return (len(self.data),)

        return ()

    @property
    def is_tensor(self):
        """Return whether the value carries element data."""
        return self.kind in {ARRAY, VIEW}

    def __repr__(self):
        if self.kind == VIEW:
            return f"Value(view<{self.data.shape}>, {self.dtype})"

        if self.kind == ARRAY:
            return f"Value(array<{self.data.shape}>, {self.data.dtype})"

        if self.kind == SCALAR:
            return f"Value(scalar, {self.dtype}={self.data!r})"

        if self.kind == POINTER:
            return f"Value(pointer, {self.data!r})"
        return f"Value({self.kind}, {self.data!r})"


def scalar(type_, value):
    """Build a scalar value."""
    return Value(type_, SCALAR, value)


def array(type_, data):
    """Build an eager array value."""
    return Value(type_, ARRAY, data)


def view(type_, view_):
    """Build a lazy view value."""
    return Value(type_, VIEW, view_)


def pointer(type_, pointer_):
    """Build a pointer value."""
    return Value(type_, POINTER, pointer_)


def tuple_(type_, items):
    """Build a tuple value."""
    return Value(type_, TUPLE, tuple(items))


def materialize(value, context=None):
    """Return a NumPy array (or scalar) holding the numeric content of ``value``.

    :param value: The runtime value to materialize.
    :param context: A :class:`~ninetoothed.interpret.memory.AccessContext`.
    :return: A ``numpy.ndarray`` for tensors, or a NumPy scalar for scalars.
    """
    if value.kind == ARRAY:
        return value.data

    if value.kind == SCALAR:
        return np.asarray(value.data)

    if value.kind == VIEW:
        if context is None:
            raise UnsupportedDTypeError(
                "Materializing a lazy tensor view requires an access context."
            )

        return context.read(value.data)

    if value.kind == POINTER:
        return np.asarray(value.data.offset)

    raise UnsupportedDTypeError(f"Cannot materialize a `{value.kind}` value.")


def is_symbolic(value):
    """Return whether the value is an unresolved symbolic expression."""
    return value.kind == SYMBOL


def symbolic(type_, expression):
    """Build a value that wraps an unevaluated symbolic expression."""
    return Value(type_, SYMBOL, expression)


__all__ = [
    "ARRAY",
    "POINTER",
    "SCALAR",
    "SYMBOL",
    "TUPLE",
    "VIEW",
    "Pointer",
    "Value",
    "array",
    "is_symbolic",
    "materialize",
    "pointer",
    "scalar",
    "symbolic",
    "tuple_",
    "view",
    "Expression",
]
