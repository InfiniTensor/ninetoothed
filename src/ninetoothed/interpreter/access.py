"""Program-instance memory access mapping derived from arranged tensor views.

The mapping reuses :func:`ninetoothed.eval._eval`, which evaluates an arranged
``Tensor`` into the flat offsets of its source storage and marks masked elements
with ``-1``. The leading dimensions of the evaluated view are the program
instances; the remaining dimensions are the value an application observes.
"""

import numpy as np

from ninetoothed.eval import _eval
from ninetoothed.interpreter.errors import UnsupportedAccessError

_MASKED_OFFSET = -1


class TensorAccess:
    """The offsets of one tensor parameter for every program instance.

    :param name: The application parameter name.
    :param offsets: The flat offsets, shaped ``(program instances, *value shape)``.
    :param program_shape: The shape of the program-instance domain.
    :param value_shape: The shape of the value an application observes.
    :param other: The fill value used for masked elements.
    """

    def __init__(self, *, name, offsets, program_shape, value_shape, other=None):
        self.name = name
        self.offsets = offsets
        self.program_shape = tuple(program_shape)
        self.value_shape = tuple(value_shape)
        self.other = other

    @property
    def num_programs(self) -> int:
        """Return the number of program instances the mapping covers."""
        return int(self.offsets.shape[0]) if self.offsets.ndim else 1

    def window(self, program_id: int):
        """Return the offsets of one program instance.

        :param program_id: The linearized program instance index.
        :return: The offsets and the access mask of the program instance.
        """
        if program_id < 0 or program_id >= self.num_programs:
            raise UnsupportedAccessError(
                f"Tensor `{self.name}` is not mapped for program instance "
                f"{program_id}; the arrangement covers "
                f"{self.num_programs} program instances."
            )

        offsets = self.offsets[program_id]
        mask = offsets != _MASKED_OFFSET

        return offsets, mask


def derive_access(name, arranged, subs, *, other=None) -> TensorAccess:
    """Derive the per-program-instance access mapping of an arranged tensor.

    :param name: The application parameter name.
    :param arranged: The arranged ``Tensor`` view produced by the arrangement.
    :param subs: The substitution table resolving source shapes and symbols.
    :param other: The fill value used for masked elements.
    :return: The access mapping.
    :raises UnsupportedAccessError: When the arrangement cannot be evaluated.
    """
    if getattr(arranged.source, "jagged_dim", None) is not None:
        raise UnsupportedAccessError(
            f"Tensor `{name}` is jagged; jagged tensors are not supported by the "
            "CPU interpreter."
        )

    try:
        evaluated = _eval(arranged, subs)
    except Exception as exc:
        raise UnsupportedAccessError(
            f"Cannot evaluate the arrangement of tensor `{name}` on the CPU: {exc}."
        ) from exc

    evaluated = np.asarray(evaluated, dtype=np.intp)
    program_ndim = _program_ndim(arranged)
    program_shape = tuple(int(dim) for dim in evaluated.shape[:program_ndim])
    value_shape = tuple(int(dim) for dim in evaluated.shape[program_ndim:])
    offsets = evaluated.reshape(-1, *value_shape)

    return TensorAccess(
        name=name,
        offsets=offsets,
        program_shape=program_shape,
        value_shape=value_shape,
        other=other,
    )


def _program_ndim(arranged) -> int:
    """Return how many leading view dimensions select program instances."""
    if not _dtype_level_shapes(arranged):
        return 0

    return int(arranged.ndim)


def _dtype_level_shapes(tensor) -> tuple[tuple, ...]:
    shapes = []
    current = getattr(tensor, "dtype", None)

    while isinstance(current, type(tensor)):
        shapes.append(tuple(current.shape))
        current = getattr(current, "dtype", None)

    return tuple(shapes)


__all__ = ["TensorAccess", "derive_access"]
