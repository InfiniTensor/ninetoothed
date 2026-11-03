"""CPU memory model shared by interpreter memory accesses.

Storage is modelled as a one-dimensional buffer per tensor argument, so every
``mem.load`` and ``mem.store`` is a gather or a scatter driven by the offsets and
the mask derived from the arrangement. Masked lanes never touch the storage, even
when their offsets are out of bounds.
"""

import numpy as np

from ninetoothed.interpreter.errors import InvalidArgumentError, InvalidProgramError

_TORCH_MODULE = "torch"


def storage_view(value, *, name: str):
    """Return a flat NumPy view of an application tensor argument.

    :param value: A NumPy array or a PyTorch CPU tensor.
    :param name: The application parameter name, used in error messages.
    :return: A one-dimensional NumPy view sharing memory with ``value``.
    :raises InvalidArgumentError: When the argument is not a supported CPU tensor.
    """
    module = type(value).__module__.split(".")[0]

    if module == _TORCH_MODULE:
        return _torch_storage_view(value, name=name)

    if isinstance(value, np.ndarray):
        if not value.flags["C_CONTIGUOUS"]:
            raise InvalidArgumentError(
                f"Tensor `{name}` must be C-contiguous to be interpreted on the CPU."
            )

        return value.reshape(-1)

    raise InvalidArgumentError(
        f"Tensor `{name}` must be a NumPy array or a PyTorch CPU tensor, "
        f"got `{type(value).__name__}`."
    )


def _torch_storage_view(value, *, name: str):
    device = getattr(value, "device", None)

    if device is not None and str(device) != "cpu":
        raise InvalidArgumentError(
            f"Tensor `{name}` is on `{device}`; the CPU interpreter only accepts "
            "CPU tensors."
        )

    if not value.is_contiguous():
        raise InvalidArgumentError(
            f"Tensor `{name}` must be C-contiguous to be interpreted on the CPU."
        )

    return value.detach().numpy().reshape(-1)


class TensorBuffer:
    """A flat CPU buffer backing one tensor argument.

    :param name: The application parameter name.
    :param storage: The one-dimensional storage view.
    :param shape: The shape of the semantic tensor, used for validation.
    """

    def __init__(self, *, name: str, storage, shape):
        self.name = name
        self.storage = storage
        self.shape = tuple(shape)
        self.dtype = storage.dtype

    @property
    def size(self) -> int:
        """Return the number of elements in the buffer."""
        return int(self.storage.size)

    def gather(self, offsets, mask=None, other=None):
        """Read the elements selected by offsets and mask.

        :param offsets: The flat element offsets, shaped like the result.
        :param mask: The boolean access mask, or ``None`` for an unmasked access.
        :param other: The fill value used for masked lanes.
        :return: The gathered values.
        """
        shape = offsets.shape
        offsets = np.asarray(offsets, dtype=np.intp)
        mask = _normalized_mask(mask, shape)

        _check_bounds(self.name, offsets, mask, size=self.size)

        safe_offsets = np.where(mask, offsets, 0) if mask is not None else offsets
        values = np.take(self.storage, safe_offsets)

        if mask is not None and not bool(mask.all()):
            fill = _fill_value(other, dtype=self.dtype)

            values = np.where(mask, values, fill)

        return values.astype(self.dtype, copy=False)

    def scatter(self, offsets, mask, values):
        """Write the values selected by offsets and mask.

        :param offsets: The flat element offsets, shaped like the values.
        :param mask: The boolean access mask, or ``None`` for an unmasked access.
        :param values: The values to write.
        """
        offsets = np.asarray(offsets, dtype=np.intp)
        mask = _normalized_mask(mask, offsets.shape)

        _check_bounds(self.name, offsets, mask, size=self.size)

        stored = np.broadcast_to(np.asarray(values), offsets.shape)
        flat_offsets = offsets.reshape(-1)
        flat_values = stored.reshape(-1)

        if mask is None:
            self.storage[flat_offsets] = flat_values.astype(self.dtype, copy=False)

            return

        selected = mask.reshape(-1)

        self.storage[flat_offsets[selected]] = flat_values[selected].astype(
            self.dtype, copy=False
        )


def _normalized_mask(mask, shape):
    if mask is None:
        return None

    mask = np.asarray(mask, dtype=bool)

    if mask.shape != tuple(shape):
        mask = np.broadcast_to(mask, shape)

    return mask


def _check_bounds(name: str, offsets, mask, *, size: int) -> None:
    if offsets.size == 0:
        return

    checked = offsets if mask is None else offsets[mask]

    if checked.size == 0:
        return

    if checked.min() < 0 or checked.max() >= size:
        raise InvalidProgramError(
            f"Tensor `{name}` was accessed out of bounds: offsets span "
            f"[{int(checked.min())}, {int(checked.max())}] while the buffer holds "
            f"{size} elements."
        )


def _fill_value(other, *, dtype):
    if other is None:
        return np.zeros((), dtype=dtype)

    if isinstance(other, float) and other == float("-inf"):
        return np.asarray(-np.inf, dtype=dtype)

    if isinstance(other, float) and other == float("inf"):
        return np.asarray(np.inf, dtype=dtype)

    if dtype == np.dtype(np.bool_):
        return np.asarray(bool(other), dtype=dtype)

    return np.asarray(other, dtype=dtype)


__all__ = ["TensorBuffer", "storage_view"]
