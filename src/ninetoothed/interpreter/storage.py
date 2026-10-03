"""Copy numeric view graphs and restore them without reading allocation gaps."""

import math
from dataclasses import dataclass

import numpy as np


def storage_groups(inputs):
    """Group overlapping address spans in stable input order.

    Span overlap is conservative for strided views. Actual byte coordinates
    still distinguish disjoint elements; touching allocations stay separate.
    """
    records = []
    order = {name: index for index, name in enumerate(inputs)}

    for name, array in inputs.items():
        if not isinstance(array, np.ndarray):
            continue

        address = array.__array_interface__["data"][0]
        extents = (
            [(size - 1) * stride for size, stride in zip(array.shape, array.strides)]
            if array.size
            else []
        )
        low = address + sum(min(0, extent) for extent in extents)
        high = (
            address
            + sum(max(0, extent) for extent in extents)
            + (array.itemsize if array.size else 0)
        )
        records.append((name, array, address, low, high))

    groups = []

    for record in sorted(records, key=lambda item: item[3]):
        if groups and record[3] < groups[-1][1]:
            groups[-1][1] = max(groups[-1][1], record[4])
            groups[-1][2].append(record)
        else:
            groups.append([record[3], record[4], [record]])

    for group in groups:
        group[2].sort(key=lambda record: order[record[0]])
    return sorted(groups, key=lambda group: order[group[2][0][0]])


@dataclass
class StorageCopy:
    """Independent storage plus descriptors that preserve the original views."""

    values: dict
    buffers: dict
    bindings: dict
    aliases: dict
    shared: bool


def restore_view(buffer, binding):
    """Validate every accessible byte before creating a view over numeric data."""
    dtype = np.dtype(binding["dtype"])
    shape, strides, offset = (
        tuple(binding["shape"]),
        tuple(binding["strides"]),
        binding["offset"],
    )

    if (
        dtype.kind not in "biufc"
        or buffer.dtype != np.uint8
        or buffer.ndim != 1
        or not buffer.flags.c_contiguous
    ):
        raise TypeError(
            "Shared replay storage must be contiguous bytes with a numeric view dtype."
        )

    if (
        type(offset) is not int
        or len(shape) != len(strides)
        or any(type(size) is not int or size < 0 for size in shape)
        or any(type(stride) is not int for stride in strides)
        or type(binding["writeable"]) is not bool
    ):
        raise ValueError(
            "Shared replay view has invalid shape, strides, offset or permission."
        )

    extents = (
        [(size - 1) * stride for size, stride in zip(shape, strides)]
        if math.prod(shape)
        else []
    )
    low = offset + sum(min(0, extent) for extent in extents)
    high = (
        offset
        + sum(max(0, extent) for extent in extents)
        + (dtype.itemsize if math.prod(shape) else 0)
    )

    if low < 0 or high > buffer.nbytes or offset < 0 or offset > buffer.nbytes:
        raise ValueError(
            "Shared replay view addresses bytes outside its saved storage."
        )

    view = np.ndarray(shape, dtype=dtype, buffer=buffer, offset=offset, strides=strides)
    view.flags.writeable = binding["writeable"]

    return view


def copy_storage(inputs):
    """Clone aliasing views without changing originals or copying padding bytes."""
    for value in inputs.values():
        if isinstance(value, np.ndarray) and value.dtype.kind not in "biufc":
            raise TypeError("Differential replay requires numeric arrays.")

    values, buffers, bindings, aliases, owners = dict(inputs), {}, {}, {}, {}
    shared = False

    for index, (low, high, records) in enumerate(storage_groups(inputs)):
        key = f"storage_{index}"
        buffer = np.zeros(
            max(high - low, max(record[1].itemsize for record in records)),
            dtype=np.uint8,
        )
        buffers[key] = buffer
        shared = shared or len({id(record[1]) for record in records}) > 1

        for name, array, address, _low, _high in records:
            if id(array) in owners:
                source = owners[id(array)]
                aliases[name] = source
                values[name] = values[source]
                continue

            binding = {
                "storage": key,
                "offset": address - low,
                "shape": list(array.shape),
                "strides": list(array.strides),
                "dtype": str(array.dtype),
                "writeable": bool(array.flags.writeable),
                "scalar": False,
            }
            view = restore_view(buffer, dict(binding, writeable=True))
            view[...] = array
            view.flags.writeable = array.flags.writeable
            values[name], bindings[name], owners[id(array)] = view, binding, name

    return StorageCopy(values, buffers, bindings, aliases, shared)
