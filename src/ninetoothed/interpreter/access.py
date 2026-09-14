"""Record checked accesses using stable storage names and relative byte ranges."""

from bisect import bisect_left
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from .storage import storage_groups


@dataclass(frozen=True)
class MemoryAccess:
    """One observed read/write; addresses are relative, never process pointers."""

    kind: str
    storage: str
    byte_ranges: tuple
    overlapping_lanes: bool = False


def merge_ranges(ranges):
    """Return sorted, disjoint half-open byte intervals."""
    merged = []

    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        elif start < end:
            merged.append((start, end))
    return tuple(merged)


class WrittenIntervals:
    """Disjoint byte intervals with the last observed writer for each interval.

    Contiguous vector accesses stay compact; sequential scalar writes append.
    A partial overwrite retains the unaffected left and right interval pieces.
    """

    def __init__(self):
        self.intervals = []

    def _position(self, start):
        position = bisect_left(self.intervals, start, key=lambda interval: interval[0])

        if position and self.intervals[position - 1][1] > start:
            position -= 1
        return position

    def readers(self, start, end):
        position = self._position(start)

        while position < len(self.intervals):
            lower, upper, writer = self.intervals[position]

            if lower >= end:
                break

            yield max(lower, start), min(upper, end), writer
            position += 1

    def write(self, start, end, writer):
        first = self._position(start)
        last = first
        retained = []

        while last < len(self.intervals) and self.intervals[last][0] < end:
            lower, upper, previous = self.intervals[last]

            if lower < start:
                retained.append((lower, start, previous))

            if upper > end:
                retained.append((end, upper, previous))

            last += 1

        retained.append((start, end, writer))
        self.intervals[first:last] = sorted(retained)


class MemoryRecorder:
    """Observe runtime accesses only, pausing during trace/watch snapshots.

    Input spans that overlap receive one deterministic storage namespace.
    Exact byte offsets distinguish disjoint strided elements within that span.
    Actual addresses exist only while computing these relative coordinates.
    """

    def __init__(self, inputs):
        arrays = {
            name: value
            for name, value in inputs.items()
            if isinstance(value, np.ndarray) and value.size
        }
        self.allocations = {}

        for low, _high, records in storage_groups(arrays):
            name = records[0][0]

            for _name, array, address, _low, _high in records:
                self.allocations[id(array)] = (
                    "storage:" + name,
                    address - low,
                )

        self.pending = None

    @contextmanager
    def capture(self):
        previous, events = self.pending, []
        self.pending = events

        try:
            yield events
        finally:
            self.pending = previous

    @contextmanager
    def paused(self):
        previous = self.pending
        self.pending = None

        try:
            yield
        finally:
            self.pending = previous

    def unknown(self):
        if self.pending is not None:
            self.pending.append(MemoryAccess("unknown", "", ()))

    def access(self, kind, array, coordinates, mask, *, linear=False):
        if self.pending is None:
            return

        valid = np.asarray(mask, dtype=bool)

        if not np.any(valid):
            return

        allocation = self.allocations.get(id(array))

        if allocation is None:
            self.unknown()

            return

        storage, base = allocation
        offsets = np.full(valid.shape, base, dtype=np.int64)
        strides = (array.itemsize,) if linear else array.strides

        for coordinate, stride in zip(coordinates, strides):
            active = np.broadcast_to(coordinate, valid.shape)[valid].astype(
                np.int64, copy=False
            )
            offsets[valid] += active * stride

        starts = np.sort(offsets[valid].reshape(-1))
        overlap = kind == "write" and bool(np.any(np.diff(starts) < array.itemsize))
        # Find contiguous runs with vector operations, not one Python object per byte.
        breaks = np.flatnonzero(np.diff(starts) > array.itemsize)
        first = np.concatenate(([0], breaks + 1))
        last = np.concatenate((breaks, [len(starts) - 1]))
        ranges = tuple(
            (int(starts[a]), int(starts[b]) + array.itemsize)
            for a, b in zip(first, last)
        )
        self.pending.append(MemoryAccess(kind, storage, ranges, overlap))

    def source(self, kind, array, indices):
        """Observe a checked scalar or partial source index without reading data."""
        if self.pending is None:
            return

        tail = array.shape[len(indices) :]
        coordinates = tuple(np.full(tail, index, dtype=np.int64) for index in indices)
        coordinates += tuple(np.indices(tail, dtype=np.int64))
        self.access(kind, array, coordinates, np.ones(tail, dtype=bool))
