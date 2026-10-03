"""Compare recorded active byte coverage with an independent byte-set oracle."""

import numpy as np
import pytest

from ninetoothed.interpreter.access import MemoryAccess, MemoryRecorder


def _view(case):
    owner = np.zeros(512, dtype=np.uint8)
    specifications = {
        "contiguous": ((3, 5), (20, 4), 32),
        "reversed": ((3, 5), (-20, -4), 88),
        "strided": ((3, 5), (48, 8), 16),
        "transpose": ((5, 3), (4, 20), 32),
        "broadcast": ((3, 5), (0, 4), 32),
        "partial_byte_overlap": ((3, 5), (5, 1), 32),
        "scalar": ((), (), 32),
        "empty": ((3, 0), (4, 4), 32),
    }
    shape, strides, offset = specifications[case]
    view = np.ndarray(
        shape, dtype=np.int32, buffer=owner, offset=offset, strides=strides
    )

    return owner, view, offset


def _oracle(array, coordinates, mask, base, linear):
    valid = np.asarray(mask, dtype=bool)
    strides = (array.itemsize,) if linear else array.strides
    coverage = []

    for lane in np.ndindex(valid.shape):
        if not valid[lane]:
            continue

        start = base

        for coordinate, stride in zip(coordinates, strides):
            start += int(np.broadcast_to(coordinate, valid.shape)[lane]) * stride

        coverage.extend(range(start, start + array.itemsize))

    ranges = []

    for byte in sorted(set(coverage)):
        if ranges and ranges[-1][1] == byte:
            ranges[-1] = (ranges[-1][0], byte + 1)
        else:
            ranges.append((byte, byte + 1))

    return tuple(ranges), len(coverage) != len(set(coverage))


@pytest.mark.parametrize("kind", ("read", "write"))
@pytest.mark.parametrize("mask_kind", ("full", "sparse", "empty"))
@pytest.mark.parametrize(
    "case",
    (
        "contiguous",
        "reversed",
        "strided",
        "transpose",
        "broadcast",
        "partial_byte_overlap",
        "scalar",
        "empty",
    ),
)
def test_recorded_ranges_match_independent_byte_union(case, mask_kind, kind):
    owner, view, base = _view(case)
    coordinates = np.indices(view.shape, dtype=np.int64, sparse=True)
    mask = np.ones(view.shape, dtype=bool)

    if mask_kind == "sparse":
        mask = np.arange(view.size).reshape(view.shape) % 3 == 1
    elif mask_kind == "empty":
        mask[...] = False

    ranges, overlapping = _oracle(view, coordinates, mask, base, False)
    recorder = MemoryRecorder({"owner": owner, "view": view})

    with recorder.capture() as events:
        recorder.access(kind, view, coordinates, mask)

    assert events == (
        [MemoryAccess(kind, "storage:owner", ranges, kind == "write" and overlapping)]
        if ranges
        else []
    )
    np.testing.assert_array_equal(owner, np.zeros_like(owner))


@pytest.mark.parametrize("kind", ("read", "write"))
@pytest.mark.parametrize("mask", (True, [True, False, True, True, False]))
def test_linear_duplicate_indices_and_broadcast_masks(kind, mask):
    array = np.arange(8, dtype=np.int32)
    offsets, valid = np.broadcast_arrays(np.array([0, 2, 2, 7, 1]), np.asarray(mask))
    coordinates = (offsets,)
    expected, overlapping = _oracle(array, coordinates, valid, 0, True)
    recorder = MemoryRecorder({"buffer": array})

    with recorder.capture() as events:
        recorder.access(kind, array, coordinates, valid, linear=True)

    assert events == [
        MemoryAccess(kind, "storage:buffer", expected, kind == "write" and overlapping)
    ]


def test_unknown_and_paused_accesses_keep_their_recording_boundaries():
    registered = np.arange(8, dtype=np.int32)
    unregistered = registered[::2]
    recorder = MemoryRecorder({"buffer": registered})

    with recorder.capture() as events:
        recorder.access("read", unregistered, (np.arange(4),), False)

        with recorder.paused():
            recorder.access("read", unregistered, (np.arange(4),), True)

        recorder.access("read", unregistered, (np.arange(4),), True)

    assert events == [MemoryAccess("unknown", "", ())]
