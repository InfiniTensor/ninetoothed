"""Identity views preserve masking, strided storage, and observable accesses."""

from types import SimpleNamespace

import numpy as np
import pytest

from ninetoothed.interpreter import interpret_program
from ninetoothed.interpreter.memory import TensorRef
from ninetoothed.ir import ssa


def _array(case):
    values = np.arange(60, dtype=np.float32).reshape(3, 4, 5)

    return {
        "contiguous": values,
        "reversed": values[::-1, :, ::-1],
        "strided": values[:, ::2, 1::2],
        "transpose": values.transpose(2, 0, 1),
        "empty": values[:, :0, :],
        "scalar": np.array(7, dtype=np.float32),
    }[case]


class _Observer:
    def __init__(self):
        self.events = []

    def access(self, kind, array, coordinates, mask):
        self.events.append(
            (kind, array, tuple(c.copy() for c in coordinates), mask.copy())
        )


@pytest.mark.parametrize(
    "case", ("contiguous", "reversed", "strided", "transpose", "empty", "scalar")
)
@pytest.mark.parametrize("explicit_layout", (False, True))
@pytest.mark.parametrize("mask_mode", ("all", "none", "alternating"))
def test_identity_read_write_and_observer(case, explicit_layout, mask_mode):
    array = _array(case)
    before = array.copy()
    mask = np.ones(array.shape, dtype=bool)

    if mask_mode == "none":
        mask[...] = False
    elif mask_mode == "alternating":
        mask = (np.arange(array.size).reshape(array.shape) % 2) == 0

    spec = None

    if explicit_layout:
        spec = SimpleNamespace(
            layout=SimpleNamespace(
                levels=(), application_shape=array.shape, view_access=None
            ),
            attrs={"other": -11},
        )

    observer = _Observer()
    ref = TensorRef(array, spec, {}, observer=observer)
    np.testing.assert_array_equal(
        ref.read(mask, other=-19), np.where(mask, before, -19)
    )
    replacement = np.arange(array.size).reshape(array.shape) + 101
    ref.write(replacement, mask)
    np.testing.assert_array_equal(array, np.where(mask, replacement, before))
    # Storage is read afresh on each access, including writes through an alias.
    array[...] = 23
    np.testing.assert_array_equal(ref.read(), np.full(array.shape, 23))
    assert [event[0] for event in observer.events] == ["read", "write", "read"]

    for _, observed_array, coordinates, _ in observer.events:
        assert observed_array is array

        for axis, coordinate in enumerate(coordinates):
            expected = np.empty(array.shape, dtype=np.int64)

            for index in np.ndindex(array.shape):
                expected[index] = index[axis]

            np.testing.assert_array_equal(coordinate, expected)

        assert len(coordinates) == array.ndim

    np.testing.assert_array_equal(observer.events[0][3], mask)
    np.testing.assert_array_equal(observer.events[1][3], mask)
    np.testing.assert_array_equal(
        observer.events[2][3], np.ones(array.shape, dtype=bool)
    )

    if array.size:
        index = tuple(size - 1 for size in array.shape)
        assert ref.extract(index) == 23


@pytest.mark.parametrize(
    "case", ("contiguous", "reversed", "strided", "transpose", "empty", "scalar")
)
def test_public_program_without_descriptors_preserves_alias_writes(case):
    array = _array(case)
    dtype = ssa.Type(kind="tensor", shape=tuple(map(str, array.shape)), dtype="float32")
    x, alias, out = (ssa.Value(name=name, type=dtype) for name in ("x", "alias", "out"))
    one = ssa.Value(name="%one", type=ssa.Type(kind="scalar", dtype="float32"))
    program = ssa.Program(
        kind="identity_alias",
        inputs=(x, alias, out),
        outputs=(out,),
        blocks=(
            ssa.Block(
                operations=(
                    ssa.Operation(
                        opcode="arith.constant", results=(one,), attrs={"value": 13}
                    ),
                    ssa.Operation(opcode="mem.store", operands=("%one", "x")),
                    ssa.Operation(opcode="mem.store", operands=("alias", "out")),
                )
            ),
        ),
    )
    output = np.full_like(array, -17)
    result = interpret_program(
        program, {"x": array, "alias": array, "out": output}, trace=True
    )
    np.testing.assert_array_equal(result.outputs["out"], np.full(array.shape, 13))
    np.testing.assert_array_equal(array, np.full(array.shape, 13))
