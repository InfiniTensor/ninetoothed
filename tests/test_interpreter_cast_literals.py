"""Execute frontend dtype string literals without treating quotes as type names."""

import numpy as np
import pytest

from ninetoothed.frontend.python import from_source
from ninetoothed.interpreter import InterpretationError, interpret_program
from ninetoothed.ir import TensorSpec

_INTEGERS = (
    ("i8", "int8"),
    ("i16", "int16"),
    ("i32", "int32"),
    ("i64", "int64"),
    ("u8", "uint8"),
    ("u16", "uint16"),
    ("u32", "uint32"),
    ("u64", "uint64"),
)
_SPELLINGS = tuple(
    (name, dtype)
    for alias, dtype in _INTEGERS
    for name in (alias, dtype, f"backend.{dtype}")
) + (
    ("bool", "bool"),
    ("fp16", "float16"),
    ("fp32", "float32"),
    ("fp64", "float64"),
)


def _run(expression, dtype):
    tensors = (
        TensorSpec(ndim=1, shape=(5,), dtype="float32", name="x"),
        TensorSpec(ndim=1, shape=(5,), dtype=dtype, name="out"),
    )
    program = from_source(
        f"def cast_literal(x, out):\n    out = x.to({expression})\n",
        tensors,
        kind="cast_literal",
    )
    assert program is not None
    x = np.array([0.0, 0.25, 1.75, 7.0, 12.5], dtype=np.float32)
    original = x.copy()
    backing = np.full(9, 3, dtype=dtype)
    out = backing[2:7]
    result = interpret_program(program, {"x": x, "out": out}, tensors=tensors)
    np.testing.assert_array_equal(result.outputs["out"], original.astype(dtype))
    np.testing.assert_array_equal(x, original)
    np.testing.assert_array_equal(backing[:2], np.full(2, 3, dtype=dtype))
    np.testing.assert_array_equal(backing[-2:], np.full(2, 3, dtype=dtype))


@pytest.mark.parametrize("spelling,dtype", _SPELLINGS)
def test_frontend_string_dtype_cast_executes(spelling, dtype):
    _run(repr(spelling), dtype)


def test_runtime_dtype_reference_remains_dynamic():
    _run("out.dtype", "int16")


def test_invalid_literal_dtype_reports_the_cast_location():
    with pytest.raises(InterpretationError, match="tensor.cast.*not_a_dtype"):
        _run(repr("not_a_dtype"), "int16")
