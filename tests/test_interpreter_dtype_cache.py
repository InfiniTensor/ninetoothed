"""Preserve dtype spelling, fallback, byte order, and diagnostic semantics."""

import gc
import weakref

import numpy as np
import pytest

from ninetoothed.interpreter.expressions import numpy_dtype


@pytest.mark.parametrize(
    "name,expected",
    (
        ("fp16", "float16"),
        ("backend.fp32", "float32"),
        ("fp64", "float64"),
        ("i1", "bool"),
        ("i8", "int8"),
        ("i16", "int16"),
        ("i32", "int32"),
        ("i64", "int64"),
        ("u8", "uint8"),
        ("u16", "uint16"),
        ("u32", "uint32"),
        ("u64", "uint64"),
        ("index", "int64"),
        ("<f4", "<f4"),
        (">f4", ">f4"),
    ),
)
def test_dtype_aliases_and_byte_order_survive_repeated_resolution(name, expected):
    for _ in range(2):
        assert numpy_dtype(name) == np.dtype(expected)


@pytest.mark.parametrize(
    "name", ("complex64", "object", "datetime64[ns]", "S4", "V16", "backend.bad_dtype")
)
def test_dtype_errors_preserve_the_original_spelling(name):
    for _ in range(2):
        with pytest.raises(ValueError) as error:
            numpy_dtype(name)

        assert str(error.value) == f"Unsupported interpreter dtype `{name}`."


def test_dtype_fallback_and_symbol_sentinels_remain_distinct():
    assert numpy_dtype(None) is None
    assert numpy_dtype(None, "float64") == np.dtype("float64")
    assert numpy_dtype(None, "complex64") == np.dtype("complex64")
    assert numpy_dtype("symbol", "float32") is None
    assert numpy_dtype("backend.none", "float32") is None


def test_dtype_cache_does_not_retain_or_freeze_user_objects():
    class Spelling:
        name = "backend.fp32"

        def __str__(self):
            return self.name

    value = Spelling()
    assert numpy_dtype(value) == np.dtype("float32")
    value.name = "backend.i32"
    assert numpy_dtype(value) == np.dtype("int32")
    reference = weakref.ref(value)
    del value
    gc.collect()
    assert reference() is None
