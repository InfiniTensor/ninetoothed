"""Real default-pipeline dot across output tiles, checked against NumPy."""

import math

import numpy as np
import pytest

import ninetoothed.language as ntl
from ninetoothed import Tensor, interpret
from ninetoothed.compiler.passes import default_pipeline
from ninetoothed.interpreter import UnsupportedOperationError, interpret_program


def matrix_tiles(a, b, out):
    return a.tile((4, 4)), b.tile((4, 4)), out.tile((4, 4))


def matrix_dot(a, b, out):
    out = ntl.dot(a, b)  # noqa: F841


def matrix_tiles_with_scratch(a, b, scratch, out):
    return tuple(tensor.tile((4, 4)) for tensor in (a, b, scratch, out))


def matrix_dot_with_nested_scratch_store(a, b, scratch, out):
    if True:
        scratch = scratch + 1  # noqa: F841

    out = ntl.dot(a, b)  # noqa: F841


def in_place_matrix_tiles(out, b):
    return out.tile((4, 4)), b.tile((4, 4))


def in_place_matrix_dot(out, b):
    out = ntl.dot(out, b)  # noqa: F841


def _tensors(dtype):
    return (
        Tensor(2, name="a", dtype=np.dtype(dtype).name, other=0),
        Tensor(2, name="b", dtype=np.dtype(dtype).name, other=0),
        Tensor(2, name="out", dtype=np.dtype(dtype).name),
    )


def _operands(shape, dtype, layout):
    rows, inner, columns = shape
    rng = np.random.default_rng(923)
    a = rng.integers(-7, 8, size=(rows, inner)).astype(dtype)
    b = rng.integers(-7, 8, size=(inner, columns)).astype(dtype)

    if np.dtype(dtype).kind == "f":
        a /= 3
        b /= 5

    if layout == "strided":
        # The same logical matrices use independent, non-contiguous storage.
        a_backing = np.full((rows * 2, inner * 2), -937, dtype=dtype)
        a_view = a_backing[::2, ::2]
        a_view[...] = a
        a = a_view
        b = b.T.copy().T
        assert not a.flags.c_contiguous
        assert not b.flags.c_contiguous

    return a, b


def _assert_equal(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype

    if expected.dtype.kind == "f":
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
    else:
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("backend", ("triton", "cuda"))
@pytest.mark.parametrize("dtype", (np.float32, np.int32), ids=("float32", "int32"))
@pytest.mark.parametrize("shape", ((7, 3, 6), (7, 4, 6), (3, 4, 7)))
@pytest.mark.parametrize("layout", ("contiguous", "strided"))
def test_default_pipeline_dot_with_multiple_output_programs(
    backend, dtype, shape, layout
):
    rows, _inner, columns = shape
    a, b = _operands(shape, dtype, layout)
    originals = (a.copy(), b.copy())
    expected = a @ b
    kernel = interpret(matrix_tiles, matrix_dot, _tensors(dtype), backend=backend)
    assert kernel.program.metadata["pass_trace"] == tuple(
        pass_.name for pass_ in default_pipeline(backend).passes
    )
    operations = kernel.program.blocks[0].operations
    assert any(op.opcode == "scf.for" for op in operations)
    assert not any(op.opcode in {"linalg.dot", "linalg.matmul"} for op in operations)
    frontend_out = np.full(expected.shape, -731, dtype=dtype)
    interpret_program(
        kernel.frontend_program,
        {"a": a, "b": b, "out": frontend_out},
        tensors=kernel.tensors,
        symbols=kernel.meta,
    )
    _assert_equal(frontend_out, expected)
    previous_out = None

    for trace in (False, True):
        backing = np.full(expected.size + 8, -731, dtype=dtype)
        out = backing[4:-4].reshape(expected.shape)
        result = interpret_program(
            kernel.program,
            {"a": a, "b": b, "out": out},
            tensors=kernel.tensors,
            symbols=kernel.meta,
            trace=trace,
        )
        _assert_equal(result.outputs["out"], expected)
        np.testing.assert_array_equal(backing[:4], np.full(4, -731, dtype=dtype))
        np.testing.assert_array_equal(backing[-4:], np.full(4, -731, dtype=dtype))

        if previous_out is not None:
            np.testing.assert_array_equal(out, previous_out)

        previous_out = out.copy()

        if not trace:
            assert not result.trace
            continue

        stores = [event for event in result.trace if event.opcode == "mem.store"]
        assert len(stores) == rows * columns
        column_tiles = math.ceil(columns / 4)
        assert {event.program_id for event in stores} == {
            (program, 0, 0) for program in range(math.ceil(rows / 4) * column_tiles)
        }
        written = []

        for event in stores:
            tile_row, tile_column = divmod(event.program_id[0], column_tiles)
            row, column = event.lane
            written.append((tile_row * 4 + row, tile_column * 4 + column))
            effective_mask = np.asarray(event.mask["value"], dtype=bool)
            assert effective_mask.sum() == 1
            assert effective_mask[event.lane]

        assert len(written) == len(set(written))
        assert set(written) == set(np.ndindex(expected.shape))

    np.testing.assert_array_equal(a, originals[0])
    np.testing.assert_array_equal(b, originals[1])


@pytest.mark.parametrize("backend", ("triton", "cuda"))
@pytest.mark.parametrize("shape", ((3, 5, 2), (7, 5, 6)))
def test_dot_with_unreduced_k_programs_is_rejected_before_output_writes(backend, shape):
    # This arrangement launches independent K tiles but has no operation that
    # combines their partial products. It does not describe a complete matmul.
    a, b = _operands(shape, np.float32, "contiguous")
    rows, _inner, columns = shape
    out = np.full((rows, columns), -731, dtype=np.float32)
    originals = (a.copy(), b.copy(), out.copy())
    kernel = interpret(matrix_tiles, matrix_dot, _tensors(np.float32), backend=backend)

    with pytest.raises(UnsupportedOperationError):
        kernel(a, b, out)

    np.testing.assert_array_equal(a, originals[0])
    np.testing.assert_array_equal(b, originals[1])
    np.testing.assert_array_equal(out, originals[2])


@pytest.mark.parametrize("backend", ("triton", "cuda"))
def test_dot_rejects_output_aliasing_an_input_before_writes(backend):
    a, b = _operands((7, 4, 4), np.float32, "contiguous")
    originals = (a.copy(), b.copy())
    kernel = interpret(matrix_tiles, matrix_dot, _tensors(np.float32), backend=backend)

    with pytest.raises(UnsupportedOperationError, match=r"entry:\d+:mem\.store"):
        kernel(a, b, a)

    np.testing.assert_array_equal(a, originals[0])
    np.testing.assert_array_equal(b, originals[1])


@pytest.mark.parametrize("backend", ("triton", "cuda"))
@pytest.mark.parametrize("strides", ((24, 1), (0, 4)), ids=("partial_bytes", "zero"))
def test_dot_rejects_overlapping_output_storage_before_writes(backend, strides):
    a, b = _operands((7, 3, 6), np.float32, "contiguous")
    backing = np.full(256, 0x5A, dtype=np.uint8)
    out = np.ndarray((7, 6), dtype=np.float32, buffer=backing, strides=strides)
    originals = (a.copy(), b.copy(), backing.copy())
    kernel = interpret(matrix_tiles, matrix_dot, _tensors(np.float32), backend=backend)

    with pytest.raises(UnsupportedOperationError, match=r"entry:\d+:mem\.store"):
        kernel(a, b, out)

    np.testing.assert_array_equal(a, originals[0])
    np.testing.assert_array_equal(b, originals[1])
    np.testing.assert_array_equal(backing, originals[2])


@pytest.mark.parametrize("backend", ("triton", "cuda"))
@pytest.mark.parametrize(
    "reverse", (False, True), ids=("positive_stride", "negative_stride")
)
def test_dot_accepts_independent_strided_output_and_preserves_guards(backend, reverse):
    a, b = _operands((7, 3, 6), np.float32, "strided")
    expected = a @ b
    backing = np.full((15, 13), -731, dtype=np.float32)
    out = backing[1::2, 1::2]

    if reverse:
        out = out[::-1, ::-1]

    assert out.shape == expected.shape
    assert not out.flags.c_contiguous
    protected = np.ones(backing.shape, dtype=bool)
    protected[1::2, 1::2] = False
    originals = (a.copy(), b.copy(), backing.copy())
    kernel = interpret(
        matrix_tiles, matrix_dot, _tensors(np.float32), backend=backend, trace=True
    )
    kernel(a, b, out)
    _assert_equal(out, expected)
    np.testing.assert_array_equal(a, originals[0])
    np.testing.assert_array_equal(b, originals[1])
    np.testing.assert_array_equal(backing[protected], originals[2][protected])


@pytest.mark.parametrize("backend", ("triton", "cuda"))
def test_dot_rejects_nested_store_before_replaying_output_lanes(backend):
    a, b = _operands((7, 3, 6), np.float32, "contiguous")
    tensors = tuple(
        Tensor(2, name=name, dtype="float32", other=0)
        for name in ("a", "b", "scratch", "out")
    )
    kernel = interpret(
        matrix_tiles_with_scratch,
        matrix_dot_with_nested_scratch_store,
        tensors,
    )
    frontend_inputs = {
        "a": a.copy(),
        "b": b.copy(),
        "scratch": np.zeros((7, 6), dtype=np.float32),
        "out": np.full((7, 6), -731, dtype=np.float32),
    }
    interpret_program(
        kernel.frontend_program,
        frontend_inputs,
        tensors=kernel.tensors,
        symbols=kernel.meta,
    )
    np.testing.assert_array_equal(
        frontend_inputs["scratch"], np.ones((7, 6), dtype=np.float32)
    )
    _assert_equal(frontend_inputs["out"], a @ b)
    kernel = interpret(
        matrix_tiles_with_scratch,
        matrix_dot_with_nested_scratch_store,
        tensors,
        backend=backend,
    )
    target_inputs = {
        "a": a.copy(),
        "b": b.copy(),
        "scratch": np.zeros((7, 6), dtype=np.float32),
        "out": np.full((7, 6), -731, dtype=np.float32),
    }
    originals = {name: value.copy() for name, value in target_inputs.items()}

    with pytest.raises(UnsupportedOperationError):
        kernel(**target_inputs)

    for name, original in originals.items():
        np.testing.assert_array_equal(target_inputs[name], original)


@pytest.mark.parametrize("backend", ("triton", "cuda"))
@pytest.mark.parametrize("storage", ("lhs_alias", "partial_bytes", "zero_stride"))
def test_single_program_dot_rejects_output_storage_before_writes(backend, storage):
    a, b = _operands((3, 4, 4), np.float32, "contiguous")

    if storage == "lhs_alias":
        backing, out = a, a
    else:
        backing = np.full(256, 0x5A, dtype=np.uint8)
        strides = (16, 1) if storage == "partial_bytes" else (0, 4)
        out = np.ndarray((3, 4), dtype=np.float32, buffer=backing, strides=strides)

    originals = (a.copy(), b.copy(), backing.copy())
    kernel = interpret(matrix_tiles, matrix_dot, _tensors(np.float32), backend=backend)

    with pytest.raises(UnsupportedOperationError, match=r"entry:\d+:mem\.store"):
        kernel(a, b, out)

    np.testing.assert_array_equal(a, originals[0])
    np.testing.assert_array_equal(b, originals[1])
    np.testing.assert_array_equal(backing, originals[2])


@pytest.mark.parametrize("backend", ("triton", "cuda"))
@pytest.mark.parametrize(
    "reverse", (False, True), ids=("positive_stride", "negative_stride")
)
def test_single_program_dot_accepts_independent_strided_output(backend, reverse):
    a, b = _operands((3, 4, 4), np.float32, "strided")
    expected = a @ b
    backing = np.full((7, 9), -731, dtype=np.float32)
    out = backing[1::2, 1::2]

    if reverse:
        out = out[::-1, ::-1]

    assert out.shape == expected.shape
    assert not out.flags.c_contiguous
    protected = np.ones(backing.shape, dtype=bool)
    protected[1::2, 1::2] = False
    originals = (a.copy(), b.copy(), backing.copy())
    kernel = interpret(
        matrix_tiles, matrix_dot, _tensors(np.float32), backend=backend, trace=True
    )
    kernel(a, b, out)
    _assert_equal(out, expected)
    np.testing.assert_array_equal(a, originals[0])
    np.testing.assert_array_equal(b, originals[1])
    np.testing.assert_array_equal(backing[protected], originals[2][protected])


@pytest.mark.parametrize("backend", ("triton", "cuda"))
def test_single_program_dot_rejects_output_binding_used_as_lhs(backend):
    out, b = _operands((3, 4, 4), np.float32, "contiguous")
    originals = (out.copy(), b.copy())
    kernel = interpret(
        in_place_matrix_tiles,
        in_place_matrix_dot,
        (
            Tensor(2, name="out", dtype="float32", other=0),
            Tensor(2, name="b", dtype="float32", other=0),
        ),
        backend=backend,
    )

    with pytest.raises(UnsupportedOperationError, match=r"entry:\d+:mem\.store"):
        kernel(out, b)

    np.testing.assert_array_equal(out, originals[0])
    np.testing.assert_array_equal(b, originals[1])
