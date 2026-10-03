"""CPU execution of real arrangements and frontend-lowered applications."""

import os
import subprocess
import sys

import numpy as np
import pytest

import ninetoothed.language as ntl
from ninetoothed import Tensor, interpret
from ninetoothed.compiler.passes import lower_for_target
from ninetoothed.interpreter import UnsupportedOperationError


def _vectors(x, y, out):
    return tuple(tensor.tile((4,)) for tensor in (x, y, out))


def _add(x, y, out):
    out = x + y  # noqa: F841


def _unary_vectors(x, out):
    return x.tile((4,)), out.tile((4,))


def _comparison(x, out):
    out = (x > 0) & (x < 4)  # noqa: F841


def _broadcast_tiles(x, bias, out):
    return x.tile((1, 8)), bias.tile((8,)), out.tile((1, 8))


def _broadcast_add(x, bias, out):
    out = x + bias  # noqa: F841


def _row_tiles(x, out):
    return x.tile((1, 8)), out.tile((1, 1))


def _row_sum(x, out):
    out = ntl.sum(x, 1)  # noqa: F841


def _squeezed_row_tiles(x, out):
    return x.tile((1, 512)), out.tile((1,))


def _nested_row_tiles(x, out):
    return x.tile((1, 4)).tile((1, -1)), out.tile((1, 1))


def _nested_row_sum(x, out):
    accumulator = ntl.zeros(out.shape, dtype=out.dtype)

    for i in range(x.shape[1]):
        accumulator += ntl.sum(x[0, i], axis=-1)

    out = accumulator  # noqa: F841


def _control_tiles(x, positive, out):
    return x.tile((4,)), positive, out.tile((4,))


def _control_flow(x, positive, out):
    accumulator = x

    for i in range(3):
        if positive:
            accumulator = accumulator + i
        else:
            accumulator = accumulator - i

    out = accumulator  # noqa: F841


def _softmax_tiles(x, out):
    return x.tile((1, 8)), out.tile((1, 8))


def _softmax(x, out):
    shifted = x - ntl.max(x, 1)[:, None]
    numerator = ntl.exp(shifted)
    out = numerator / ntl.sum(numerator, 1)[:, None]  # noqa: F841


def _matrix_tiles(a, b, out):
    return a.tile((4, 4)), b.tile((4, 4)), out.tile((4, 4))


def _dot(a, b, out):
    out = ntl.dot(a, b)  # noqa: F841


def _whole_matrices(a, b, out):
    return a, b, out


def _transpose_tiles(x, out):
    return x.tile((4, 4)), out.tile((4, 4))


def _whole_transpose(x, out):
    return x, out


def _transpose(x, out):
    out = ntl.trans(x)  # noqa: F841


def _descriptor(ndim, name, dtype="float32", **kwargs):
    return Tensor(ndim, name=name, dtype=dtype, **kwargs)


def _check(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype

    if expected.dtype.kind == "f":
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
    else:
        np.testing.assert_array_equal(actual, expected)


def _run_and_compare(handle, arguments, expected, *, optimize):
    originals = [
        arg.copy() if isinstance(arg, np.ndarray) else arg for arg in arguments
    ]
    before = handle(*arguments)
    _check(arguments[-1], expected)
    assert "out" in before.outputs
    _check(before.outputs["out"], expected)

    if optimize:
        for backend in ("cuda", "triton"):
            program = lower_for_target(
                handle.program, backend=backend, tensors=handle.tensors
            )
            assert "ssa.canonicalize" in program.metadata["pass_trace"]
            assert f"ssa.{backend}.optimize_schedule" in program.metadata["pass_trace"]
            optimized_arguments = [
                arg.copy() if isinstance(arg, np.ndarray) else arg for arg in originals
            ]
            after = handle.with_program(program)(*optimized_arguments)
            _check(optimized_arguments[-1], expected)
            _check(after.outputs["out"], before.outputs["out"])

    for argument, original in zip(arguments[:-1], originals[:-1]):
        if isinstance(argument, np.ndarray):
            np.testing.assert_array_equal(argument, original)


@pytest.mark.parametrize("dtype", (np.float32, np.int32))
@pytest.mark.parametrize("size", (1, 8, 11))
def test_elementwise_and_nondivisible_tail(dtype, size):
    x = np.arange(size, dtype=dtype) - 3
    y = np.arange(size, dtype=dtype) * 2
    # Both ends are guarded: the output is a view into a larger allocation.
    backing = np.full(size + 2, -731, dtype=dtype)
    out = backing[1:-1]
    handle = interpret(
        _vectors,
        _add,
        tuple(_descriptor(1, name, np.dtype(dtype).name) for name in ("x", "y", "out")),
    )
    _run_and_compare(handle, [x, y, out], x + y, optimize=True)
    np.testing.assert_array_equal(backing[[0, -1]], np.array([-731, -731], dtype=dtype))


def test_bool_results_are_exact():
    x = np.arange(-2, 8, dtype=np.int32)
    out = np.empty(x.shape, dtype=np.bool_)
    handle = interpret(
        _unary_vectors,
        _comparison,
        (_descriptor(1, "x", "int32"), _descriptor(1, "out", "bool")),
    )
    _run_and_compare(handle, [x, out], (x > 0) & (x < 4), optimize=True)


def test_broadcast_reuses_bias_for_each_row():
    x = np.arange(15, dtype=np.float32).reshape(3, 5) / 7
    bias = np.array([3, -4, 2, 7, -1], dtype=np.float32)
    out = np.empty_like(x)
    handle = interpret(
        _broadcast_tiles,
        _broadcast_add,
        (_descriptor(2, "x"), _descriptor(1, "bias"), _descriptor(2, "out")),
    )
    _run_and_compare(handle, [x, bias, out], x + bias, optimize=True)


@pytest.mark.parametrize("dtype", (np.float32, np.int32))
def test_row_reduction_ignores_padded_lanes(dtype):
    x = (np.arange(15).reshape(3, 5) - 9).astype(dtype)
    out = np.empty((3, 1), dtype=dtype)
    handle = interpret(
        _row_tiles,
        _row_sum,
        (
            _descriptor(2, "x", np.dtype(dtype).name, other=0),
            _descriptor(2, "out", np.dtype(dtype).name),
        ),
    )
    expected = np.sum(x, axis=1, keepdims=True, dtype=dtype)
    _run_and_compare(handle, [x, out], expected, optimize=True)


@pytest.mark.parametrize("dtype", (np.float32, np.int32))
def test_row_reduction_maps_singleton_program_axes_to_a_vector_output(dtype):
    rng = np.random.default_rng(2026)
    x = rng.integers(-10, 10, size=(7, 257)).astype(dtype)
    out = np.full(7, -731, dtype=dtype)
    handle = interpret(
        _squeezed_row_tiles,
        _row_sum,
        (
            _descriptor(2, "x", np.dtype(dtype).name, other=0),
            _descriptor(1, "out", np.dtype(dtype).name),
        ),
    )
    expected = np.sum(x, axis=1, dtype=dtype)
    _run_and_compare(handle, [x, out], expected, optimize=True)


@pytest.mark.parametrize("positive", (False, True))
def test_nested_if_and_for_carry_values(positive):
    x = np.arange(11, dtype=np.float32) / 3
    out = np.empty_like(x)
    handle = interpret(
        _control_tiles,
        _control_flow,
        (
            _descriptor(1, "x"),
            _descriptor(0, "positive", "bool", constexpr=True),
            _descriptor(1, "out"),
        ),
    )
    expected = x + (3 if positive else -3)
    _run_and_compare(handle, [x, positive, out], expected, optimize=True)


def test_softmax_uses_negative_infinity_for_padding():
    x = np.array([[1000, 999, 998, 997, 996], [-4, -3, -2, -1, 0]], dtype=np.float32)
    out = np.empty_like(x)
    handle = interpret(
        _softmax_tiles,
        _softmax,
        (_descriptor(2, "x", other=float("-inf")), _descriptor(2, "out")),
    )
    exponentials = np.exp(x - np.max(x, axis=1, keepdims=True))
    expected = exponentials / np.sum(exponentials, axis=1, keepdims=True)
    _run_and_compare(handle, [x, out], expected, optimize=True)
    np.testing.assert_allclose(out.sum(axis=1), 1, rtol=1e-3, atol=1e-3)


def test_dot_on_masked_matrix_tiles():
    a = (np.arange(12, dtype=np.float32).reshape(3, 4) - 7) / 3
    b = (np.arange(8, dtype=np.float32).reshape(4, 2) + 2) / 5
    out = np.empty((3, 2), dtype=np.float32)
    handle = interpret(
        _matrix_tiles,
        _dot,
        (
            _descriptor(2, "a", other=0),
            _descriptor(2, "b", other=0),
            _descriptor(2, "out"),
        ),
    )
    _run_and_compare(handle, [a, b, out], a @ b, optimize=False)


def test_cpu_path_does_not_import_gpu_backends():
    source = """
import importlib.abc
import sys

class BlockGPUImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'triton', 'tilelang', 'cupy', 'pycuda'}:
            raise AssertionError('CPU interpreter imported ' + fullname)

sys.meta_path.insert(0, BlockGPUImports())
from tests.test_interpreter_applications import test_broadcast_reuses_bias_for_each_row
test_broadcast_reuses_bias_for_each_row()
"""
    completed = subprocess.run(
        [sys.executable, "-c", source],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_tiled_trace_visits_each_program_and_is_reproducible():
    x = np.arange(11, dtype=np.float32)
    y = np.ones(11, dtype=np.float32)
    out = np.full_like(x, -99)
    handle = interpret(
        _vectors,
        _add,
        tuple(_descriptor(1, name) for name in ("x", "y", "out")),
        trace=True,
    )
    first = handle(x, y, out)
    second = handle(x, y, np.full_like(x, -99))
    assert first.trace == second.trace
    assert {event.program_id for event in first.trace} == {
        (0, 0, 0),
        (1, 0, 0),
        (2, 0, 0),
    }
    assert sum(event.opcode == "mem.store" for event in first.trace) == 3
    _check(out, x + y)


def test_arranged_loads_and_stores_follow_noncontiguous_numpy_strides():
    x = np.arange(44, dtype=np.float32)[1::4]
    y = np.arange(11, dtype=np.float32)[::-1]
    backing = np.full(24, -731, dtype=np.float32)
    out = backing[1:23:2]
    assert not x.flags.c_contiguous
    assert not y.flags.c_contiguous
    assert not out.flags.c_contiguous
    handle = interpret(
        _vectors,
        _add,
        tuple(_descriptor(1, name) for name in ("x", "y", "out")),
    )
    _run_and_compare(handle, [x, y, out], x + y, optimize=True)
    untouched = np.ones(backing.shape, dtype=bool)
    untouched[1:23:2] = False
    np.testing.assert_array_equal(backing[untouched], -731)


def test_nested_arrangement_reduces_across_multiple_inner_tiles():
    x = (np.arange(33, dtype=np.float32).reshape(3, 11) - 16) / 4
    out = np.empty((3, 1), dtype=np.float32)
    handle = interpret(
        _nested_row_tiles,
        _nested_row_sum,
        (_descriptor(2, "x", other=0), _descriptor(2, "out")),
        trace=True,
    )
    expected = np.sum(x, axis=1, keepdims=True)
    _run_and_compare(handle, [x, out], expected, optimize=True)


def test_trace_reports_the_effective_arrangement_tail_mask():
    x = np.arange(11, dtype=np.float32)
    y = np.ones_like(x)
    out = np.full_like(x, -99)
    handle = interpret(
        _vectors,
        _add,
        tuple(_descriptor(1, name) for name in ("x", "y", "out")),
        trace=True,
    )
    result = handle(x, y, out)
    stores = [event for event in result.trace if event.opcode == "mem.store"]
    assert [event.mask["value"] for event in stores] == [
        [True, True, True, True],
        [True, True, True, True],
        [True, True, True, False],
    ]
    assert stores[-1].mask["dtype"] == "bool"
    assert stores[-1].mask["shape"] == [4]
    _check(out, x + y)


@pytest.mark.parametrize("backend", ("cuda", "triton"))
def test_high_level_backend_option_executes_the_real_pass_pipeline(backend):
    handle = interpret(
        _vectors,
        _add,
        tuple(_descriptor(1, name) for name in ("x", "y", "out")),
        backend=backend,
    )
    assert handle.program.metadata["target_backend"] == backend
    assert f"ssa.{backend}.optimize_schedule" in handle.program.metadata["pass_trace"]
    assert "pass_trace" not in handle.frontend_program.metadata
    x = np.arange(11, dtype=np.float32)
    out = np.empty_like(x)
    handle(x, x, out)
    _check(out, x * 2)


@pytest.mark.parametrize("backend", ("cuda", "triton"))
@pytest.mark.parametrize("dtype", (np.float32, np.int32))
@pytest.mark.parametrize("shape", ((3, 1, 2), (3, 3, 2), (2, 4, 3)))
def test_fixed_k_tiled_dot_matches_numpy_after_target_decomposition(
    backend, dtype, shape
):
    rows, inner, columns = shape
    rng = np.random.default_rng(2026)
    a = rng.integers(-7, 8, size=(rows, inner)).astype(dtype)
    b = rng.integers(-7, 8, size=(inner, columns)).astype(dtype)

    if dtype == np.float32:
        a /= 3
        b /= 5

    expected = a @ b
    tensors = (
        _descriptor(2, "a", np.dtype(dtype).name, other=0),
        _descriptor(2, "b", np.dtype(dtype).name, other=0),
        _descriptor(2, "out", np.dtype(dtype).name),
    )
    raw = interpret(_matrix_tiles, _dot, tensors)
    raw_out = np.empty_like(expected)
    raw(a, b, raw_out)
    _check(raw_out, expected)
    handle = interpret(_matrix_tiles, _dot, tensors, backend=backend, trace=True)
    assert handle.program.metadata["linalg_decomposed"]
    operations = handle.program.blocks[0].operations
    assert any(op.opcode == "shape.dim" and op.attrs["dim"] == -1 for op in operations)
    assert not any(op.opcode == "linalg.dot" for op in operations)
    backing = np.full(expected.size + 2, -731, dtype=dtype)
    out = backing[1:-1].reshape(expected.shape)
    result = handle(a, b, out)
    _check(out, expected)
    np.testing.assert_array_equal(backing[[0, -1]], np.array([-731, -731], dtype=dtype))
    stores = [event for event in result.trace if event.opcode == "mem.store"]
    assert len(stores) == rows * columns
    assert {event.lane for event in stores} == set(np.ndindex(expected.shape))

    for event in stores:
        mask = np.asarray(event.mask["value"])
        assert mask.sum() == 1
        assert mask[event.lane]

    repeated = handle(a, b, np.full_like(expected, -731))
    assert result.trace == repeated.trace


@pytest.mark.parametrize("backend", ("cuda", "triton"))
@pytest.mark.parametrize("dtype", (np.float32, np.int32))
@pytest.mark.parametrize("inner", (1, 3, 7))
def test_untiled_non_square_dot_executes_every_output_lane(backend, dtype, inner):
    a = (np.arange(3 * inner).reshape(3, inner) - 4).astype(dtype)
    b = (np.arange(inner * 2).reshape(inner, 2) + 2).astype(dtype)
    out = np.full((3, 2), -731, dtype=dtype)
    handle = interpret(
        _whole_matrices,
        _dot,
        tuple(_descriptor(2, name, np.dtype(dtype).name) for name in ("a", "b", "out")),
        backend=backend,
    )
    handle(a, b, out)
    _check(out, a @ b)


@pytest.mark.parametrize("backend", ("cuda", "triton"))
@pytest.mark.parametrize("arrangement", (_transpose_tiles, _whole_transpose))
@pytest.mark.parametrize("dtype", (np.float32, np.int32))
def test_non_square_transpose_matches_numpy_after_decomposition(
    backend, arrangement, dtype
):
    x = np.arange(6, dtype=dtype).reshape(3, 2)
    out = np.full((2, 3), -731, dtype=dtype)
    handle = interpret(
        arrangement,
        _transpose,
        (
            _descriptor(2, "x", np.dtype(dtype).name),
            _descriptor(2, "out", np.dtype(dtype).name),
        ),
        backend=backend,
    )
    handle(x, out)
    _check(out, x.T)


@pytest.mark.parametrize("backend", ("cuda", "triton"))
@pytest.mark.parametrize("shape", ((3, 5, 2),))
def test_unreduced_k_program_dot_is_rejected_without_partial_writes(backend, shape):
    rows, inner, columns = shape
    out = np.full((rows, columns), -731, dtype=np.float32)
    handle = interpret(
        _matrix_tiles,
        _dot,
        (
            _descriptor(2, "a", other=0),
            _descriptor(2, "b", other=0),
            _descriptor(2, "out"),
        ),
        backend=backend,
    )

    with pytest.raises(UnsupportedOperationError, match="multiple arranged programs"):
        handle(
            np.ones((rows, inner), dtype=np.float32),
            np.ones((inner, columns), dtype=np.float32),
            out,
        )

    np.testing.assert_array_equal(out, np.full(out.shape, -731, dtype=np.float32))
