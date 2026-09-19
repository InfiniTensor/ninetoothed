"""Tests for the CPU reference interpreter (``ninetoothed.interpret``).

These tests are deliberately GPU-free: they need only NumPy, so they run on any
machine and in CI without a CUDA device.  They are organised as

* *semantics* — the interpreter must agree with a NumPy reference;
* *layout* — the resolved access map must agree with the compiler's own
  :func:`ninetoothed.eval._eval`, which is the authoritative view of the same
  mapping;
* *invariants* — masked accesses must never touch the backing buffer;
* *diagnostics* — failures must be reported against a specific SSA operation;
* *tooling* — tracing, differential comparison and reproduction rendering;
* *bisection* — per-pass semantic comparison and minimal reproductions.
"""

# A NineToothed application writes its result by assigning to the output
# parameter, and the frontend lowers that assignment to `mem.store`.  The
# assignment is therefore meaningful even though Python never reads it back,
# which is why every application body carries a per-line noqa marker.

import dataclasses
import json
import subprocess
import sys

import numpy as np
import pytest

import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor
from ninetoothed.compiler.passes import Pass, create_default_registry
from ninetoothed.eval import _eval
from ninetoothed.interpret import (
    UNSUPPORTED_OPERATIONS,
    MissingSymbolError,
    ProgramDomainError,
    Tracer,
    UnsupportedAccessError,
    UnsupportedDTypeError,
    UnsupportedOperationError,
    access_mask,
    access_offsets,
    build_reproduction,
    compare_interpretations,
    compare_passes,
    compare_pipeline,
    format_support_matrix,
    interpret,
    interpret_program,
    random_inputs,
    render_reproduction,
    supported_opcodes,
)

BLOCK_SIZE = Symbol("BLOCK_SIZE", meta=True)
WIDTH = Symbol("WIDTH", meta=True)
NEG_INF = float("-inf")


def tiled(x, out, BLOCK_SIZE=BLOCK_SIZE):
    return x.tile((BLOCK_SIZE,)), out.tile((BLOCK_SIZE,))


def row_tiled(x, out, WIDTH=WIDTH):
    return x.tile((1, WIDTH)), out.tile((1, WIDTH))


def _double_plus_one(x, out):
    out = x * 2.0 + 1.0  # noqa: F841


def _double(x, out):
    out = x * 2.0  # noqa: F841


def _row_sum(x, out):
    out = ntl.sum(x, axis=1)  # noqa: F841


def _softmax_application(x, out):
    shifted = x - ntl.max(x, axis=1)[:, None]
    numerator = ntl.exp(shifted)
    out = numerator / ntl.sum(numerator, axis=1)[:, None]  # noqa: F841


BLOCK_SIZE_M = Symbol("BLOCK_SIZE_M", meta=True)
BLOCK_SIZE_N = Symbol("BLOCK_SIZE_N", meta=True)
BLOCK_SIZE_K = Symbol("BLOCK_SIZE_K", meta=True)


def _matmul_arrangement(
    lhs,
    rhs,
    output,
    BLOCK_SIZE_M=BLOCK_SIZE_M,
    BLOCK_SIZE_N=BLOCK_SIZE_N,
    BLOCK_SIZE_K=BLOCK_SIZE_K,
):
    """Mirrors ``tests/test_matmul.py``: nested tiles, expand and squeeze."""
    output_tiled = output.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))

    lhs_tiled = (
        lhs.tile((BLOCK_SIZE_M, BLOCK_SIZE_K))
        .tile((1, -1))
        .expand((-1, output_tiled.shape[1]))
    )
    lhs_tiled.dtype = lhs_tiled.dtype.squeeze(0)

    rhs_tiled = (
        rhs.tile((BLOCK_SIZE_K, BLOCK_SIZE_N))
        .tile((-1, 1))
        .expand((output_tiled.shape[0], -1))
    )
    rhs_tiled.dtype = rhs_tiled.dtype.squeeze(1)

    return lhs_tiled, rhs_tiled, output_tiled


def _matmul_application(lhs, rhs, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)

    for k in range(lhs.shape[0]):
        accumulator += ntl.dot(lhs[k], rhs[k])

    output = accumulator


def run(arrangement, application, arrays, *, symbols, tensors=None, **kwargs):
    """Interpret an application over concrete arrays."""
    if tensors is None:
        tensors = tuple(Tensor(np.asarray(array).ndim) for array in arrays)

    return interpret(
        arrangement,
        application,
        tensors=tensors,
        inputs=tuple(np.asarray(array) for array in arrays),
        symbols=symbols,
        **kwargs,
    )


# -------------------------------- semantics --------------------------------


def test_elementwise_tile_with_masked_tail():
    x = np.arange(10, dtype=np.float32)
    out = np.zeros_like(x)

    def application(x, out):
        out = x * 2.0 + 1.0  # noqa: F841

    result = run(tiled, application, (x, out), symbols={"BLOCK_SIZE": 4})

    assert result.launch_shape == (3,)
    np.testing.assert_array_equal(result.output("out"), x * 2.0 + 1.0)


def test_elementwise_does_not_widen_float32():
    x = np.full(4, 0.1, dtype=np.float32)
    out = np.zeros_like(x)

    def application(x, out):
        out = x + x  # noqa: F841

    result = run(tiled, application, (x, out), symbols={"BLOCK_SIZE": 4})
    produced = result.output("out")

    assert produced.dtype == np.float32
    np.testing.assert_array_equal(
        produced, (x.astype(np.float32) + x).astype(np.float32)
    )


def test_masked_elements_never_touch_the_buffer():
    """The tail of a partial tile must stay exactly as the caller left it."""
    x = np.arange(10, dtype=np.float32)
    out = np.full(10, -999.0, dtype=np.float32)

    def application(x, out):
        out = x * 2.0  # noqa: F841

    result = run(tiled, application, (x, out), symbols={"BLOCK_SIZE": 4})

    assert list(result.output("out")[10:]) == []
    np.testing.assert_array_equal(result.output("out"), x * 2.0)


def test_broadcast_row_scalar_and_row_vector():
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = np.zeros_like(x)

    def application(x, out):
        offset = ntl.full((1,), 1.0, dtype=ntl.float32)
        out = x + offset + 2.0  # noqa: F841

    result = run(row_tiled, application, (x, out), symbols={"WIDTH": 4})
    np.testing.assert_array_equal(result.output("out"), x + 3.0)


def test_row_reduction():
    x = np.arange(28, dtype=np.float32).reshape(4, 7)
    out = np.zeros(4, dtype=np.float32)

    def application(x, out):
        out = ntl.sum(x, axis=1)  # noqa: F841

    result = run(row_tiled, application, (x, out), symbols={"WIDTH": 7})

    assert result.launch_shape == (4,)
    np.testing.assert_array_equal(result.output("out"), x.sum(axis=1))


def test_row_max_and_min():
    x = np.array([[3.0, -1.0, 7.0], [0.0, 5.0, 2.0]], dtype=np.float32)
    out = np.zeros(2, dtype=np.float32)

    def application_max(x, out):
        out = ntl.max(x, axis=1)  # noqa: F841

    result = run(row_tiled, application_max, (x, out.copy()), symbols={"WIDTH": 3})
    np.testing.assert_array_equal(result.output("out"), x.max(axis=1))

    def application_min(x, out):
        out = ntl.min(x, axis=1)  # noqa: F841

    result = run(row_tiled, application_min, (x, out.copy()), symbols={"WIDTH": 3})
    np.testing.assert_array_equal(result.output("out"), x.min(axis=1))


def test_loop_carried_row_sum():
    """Mirrors ``tests/test_generation.py::test_loop_carried_row_sum``."""
    x = np.arange(21, dtype=np.float32).reshape(3, 7)
    out = np.zeros((3, 1), dtype=np.float32)

    def arrangement(x, out, BLOCK_SIZE=BLOCK_SIZE):
        return x.tile((1, BLOCK_SIZE)).tile((1, -1)), out.tile((1, 1))

    def application(x, out):
        acc = ntl.zeros(out.shape, dtype=out.dtype)

        for i in range(x.shape[1]):
            acc += ntl.sum(x[0, i], axis=-1)

        out = acc

    result = run(
        arrangement,
        application,
        (x, out),
        symbols={"BLOCK_SIZE": 7},
        tensors=(Tensor(2, other=0), Tensor(2)),
    )

    assert result.launch_shape == (3, 1)
    np.testing.assert_array_equal(result.output("out"), x.sum(axis=-1, keepdims=True))


def test_softmax_uses_other_for_masked_lanes():
    """A masked lane must contribute ``other`` (``-inf``), not a stale value."""
    x = np.random.default_rng(0).random((3, 11), dtype=np.float32)
    out = np.zeros_like(x)

    def application(x, out):
        shifted = x - ntl.max(x, axis=1)[:, None]
        numerator = ntl.exp(shifted)
        out = numerator / ntl.sum(numerator, axis=1)[:, None]  # noqa: F841

    result = run(
        row_tiled,
        application,
        (x, out),
        symbols={"WIDTH": 16},
        tensors=(Tensor(2, other=NEG_INF), Tensor(2)),
    )
    shifted = x - x.max(axis=1, keepdims=True)
    expected = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)

    np.testing.assert_allclose(result.output("out"), expected, rtol=1e-5, atol=1e-6)


def test_layernorm_two_reductions_and_rsqrt():
    x = np.random.default_rng(1).standard_normal((4, 8)).astype(np.float32)
    out = np.zeros_like(x)

    def application(x, out):
        width = x.shape[1]
        mean = ntl.sum(x, axis=1) / width
        mean_square = ntl.sum(x * x, axis=1) / width
        variance = mean_square - mean * mean
        out = (x - mean[:, None]) * ntl.rsqrt(variance[:, None] + 1e-5)  # noqa: F841

    result = run(row_tiled, application, (x, out), symbols={"WIDTH": 8})
    expected = (x - x.mean(axis=1, keepdims=True)) / np.sqrt(
        x.var(axis=1, keepdims=True) + 1e-5
    )

    np.testing.assert_allclose(result.output("out"), expected, rtol=1e-4, atol=1e-5)


def test_integer_arithmetic_is_bit_exact():
    """Integer results must match the C semantics the CUDA backend emits."""
    x = np.array([-7, -1, 0, 1, 2, 13, 100, -100, 5], dtype=np.int32)
    out = np.zeros_like(x)

    def application(x, out):
        out = (x * 3 - 7) % 5  # noqa: F841

    result = run(tiled, application, (x, out), symbols={"BLOCK_SIZE": 4})

    # C-style remainder: the sign follows the dividend, unlike Python's `%`.
    assert result.output("out").dtype == np.int32
    np.testing.assert_array_equal(result.output("out"), np.fmod(x * 3 - 7, 5))


def test_bool_comparison_and_select():
    x = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 0.0], dtype=np.float32)
    y = np.array([4.0, 2.0, 2.0, 1.0, 9.0, 0.0], dtype=np.float32)
    out = np.zeros(6, dtype=np.float32)

    def application(x, y, out):
        out = ntl.where(x > y, x, y)  # noqa: F841

    result = run(
        lambda a, b, c, BLOCK_SIZE=BLOCK_SIZE: (
            a.tile((BLOCK_SIZE,)),
            b.tile((BLOCK_SIZE,)),
            c.tile((BLOCK_SIZE,)),
        ),
        application,
        (x, y, out),
        symbols={"BLOCK_SIZE": 6},
    )

    np.testing.assert_array_equal(result.output("out"), np.maximum(x, y))


def test_cast_between_dtypes():
    x = np.array([1.7, -2.9, 3.5, 0.4], dtype=np.float32)
    out = np.zeros(4, dtype=np.int32)

    def application(x, out):
        out = x.to(ntl.int32)  # noqa: F841

    result = run(tiled, application, (x, out), symbols={"BLOCK_SIZE": 4})

    # C-style truncation towards zero.
    np.testing.assert_array_equal(result.output("out"), np.trunc(x).astype(np.int32))


BLOCK_SIZE_M = Symbol("BLOCK_SIZE_M", meta=True)
BLOCK_SIZE_N = Symbol("BLOCK_SIZE_N", meta=True)
BLOCK_SIZE_K = Symbol("BLOCK_SIZE_K", meta=True)


def matmul_arrangement(
    lhs,
    rhs,
    output,
    BLOCK_SIZE_M=BLOCK_SIZE_M,
    BLOCK_SIZE_N=BLOCK_SIZE_N,
    BLOCK_SIZE_K=BLOCK_SIZE_K,
):
    """Mirror the arrangement from ``tests/test_matmul.py`` (expand and squeeze)."""
    output_tiled = output.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))

    lhs_tiled = (
        lhs.tile((BLOCK_SIZE_M, BLOCK_SIZE_K))
        .tile((1, -1))
        .expand((-1, output_tiled.shape[1]))
    )
    lhs_tiled.dtype = lhs_tiled.dtype.squeeze(0)

    rhs_tiled = (
        rhs.tile((BLOCK_SIZE_K, BLOCK_SIZE_N))
        .tile((-1, 1))
        .expand((output_tiled.shape[0], -1))
    )
    rhs_tiled.dtype = rhs_tiled.dtype.squeeze(1)

    return lhs_tiled, rhs_tiled, output_tiled


def test_matmul_with_expand_and_squeeze():
    """Exercises nested tiles, expand, squeeze, linalg.dot and scf.for."""
    lhs = np.random.default_rng(0).standard_normal((4, 8)).astype(np.float32)
    rhs = np.random.default_rng(1).standard_normal((8, 4)).astype(np.float32)
    output = np.zeros((4, 4), dtype=np.float32)

    def application(lhs, rhs, output):
        accumulator = ntl.zeros(output.shape, dtype=ntl.float32)

        for k in range(lhs.shape[0]):
            accumulator += ntl.dot(lhs[k], rhs[k])

        output = accumulator

    result = interpret(
        matmul_arrangement,
        application,
        tensors=(Tensor(2), Tensor(2), Tensor(2)),
        inputs=(lhs, rhs, output),
        symbols={"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 4, "BLOCK_SIZE_K": 8},
    )

    np.testing.assert_allclose(result.output("output"), lhs @ rhs, rtol=1e-5, atol=1e-5)


def test_matmul_with_masked_k_tail():
    """A K block that does not divide the reduction length must not read garbage."""
    lhs = np.random.default_rng(2).standard_normal((2, 5)).astype(np.float32)
    rhs = np.random.default_rng(3).standard_normal((5, 3)).astype(np.float32)
    output = np.zeros((2, 3), dtype=np.float32)

    def application(lhs, rhs, output):
        accumulator = ntl.zeros(output.shape, dtype=ntl.float32)

        for k in range(lhs.shape[0]):
            accumulator += ntl.dot(lhs[k], rhs[k])

        output = accumulator

    result = interpret(
        matmul_arrangement,
        application,
        tensors=(Tensor(2, other=0), Tensor(2, other=0), Tensor(2)),
        inputs=(lhs, rhs, output),
        symbols={"BLOCK_SIZE_M": 2, "BLOCK_SIZE_N": 3, "BLOCK_SIZE_K": 8},
    )

    np.testing.assert_allclose(result.output("output"), lhs @ rhs, rtol=1e-5, atol=1e-5)


def test_transpose():
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = np.zeros(2, dtype=np.float32)

    def arrangement(x, out, BLOCK_SIZE=BLOCK_SIZE):
        return x.tile((1, BLOCK_SIZE)), out.tile((1,))

    def application(x, out):
        out = ntl.sum(ntl.trans(x), axis=0)  # noqa: F841

    result = run(
        arrangement,
        application,
        (x, out),
        symbols={"BLOCK_SIZE": 3},
        tensors=(Tensor(2), Tensor(1)),
    )

    np.testing.assert_allclose(result.output("out"), x.T.sum(axis=0), rtol=1e-6)


# --------------------------------- layout ----------------------------------


def _expected_layout(arrangement, arrays, symbols_map, name, index):
    """Return the compiler's own access map for ``name`` (mask folded to -1)."""
    descriptors = tuple(Tensor(np.asarray(array).ndim) for array in arrays)
    arranged = arrangement(*descriptors)
    target = dict(zip(("x", "y", "out"), arranged))
    arranged_tensor = target[name]
    substitutions = {arranged_tensor.source: {"shape": np.asarray(arrays[0]).shape}}
    substitutions.update(symbols_map)
    offsets = _eval(arranged_tensor, substitutions)

    return offsets if index == 0 else offsets >= 0


@pytest.mark.parametrize(
    "label,arrangement,application,shape,symbols_map",
    [
        (
            "1d tile with masked tail",
            tiled,
            lambda x, out: None,
            (10,),
            {BLOCK_SIZE: 4},
        ),
        ("2d row tile", row_tiled, lambda x, out: None, (4, 7), {WIDTH: 5}),
    ],
)
def test_access_map_matches_the_compiler(
    label, arrangement, application, shape, symbols_map
):
    """The interpreter's layout must agree with ``ninetoothed.eval._eval``."""
    del label, application

    x = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    out = np.zeros_like(x)

    def noop(x, out):
        out = x + 0.0  # noqa: F841

    result = run(
        arrangement, noop, (x, out), symbols={str(k): v for k, v in symbols_map.items()}
    )
    mine = np.where(access_mask(result, "x"), access_offsets(result, "x"), -1)
    theirs = _expected_layout(arrangement, (x, out), symbols_map, "x", 0)

    assert mine.shape == theirs.shape
    np.testing.assert_array_equal(mine, theirs)


def test_nested_tile_access_map_covers_both_levels():
    x = np.arange(32, dtype=np.float32).reshape(4, 8)
    out = np.zeros(4, dtype=np.float32)

    def arrangement(x, out, BLOCK_SIZE=BLOCK_SIZE):
        return x.tile((1, BLOCK_SIZE)).tile((1, -1)), out.tile((1,))

    def application(x, out):
        out = ntl.sum(x[0, 0], axis=-1)  # noqa: F841

    result = run(
        arrangement,
        application,
        (x, out),
        symbols={"BLOCK_SIZE": 8},
        tensors=(Tensor(2), Tensor(1)),
    )
    mine = np.where(access_mask(result, "x"), access_offsets(result, "x"), -1)
    theirs = _expected_layout(arrangement, (x, out), {BLOCK_SIZE: 8}, "x", 0)

    assert mine.shape == theirs.shape == (4, 1, 1, 1, 1, 8)
    np.testing.assert_array_equal(mine, theirs)


def test_masks_are_reported_per_program_instance():
    x = np.arange(9, dtype=np.float32)
    out = np.zeros_like(x)

    def application(x, out):
        out = x + 1.0  # noqa: F841

    result = run(tiled, application, (x, out), symbols={"BLOCK_SIZE": 4})
    mask = access_mask(result, "x")

    assert mask.shape == (3, 4)
    assert mask.tolist() == [
        [True, True, True, True],
        [True, True, True, True],
        [True, False, False, False],
    ]


# ------------------------------ invariants ---------------------------------


def test_unmasked_out_of_bounds_store_is_reported():
    """An access the layout claims is in bounds but which is not must not be silent."""
    from ninetoothed.interpret.expr import parse_expression
    from ninetoothed.interpret.memory import AccessContext, TensorRuntime

    # A tensor whose declared extent (8) is larger than its backing buffer (4):
    # the layout says every access is in bounds, so the interpreter must notice
    # that the buffer disagrees rather than silently corrupting memory.
    tensor = TensorRuntime(
        name="t",
        buffer=np.zeros(4, dtype=np.float32),
        dtype="float32",
        levels=((8,),),
        source_shape=(8,),
        source_strides=(1,),
        view_shape=(1,),
        template=None,
        view_offsets=parse_expression("index"),
        view_mask=parse_expression("True"),
    )
    context = AccessContext({}, 0)

    with pytest.raises(UnsupportedAccessError, match="outside the buffer"):
        context.write(tensor.root_view(0), np.zeros(8, dtype=np.float32))

    with pytest.raises(UnsupportedAccessError, match="outside the buffer"):
        context.read(tensor.root_view(0))


def test_masked_out_of_bounds_access_is_allowed():
    """A masked-out lane may point anywhere, because it is never dereferenced."""
    from ninetoothed.interpret.expr import parse_expression
    from ninetoothed.interpret.memory import AccessContext, TensorRuntime

    tensor = TensorRuntime(
        name="t",
        buffer=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        dtype="float32",
        levels=((8,),),
        source_shape=(8,),
        source_strides=(1,),
        view_shape=(1,),
        template=None,
        view_offsets=parse_expression("index"),
        view_mask=parse_expression("index < 4"),
        other=-1.0,
    )
    context = AccessContext({}, 0)
    values = context.read(tensor.root_view(0))

    np.testing.assert_array_equal(
        values, np.array([1.0, 2.0, 3.0, 4.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
    )


def test_symbol_can_be_bound_by_parameter_name():
    x = np.arange(8, dtype=np.float32)
    out = np.zeros_like(x)

    def application(x, out):
        out = x * 2.0  # noqa: F841

    by_symbol = run(tiled, application, (x, out.copy()), symbols={"BLOCK_SIZE": 4})
    by_parameter = run(tiled, application, (x, out.copy()), symbols={"BLOCK_SIZE": 4})

    np.testing.assert_array_equal(by_symbol.output("out"), by_parameter.output("out"))


def test_unknown_symbol_is_rejected_with_a_hint():
    x = np.arange(8, dtype=np.float32)
    out = np.zeros_like(x)

    def application(x, out):
        out = x * 2.0  # noqa: F841

    with pytest.raises(MissingSymbolError, match="Unknown symbol"):
        run(tiled, application, (x, out), symbols={"NOT_A_SYMBOL": 4})


def test_mismatched_launch_domains_are_rejected():
    x = np.arange(10, dtype=np.float32)
    out = np.zeros(10, dtype=np.float32)

    def arrangement(x, out, BLOCK_SIZE=BLOCK_SIZE):
        return x.tile((BLOCK_SIZE,)), out.tile((1,))

    def application(x, out):
        out = ntl.sum(x)  # noqa: F841

    with pytest.raises(ProgramDomainError, match="program instance"):
        run(arrangement, application, (x, out), symbols={"BLOCK_SIZE": 4})


def test_unsupported_dtype_is_rejected_with_context():
    import ninetoothed

    x = np.arange(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)

    def application(x, out):
        out = x + 1.0  # noqa: F841

    with pytest.raises(UnsupportedDTypeError, match="bfloat16"):
        interpret(
            tiled,
            application,
            tensors=(Tensor(1, dtype=ninetoothed.bfloat16), Tensor(1)),
            inputs=(x, out),
            symbols={"BLOCK_SIZE": 4},
        )


def test_unsupported_opcode_names_the_operation():
    """An unhandled opcode must be reported against its SSA location."""
    from ninetoothed.interpret.registry import require_handler

    with pytest.raises(UnsupportedOperationError, match="totally.made.up"):
        require_handler("totally.made.up", location="entry:7")


def test_target_intrinsics_are_reported_clearly():
    from ninetoothed.interpret.registry import require_handler

    with pytest.raises(UnsupportedOperationError, match="cdiv"):
        require_handler("call.cdiv", location="entry:3")


def _import_environment():
    """Return an environment where a bare ``python -c`` can import the package.

    ``PYTHONPATH`` is replaced rather than extended, so a stub or a partially
    installed Triton on the developer's machine cannot make the check pass by
    accident.
    """
    import os
    import pathlib

    environment = dict(os.environ)
    source = pathlib.Path(__file__).resolve().parents[1] / "src"

    if source.is_dir():
        environment["PYTHONPATH"] = str(source)

    return environment


def test_importing_the_interpreter_pulls_in_no_cuda_runtime():
    """The interpreter must be usable where no CUDA runtime exists at all."""
    script = (
        "import sys\n"
        "import ninetoothed.interpret\n"
        "leaked = sorted(\n"
        "    {name.split('.')[0] for name in sys.modules}\n"
        "    & {'triton', 'torch', 'tilelang'}\n"
        ")\n"
        "print(','.join(leaked))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env=_import_environment(),
    )

    assert completed.stdout.strip() == "", completed.stdout


def test_support_matrix_declares_known_gaps():
    matrix = format_support_matrix()

    assert "arith.add" in matrix
    assert "mem.store" in matrix
    assert "reduce.sum" in matrix
    assert "math.rand" in UNSUPPORTED_OPERATIONS
    assert "mem.atomic_add" in UNSUPPORTED_OPERATIONS


def test_every_frontend_opcode_is_covered_or_declared():
    """Guard against the frontend growing an opcode the interpreter silently drops."""
    covered = set(supported_opcodes()) | set(UNSUPPORTED_OPERATIONS)
    expected = {
        "arith.add",
        "arith.sub",
        "arith.mul",
        "arith.div",
        "arith.floordiv",
        "arith.mod",
        "arith.pow",
        "arith.constant",
        "arith.neg",
        "arith.pos",
        "arith.not",
        "arith.invert",
        "arith.and",
        "arith.or",
        "arith.maximum",
        "arith.minimum",
        "arith.bitwise_and",
        "arith.bitwise_or",
        "arith.bitwise_xor",
        "arith.bitwise_left_shift",
        "arith.bitwise_right_shift",
        "cmp.eq",
        "cmp.ne",
        "cmp.lt",
        "cmp.le",
        "cmp.gt",
        "cmp.ge",
        "math.exp",
        "math.exp2",
        "math.log",
        "math.sqrt",
        "math.rsqrt",
        "math.tanh",
        "math.pow",
        "select.where",
        "tensor.zeros",
        "tensor.full",
        "tensor.extract",
        "tensor.view",
        "tensor.cast",
        "tensor.stride",
        "shape.dim",
        "index.offset",
        "symbol.attr",
        "tuple.construct",
        "mem.load",
        "mem.store",
        "mem.data_ptr",
        "mem.atomic_add",
        "reduce.sum",
        "reduce.max",
        "reduce.min",
        "linalg.dot",
        "linalg.matmul",
        "linalg.transpose",
        "scf.for",
        "scf.if",
        "scf.yield",
    }

    missing = sorted(expected - covered)

    assert expected <= covered, f"Missing opcode(s): {missing}."


# -------------------------------- tooling ----------------------------------


def test_trace_records_opcodes_masks_and_program_ids():
    x = np.arange(9, dtype=np.float32)
    out = np.zeros_like(x)

    def application(x, out):
        out = x + 1.0  # noqa: F841

    result = run(
        tiled,
        application,
        (x, out),
        symbols={"BLOCK_SIZE": 4},
        tracer=Tracer(opcodes={"mem.store"}),
    )
    events = result.trace_events(opcode="mem.store")

    assert [event.program_id for event in events] == [0, 1, 2]
    assert [event.mask for event in events] == [
        "4/4 active",
        "4/4 active",
        "1/4 active",
    ]
    assert "mem.store" in result.render_trace()


def test_trace_snapshots_can_be_serialized():
    x = np.arange(4, dtype=np.float32)
    out = np.zeros_like(x)

    def application(x, out):
        out = x + 1.0  # noqa: F841

    result = run(
        tiled,
        application,
        (x, out),
        symbols={"BLOCK_SIZE": 4},
        tracer=Tracer(),
    )
    payload = json.loads(json.dumps(result.trace[0].to_dict(), default=str))

    assert payload["opcode"] == "arith.constant"
    assert "operands" in payload and "results" in payload
    assert any(event.opcode == "arith.add" for event in result.trace)


def test_interpret_program_reproduces_a_run():
    x = np.arange(10, dtype=np.float32)
    out = np.zeros_like(x)

    def application(x, out):
        out = x * 3.0  # noqa: F841

    original = run(tiled, application, (x, out), symbols={"BLOCK_SIZE": 4})
    buffers = {
        name: tensor.buffer.reshape(tuple(int(dim) for dim in tensor.source_shape))
        for name, tensor in original.memory.tensors.items()
    }
    replayed = interpret_program(original.program, buffers, symbols=original.symbols)

    np.testing.assert_array_equal(replayed.output("out"), original.output("out"))


def test_compare_interpretations_detects_a_perturbation():
    x = np.arange(10, dtype=np.float32)
    out = np.zeros_like(x)

    def application(x, out):
        out = x * 2.0  # noqa: F841

    reference = run(tiled, application, (x, out), symbols={"BLOCK_SIZE": 4})
    suspect = run(tiled, application, (x, out.copy()), symbols={"BLOCK_SIZE": 4})
    suspect.outputs["out"] = suspect.outputs["out"].copy()
    suspect.outputs["out"][0] += 1.0

    diff = compare_interpretations(reference, suspect, label="perturbed")

    assert not diff.matches
    assert diff.mismatching_outputs == ("out",)
    assert diff.outputs["out"].mismatches == 1
    assert "[DIFF ]" in diff.render()
    assert json.loads(diff.to_json())["mismatching_outputs"] == ["out"]


def test_compare_interpretations_accepts_identical_runs():
    x = np.arange(10, dtype=np.float32)
    out = np.zeros_like(x)

    def application(x, out):
        out = x * 2.0  # noqa: F841

    left = run(tiled, application, (x, out), symbols={"BLOCK_SIZE": 4}, tracer=Tracer())
    right = run(
        tiled, application, (x, out.copy()), symbols={"BLOCK_SIZE": 4}, tracer=Tracer()
    )
    diff = compare_interpretations(left, right, label="same run twice")

    assert diff.matches
    assert diff.trace is not None and diff.trace.matches


def test_compare_pipeline_is_semantics_preserving():
    x = np.random.default_rng(3).random((3, 11), dtype=np.float32)
    out = np.zeros_like(x)

    def application(x, out):
        shifted = x - ntl.max(x, axis=1)[:, None]
        numerator = ntl.exp(shifted)
        out = numerator / ntl.sum(numerator, axis=1)[:, None]  # noqa: F841

    diff = compare_pipeline(
        row_tiled,
        application,
        tensors=(Tensor(2, other=NEG_INF), Tensor(2)),
        inputs=(x, out),
        symbols={"WIDTH": 16},
        pipeline=[],
        trace=True,
    )

    assert diff.matches, diff.render()


def test_trace_divergence_is_located():
    x = np.arange(10, dtype=np.float32)
    out = np.zeros_like(x)

    def application(x, out):
        out = x * 2.0  # noqa: F841

    narrow = run(
        tiled, application, (x, out.copy()), symbols={"BLOCK_SIZE": 4}, tracer=Tracer()
    )
    wide = run(
        tiled, application, (x, out.copy()), symbols={"BLOCK_SIZE": 8}, tracer=Tracer()
    )
    diff = compare_interpretations(narrow, wide, label="mask widths")

    assert not diff.matches
    assert diff.trace is not None
    assert "mask" in diff.trace.divergences[0].reason


def test_reproduction_snippet_contains_the_essentials():
    x = np.arange(10, dtype=np.float32)
    out = np.zeros_like(x)

    def application(x, out):
        out = x * 2.0  # noqa: F841

    result = run(tiled, application, (x, out), symbols={"BLOCK_SIZE": 4})
    snippet = render_reproduction(result, label="elementwise")

    assert "launch shape : (3,)" in snippet
    assert "ssa @" in snippet
    assert "BUFFERS = {" in snippet
    assert "SYMBOLS = {" in snippet
    assert "def application" in snippet
    # The snippet must be syntactically valid Python.
    compile(snippet, "<reproduction>", "exec")


def test_access_map_can_be_restricted_to_one_instance():
    from ninetoothed.interpret import access_map

    x = np.arange(10, dtype=np.float32)
    out = np.zeros_like(x)

    def application(x, out):
        out = x * 2.0  # noqa: F841

    result = run(tiled, application, (x, out), symbols={"BLOCK_SIZE": 4})
    subset = access_map(result, "x", program_ids=[2])

    assert set(subset) == {2}
    offsets, mask = subset[2]
    # Offsets 8 and 9 are real elements; 10 and 11 are the masked tail.
    assert offsets.tolist() == [8, 9, 10, 11]
    assert mask.tolist() == [True, True, False, False]


# ------------------------ pass bisection and reproductions -------------------------


class _WrongConstant(Pass):
    """A deliberately broken pass: it bumps the first float constant.

    Registering it into a private registry is how the bisection is tested — a pass
    that is *not* semantics preserving must be named as the culprit.
    """

    name = "test.wrong_constant"

    def run(self, program, context):
        changed = False

        def fix(block):
            nonlocal changed
            operations = []

            for operation in block.operations:
                if not changed and operation.opcode == "arith.constant":
                    value = operation.attrs.get("value")

                    if isinstance(value, float):
                        operation = dataclasses.replace(
                            operation, attrs={**operation.attrs, "value": value + 1.0}
                        )
                        changed = True

                operations.append(operation)

            return dataclasses.replace(block, operations=tuple(operations))

        return dataclasses.replace(
            program, blocks=tuple(fix(block) for block in program.blocks)
        )


def _registry_with_wrong_constant():
    registry = create_default_registry()
    registry.register(_WrongConstant)

    return registry


@pytest.mark.parametrize(
    "name,arrangement,application,tensors,shapes,symbols,seed",
    [
        (
            "elementwise-masked-tail",
            tiled,
            _double_plus_one,
            (Tensor(1), Tensor(1)),
            [(10,), (10,)],
            {"BLOCK_SIZE": 4},
            11,
        ),
        (
            "row-reduction",
            row_tiled,
            _row_sum,
            (Tensor(2), Tensor(1)),
            [(4, 7), (4,)],
            {"WIDTH": 7},
            22,
        ),
        (
            "softmax",
            row_tiled,
            _softmax_application,
            # The `other=-inf` value keeps the masked lanes out of the denominator.
            (Tensor(2, other=NEG_INF), Tensor(2)),
            [(3, 11), (3, 11)],
            {"WIDTH": 16},
            33,
        ),
    ],
)
def test_default_pipeline_is_semantics_preserving(
    name, arrangement, application, tensors, shapes, symbols, seed
):
    """Three representative programs must survive the default pass pipeline."""
    generated = random_inputs([(shape, "float32") for shape in shapes], seed=seed)
    arrays = (generated[0], np.zeros_like(generated[1]))
    diff = compare_passes(
        arrangement,
        application,
        tensors=tensors,
        inputs=arrays,
        symbols=symbols,
        label=name,
    )

    assert len(diff.stages) == len(diff.pass_names) + 1
    assert diff.pass_names[0] == "ssa.canonicalize"
    assert diff.matches, diff.render()
    assert diff.first_divergence is None
    assert diff.localize() is None


def test_matmul_survives_the_whole_default_pipeline():
    """The repo's own matmul: nested tiles, expand/squeeze, `linalg.dot`, a loop."""
    lhs, rhs, output = random_inputs(
        [((4, 5), "float32"), ((5, 3), "float32"), ((4, 3), "float32")], seed=7
    )
    output[...] = 0.0

    diff = compare_passes(
        _matmul_arrangement,
        _matmul_application,
        tensors=(Tensor(2), Tensor(2), Tensor(2)),
        inputs=(lhs, rhs, output),
        symbols={"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 4, "BLOCK_SIZE_K": 8},
        label="matmul",
    )

    assert diff.executed, diff.render()
    assert diff.matches, diff.render()
    # Every stage agrees with the frontend program, and the frontend program
    # agrees with NumPy exactly.
    np.testing.assert_array_equal(
        diff.baseline.interpretation.output("output"), lhs @ rhs
    )


def test_compare_passes_reports_stages_it_cannot_execute():
    """An unexecutable stage is an unknown, never a silent match."""
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = np.zeros_like(x)

    # Declaring the tensors as rank-1 makes every stage fail to resolve `WIDTH`,
    # which is exactly the situation a partially covered pipeline hits.
    diff = compare_passes(
        row_tiled,
        _double,
        tensors=(Tensor(1), Tensor(1)),
        inputs=(x, out),
        symbols={"WIDTH": 4},
        pipeline=["ssa.canonicalize"],
        label="unresolvable",
    )

    assert len(diff.stages) == 2
    assert not diff.executed
    # A scan that never ran its baseline has validated nothing.
    assert not diff.matches
    assert diff.first_divergence is None
    assert "MissingSymbolError" in diff.stages[0].error
    assert "UNKNOWN" in diff.render()
    assert diff.to_dict()["executed"] is False


def test_pipeline_diff_never_reports_a_match_without_a_baseline():
    from ninetoothed.interpret import PassStage, PipelineDiff

    stage = PassStage(index=0, name="", error="UnsupportedOperationError: nope")
    diff = PipelineDiff(stages=(stage,), pass_names=())

    assert not diff.executed
    assert not diff.matches
    assert diff.localize() is None


def test_compare_passes_names_the_first_diverging_pass():
    x = np.arange(8, dtype=np.float32)
    out = np.zeros_like(x)
    diff = compare_passes(
        tiled,
        _double,
        tensors=(Tensor(1), Tensor(1)),
        inputs=(x, out),
        symbols={"BLOCK_SIZE": 4},
        pipeline=[_WrongConstant.name],
        pass_registry=_registry_with_wrong_constant(),
        label="elementwise",
    )

    assert not diff.matches
    assert diff.diverging_passes == (_WrongConstant.name,)
    assert diff.first_divergence.name == _WrongConstant.name
    assert "first diverging pass" in diff.render()
    # The baseline stage must be the program the frontend produced.
    assert diff.baseline.name == "" and diff.baseline.matches_baseline is False


def test_localization_names_the_instance_and_the_store():
    x = np.arange(8, dtype=np.float32)
    out = np.zeros_like(x)
    diff = compare_passes(
        tiled,
        _double,
        tensors=(Tensor(1), Tensor(1)),
        inputs=(x, out),
        symbols={"BLOCK_SIZE": 4},
        pipeline=[_WrongConstant.name],
        pass_registry=_registry_with_wrong_constant(),
        label="elementwise",
    )
    localization = diff.localize()

    assert localization is not None
    assert localization.pass_name == _WrongConstant.name
    assert localization.output == "out"
    # The value 2.0 became 3.0 because the broken pass bumped the constant.
    assert localization.expected == 2.0
    assert localization.actual == 3.0
    # Element 1 of `out` is written by program instance 0 (the first tile).
    assert localization.program_id == 0
    assert localization.operations
    assert localization.operations[0][0] == "mem.store"
    assert "mem.store" in localization.render()
    assert json.loads(json.dumps(localization.to_dict()))["output"] == "out"


def test_reproduction_carries_ssa_data_shape_dtype_and_seed():
    x, out = random_inputs([((6,), "float32"), ((6,), "float32")], seed=17)
    out[...] = 0.0

    def application(x, out):
        out = x * 2.0  # noqa: F841

    result = run(tiled, application, (x, out), symbols={"BLOCK_SIZE": 4}, seed=17)
    reproduction = build_reproduction(result, label="elementwise")
    payload = reproduction.to_dict()

    assert "ssa @" in payload["ssa"]
    assert payload["seed"] == 17
    assert payload["launch_shape"] == [2]
    assert payload["inputs"][0]["name"] == "x"
    assert payload["inputs"][0]["shape"] == [6]
    assert payload["inputs"][0]["dtype"] == "float32"
    assert len(payload["inputs"][0]["data"]) == 6

    snippet = reproduction.render()
    assert "SEED = 17" in snippet
    assert "BUFFER_SPECS = {" in snippet
    assert "# shape (6,), dtype float32" in snippet
    # The snippet must be syntactically valid Python.
    compile(snippet, "<reproduction>", "exec")


def test_pipeline_diff_serializes_and_renders_a_reproduction():
    x = np.arange(8, dtype=np.float32)
    out = np.zeros_like(x)
    diff = compare_passes(
        tiled,
        _double,
        tensors=(Tensor(1), Tensor(1)),
        inputs=(x, out),
        symbols={"BLOCK_SIZE": 4},
        pipeline=[_WrongConstant.name],
        pass_registry=_registry_with_wrong_constant(),
        label="elementwise",
    )

    payload = json.loads(diff.to_json())

    assert payload["first_divergence"] == _WrongConstant.name
    assert payload["matches"] is False
    assert [stage["label"] for stage in payload["stages"]] == [
        "<frontend>",
        _WrongConstant.name,
    ]

    snippet = diff.reproduction().render()

    assert "passes       : test.wrong_constant" in snippet
    compile(snippet, "<reproduction>", "exec")


def test_a_target_intrinsic_can_be_adopted_by_registering_a_handler():
    """A locally registered `call.*` handler must win over the built-in refusal."""
    from ninetoothed.interpret import format_support_matrix, register
    from ninetoothed.interpret.operations.common import (
        bind_data,
        materialize,
        operands,
    )
    from ninetoothed.interpret.registry import spec_for

    if spec_for("call.logaddexp") is None:

        @register("call.logaddexp", category="call", summary="log(exp(a) + exp(b)).")
        def _handle_logaddexp(state, operation):
            left, right = operands(state, operation)
            bind_data(
                state,
                operation,
                np.logaddexp(
                    materialize(left, state.context), materialize(right, state.context)
                ),
            )

    assert "call.logaddexp" in format_support_matrix()

    x = np.linspace(-2.0, 2.0, 6, dtype=np.float32)
    y = np.linspace(2.0, -2.0, 6, dtype=np.float32)
    out = np.zeros_like(x)

    def arrangement(x, y, out, BLOCK_SIZE=BLOCK_SIZE):
        return (
            x.tile((BLOCK_SIZE,)),
            y.tile((BLOCK_SIZE,)),
            out.tile((BLOCK_SIZE,)),
        )

    def application(x, y, out):
        out = ntl.logaddexp(x, y)  # noqa: F841

    result = run(
        arrangement,
        application,
        (x, y, out),
        symbols={"BLOCK_SIZE": 4},
    )

    # Without the registration this would have raised UnsupportedOperationError.
    np.testing.assert_array_equal(result.output("out"), np.logaddexp(x, y))


def test_random_inputs_are_deterministic_and_dtype_exact():
    first = random_inputs([((4,), "float32"), ((4,), "int32"), ((4,), "bool")], seed=5)
    second = random_inputs([((4,), "float32"), ((4,), "int32"), ((4,), "bool")], seed=5)

    assert [array.dtype for array in first] == [
        np.dtype("float32"),
        np.dtype("int32"),
        np.dtype("bool"),
    ]

    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)

    with pytest.raises(UnsupportedDTypeError):
        random_inputs([((4,), "bfloat16")], seed=1)
