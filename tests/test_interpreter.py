"""Tests for the CPU reference interpreter.

The interpreter consumes the SSA program a backend would receive, so these tests
cover the supported operations, the CPU memory model, the structured control
flow, the execution trace, the interactive debugging session, the reproduction
export, and the differential comparisons against NumPy, PyTorch, and the
production backend when a device is available.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor
from ninetoothed.interpreter import (
    Breakpoint,
    Interpreter,
    InvalidArgumentError,
    PassComparison,
    PassStep,
    TraceOptions,
    UnsupportedDTypeError,
    UnsupportedOperationError,
    compare_passes,
    export_reproduction,
    interpret,
)
from tests.utils import get_available_devices

ELEMENTWISE_BLOCK_SIZE = Symbol("ELEMENTWISE_BLOCK_SIZE", constexpr=True)
MASKED_BLOCK_SIZE = Symbol("MASKED_BLOCK_SIZE", constexpr=True)
BROADCAST_BLOCK_SIZE = Symbol("BROADCAST_BLOCK_SIZE", constexpr=True)
ROW_BLOCK_SIZE = Symbol("ROW_BLOCK_SIZE", constexpr=True)
SCALAR_BLOCK_SIZE = Symbol("SCALAR_BLOCK_SIZE", constexpr=True)
MATMUL_BLOCK_SIZE_M = Symbol("MATMUL_BLOCK_SIZE_M", meta=True)
MATMUL_BLOCK_SIZE_N = Symbol("MATMUL_BLOCK_SIZE_N", meta=True)
MATMUL_BLOCK_SIZE_K = Symbol("MATMUL_BLOCK_SIZE_K", meta=True)


def elementwise_arrangement(lhs, rhs, output):
    return (
        lhs.tile((ELEMENTWISE_BLOCK_SIZE,)),
        rhs.tile((ELEMENTWISE_BLOCK_SIZE,)),
        output.tile((ELEMENTWISE_BLOCK_SIZE,)),
    )


def elementwise_add(lhs, rhs, output):
    output = lhs + rhs  # noqa: F841


def elementwise_multiply(lhs, rhs, output):
    output = lhs * rhs  # noqa: F841


def elementwise_and(lhs, rhs, output):
    output = lhs & rhs  # noqa: F841


def elementwise_affine(lhs, rhs, output):
    output = lhs * 2 + rhs  # noqa: F841


def masked_arrangement(input, output):
    return input.tile((MASKED_BLOCK_SIZE,)), output.tile((MASKED_BLOCK_SIZE,))


def masked_double(input, output):
    output = input * 2  # noqa: F841


def broadcast_arrangement(input, bias, output):
    return (
        input.tile((BROADCAST_BLOCK_SIZE, BROADCAST_BLOCK_SIZE)),
        bias.tile((1, BROADCAST_BLOCK_SIZE)),
        output.tile((BROADCAST_BLOCK_SIZE, BROADCAST_BLOCK_SIZE)),
    )


def broadcast_application(input, bias, output):
    output = input + bias  # noqa: F841


def scalar_broadcast_arrangement(input, alpha, output):
    return input.tile((SCALAR_BLOCK_SIZE,)), alpha, output.tile((SCALAR_BLOCK_SIZE,))


def scalar_broadcast_application(input, alpha, output):
    output = input * alpha  # noqa: F841


def row_reduction_arrangement(input, output):
    return input.tile((1, ROW_BLOCK_SIZE)), output.tile((1, ROW_BLOCK_SIZE))


def row_sum_arrangement(input, output):
    return input.tile((1, ROW_BLOCK_SIZE)), output.tile((1, 1))


def row_sum_application(input, output):
    output = ntl.sum(input, axis=1)  # noqa: F841


def softmax_rows_application(input, output):
    shifted = input - ntl.max(input, axis=1)
    numerator = ntl.exp(shifted)
    output = numerator / ntl.sum(numerator, axis=1)  # noqa: F841


def branch_arrangement(input, output):
    return input.tile((ELEMENTWISE_BLOCK_SIZE,)), output.tile((ELEMENTWISE_BLOCK_SIZE,))


def branch_application(input, output):
    doubled = input * 2

    if ntl.sum(input) > 0:
        output = doubled  # noqa: F841
    else:
        output = -doubled  # noqa: F841


def matmul_arrangement(
    lhs,
    rhs,
    output,
    BLOCK_SIZE_M=MATMUL_BLOCK_SIZE_M,
    BLOCK_SIZE_N=MATMUL_BLOCK_SIZE_N,
    BLOCK_SIZE_K=MATMUL_BLOCK_SIZE_K,
):
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


def matmul_application(lhs, rhs, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)

    for k in range(lhs.shape[0]):
        accumulator += ntl.dot(lhs[k], rhs[k])

    output = accumulator.to(ntl.float16)  # noqa: F841


def cast_to_bfloat16_application(input, output):
    output = input.to(ntl.bfloat16)  # noqa: F841


def cast_to_float8_application(input, output):
    output = input.to(ntl.float8e4nv)  # noqa: F841


def atomic_application(input, output):
    ntl.atomic_add(output, input)


def elementwise_kernel(application, *, name, meta=None, run_passes=True, **kwargs):
    return interpret(
        elementwise_arrangement,
        application,
        (Tensor(1), Tensor(1), Tensor(1)),
        kernel_name=name,
        meta=meta,
        run_passes=run_passes,
        **kwargs,
    )


def row_reduction_kernel(
    application,
    *,
    name,
    arrangement=row_reduction_arrangement,
    other=None,
    run_passes=True,
):
    return interpret(
        arrangement,
        application,
        (Tensor(2, other=other), Tensor(2)),
        kernel_name=name,
        meta={"ROW_BLOCK_SIZE": 256},
        run_passes=run_passes,
    )


def _reference_softmax(input):
    shifted = input - input.max(axis=1, keepdims=True)
    numerator = np.exp(shifted)

    return numerator / numerator.sum(axis=1, keepdims=True)


def test_elementwise_float32_matches_numpy():
    kernel = elementwise_kernel(
        elementwise_add,
        name="elementwise_float32",
        meta={"ELEMENTWISE_BLOCK_SIZE": 256},
    )

    rng = np.random.default_rng(0)
    lhs = rng.normal(size=1000).astype(np.float32)
    rhs = rng.normal(size=1000).astype(np.float32)
    output = np.zeros(1000, dtype=np.float32)

    kernel(lhs, rhs, output)

    assert np.array_equal(output, lhs + rhs)


def test_elementwise_int32_matches_numpy():
    kernel = elementwise_kernel(
        elementwise_multiply,
        name="elementwise_int32",
        meta={"ELEMENTWISE_BLOCK_SIZE": 256},
    )

    rng = np.random.default_rng(1)
    lhs = rng.integers(-1000, 1000, size=1000).astype(np.int32)
    rhs = rng.integers(-1000, 1000, size=1000).astype(np.int32)
    output = np.zeros(1000, dtype=np.int32)

    kernel(lhs, rhs, output)

    assert output.dtype == np.int32
    assert np.array_equal(output, lhs * rhs)


def test_elementwise_bool_matches_numpy():
    kernel = elementwise_kernel(
        elementwise_and,
        name="elementwise_bool",
        meta={"ELEMENTWISE_BLOCK_SIZE": 4},
    )

    lhs = np.array([True, True, False, False, True, False])
    rhs = np.array([True, False, True, False, True, True])
    output = np.zeros(6, dtype=bool)

    kernel(lhs, rhs, output)

    assert output.dtype == np.bool_
    assert np.array_equal(output, lhs & rhs)


def test_literal_operands_keep_the_operand_dtype():
    kernel = elementwise_kernel(
        elementwise_affine,
        name="elementwise_affine",
        meta={"ELEMENTWISE_BLOCK_SIZE": 4},
    )

    lhs = np.array([0.5, 1.5, -2.25, 3.75], dtype=np.float32)
    rhs = np.ones(4, dtype=np.float32)
    output = np.zeros(4, dtype=np.float32)

    kernel(lhs, rhs, output)

    assert output.dtype == np.float32
    assert np.array_equal(output, lhs * 2 + rhs)


def test_broadcast_row_matches_numpy():
    kernel = interpret(
        broadcast_arrangement,
        broadcast_application,
        (Tensor(2), Tensor(2), Tensor(2)),
        kernel_name="broadcast_row",
        meta={"BROADCAST_BLOCK_SIZE": 8},
    )

    rng = np.random.default_rng(2)
    input = rng.normal(size=(8, 6)).astype(np.float32)
    bias = rng.normal(size=(1, 6)).astype(np.float32)
    output = np.zeros((8, 6), dtype=np.float32)

    kernel(input, bias, output)

    assert np.array_equal(output, input + bias)


def test_broadcast_scalar_matches_numpy():
    kernel = interpret(
        scalar_broadcast_arrangement,
        scalar_broadcast_application,
        (Tensor(1), Tensor(0), Tensor(1)),
        kernel_name="broadcast_scalar",
        meta={"SCALAR_BLOCK_SIZE": 4},
    )

    input = np.arange(10, dtype=np.float32)
    output = np.zeros(10, dtype=np.float32)

    kernel(input, np.float32(2.5), output)

    assert np.array_equal(output, input * np.float32(2.5))


def test_masked_load_and_store_match_numpy():
    kernel = interpret(
        masked_arrangement,
        masked_double,
        (Tensor(1), Tensor(1)),
        kernel_name="masked",
        meta={"MASKED_BLOCK_SIZE": 256},
    )

    input = np.arange(1000, dtype=np.float32)
    output = np.full(1000, -1.0, dtype=np.float32)

    kernel(input, output)

    assert np.array_equal(output, input * 2)
    assert bool((output >= 0).all())


def test_masked_lanes_never_touch_out_of_bounds_storage():
    size = 1000
    padding = 24

    kernel = interpret(
        masked_arrangement,
        masked_double,
        (Tensor(1), Tensor(1)),
        kernel_name="masked_storage",
        meta={"MASKED_BLOCK_SIZE": 256},
    )

    input_storage = np.full(size + padding, 12345.0, dtype=np.float32)
    output_storage = np.full(size + padding, -7.0, dtype=np.float32)
    input = input_storage[:size]
    output = output_storage[:size]

    input[:] = np.arange(size, dtype=np.float32)

    kernel(input, output)

    assert np.array_equal(output, input * 2)
    assert np.array_equal(
        output_storage[size:], np.full(padding, -7.0, dtype=np.float32)
    )


def test_row_reduction_matches_numpy():
    kernel = row_reduction_kernel(
        row_sum_application, name="row_sum", arrangement=row_sum_arrangement
    )

    rng = np.random.default_rng(3)
    input = rng.normal(size=(5, 200)).astype(np.float32)
    output = np.zeros((5, 1), dtype=np.float32)

    kernel(input, output)

    expected = input.sum(axis=1, keepdims=True)

    assert output.shape == (5, 1)
    assert np.allclose(output, expected, rtol=1e-4, atol=1e-4)


def test_softmax_matches_numpy():
    kernel = row_reduction_kernel(
        softmax_rows_application, name="softmax_numpy", other=float("-inf")
    )

    rng = np.random.default_rng(4)
    input = rng.normal(size=(37, 129)).astype(np.float32)
    output = np.zeros_like(input)

    kernel(input, output)

    assert np.allclose(output, _reference_softmax(input), rtol=1e-4, atol=1e-5)


def test_softmax_matches_torch():
    kernel = row_reduction_kernel(
        softmax_rows_application, name="softmax_torch", other=float("-inf")
    )

    rng = np.random.default_rng(5)
    input = torch.from_numpy(rng.normal(size=(37, 129)).astype(np.float32))
    output = torch.zeros_like(input)

    kernel(input, output)

    expected = torch.softmax(input, dim=-1)

    assert torch.allclose(output, expected, rtol=1e-3, atol=1e-3)


def test_control_flow_branch_matches_numpy():
    kernel = interpret(
        branch_arrangement,
        branch_application,
        (Tensor(1), Tensor(1)),
        kernel_name="branch",
        meta={"ELEMENTWISE_BLOCK_SIZE": 4},
    )

    positive = np.array([1.0, -2.0, 3.0, 4.0], dtype=np.float32)

    for input, sign in ((positive, 1.0), (-positive, -1.0)):
        output = np.zeros_like(input)

        kernel(input, output)

        assert np.array_equal(output, sign * input * 2)


def test_control_flow_loop_matches_numpy():
    kernel = interpret(
        matmul_arrangement,
        matmul_application,
        (Tensor(2), Tensor(2), Tensor(2)),
        kernel_name="matmul",
    )

    rng = np.random.default_rng(6)
    lhs = rng.normal(size=(128, 160)).astype(np.float16)
    rhs = rng.normal(size=(160, 96)).astype(np.float16)
    output = np.zeros((128, 96), dtype=np.float16)

    kernel(lhs, rhs, output)

    expected = (lhs.astype(np.float32) @ rhs.astype(np.float32)).astype(np.float16)

    assert output.dtype == np.float16
    assert np.allclose(output, expected, rtol=1e-3, atol=1e-3)


def test_pytorch_cpu_tensors_are_accepted():
    kernel = elementwise_kernel(
        elementwise_add, name="torch_arguments", meta={"ELEMENTWISE_BLOCK_SIZE": 256}
    )

    lhs = torch.rand(1000)
    rhs = torch.rand(1000)
    output = torch.empty(1000)

    returned = kernel(lhs, rhs, output)

    assert torch.allclose(output, lhs + rhs)
    assert np.array_equal(returned, output.numpy())


def test_invalid_arguments_are_rejected():
    kernel = elementwise_kernel(
        elementwise_add, name="invalid_arguments", meta={"ELEMENTWISE_BLOCK_SIZE": 4}
    )

    lhs = np.zeros(4, dtype=np.float32)
    output = np.zeros(4, dtype=np.float32)

    with pytest.raises(InvalidArgumentError):
        kernel([0.0, 0.0, 0.0, 0.0], lhs, output)

    with pytest.raises(InvalidArgumentError):
        kernel(np.zeros((4, 1), dtype=np.float32), lhs, output)

    with pytest.raises(InvalidArgumentError):
        kernel(np.zeros(8, dtype=np.float32)[::2], lhs, output)

    with pytest.raises(InvalidArgumentError):
        kernel(lhs)


def test_application_without_a_retrievable_source_is_reported():
    namespace = {}
    exec("def application(lhs, rhs, output):\n    output = lhs + rhs\n", namespace)

    with pytest.raises(ValueError, match="not retrievable"):
        interpret(
            elementwise_arrangement,
            namespace["application"],
            (Tensor(1), Tensor(1), Tensor(1)),
        )


def test_unknown_symbol_is_rejected():
    with pytest.raises(InvalidArgumentError):
        interpret(
            elementwise_arrangement,
            elementwise_add,
            (Tensor(1), Tensor(1), Tensor(1)),
            kernel_name="unknown_symbol",
            meta={"NOT_A_SYMBOL": 4},
        )


def test_missing_constexpr_symbol_is_reported():
    kernel = interpret(
        elementwise_arrangement,
        elementwise_add,
        (Tensor(1), Tensor(1), Tensor(1)),
        kernel_name="missing_symbol",
    )

    with pytest.raises(InvalidArgumentError):
        kernel(
            np.zeros(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
        )


def _optimization_case(name):
    """Return an application and its arguments for the pipeline comparisons.

    :param name: The case name.
    :return: The arrangement, the application, the tensors, the meta values, an
        argument factory, and the expected output.
    """
    if name == "elementwise":

        def arguments():
            rng = np.random.default_rng(7)
            lhs = rng.normal(size=1000).astype(np.float32)
            rhs = rng.normal(size=1000).astype(np.float32)

            return lhs, rhs, np.zeros(1000, dtype=np.float32)

        lhs, rhs, _ = arguments()

        return (
            elementwise_arrangement,
            elementwise_add,
            (Tensor(1), Tensor(1), Tensor(1)),
            {"ELEMENTWISE_BLOCK_SIZE": 128},
            arguments,
            lhs + rhs,
        )

    if name == "masked":

        def arguments():
            rng = np.random.default_rng(7)

            return (
                rng.normal(size=1000).astype(np.float32),
                np.zeros(1000, dtype=np.float32),
            )

        input, _ = arguments()

        return (
            masked_arrangement,
            masked_double,
            (Tensor(1), Tensor(1)),
            {"MASKED_BLOCK_SIZE": 256},
            arguments,
            input * 2,
        )

    if name == "broadcast":

        def arguments():
            rng = np.random.default_rng(7)

            return (
                rng.normal(size=(8, 6)).astype(np.float32),
                rng.normal(size=(1, 6)).astype(np.float32),
                np.zeros((8, 6), dtype=np.float32),
            )

        input, bias, _ = arguments()

        return (
            broadcast_arrangement,
            broadcast_application,
            (Tensor(2), Tensor(2), Tensor(2)),
            {"BROADCAST_BLOCK_SIZE": 8},
            arguments,
            input + bias,
        )

    if name == "row_reduction":

        def arguments():
            rng = np.random.default_rng(7)

            return (
                rng.normal(size=(5, 200)).astype(np.float32),
                np.zeros((5, 1), dtype=np.float32),
            )

        input, _ = arguments()

        return (
            row_sum_arrangement,
            row_sum_application,
            (Tensor(2), Tensor(2)),
            {"ROW_BLOCK_SIZE": 256},
            arguments,
            input.sum(axis=1, keepdims=True),
        )

    raise AssertionError(f"Unknown optimization case `{name}`.")


@pytest.mark.parametrize(
    "name", ("elementwise", "masked", "broadcast", "row_reduction")
)
def test_optimized_and_unoptimized_ssa_agree(name):
    arrangement, application, tensors, meta, arguments, expected = _optimization_case(
        name
    )
    optimized_kernel = interpret(
        arrangement, application, tensors, kernel_name=f"optimization_{name}", meta=meta
    )
    raw_kernel = interpret(
        arrangement,
        application,
        tensors,
        kernel_name=f"optimization_{name}",
        meta=meta,
        run_passes=False,
    )

    optimized_arguments = list(arguments())
    raw_arguments = list(arguments())

    optimized_kernel(*optimized_arguments)
    raw_kernel(*raw_arguments)

    assert optimized_kernel.pass_trace
    assert raw_kernel.pass_trace == ()
    assert np.allclose(optimized_arguments[-1], expected, rtol=1e-4, atol=1e-4)
    assert np.allclose(raw_arguments[-1], expected, rtol=1e-4, atol=1e-4)
    assert np.allclose(optimized_arguments[-1], raw_arguments[-1], rtol=1e-4, atol=1e-4)

    assert raw_kernel.program.kind == optimized_kernel.program.kind


@pytest.mark.parametrize(
    "name", ("elementwise", "masked", "broadcast", "row_reduction")
)
def test_every_pass_preserves_the_front_end_semantics(name):
    arrangement, application, tensors, meta, arguments, expected = _optimization_case(
        name
    )

    comparison = compare_passes(
        arrangement,
        application,
        tensors,
        arguments,
        meta=meta,
        kernel_name=f"diagnosed_{name}",
    )

    assert comparison.first_mismatch is None
    assert len(comparison.steps) > 1
    assert comparison.steps[0].pass_name is None
    assert comparison.steps[0].passes == ()

    for step in comparison.steps:
        assert step.matches
        assert step.interpretation is not None

    assert "Every pass preserved" in comparison.format()

    assert np.allclose(
        Interpreter(comparison.steps[-1].interpretation, meta=meta)(*arguments()),
        expected,
        rtol=1e-4,
        atol=1e-4,
    )


def test_pass_comparison_reports_the_first_diverging_pass():
    steps = (
        PassStep(
            pass_name=None,
            passes=(),
            matches=True,
            max_difference=0.0,
            interpretation=None,
        ),
        PassStep(
            pass_name="ssa.first",
            passes=("ssa.first",),
            matches=True,
            max_difference=0.0,
            interpretation=None,
        ),
        PassStep(
            pass_name="ssa.second",
            passes=("ssa.first", "ssa.second"),
            matches=False,
            max_difference=1.5,
            interpretation=None,
        ),
        PassStep(
            pass_name="ssa.third",
            passes=("ssa.first", "ssa.second", "ssa.third"),
            matches=False,
            max_difference=1.5,
            interpretation=None,
        ),
    )
    comparison = PassComparison(steps=steps)

    assert comparison.first_mismatch is steps[2]
    assert "`ssa.second`" in comparison.format()
    assert "max |difference| = 1.500e+00" in comparison.format()


def test_trace_reports_program_id_operation_values_and_mask():
    kernel = interpret(
        masked_arrangement,
        masked_double,
        (Tensor(1), Tensor(1)),
        kernel_name="traced",
        meta={"MASKED_BLOCK_SIZE": 4},
        trace=True,
    )

    input = np.arange(10, dtype=np.float32)
    output = np.zeros(10, dtype=np.float32)

    kernel(input, output)

    trace = kernel.last_trace

    assert len(trace) > 0
    assert {event.program_id for event in trace} == {0, 1, 2}

    for event in trace:
        assert event.opcode
        assert "entry:" in event.location
        assert event.inputs or event.outputs

    stores = [event for event in trace if event.opcode == "mem.store"]

    assert len(stores) == 3
    assert stores[0].mask is not None
    assert "False" not in stores[0].mask
    assert "False" in stores[2].mask
    assert "True" in stores[2].mask
    assert "entry:" in stores[0].format()

    wide = interpret(
        masked_arrangement,
        masked_double,
        (Tensor(1), Tensor(1)),
        kernel_name="traced_mask_wide",
        meta={"MASKED_BLOCK_SIZE": 256},
        trace=True,
        trace_options=TraceOptions(
            program_ids=(3,), opcodes=("mem.",), max_elements=256
        ),
    )

    wide(np.arange(1000, dtype=np.float32), np.zeros(1000, dtype=np.float32))

    (tail,) = list(wide.last_trace)

    assert tail.mask.count("True") == 232
    assert tail.mask.count("False") == 24


def test_trace_can_be_filtered_by_program_id_and_opcode():
    kernel = interpret(
        masked_arrangement,
        masked_double,
        (Tensor(1), Tensor(1)),
        kernel_name="traced_filtered",
        meta={"MASKED_BLOCK_SIZE": 4},
        trace=True,
    )

    kernel(np.arange(10, dtype=np.float32), np.zeros(10, dtype=np.float32))

    trace = kernel.last_trace

    assert {event.program_id for event in trace.filter(program_ids=(1,))} == {1}
    assert all(
        event.opcode.startswith("mem.") for event in trace.filter(opcodes=("mem.",))
    )
    assert trace.filter(program_ids=(99,)).events == []
    assert "array(" in trace.format()


def test_trace_options_restrict_the_recorded_events():
    options = TraceOptions(program_ids=(0,), opcodes=("arith.",), keep_values=True)

    kernel = interpret(
        masked_arrangement,
        masked_double,
        (Tensor(1), Tensor(1)),
        kernel_name="traced_options",
        meta={"MASKED_BLOCK_SIZE": 4},
        trace=True,
        trace_options=options,
    )

    kernel(np.arange(10, dtype=np.float32), np.zeros(10, dtype=np.float32))

    trace = kernel.last_trace

    assert len(trace) > 0

    for event in trace:
        assert event.program_id == 0
        assert event.opcode.startswith("arith.")
        assert event.values is not None


def test_trace_is_reproducible():
    def run():
        kernel = interpret(
            masked_arrangement,
            masked_double,
            (Tensor(1), Tensor(1)),
            kernel_name="traced_reproducible",
            meta={"MASKED_BLOCK_SIZE": 4},
            trace=True,
        )
        kernel(np.arange(10, dtype=np.float32), np.zeros(10, dtype=np.float32))

        return kernel.last_trace.format()

    assert run() == run()


def test_unsupported_operation_reports_the_opcode_and_location():
    kernel = interpret(
        masked_arrangement,
        atomic_application,
        (Tensor(1), Tensor(1)),
        kernel_name="atomic",
        meta={"MASKED_BLOCK_SIZE": 4},
    )

    with pytest.raises(UnsupportedOperationError) as info:
        kernel(np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32))

    assert info.value.opcode == "mem.atomic_add"
    assert "entry:" in info.value.location
    assert "mem.atomic_add" in str(info.value)
    assert info.value.location in str(info.value)


@pytest.mark.parametrize(
    "application, dtype",
    (
        (cast_to_bfloat16_application, "bfloat16"),
        (cast_to_float8_application, "float8"),
    ),
)
def test_unsupported_dtype_is_reported(application, dtype):
    kernel = interpret(
        masked_arrangement,
        application,
        (Tensor(1), Tensor(1)),
        kernel_name="unsupported_dtype",
        meta={"MASKED_BLOCK_SIZE": 4},
    )

    with pytest.raises(UnsupportedDTypeError) as info:
        kernel(np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32))

    assert dtype in str(info.value)


def test_support_matrix_lists_supported_and_unsupported_operations():
    kernel = elementwise_kernel(
        elementwise_add, name="support_matrix", meta={"ELEMENTWISE_BLOCK_SIZE": 4}
    )

    supported = {entry.opcode for entry in kernel.supported_operations()}

    assert {
        "arith.",
        "arith.constant",
        "cmp.",
        "linalg.dot",
        "linalg.matmul",
        "math.",
        "mem.store",
        "reduce.",
        "scf.for",
        "scf.if",
        "select.where",
        "tensor.cast",
    } <= supported

    for entry in kernel.supported_operations():
        assert entry.summary
        assert entry.dtypes

    unsupported = dict(kernel.unsupported_operations())

    assert "mem.atomic_add" in unsupported
    assert "mem.data_ptr" in unsupported
    assert "mem.load" in unsupported
    assert "math.rand" in unsupported

    for reason in unsupported.values():
        assert reason


# ---------------------------------------------------------------------------
# Interactive debugging.
# ---------------------------------------------------------------------------


def debug_arguments():
    """Return the arguments of the masked case the debugging tests step."""
    return np.arange(10, dtype=np.float32), np.zeros(10, dtype=np.float32)


def debug_session(*, breakpoints=(), watch=()):
    """Return an interactive session over the masked elementwise case."""
    kernel = interpret(
        masked_arrangement,
        masked_double,
        (Tensor(1), Tensor(1)),
        kernel_name="debugged",
        meta={"MASKED_BLOCK_SIZE": 4},
    )

    return kernel.debug(*debug_arguments(), breakpoints=breakpoints, watch=watch)


def unexported_arrangement(input, output):
    return input.tile((MASKED_BLOCK_SIZE,)), output.tile((MASKED_BLOCK_SIZE,))


def _unexported_double(value):
    return value * 2


def unexported_application(input, output):
    output = _unexported_double(input)  # noqa: F841


def test_debug_session_steps_through_every_operation():
    session = debug_session()
    stops = []

    while True:
        stop = session.step()

        if stop is None:
            break

        stops.append(stop)

    assert [stop.index for stop in stops] == list(range(9))
    assert [stop.program_id for stop in stops] == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert len(session) == 9
    assert np.array_equal(session.output, np.arange(10, dtype=np.float32) * 2)


def test_debug_step_reports_the_operands_the_mask_and_the_location():
    session = debug_session()
    stop = session.step()

    assert (stop.location, stop.opcode, stop.program_id) == (
        "entry:0:arith.constant",
        "arith.constant",
        0,
    )

    while stop.opcode != "mem.store":
        stop = session.step()

    assert stop.operands == ("%1", "output")
    assert set(stop.inputs) == {"%1", "output"}
    assert "array(" in stop.inputs["%1"]
    assert stop.mask is not None
    assert stop.mask.count("True") == 4
    assert "entry:2:mem.store" in stop.format()


def test_debug_value_returns_the_value_of_a_name_at_the_stop():
    session = debug_session()

    while session.step().opcode != "mem.store":
        pass

    doubled = session.value("%1")

    assert np.array_equal(doubled, np.array([0, 2, 4, 6], dtype=np.float32))
    assert session.value("%undefined") is None


def test_debug_value_is_a_snapshot_that_later_stops_do_not_change():
    session = debug_session()

    while session.step().opcode != "mem.store":
        pass

    before = session.value("output")

    assert np.array_equal(before, np.zeros(4, dtype=np.float32))

    output = session.finish()

    assert np.array_equal(before, np.zeros(4, dtype=np.float32))
    assert np.array_equal(output, np.arange(10, dtype=np.float32) * 2)


def test_debug_breakpoint_stops_on_a_program_id():
    session = debug_session(breakpoints=(Breakpoint(program_ids=(1,)),))

    stop = session.resume()

    assert (stop.index, stop.program_id, stop.opcode) == (3, 1, "arith.constant")
    assert session.resume().index == 4
    assert session.resume().index == 5
    assert session.resume() is None


def test_debug_breakpoint_stops_on_an_opcode_prefix():
    session = debug_session(breakpoints=(Breakpoint(opcodes=("mem.",)),))

    assert [session.resume().index for _ in range(3)] == [2, 5, 8]
    assert session.resume() is None


def test_debug_breakpoint_combines_a_program_id_with_a_location():
    session = debug_session(
        breakpoints=(Breakpoint(program_ids=(2,), locations=("entry:2:mem.store",)),)
    )

    stop = session.resume()

    assert (stop.index, stop.program_id) == (8, 2)
    assert stop.mask.count("True") == 2
    assert stop.mask.count("False") == 2
    assert session.resume() is None


def test_debug_watch_reports_a_value_at_every_stop():
    session = debug_session(watch=("%1",))
    stored = []

    while True:
        stop = session.step()

        if stop is None:
            break

        assert set(stop.watched) == {"%1"}

        if stop.opcode == "mem.store":
            assert "array(" in stop.watched["%1"]
            stored.append(session.value("%1").copy())

    assert np.array_equal(stored[0], np.array([0, 2, 4, 6], dtype=np.float32))
    assert np.array_equal(stored[1], np.array([8, 10, 12, 14], dtype=np.float32))
    assert np.array_equal(stored[2], np.array([16, 18, 0, 0], dtype=np.float32))


def test_debug_watch_reports_an_unbound_name_and_can_be_extended():
    session = debug_session()

    session.watch("%0", "output")

    first = session.step()

    assert set(first.watched) == {"%0", "output"}
    assert first.watched["%0"] == "<unbound>"
    assert "array(" in first.watched["output"]

    second = session.step()

    assert second.watched["%0"] == "2"


def test_debug_history_lists_the_stops_and_reset_restarts_the_session():
    session = debug_session(breakpoints=(Breakpoint(opcodes=("mem.store",)),))

    session.resume()
    session.resume()

    history = session.format_history()

    assert len(session) == 2
    assert "program 0" in history
    assert "program 1" in history
    assert history.count("mem.store") == 2

    session.reset()

    assert len(session) == 0
    assert session.format_history() == ""
    assert session.resume().index == 2


def test_debug_session_output_matches_a_plain_run_of_a_looping_application():
    meta = {
        "MATMUL_BLOCK_SIZE_M": 32,
        "MATMUL_BLOCK_SIZE_N": 32,
        "MATMUL_BLOCK_SIZE_K": 32,
    }
    kernel = interpret(
        matmul_arrangement,
        matmul_application,
        (Tensor(2), Tensor(2), Tensor(2)),
        kernel_name="debugged_loop",
        meta=meta,
    )

    plain = kernel(*matmul_arguments())

    session = kernel.debug(
        *matmul_arguments(), breakpoints=(Breakpoint(opcodes=("mem.store",)),), **meta
    )
    stops = 0

    while session.resume() is not None:
        stops += 1

    assert stops == 4
    assert np.array_equal(session.finish(), plain)


def matmul_arguments():
    """Return a fresh set of the small half-precision matmul case."""
    rng = np.random.default_rng(5)

    return [
        rng.normal(size=(64, 96)).astype(np.float16),
        rng.normal(size=(96, 48)).astype(np.float16),
        np.zeros((64, 48), dtype=np.float16),
    ]


# ---------------------------------------------------------------------------
# Reproduction export.
# ---------------------------------------------------------------------------


def export_masked_reproduction(directory, *, expected=None, actual=None):
    """Export the masked elementwise case into one reproduction directory."""
    return export_reproduction(
        directory,
        name="masked",
        arrangement=masked_arrangement,
        application=masked_double,
        tensors=(Tensor(1), Tensor(1)),
        arguments=(
            np.arange(10, dtype=np.float32),
            np.zeros(10, dtype=np.float32),
        ),
        meta={"MASKED_BLOCK_SIZE": 4},
        expected=expected,
        actual=actual,
        seed=20260916,
        reason="the reference disagrees with the interpreter",
    )


def test_reproduction_writes_the_files_needed_to_rerun_the_case(tmp_path):
    reproduction = export_masked_reproduction(
        tmp_path / "repro",
        expected=np.arange(10, dtype=np.float32) * 2,
        actual=np.arange(10, dtype=np.float32) * 2,
    )
    names = sorted(path.name for path in reproduction.files)

    assert names == [
        "actual.npy",
        "case.json",
        "expected.npy",
        "inputs.npz",
        "passes.txt",
        "reproduce.py",
        "ssa_frontend.txt",
        "ssa_optimized.txt",
    ]
    assert all(path.exists() for path in reproduction.files)
    assert reproduction.unresolved == ()
    assert "reproduction of" in reproduction.format()
    assert "the reference disagrees" in reproduction.format()

    source = (reproduction.directory / "reproduce.py").read_text(encoding="utf-8")

    compile(source, "reproduce.py", "exec")


def test_reproduction_records_the_metadata_of_the_case(tmp_path):
    expected = np.arange(10, dtype=np.float32) * 2
    reproduction = export_masked_reproduction(
        tmp_path / "repro", expected=expected, actual=expected + 0.5
    )
    directory = reproduction.directory
    case = json.loads((directory / "case.json").read_text(encoding="utf-8"))

    assert case["case"] == "masked"
    assert case["kernel_name"] == "masked"
    assert case["seed"] == 20260916
    assert case["tolerance"] == {"rtol": 1e-3, "atol": 1e-3}
    assert case["max_absolute_difference"] == pytest.approx(0.5)
    assert case["mismatched_elements"] == 10
    assert case["arguments"][0]["name"] == "argument_0"
    assert case["arguments"][0]["shape"] == [10]
    assert case["arguments"][0]["dtype"] == "float32"
    assert case["tensors"][0]["ndim"] == 1
    assert case["meta"] == {"MASKED_BLOCK_SIZE": "4"}
    assert (
        case["passes"]
        == (directory / "passes.txt").read_text(encoding="utf-8").splitlines()
    )
    assert case["unresolved_names"] == []
    assert np.array_equal(np.load(directory / "expected.npy"), expected)
    assert np.array_equal(np.load(directory / "actual.npy"), expected + 0.5)

    with np.load(directory / "inputs.npz") as archive:
        assert sorted(archive.files) == ["argument_0", "argument_1"]
        assert np.array_equal(archive["argument_0"], np.arange(10, dtype=np.float32))
        assert np.array_equal(archive["argument_1"], np.zeros(10, dtype=np.float32))


def test_reproduction_script_reruns_the_case_and_reports_the_match(tmp_path):
    expected = np.arange(10, dtype=np.float32) * 2
    reproduction = export_masked_reproduction(
        tmp_path / "repro", expected=expected, actual=expected
    )
    completed = subprocess.run(
        [sys.executable, str(reproduction.directory / "reproduce.py")],
        env=dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE"),
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "seed: 20260916" in completed.stdout
    assert "interpreter matches the recorded result: True" in completed.stdout


def test_reproduction_reports_the_names_it_cannot_export(tmp_path):
    reproduction = export_reproduction(
        tmp_path / "repro",
        name="unexported",
        arrangement=unexported_arrangement,
        application=unexported_application,
        tensors=(Tensor(1), Tensor(1)),
        arguments=(np.arange(4, dtype=np.float32), np.zeros(4, dtype=np.float32)),
        meta={"MASKED_BLOCK_SIZE": 4},
    )
    source = (reproduction.directory / "reproduce.py").read_text(encoding="utf-8")

    assert reproduction.unresolved == ("_unexported_double",)
    assert "_unexported_double" in reproduction.format()
    assert "could not be exported" in source
    assert "_unexported_double = " not in source

    compile(source, "reproduce.py", "exec")


def test_reproduction_without_a_reference_omits_the_result_files(tmp_path):
    reproduction = export_masked_reproduction(tmp_path / "repro")
    names = sorted(path.name for path in reproduction.files)
    case = json.loads(
        (reproduction.directory / "case.json").read_text(encoding="utf-8")
    )

    assert "actual.npy" not in names
    assert "expected.npy" not in names
    assert case["expected"] is None
    assert case["actual"] is None
    assert case["max_absolute_difference"] is None
    assert case["mismatched_elements"] is None


def test_reproduction_records_a_shape_mismatch(tmp_path):
    reproduction = export_masked_reproduction(
        tmp_path / "repro",
        expected=np.zeros(10, dtype=np.float32),
        actual=np.zeros(4, dtype=np.float32),
    )
    case = json.loads(
        (reproduction.directory / "case.json").read_text(encoding="utf-8")
    )

    assert case["shape_mismatch"] == [[10], [4]]
    assert case["max_absolute_difference"] is None
    assert case["mismatched_elements"] is None


CUDA_FREE_SCRIPT = """
import sys

import numpy as np
import torch

import ninetoothed
from ninetoothed import Symbol, Tensor

assert not torch.cuda.is_available()

# Importing torch on a CUDA-enabled build already pulls in a few torch.cuda
# submodules, so this baseline records what plain import torch costs us.
cuda_baseline = frozenset(name for name in sys.modules if name.startswith("torch.cuda"))

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(lhs, rhs, output):
    return (
        lhs.tile((BLOCK_SIZE,)),
        rhs.tile((BLOCK_SIZE,)),
        output.tile((BLOCK_SIZE,)),
    )


def application(lhs, rhs, output):
    output = lhs * 2 + rhs


kernel = ninetoothed.interpret(
    arrangement,
    application,
    (Tensor(1), Tensor(1), Tensor(1)),
    kernel_name="cuda_free",
    meta={"BLOCK_SIZE": 4},
)

lhs = np.arange(10, dtype=np.float32)
rhs = np.ones(10, dtype=np.float32)
output = np.zeros(10, dtype=np.float32)

kernel(lhs, rhs, output)

assert np.array_equal(output, lhs * 2 + rhs), output

torch_lhs = torch.arange(10, dtype=torch.float32)
torch_rhs = torch.ones(10, dtype=torch.float32)
torch_output = torch.zeros(10, dtype=torch.float32)

kernel(torch_lhs, torch_rhs, torch_output)

assert torch.equal(torch_output, torch_lhs * 2 + torch_rhs), torch_output
cuda_modules = frozenset(name for name in sys.modules if name.startswith("torch.cuda"))
assert cuda_modules <= cuda_baseline, cuda_modules - cuda_baseline

print("interpreter-ok")
"""


def test_interpretation_runs_without_any_cuda_device(tmp_path):
    # An empty CUDA_VISIBLE_DEVICES is a no-op on Windows, so "-1" is used to hide
    # the device on every platform.
    environment = dict(
        os.environ, CUDA_VISIBLE_DEVICES="-1", KMP_DUPLICATE_LIB_OK="TRUE"
    )

    # The frontend lowers Python source, and `inspect` cannot retrieve the source
    # of a function defined through `python -c` on every interpreter, so the script
    # is written to a file first.
    script = tmp_path / "cuda_free.py"
    script.write_text(CUDA_FREE_SCRIPT, encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(script)],
        env=environment,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "interpreter-ok" in completed.stdout


def _backend_cases():
    """Return the differential cases shared with the production backend."""
    return (
        (
            "elementwise",
            elementwise_arrangement,
            elementwise_add,
            (Tensor(1), Tensor(1), Tensor(1)),
            {"ELEMENTWISE_BLOCK_SIZE": 256},
            lambda: (torch.rand(1000), torch.rand(1000), torch.zeros(1000)),
            (1e-3, 1e-3),
        ),
        (
            "masked",
            masked_arrangement,
            masked_double,
            (Tensor(1), Tensor(1)),
            {"MASKED_BLOCK_SIZE": 256},
            lambda: (torch.rand(1000), torch.zeros(1000)),
            (1e-3, 1e-3),
        ),
        (
            "broadcast",
            broadcast_arrangement,
            broadcast_application,
            (Tensor(2), Tensor(2), Tensor(2)),
            {"BROADCAST_BLOCK_SIZE": 8},
            lambda: (torch.rand(8, 6), torch.rand(1, 6), torch.zeros(8, 6)),
            (1e-3, 1e-3),
        ),
        (
            "row_reduction",
            row_sum_arrangement,
            row_sum_application,
            (Tensor(2), Tensor(2)),
            {"ROW_BLOCK_SIZE": 256},
            lambda: (torch.rand(5, 200), torch.zeros(5, 1)),
            (1e-3, 1e-3),
        ),
        (
            "softmax",
            row_reduction_arrangement,
            softmax_rows_application,
            (Tensor(2, other=float("-inf")), Tensor(2)),
            {"ROW_BLOCK_SIZE": 256},
            lambda: (torch.rand(37, 129), torch.zeros(37, 129)),
            (1e-3, 1e-3),
        ),
        (
            "matmul",
            matmul_arrangement,
            matmul_application,
            (Tensor(2), Tensor(2), Tensor(2)),
            {},
            lambda: (
                torch.randn(128, 160).to(torch.float16),
                torch.randn(160, 96).to(torch.float16),
                torch.zeros(128, 96, dtype=torch.float16),
            ),
            (2e-2, 2e-2),
        ),
    )


@pytest.mark.parametrize("name", [case[0] for case in _backend_cases()])
@pytest.mark.parametrize("device", get_available_devices())
def test_matches_backend(name, device):
    if device != "cuda":
        pytest.skip("The backend differential test needs a CUDA device.")

    case = next(case for case in _backend_cases() if case[0] == name)
    _, arrangement, application, tensors, meta, inputs, tolerance = case

    interpreter_kernel = interpret(
        arrangement, application, tensors, kernel_name=name, meta=meta
    )
    backend_kernel = ninetoothed.make(
        arrangement, application, tensors, kernel_name=name, max_num_configs=1
    )

    interpreter_arguments = list(inputs())
    backend_arguments = [tensor.clone().to(device) for tensor in interpreter_arguments]

    interpreter_kernel(*interpreter_arguments)
    backend_kernel(*backend_arguments, **meta)

    expected = interpreter_arguments[-1].to(device)
    actual = backend_arguments[-1]

    rtol, atol = tolerance

    assert torch.allclose(expected, actual, rtol=rtol, atol=atol), (
        (expected - actual).abs().max()
    )
