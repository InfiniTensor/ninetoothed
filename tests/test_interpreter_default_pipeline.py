"""Check every real default SSA pass against an independently validated reference."""

from functools import partial

import numpy as np
import pytest

from ninetoothed import interpret
from ninetoothed.backends.core import Target
from ninetoothed.compiler.passes import Context, default_pipeline, lower_for_target
from ninetoothed.interpreter import interpret_program
from ninetoothed.interpreter.debugger import check_passes
from ninetoothed.ir import ssa

from .test_interpreter_applications import (
    _add,
    _broadcast_add,
    _broadcast_tiles,
    _check,
    _comparison,
    _descriptor,
    _row_sum,
    _row_tiles,
    _unary_vectors,
    _vectors,
)


@pytest.fixture(
    params=(
        "elementwise_int32",
        "broadcast_float32",
        "reduction_float32",
        "comparison_bool",
    )
)
def application_case(request):
    if request.param == "elementwise_int32":
        x = np.arange(-5, 6, dtype=np.int32)
        y = np.arange(11, dtype=np.int32) * 3
        inputs = {"x": x, "y": y, "out": np.full_like(x, -731)}
        tensors = tuple(_descriptor(1, name, "int32") for name in inputs)
        arrangement, application, expected = _vectors, _add, x + y
    elif request.param == "broadcast_float32":
        x = np.arange(15, dtype=np.float32).reshape(3, 5) / 7
        bias = np.array([3, -4, 2, 7, -1], dtype=np.float32)
        inputs = {"x": x, "bias": bias, "out": np.full_like(x, -731)}
        tensors = (_descriptor(2, "x"), _descriptor(1, "bias"), _descriptor(2, "out"))
        arrangement, application, expected = _broadcast_tiles, _broadcast_add, x + bias
    elif request.param == "reduction_float32":
        x = (np.arange(15, dtype=np.float32).reshape(3, 5) - 9) / 7
        inputs = {"x": x, "out": np.full((3, 1), -731, dtype=np.float32)}
        tensors = (_descriptor(2, "x", other=0), _descriptor(2, "out"))
        arrangement, application = _row_tiles, _row_sum
        expected = np.sum(x, axis=1, keepdims=True, dtype=np.float32)
    else:
        assert request.param == "comparison_bool"
        x = np.arange(-2, 9, dtype=np.int32)
        inputs = {"x": x, "out": np.ones(x.shape, dtype=np.bool_)}
        tensors = (_descriptor(1, "x", "int32"), _descriptor(1, "out", "bool"))
        arrangement, application, expected = (
            _unary_vectors,
            _comparison,
            (x > 0) & (x < 4),
        )

    return interpret(arrangement, application, tensors), inputs, expected


@pytest.mark.parametrize(
    "backend", (Target.TRITON, Target.CUDA), ids=lambda target: target.value
)
def test_each_default_pass_preserves_the_original_application(
    backend, application_case
):
    kernel, inputs, expected = application_case
    originals = {name: value.copy() for name, value in inputs.items()}
    reference = kernel.frontend_program
    before = interpret_program(
        reference,
        {name: value.copy() for name, value in inputs.items()},
        tensors=kernel.tensors,
        symbols=kernel.meta,
    )
    _check(before.outputs["out"], expected)
    pipeline = default_pipeline(backend)
    assert pipeline.passes
    assert pipeline.spec.mode == "default"
    context = Context(
        backend=backend,
        compiler_options={},
        kernel_metadata={},
        tensors=kernel.tensors,
        pass_options=pipeline.spec.pass_options,
        pipeline_spec=pipeline.spec,
    )
    transformed = []

    def run_pass(pass_, program):
        result = pass_.run(program, context)
        ssa.verify_program(result)
        transformed.append(result)

        return result

    checks = tuple((pass_.name, partial(run_pass, pass_)) for pass_ in pipeline.passes)
    report = check_passes(
        reference,
        checks,
        inputs,
        tensors=kernel.tensors,
        symbols=kernel.meta,
    )
    assert report.passed, report
    assert report.first_bad_pass is None
    assert report.checked_passes == tuple(pass_.name for pass_ in pipeline.passes)
    assert len(transformed) == len(pipeline.passes)
    lowered = lower_for_target(reference, backend=backend, tensors=kernel.tensors)
    assert lowered.metadata["pipeline_selection"]["mode"] == "default"
    assert lowered.metadata["pass_trace"] == report.checked_passes
    assert ssa.render(transformed[-1]) == ssa.render(lowered)

    for name, value in inputs.items():
        np.testing.assert_array_equal(value, originals[name])
