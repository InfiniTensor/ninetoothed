"""Keep CPU preparation aligned with the verified compiler frontend boundary."""

from dataclasses import replace

import numpy as np
import pytest

from ninetoothed import Symbol, Tensor, interpret
from ninetoothed.compiler import Compiler, CompileRequest
from ninetoothed.errors import LoweringError
from ninetoothed.ir import ssa
from ninetoothed.naming import remove_prefixes


def _copy(x, out):
    out = x  # noqa: F841


def _layout(x, out):
    return x.tile((4,)), out.tile((4,))


def _named_layout(x, out):
    return x.axes("n").tile((4,)), out.axes("n").tile((4,))


def _annotated_named_copy(
    x: Tensor(1, dtype="float32", name="x").axes("n").tile((4,)),
    out: Tensor(1, dtype="float32", name="out").axes("n").tile((4,)),
):
    out = x  # noqa: F841


def _descriptors():
    return tuple(Tensor(1, dtype="float32", name=name) for name in ("x", "out"))


def _check_copy(handle):
    x = np.arange(17, dtype=np.float32)[::-1]
    expected = x.copy()
    backing = np.full(36, -731, dtype=np.float32)
    out = backing[1:35:2]
    result = handle(x, out)
    np.testing.assert_array_equal(result.outputs["out"], expected)
    np.testing.assert_array_equal(x, expected)
    np.testing.assert_array_equal(backing[::2], -731)
    assert backing[-1] == -731


@pytest.mark.parametrize("named", (False, True), ids=("legacy", "named"))
def test_interpreter_uses_the_same_verified_frontend_program(named):
    layout = _named_layout if named else _layout
    tensors = _descriptors()
    module = Compiler().frontend(
        CompileRequest(arrangement=layout, application=_copy, tensors=tensors)
    )
    handle = interpret(layout, _copy, tensors)
    assert handle.frontend_program == module.program
    assert handle.tensors == module.tensors
    assert handle.parameters == module.public_args
    assert handle.frontend_program.metadata["shape_validation"] == "strict"
    assert "external_calls" in handle.frontend_program.metadata
    assert (module.named_layout is not None) == named
    _check_copy(handle)


def test_annotated_named_application_uses_the_compiler_frontend():
    module = Compiler().frontend(CompileRequest(application=_annotated_named_copy))
    handle = interpret(_annotated_named_copy)
    assert module.named_layout is not None
    assert handle.frontend_program == module.program
    _check_copy(handle)


def test_interpreter_rejects_the_same_invalid_store_shape_as_the_compiler():
    def invalid_layout(x, out):
        return x.tile((4,)), out.tile((8,))

    tensors = _descriptors()
    request = CompileRequest(
        arrangement=invalid_layout, application=_copy, tensors=tensors
    )

    for prepare in (
        lambda: Compiler().frontend(request),
        lambda: interpret(invalid_layout, _copy, tensors),
    ):
        with pytest.raises((LoweringError, ssa.VerificationError), match="broadcast"):
            prepare()


def test_arrangement_runs_once_and_does_not_mutate_user_descriptors():
    calls = []

    def layout(x, out):
        calls.append(1)

        return _layout(x, out)

    tensors = _descriptors()
    handle = interpret(layout, _copy, tensors)
    assert calls == [1]
    assert all(not tensor._history for tensor in tensors)
    _check_copy(handle)
    assert calls == [1]


@pytest.mark.parametrize("binding", ("default", "short", "full", "both"))
def test_meta_defaults_match_the_module_and_explicit_names_take_precedence(binding):
    block = Symbol(
        "BLOCK_SIZE", meta=True, lower_bound=300, upper_bound=1024, power_of_two=True
    )

    def layout(x, out):
        return x.tile((block,)), out.tile((block,))

    tensors = _descriptors()
    module = Compiler().frontend(
        CompileRequest(arrangement=layout, application=_copy, tensors=tensors)
    )
    full = str(block)
    short = remove_prefixes(full)
    assert module.meta_defaults[full] == 512
    options = {
        "default": {},
        "short": {short: 1024},
        "full": {full: 1024},
        "both": {full: 512, short: 1024},
    }
    supplied = options[binding]
    original = dict(supplied)
    handle = interpret(layout, _copy, tensors, meta=supplied)
    expected = 512 if binding in {"default", "both"} else 1024
    assert handle.meta[full] == expected
    assert supplied == original
    _check_copy(handle)


def test_transformed_program_reuses_frontend_and_layout_bindings():
    handle = interpret(_layout, _copy, _descriptors())
    original = handle.frontend_program
    transformed = replace(handle.program, kind="transformed_copy")
    replay = handle.with_program(transformed)
    assert handle.program is original
    assert replay.program is transformed
    assert replay.frontend_program is original
    assert replay.parameters == handle.parameters
    assert replay.tensors is handle.tensors
    assert replay.meta == handle.meta
    _check_copy(replay)
