"""Focused tests for Triton emitter syntax mappings."""

from ninetoothed.backends.emitters.ssa import _nested_python_conjunction
from ninetoothed.backends.emitters.triton import TritonTarget
from ninetoothed.frontend.types import _interleave_type
from ninetoothed.ir import ssa


def test_triton_interleave_call_is_namespace_qualified():
    target = TritonTarget()

    assert target.call("interleave", ("left", "right")) == (
        "tl.interleave(left, right)"
    )


def test_interleave_type_doubles_the_last_dimension():
    type_ = ssa.Type(kind="tensor", shape=("16", "32"), dtype="bfloat16")

    assert _interleave_type(type_, type_).shape == ("16", "64")


def test_triton_transpose_call_is_namespace_qualified():
    target = TritonTarget()

    assert target.call("trans", ("value",)) == "tl.trans(value)"


def test_triton_bitcast_preserves_raw_bits():
    target = TritonTarget()

    assert target.bitcast("float32", "value") == ("value.to(tl.float32, bitcast=True)")


def test_triton_runtime_stride_guard_uses_binary_conjunctions():
    predicate = _nested_python_conjunction(("(a == 1)", "(b == 1)", "(c == 1)"))

    assert predicate == "(((a == 1) and (b == 1)) and (c == 1))"
