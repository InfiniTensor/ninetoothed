"""Retain dynamic expression semantics while reusing bounded evaluation plans."""

import gc
import weakref

import numpy as np
import pytest

from ninetoothed.interpreter.expressions import evaluate
from ninetoothed.ir import IndexExpr


def constant(value):
    return IndexExpr(op="constant", value=value)


def symbol(name):
    return IndexExpr(op="symbol", value=name)


@pytest.mark.parametrize("literal", (True, 1, 1.0, 0, 0.0, -0.0))
def test_equal_literals_keep_their_types_and_signed_zero(literal):
    expressions = [
        IndexExpr(op="mul", operands=(constant(value), symbol("x")))
        for value in (True, 1, 1.0, 0, 0.0, -0.0, literal)
    ]

    for expression in expressions:
        value = expression.operands[0].value
        actual = evaluate(expression, {"x": 1})
        expected = value * 1
        assert type(actual) is type(expected)
        assert actual == expected

        if expected == 0:
            assert np.signbit(actual) == np.signbit(expected)

        direct = evaluate(expression.operands[0], {})
        assert type(direct) is type(value)


def test_reused_plan_reads_changed_symbols_and_does_not_retain_input_arrays():
    expression = IndexExpr.parse("x * factor + offset")
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    inputs = {"x": x, "factor": 2, "offset": 1}
    np.testing.assert_array_equal(evaluate(expression, inputs), x * 2 + 1)
    x[:] = -3
    inputs.update(factor=4, offset=-2)
    np.testing.assert_array_equal(evaluate(expression, inputs), x * 4 - 2)
    reference = weakref.ref(x)
    del x
    inputs["x"] = np.arange(5, dtype=np.int32)
    gc.collect()
    assert reference() is None
    np.testing.assert_array_equal(evaluate(expression, inputs), np.arange(5) * 4 - 2)


def test_unhashable_constant_remains_live_without_global_retention():
    array = np.array([2, 3], dtype=np.int32)
    expression = IndexExpr(op="add", operands=(constant(array), symbol("x")))
    np.testing.assert_array_equal(evaluate(expression, {"x": 1}), [3, 4])
    array[:] = 9
    np.testing.assert_array_equal(evaluate(expression, {"x": 3}), [12, 12])
    reference = weakref.ref(array)
    del expression, array
    gc.collect()
    assert reference() is None


def test_reused_plan_preserves_left_to_right_error_order():
    expression = IndexExpr.parse("missing + unsupported(1)")

    for _ in range(2):
        with pytest.raises(ValueError, match="Unbound layout symbol `missing`"):
            evaluate(expression, {})

        with pytest.raises(ValueError, match="Unsupported layout expression"):
            evaluate(expression, {"missing": 0})


def test_plan_cache_releases_old_program_expressions():
    references = []

    for index in range(1024):
        expression = IndexExpr(op="add", operands=(symbol("x"), constant(index)))
        references.append(weakref.ref(expression))
        assert evaluate(expression, {"x": 2}) == index + 2

    del expression
    gc.collect()
    assert sum(reference() is not None for reference in references) <= 256
