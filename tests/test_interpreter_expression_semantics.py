"""Expression plans preserve dynamic behavior and the recursive numeric semantics."""

import gc
import operator
import random
import weakref

import numpy as np
import pytest

from ninetoothed.interpreter import expressions
from ninetoothed.ir import IndexExpr


def _constant(value):
    return IndexExpr(op="constant", value=value)


def _symbol(name):
    return IndexExpr(op="symbol", value=name)


@pytest.mark.parametrize(
    "source",
    (
        "(2 + 3) * x",
        "((True + 2) * (7 - 4)) + x",
        "(3 < 4) + x",
        "(-(2 + 5)) * x",
        "(+(False + 7)) - x",
        "((0 * 7) + 0) * x",
    ),
)
@pytest.mark.parametrize("value", (True, 1, -0.0, 2.5))
def test_constant_subtrees_preserve_result_types_and_signed_zero(source, value):
    expression = IndexExpr.parse(source)
    expected = expressions._evaluate(expression, {"x": value})
    actual = expressions.evaluate(expression, {"x": value})
    assert type(actual) is type(expected)
    assert actual == expected

    if expected == 0:
        assert np.signbit(actual) == np.signbit(expected)


@pytest.mark.parametrize("seed", range(8))
def test_random_integer_expression_trees_match_recursive_evaluation(seed):
    rng = random.Random(seed)

    def tree(depth):
        if not depth or rng.random() < 0.3:
            return (
                _symbol("x")
                if rng.random() < 0.3
                else _constant(rng.choice([False, True, -7, -1, 0, 2, 11]))
            )

        return IndexExpr(
            op=rng.choice(["add", "sub", "mul", "lt", "ge"]),
            operands=(tree(depth - 1), tree(depth - 1)),
        )

    for _ in range(60):
        expression = tree(4)

        for value in (-3, 0, 5, np.arange(5, dtype=np.int32)):
            try:
                expected = expressions._evaluate(expression, {"x": value})
            except (TypeError, ValueError) as error:
                with pytest.raises(type(error)) as actual_error:
                    expressions.evaluate(expression, {"x": value})

                assert str(actual_error.value) == str(error)
                continue

            actual = expressions.evaluate(expression, {"x": value})
            assert type(actual) is type(expected)
            np.testing.assert_array_equal(actual, expected)


def test_changed_symbols_arrays_and_their_lifetime_remain_dynamic():
    expression = IndexExpr.parse("((2 + 3) * (7 - 4)) * x + (9 - 8)")
    x = np.arange(7, dtype=np.float32)
    inputs = {"x": x}
    np.testing.assert_array_equal(expressions.evaluate(expression, inputs), x * 15 + 1)
    x[:] = -2
    np.testing.assert_array_equal(expressions.evaluate(expression, inputs), x * 15 + 1)
    reference = weakref.ref(x)
    del x
    inputs["x"] = np.arange(3, dtype=np.int32)
    gc.collect()
    assert reference() is None
    np.testing.assert_array_equal(
        expressions.evaluate(expression, inputs), inputs["x"] * 15 + 1
    )


def test_custom_binary_operator_is_never_executed_during_plan_construction(monkeypatch):
    calls = []

    def observed(a, b):
        calls.append((a, b))

        return operator.add(a, b)

    monkeypatch.setitem(expressions.BINARY, "add", observed)
    expression = IndexExpr.parse("missing + (2 + 3)")

    with pytest.raises(ValueError, match="Unbound layout symbol `missing`"):
        expressions.evaluate(expression, {})

    assert calls == []
    assert expressions.evaluate(expression, {"missing": 7}) == 12
    assert calls == [(2, 3), (7, 5)]
    expressions._compiled_expression.cache_clear()


@pytest.mark.parametrize(
    "source,setting",
    (("missing + (1.0 / 0.0)", "divide"), ("missing + (0.0 / 0.0)", "invalid")),
)
def test_warning_policy_and_left_to_right_errors_remain_live(source, setting):
    expression = IndexExpr.parse(source)

    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="Unbound layout symbol `missing`"):
            expressions.evaluate(expression, {})

    with np.errstate(all="ignore"):
        expressions.evaluate(expression, {"missing": 0})

    with np.errstate(**{setting: "raise"}):
        with pytest.raises(FloatingPointError):
            expressions.evaluate(expression, {"missing": 0})


def test_numpy_integer_overflow_policy_remains_live():
    expression = IndexExpr(
        op="add", operands=(_constant(np.int64(2**63 - 1)), _constant(np.int64(1)))
    )

    with np.errstate(over="ignore"):
        assert expressions.evaluate(expression, {}) == np.int64(-(2**63))

    with np.errstate(over="raise"):
        with pytest.raises(FloatingPointError):
            expressions.evaluate(expression, {})


class _MutableInteger(int):
    def __new__(cls):
        result = super().__new__(cls, 1)
        result.calls = 0

        return result

    def __add__(self, other):
        self.calls += 1

        return int(self) + other + self.calls


def test_integer_subclasses_keep_runtime_operator_calls():
    value = _MutableInteger()
    expression = IndexExpr(op="add", operands=(_constant(value), _constant(2)))
    assert expressions.evaluate(expression, {}) == 4
    assert expressions.evaluate(expression, {}) == 5
    assert value.calls == 2


@pytest.mark.parametrize("large", (2**64, -(2**96)))
def test_large_integer_constants_preserve_unbounded_runtime_arithmetic(large):
    expression = IndexExpr(op="mul", operands=(_constant(large), _constant(large)))
    assert expressions.evaluate(expression, {}) == large * large


def test_mutable_array_constants_are_not_retained_in_plans():
    value = np.arange(3, dtype=np.int32)
    expression = IndexExpr(
        op="add", operands=(_constant(value), IndexExpr.parse("2 + 3"))
    )
    np.testing.assert_array_equal(expressions.evaluate(expression, {}), [5, 6, 7])
    value[:] = 11
    np.testing.assert_array_equal(expressions.evaluate(expression, {}), [16, 16, 16])
    reference = weakref.ref(value)
    del expression, value
    gc.collect()
    assert reference() is None
