"""Target-independent float32 arithmetic and bounded correction contracts."""

import math

import numpy as np
import pytest

from ninetoothed.compiler.math_approximations import bounded_polynomial


def _evaluate(recipe, values):
    environment = {"input": np.asarray(values, dtype=np.float32)}

    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        for step in recipe.steps:
            args = [environment[name] for name in step.operands]

            if step.opcode == "arith.constant":
                value = np.float32(step.constant)
            elif step.opcode == "arith.mul":
                value = np.multiply(*args, dtype=np.float32)
            elif step.opcode == "arith.add":
                value = np.add(*args, dtype=np.float32)
            else:
                raise AssertionError(step.opcode)

            environment[step.result] = value

    return environment[recipe.result]


def _reference(opcode, values):
    function = np.sin if opcode == "math.sin" else np.cos

    with np.errstate(invalid="ignore"):
        return function(np.asarray(values, dtype=np.float64)).astype(np.float32)


def _rounded_error_bound(recipe):
    """Conservative absolute forward bound for the separate f32 operations.

    Track magnitude/error against exact Taylor coefficients, including the
    rounded x*x argument and each multiply/add; use an absolute subnormal
    rounding allowance. The alternating remainder is added only at the end.
    This bound assumes IEEE round-to-nearest rather than vendor fast math.
    """
    unit = 2.0**-24
    tiny = 2.0**-150
    bound = recipe.domain.upper
    square = bound * bound
    square_error = unit * square + tiny
    parity = int(recipe.multiply_input)
    exact = [
        (-1.0) ** k / math.factorial(2 * k + parity)
        for k in range(recipe.degree, -1, -1)
    ]
    magnitude = abs(exact[0])
    float_magnitude = abs(recipe.coefficients[0])
    error = abs(recipe.coefficients[0] - exact[0])

    for coefficient, rounded in zip(exact[1:], recipe.coefficients[1:]):
        product_magnitude = float_magnitude * (square + square_error)
        error = (
            error * (square + square_error)
            + magnitude * square_error
            + unit * product_magnitude
            + tiny
            + abs(coefficient - rounded)
        )
        rounded_product = product_magnitude * (1.0 + unit) + tiny
        error += unit * (rounded_product + abs(rounded)) + tiny
        float_magnitude = (rounded_product + abs(rounded)) * (1.0 + unit) + tiny
        magnitude = magnitude * square + abs(coefficient)

    if recipe.multiply_input:
        error = error * bound + unit * float_magnitude * bound + tiny

    return error + recipe.truncation_error_bound


@pytest.mark.parametrize("opcode", ("math.sin", "math.cos"))
def test_dense_domain_separate_float32_horner_error(opcode):
    recipe = bounded_polynomial(opcode)
    values = np.linspace(recipe.domain.lower, recipe.domain.upper, 262145).astype(
        np.float32
    )
    error = float(
        np.max(np.abs(_evaluate(recipe, values) - _reference(opcode, values)))
    )
    analytic = _rounded_error_bound(recipe)
    print(f"{opcode}: dense_max_abs={error:.9g}, ieee_forward_bound={analytic:.9g}")
    assert error <= analytic
    assert analytic < 3e-5
    assert error < 2e-6
