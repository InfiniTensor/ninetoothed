"""Independent domain, roundoff and full-bit controls for exponential recipes."""

import math

import numpy as np
import pytest

from ninetoothed.compiler.math_exponential import bounded_exponential


def _evaluate(recipe, values):
    values = {"input": np.asarray(values, dtype=np.float32)}

    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        for step in recipe.steps:
            operands = [values[name] for name in step.operands]

            if step.opcode == "arith.constant":
                value = np.float32(step.constant)
            elif step.opcode == "arith.mul":
                value = np.multiply(*operands, dtype=np.float32)
            elif step.opcode == "arith.add":
                value = np.add(*operands, dtype=np.float32)
            elif step.opcode == "arith.div":
                value = np.divide(*operands, dtype=np.float32)
            else:
                raise AssertionError(step.opcode)

            values[step.result] = value

    return values[recipe.result]


def _reference(opcode, values):
    values = np.asarray(values, dtype=np.float64)

    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        if opcode == "math.exp":
            result = np.exp(values)
        elif opcode == "math.exp2":
            result = np.exp2(values)
        else:
            result = 1.0 / (1.0 + np.exp(-values))

    return result.astype(np.float32)


def _relative_error_bound(recipe):
    """Bound Horner coefficient/operation rounding and three squared results.

    Coefficient rounding, separate Horner operations, argument scaling and
    squaring each have a distinct error budget. The exact coefficient
    reference is independently reconstructed from the Chebyshev expansion.
    This bound supports the public tolerance; stricter numerical controls
    do not constitute a stronger full-domain proof.
    """
    unit = 2.0**-24
    scale = math.log(2.0) * 0.125 if recipe.opcode == "math.exp2" else 0.125
    scale_error = 8.0 * abs(abs(recipe.input_scale) - scale)
    radius = max(8.0 * scale, 8.0 * abs(recipe.input_scale))
    argument_error = scale_error + unit * radius + 2.0**-150
    real_coefficients = _chebyshev_power_coefficients()
    rounded = np.asarray(recipe.coefficients[::-1], dtype=np.float64)
    degree = len(recipe.coefficients) - 1
    operation_count = 2 * degree
    gamma = operation_count * unit / (1.0 - operation_count * unit)
    coefficient_error = sum(
        abs(value - exact) * radius**index
        for index, (value, exact) in enumerate(zip(rounded, real_coefficients))
    )
    absolute_terms = sum(
        abs(value) * (radius + argument_error) ** index
        for index, value in enumerate(rounded)
    )
    absolute = (
        coefficient_error
        + gamma * absolute_terms
        + degree * 2.0**-150
        + math.exp(radius + argument_error) * argument_error
        + recipe.scaled_polynomial_truncation_bound
    )
    relative = absolute * math.exp(radius)
    squared = (1.0 + relative) ** 8 * (1.0 + unit) ** 7 - 1.0

    if recipe.sigmoid:
        denominator = squared + unit * (1.0 + squared)

        return (1.0 + unit) / (1.0 - denominator) - 1.0

    return squared


def _chebyshev_power_coefficients():
    """Reconstruct the real recipe separately from product coefficient data."""
    bessel = [
        math.fsum(
            0.5 ** (2 * term + index)
            / (math.factorial(term) * math.factorial(term + index))
            for term in range(32)
        )
        for index in range(9)
    ]
    coefficients = np.polynomial.chebyshev.cheb2poly(
        [bessel[0], *(2.0 * value for value in bessel[1:])]
    )
    coefficients[0] = 1.0

    return coefficients


@pytest.mark.parametrize("opcode", ("math.exp", "math.exp2", "math.sigmoid"))
def test_dense_domain_uses_unchanged_public_precision(opcode):
    recipe = bounded_exponential(opcode)
    values = np.linspace(-8.0, 8.0, 262145).astype(np.float32)
    actual = _evaluate(recipe, values)
    expected = _reference(opcode, values)
    assert np.allclose(actual, expected, atol=3e-5, rtol=2e-4)
    assert np.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert np.max(np.abs(actual.astype(np.float64) / expected - 1.0)) < 3e-6
    assert _relative_error_bound(recipe) < 2e-4
    assert recipe.scaled_polynomial_truncation_bound < 2e-8
