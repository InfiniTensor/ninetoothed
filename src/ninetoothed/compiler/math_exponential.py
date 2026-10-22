"""Bounded float32 exponential recipes with explicit rounding boundaries.

A consumer must correct every active out-of-domain lane with its original
full-range primitive. These recipes do not permit fused multiply-add or change
the caller's precision tolerance, overflow policy, or padding mask.
"""

import math
from dataclasses import dataclass

from ninetoothed.compiler.math_approximations import (
    PolynomialDomain,
    PolynomialStep,
    _float32,
)

# Exp(t) on [-1, 1]: I_0(1) + 2 * sum(I_k(1) * T_k(t), k=1..8),
# converted to ascending power coefficients. Separate float32 Horner steps
# preserve the consumer's rounding contract. The constant is pinned to one
# before rounding so both signed zeros retain their exact exponential value.
_POWER_COEFFICIENTS = (
    1.0,
    0.9999999009433116,
    0.49999997254519923,
    0.16666798559827084,
    0.04166688603185045,
    0.00832859610683408,
    0.001388275939486621,
    0.0002046999335937274,
    0.000025499197525411786,
)
_TRUNCATION_BOUND = (
    2.0 * math.exp(0.25) * 0.5**9 / math.factorial(9) / (1.0 - 0.5 / 10.0)
    + abs(1.0 - 1.000000000549552)
    + 1e-13
)


@dataclass(frozen=True)
class BoundedExponential:
    opcode: str
    dtype: str
    domain: PolynomialDomain
    input_scale: float
    coefficients: tuple[float, ...]
    squarings: int
    sigmoid: bool
    steps: tuple[PolynomialStep, ...]
    result: str
    scaled_polynomial_truncation_bound: float


def _recipe(opcode, scale, *, sigmoid=False):
    coefficients = tuple(_float32(value) for value in reversed(_POWER_COEFFICIENTS))
    scale = _float32(scale)
    steps = [
        PolynomialStep("arith.constant", "scale", constant=scale),
        PolynomialStep("arith.mul", "scaled", ("input", "scale")),
        PolynomialStep("arith.constant", "polynomial_0", constant=coefficients[0]),
    ]
    previous = "polynomial_0"

    for index, coefficient in enumerate(coefficients[1:], start=1):
        product = f"product_{index}"
        constant = f"coefficient_{index}"
        result = f"polynomial_{index}"
        steps.extend(
            (
                PolynomialStep("arith.mul", product, (previous, "scaled")),
                PolynomialStep("arith.constant", constant, constant=coefficient),
                PolynomialStep("arith.add", result, (product, constant)),
            )
        )
        previous = result

    for index in range(3):
        result = f"squared_{index}"
        steps.append(PolynomialStep("arith.mul", result, (previous, previous)))
        previous = result

    if sigmoid:
        steps.extend(
            (
                PolynomialStep("arith.constant", "one", constant=1.0),
                PolynomialStep("arith.add", "denominator", ("one", previous)),
                PolynomialStep("arith.div", "result", ("one", "denominator")),
            )
        )
        previous = "result"

    return BoundedExponential(
        opcode=opcode,
        dtype="float32",
        domain=PolynomialDomain("float32", -8.0, 8.0),
        input_scale=scale,
        coefficients=coefficients,
        squarings=3,
        sigmoid=sigmoid,
        steps=tuple(steps),
        result=previous,
        # I_k(1) <= exp(1/4) * (1/2)^k / k!, and the k>=9 tail ratio
        # is at most 1/20. Include the pinned-constant displacement and a
        # binary64 conversion reserve. This is the approximation bound
        # before float32 coefficient/operation rounding and squaring.
        scaled_polynomial_truncation_bound=_TRUNCATION_BOUND,
    )


_RECIPES = {
    "math.exp": _recipe("math.exp", 0.125),
    "math.exp2": _recipe("math.exp2", math.log(2.0) * 0.125),
    "math.sigmoid": _recipe("math.sigmoid", -0.125, sigmoid=True),
}


def bounded_exponential(opcode, *, dtype="float32"):
    """Return a recipe for one canonical operation, independent of consumers."""
    return _RECIPES.get(opcode) if dtype == "float32" else None
