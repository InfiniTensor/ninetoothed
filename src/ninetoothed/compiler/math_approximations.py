"""Bounded, typed arithmetic recipes for target vector math lowering.

These recipes describe individual SSA math operations, not applications. A
consumer must retain the original full-domain primitive for every valid lane
outside the closed domain, including NaN and infinities. Each arithmetic step
rounds separately to float32; replacing multiply/add pairs with FMA is outside
this contract. Padding lanes are never part of the correction domain.
"""

import math
import struct
from dataclasses import dataclass


def _float32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


@dataclass(frozen=True)
class PolynomialDomain:
    dtype: str
    lower: float
    upper: float

    def contains(self, value):
        """Ordered comparisons deliberately reject NaN and both infinities."""
        return self.lower <= value <= self.upper


@dataclass(frozen=True)
class PolynomialStep:
    opcode: str
    result: str
    operands: tuple[str, ...] = ()
    constant: float | None = None


@dataclass(frozen=True)
class BoundedPolynomial:
    opcode: str
    dtype: str
    domain: PolynomialDomain
    coefficients: tuple[float, ...]
    multiply_input: bool
    steps: tuple[PolynomialStep, ...]
    result: str
    truncation_error_bound: float

    @property
    def degree(self):
        """Degree in x squared, rather than the final polynomial in x."""
        return len(self.coefficients) - 1


def _recipe(opcode, degree, *, multiply_input):
    bound = _float32(math.pi)
    parity = int(multiply_input)
    coefficients = tuple(
        _float32((-1.0) ** k / math.factorial(2 * k + parity))
        for k in range(degree, -1, -1)
    )
    steps = [PolynomialStep("arith.mul", "square", ("input", "input"))]
    steps.append(
        PolynomialStep("arith.constant", "polynomial_0", constant=coefficients[0])
    )
    previous = "polynomial_0"

    for index, coefficient in enumerate(coefficients[1:], start=1):
        product = f"product_{index}"
        constant = f"coefficient_{index}"
        result = f"polynomial_{index}"
        steps.extend(
            (
                PolynomialStep("arith.mul", product, (previous, "square")),
                PolynomialStep("arith.constant", constant, constant=coefficient),
                PolynomialStep("arith.add", result, (product, constant)),
            )
        )
        previous = result

    if multiply_input:
        steps.append(PolynomialStep("arith.mul", "result", (previous, "input")))
        previous = "result"

    # Alternating Taylor remainder for exact arithmetic. Float32 coefficient
    # and operation rounding are additional errors, covered separately by
    # numerical validation; this field is not a full floating-point bound.
    next_power = 2 * (degree + 1) + parity

    return BoundedPolynomial(
        opcode=opcode,
        dtype="float32",
        domain=PolynomialDomain("float32", -bound, bound),
        coefficients=coefficients,
        multiply_input=multiply_input,
        steps=tuple(steps),
        result=previous,
        truncation_error_bound=bound**next_power / math.factorial(next_power),
    )


_RECIPES = {
    "math.sin": _recipe("math.sin", 8, multiply_input=True),
    "math.cos": _recipe("math.cos", 9, multiply_input=False),
}


def bounded_polynomial(opcode, *, dtype="float32"):
    """Return a validated recipe only for a canonical float32 SSA operation.

    Dtype normalization belongs to the caller. No approximate full-domain
    argument reduction or weaker precision policy is implied by this lookup.
    """
    return _RECIPES.get(opcode) if dtype == "float32" else None
