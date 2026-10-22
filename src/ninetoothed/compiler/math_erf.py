"""Typed bounded erf arithmetic with an explicit full-domain fallback contract."""

from dataclasses import dataclass

from ninetoothed.compiler.math_approximations import (
    PolynomialDomain,
    PolynomialStep,
    _float32,
)


@dataclass(frozen=True)
class BoundedErf:
    opcode: str
    dtype: str
    domain: PolynomialDomain
    coefficients: tuple[float, ...]
    variable_scale: float
    variable_shift: float
    steps: tuple[PolynomialStep, ...]
    result: str
    exact_arithmetic_error_bound: float


def _recipe():
    # Chebyshev truncation in t=x*x/2-1, then conversion to float32 power
    # coefficients. An exact rational Chebyshev l1 comparison with the
    # 34-term erf Taylor series bounds real-arithmetic error by 1.923e-7
    # on [-2,2], including coefficient rounding. Float32 operation rounding
    # is additional and must be validated by each actual target consumer.
    coefficients = tuple(
        map(
            _float32,
            (
                7.82495117164217e-05,
                -0.0003679524234030396,
                0.0013993724714964628,
                -0.005209906492382288,
                0.017130400985479355,
                -0.048669762909412384,
                0.11947888135910034,
                -0.261111319065094,
                0.6749332547187805,
            ),
        )
    )
    steps = [
        PolynomialStep("arith.mul", "square", ("input", "input")),
        PolynomialStep("arith.constant", "scale", constant=0.5),
        PolynomialStep("arith.mul", "scaled_square", ("square", "scale")),
        PolynomialStep("arith.constant", "shift", constant=-1.0),
        PolynomialStep("arith.add", "variable", ("scaled_square", "shift")),
        PolynomialStep("arith.constant", "polynomial_0", constant=coefficients[0]),
    ]
    previous = "polynomial_0"

    for index, coefficient in enumerate(coefficients[1:], start=1):
        product, constant, result = (
            f"product_{index}",
            f"coefficient_{index}",
            f"polynomial_{index}",
        )
        steps.extend(
            (
                PolynomialStep("arith.mul", product, (previous, "variable")),
                PolynomialStep("arith.constant", constant, constant=coefficient),
                PolynomialStep("arith.add", result, (product, constant)),
            )
        )
        previous = result
    # Multiplication by the original input preserves either sign of zero.

    steps.append(PolynomialStep("arith.mul", "result", (previous, "input")))

    return BoundedErf(
        "math.erf",
        "float32",
        PolynomialDomain("float32", -2.0, 2.0),
        coefficients,
        0.5,
        -1.0,
        tuple(steps),
        "result",
        2e-7,
    )


_ERF = _recipe()


def bounded_erf(opcode, *, dtype="float32"):
    """Only active, finite in-domain lanes may consume the polynomial.

    Consumers retain the original full-range helper outside this closed
    domain, including NaN and infinities. Separate float32 multiply/add
    steps must not be reassociated or silently contracted to FMA.
    """
    return _ERF if opcode == "math.erf" and dtype == "float32" else None
