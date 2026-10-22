"""Bounded erf precision and domain contracts independent of target source."""

import math

import numpy as np

from ninetoothed.compiler.math_erf import bounded_erf


def _evaluate(values):
    recipe = bounded_erf("math.erf")
    environment = {"input": np.asarray(values, dtype=np.float32)}

    with np.errstate(all="ignore"):
        for step in recipe.steps:
            if step.opcode == "arith.constant":
                result = np.float32(step.constant)
            elif step.opcode == "arith.mul":
                result = environment[step.operands[0]] * environment[step.operands[1]]
            elif step.opcode == "arith.add":
                result = environment[step.operands[0]] + environment[step.operands[1]]
            else:
                raise AssertionError(step.opcode)

            environment[step.result] = result
    return environment[recipe.result]


def test_dense_domain_keeps_existing_extra_absolute_precision_requirement():
    values = np.linspace(-2, 2, 262145, dtype=np.float32)
    reference = np.array([math.erf(float(value)) for value in values], dtype=np.float32)
    np.testing.assert_allclose(_evaluate(values), reference, atol=1e-6, rtol=0)
