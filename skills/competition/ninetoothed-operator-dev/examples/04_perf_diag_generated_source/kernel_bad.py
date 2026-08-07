"""DELIBERATELY SUBOPTIMAL softmax kernel — used as the 'before' in the perf/diag self-test task.

Two intentional problems:
1. Missing numerical stability: no subtract-max step → exp(large value) overflows.
2. No fp16→fp32 upcast before exp: precision loss on large rows.

On triton>=3.x, problem 2 is not merely a precision issue — `tl.exp` rejects
fp16 outright (`ValueError: Expected dtype ['fp32','fp64'] but got fp16`), so
this kernel fails to *compile* on fp16 input. That hard compile error is the
diagnostic signal; the fix (kernel_fixed) upcasts to fp32 before exp. On stacks
where it does compile, problem 1 makes the result numerically wrong. Both are
detectable in the generated source (no fp32 cast visible before exp).
"""

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(x, out, BLOCK_SIZE=BLOCK_SIZE):
    return x.tile((1, BLOCK_SIZE)), out.tile((1, BLOCK_SIZE))


def application(x, out):
    # BUG 1: no subtract-max (numerical instability)
    # BUG 2: no upcast to fp32 (precision loss on fp16).
    numerator = ntl.exp(x)
    out = numerator / ntl.sum(numerator)  # noqa: F841


_TENSORS = (Tensor(2, other=float("-inf")), Tensor(2))

kernel_bad = ninetoothed.make(arrangement, application, _TENSORS)
