"""
T3: GELU activation (tanh approximation).

Follows official ninetoothed pattern exactly.
"""

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


@ninetoothed.jit
def _gelu_kernel(
    x: Tensor(2).tile((1, BLOCK_SIZE)),
    y: Tensor(2).tile((1, BLOCK_SIZE)),
):
    """GELU: 0.5*x*(1+tanh(0.7979*(x+0.044715*x^3))). tanh via exp."""
    x3 = x * x * x
    inner = 0.7978845608028654 * (x + 0.044715 * x3)
    exp2 = ntl.exp(2.0 * inner)
    tanh_inner = (exp2 - 1.0) / (exp2 + 1.0)
    y = 0.5 * x * (1.0 + tanh_inner)  # noqa: F841


def gelu(X, Y):
    """Elementwise GELU. Works on contiguous and non-contiguous inputs."""
    _gelu_kernel(X, Y, BLOCK_SIZE=X.shape[-1])
