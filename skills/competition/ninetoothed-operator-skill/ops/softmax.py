"""
Operator: softmax (online two-pass)
Pattern: Reduction over last dim via autotuning block_size
Reference: torch.softmax(input, dim=-1)

Calling convention:
    kernel = make_softmax(ndim=2)
    kernel(input, output)   # NO BLOCK_SIZE kwarg
"""
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def _arrangement(*tensors):
    block_size = BLOCK_SIZE
    dim = -1
    dims = (dim,)
    ndim = max(t.ndim for t in tensors if t.ndim != 0)
    dims = tuple(d if d >= 0 else d + ndim for d in dims)
    non_target_dims = tuple(i for i in range(ndim) if i not in dims)

    def _arrange(t):
        if t.ndim == 0:
            return t
        arranged = t.permute(non_target_dims + dims)
        arranged = arranged.flatten(start_dim=-len(dims))
        inner = tuple(1 for _ in non_target_dims) + (block_size,)
        outer = tuple(1 for _ in non_target_dims) + (-1,)
        idx = tuple(range(len(non_target_dims)))
        arranged = arranged.tile(inner)
        arranged = arranged.tile(outer)
        arranged.dtype = arranged.dtype.squeeze(idx)
        arranged.dtype.dtype = arranged.dtype.dtype.squeeze(idx)
        return arranged

    return tuple(_arrange(t) for t in tensors)


def _exp(x, dtype):
    exp_dtype = dtype if dtype != ntl.float16 else ntl.float32
    return ntl.cast(ntl.exp(ntl.cast(x, exp_dtype)), dtype)


def _application(input, output):
    dtype = output.dtype.dtype
    prev_max = ntl.cast(float("-inf"), dtype)
    denominator = ntl.cast(0, dtype)

    for i in range(input.shape[0]):
        input_i = ntl.cast(input[i], dtype)
        curr_max = ntl.cast(ntl.maximum(prev_max, ntl.max(input_i)), dtype)
        input_max_diff_exp = _exp(input_i - curr_max, dtype)
        prev_curr_max_diff_exp = _exp(prev_max - curr_max, dtype)
        denominator = denominator * prev_curr_max_diff_exp + ntl.sum(input_max_diff_exp)
        prev_max = curr_max

    for i in range(input.shape[0]):
        numerator = _exp(input[i] - prev_max, dtype)
        output[i] = numerator / denominator


import triton as _triton


class _SoftmaxKernel:
    """Wrapper that makes BLOCK_SIZE optional.

    - k(x, out)              → BLOCK_SIZE = next_power_of_2(x.shape[-1])
    - k(x, out, BLOCK_SIZE=256) → uses provided value
    """

    def __init__(self, raw_kernel):
        self._raw = raw_kernel

    def __call__(self, *args, BLOCK_SIZE=None, **kwargs):
        if BLOCK_SIZE is None:
            BLOCK_SIZE = _triton.next_power_of_2(args[0].shape[-1])
        return self._raw(*args, BLOCK_SIZE=BLOCK_SIZE, **kwargs)


def make_softmax(ndim: int = 2):
    tensors = (
        Tensor(ndim, other=float("-inf"), shape_options={"constexpr": True}),
        Tensor(ndim),
    )
    raw = ninetoothed.make(_arrangement, _application, tensors)
    return _SoftmaxKernel(raw)


kernel = make_softmax(ndim=2)
