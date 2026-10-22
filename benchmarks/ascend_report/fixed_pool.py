"""Use the pinned max-pool algorithm with one fixed scheduling configuration."""

import functools
import importlib
import types

BLOCK_SIZE = 128
NUM_WARPS = 8
NUM_STAGES = 1
MAX_NUM_CONFIGS = 1
KERNEL_NAME = "ntops_lab_max_pool2d_fixed128"


@functools.cache
def build_kernel():
    """Reuse the source arrangement and application without changing globals."""
    import ninetoothed

    source = importlib.import_module("ntops_lab.kernels.layout.avg_pool2d")
    original = source._pool2d_arrangement
    globals_ = dict(original.__globals__)
    globals_["block_size"] = lambda: BLOCK_SIZE
    arrangement = types.FunctionType(
        original.__code__,
        globals_,
        name=original.__name__,
        argdefs=original.__defaults__,
        closure=original.__closure__,
    )
    arrangement.__kwdefaults__ = original.__kwdefaults__

    return ninetoothed.make(
        functools.partial(
            arrangement,
            kernel_h=3,
            kernel_w=3,
            stride_h=2,
            stride_w=2,
            pad_h=1,
            pad_w=1,
        ),
        source._max_pool_application,
        (
            ninetoothed.Tensor(4, other=float("-inf")),
            ninetoothed.Tensor(4),
        ),
        kernel_name=KERNEL_NAME,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        max_num_configs=MAX_NUM_CONFIGS,
    )


def run(x):
    """Allocate the original k3/s2/p1 output and launch its fixed kernel once."""
    import torch

    n, c, h, w = x.shape
    out_h = (h + 2 - 3) // 2 + 1
    out_w = (w + 2 - 3) // 2 + 1
    out = torch.empty((n, c, out_h, out_w), device=x.device, dtype=x.dtype)
    build_kernel()(x, out)

    return out
