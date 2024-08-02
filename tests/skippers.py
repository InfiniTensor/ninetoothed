import pytest
import torch


def skip_if_cuda_not_available(func):
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available",
    )(func)


def skip_if_float8_e5m2_not_supported(func):
    return pytest.mark.skipif(
        not hasattr(torch, "float8_e5m2"),
        reason="`float8_e5m2` not supported",
    )(func)
