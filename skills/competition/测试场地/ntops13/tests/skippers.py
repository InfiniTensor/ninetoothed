import pytest
import torch


def skip_if_cuda_not_available(func):
    return pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )(func)
