import unittest

import torch


def skip_if_cuda_not_available(func):
    return unittest.skipIf(not torch.cuda.is_available, "CUDA is not available")(func)


def skip_if_float8_e5m2_not_supported(func):
    return unittest.skipIf(
        not hasattr(torch, "float8_e5m2"),
        "`torch` does not have `float8_e5m2`",
    )(func)
