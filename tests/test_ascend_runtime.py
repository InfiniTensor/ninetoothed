import pytest
import torch

import ninetoothed
from ninetoothed import Tensor


def _arrangement(x, y, out):
    return tuple(tensor.tile((512,)) for tensor in (x, y, out))


def _add(x, y, out):
    out = x + y  # noqa: F841


def _fused(x, y, out):
    out = (x + y) * (x - y)  # noqa: F841


@pytest.mark.parametrize("application", (_add, _fused))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
@pytest.mark.parametrize("size", (1, 513, 98432))
def test_ascend_jit_matches_torch(application, dtype, size):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    kernel = ninetoothed.make(
        _arrangement,
        application,
        (Tensor(1), Tensor(1), Tensor(1)),
        backend="ascend",
        platform="ascend-910b4",
    )
    x = torch.randn(size, device="npu", dtype=dtype)
    y = torch.randn_like(x)
    out = torch.empty_like(x)
    expected = x + y if application is _add else (x + y) * (x - y)
    for _ in range(2):
        kernel(x, y, out)
        torch.npu.synchronize()
        torch.testing.assert_close(out, expected)
