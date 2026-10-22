import pytest
import torch

import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.compiler import lower, make
from tests.utils import get_available_devices


def _tiled_arrangement(x, out):
    arranged = []

    for tensor in (x, out):
        tensor = tensor.tile((1, 256)).tile((1, -1))
        tensor.dtype = tensor.dtype.squeeze((0,))
        tensor.dtype.dtype = tensor.dtype.dtype.squeeze((0,))
        arranged.append(tensor)

    return tuple(arranged)


def _tiled_normalize(x, out):
    accumulator = ntl.zeros(x.dtype.shape, dtype=ntl.float32)

    for i in range(x.shape[0]):
        block = ntl.cast(x[i], ntl.float32)
        accumulator += block * block

    scale = ntl.rsqrt(ntl.sum(accumulator) / x.source.shape[1] + 1e-5)

    for i in range(x.shape[0]):
        out[i] = x[i] * scale


def test_tiled_reduction_uses_subscript_store_domain():
    artifact = lower(
        _tiled_arrangement,
        _tiled_normalize,
        (Tensor(2, other=0), Tensor(2)),
        backend="triton",
    )
    assert artifact.metadata["ssa_schedule"]["reduction"]["mode"] == "row-vector"
    assert "tl.sum(" in artifact.primary_source
    assert "ntl.float32" not in artifact.primary_source


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", (torch.float16, torch.float32))
@pytest.mark.parametrize("width", (17, 256, 1025))
def test_tiled_reduction_loop_runtime(device, dtype, width):
    kernel = make(
        _tiled_arrangement,
        _tiled_normalize,
        (Tensor(2, other=0), Tensor(2)),
        backend="triton",
        max_num_configs=1,
    )
    x = torch.randn((7, width * 2), device=device, dtype=dtype)[:, ::2]
    out = torch.empty((7, width), device=device, dtype=dtype)
    kernel(x, out)
    expected = x.float() * torch.rsqrt(
        x.float().square().mean(dim=1, keepdim=True) + 1e-5
    )
    torch.testing.assert_close(out, expected.to(dtype))
