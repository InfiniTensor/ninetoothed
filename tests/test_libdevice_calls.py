import pytest
import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(x, y, out):
    return x.tile((32,)), y.tile((32,)), out.tile((32,))


def fmod_application(x, y, out):
    out = ntl.libdevice.fmod(x, y)  # noqa: F841


def reduction_arrangement(x, y, out):
    return x.tile((1, 32)), y.tile((1, 32)), out.tile((1,))


def reduction_application(x, y, out):
    out = ntl.sum(ntl.libdevice.fmod(x, y), axis=1)  # noqa: F841


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fmod_preserves_libdevice_target_and_boundary_values():
    kernel = ninetoothed.make(
        arrangement,
        fmod_application,
        (Tensor(1), Tensor(1), Tensor(1)),
        max_num_configs=1,
    )
    y = torch.linspace(0.5, 3.5, 4096, device="cuda")
    x = torch.nextafter(17 * y, torch.full_like(y, -float("inf")))
    x[::2] = -x[::2]
    y[::3] = -y[::3]
    out = torch.full_like(x, -12345)
    kernel(x, y, out)
    torch.testing.assert_close(out, torch.fmod(x, y), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_libdevice_call_inside_reduction():
    kernel = ninetoothed.make(
        reduction_arrangement,
        reduction_application,
        (Tensor(2), Tensor(2, other=1.0), Tensor(1)),
        max_num_configs=1,
    )
    x = torch.randn((3, 32), device="cuda")
    y = torch.rand_like(x) + 0.5
    out = torch.full((3,), -12345.0, device="cuda")
    kernel(x, y, out)
    torch.testing.assert_close(out, torch.fmod(x, y).sum(1))
