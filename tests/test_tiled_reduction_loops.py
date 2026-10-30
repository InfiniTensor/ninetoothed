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


def _stateful_arrangement(x, side, flag, out):
    return x.tile((4,)), side.tile((4, 4)), flag, out.tile((4, 4))


def _stateful_application(x, side, flag, out):
    a = x
    b = x * 2

    if (x > 2) == flag:
        side = side + 1
        a = x + 1
        b = x * 2 + 10
    else:
        side = side + 2
        a = x + 2
        b = x * 2 + 20

    for i in range(2):
        a = a + 3
        b = b + 4

    out = a + b  # noqa: F841


def _pure_initial_application(x, side, flag, out):
    initial = x

    if flag:
        initial = x + 1
    else:
        initial = x + 2

    acc = initial

    for i in range(2):
        acc = acc + 3

    out = initial[:, None] + acc  # noqa: F841


@pytest.mark.parametrize(
    "application", (_stateful_application, _pure_initial_application)
)
@pytest.mark.parametrize("device", get_available_devices())
def test_native_carry_runtime(device, application):
    kernel = make(
        _stateful_arrangement,
        application,
        (
            Tensor(shape=(4,), dtype="float32"),
            Tensor(shape=(4, 4), dtype="float32"),
            Tensor(0, dtype="bool"),
            Tensor(shape=(4, 4), dtype="float32"),
        ),
        backend="cuda",
        max_num_configs=1,
    )
    x = torch.arange(4, device=device, dtype=torch.float32)

    for flag in (True, False):
        side = torch.zeros((4, 4), device=device)
        out = torch.empty_like(side)
        kernel(x, side, flag, out)

        if application is _pure_initial_application:
            torch.testing.assert_close(side, torch.zeros_like(side))
            torch.testing.assert_close(
                out, x[:, None] + x[None, :] + 6 + 2 * (1 if flag else 2)
            )
            continue

        condition = (x > 2) == flag
        torch.testing.assert_close(
            side, torch.where(condition, 1.0, 2.0).expand_as(side)
        )
        torch.testing.assert_close(
            out, (3 * x + torch.where(condition, 25.0, 36.0)).expand_as(out)
        )
