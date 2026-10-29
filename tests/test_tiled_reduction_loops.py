import pytest
import torch

import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor
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


def _row_arrangement(x, out):
    return x.tile((1, 8)), out.tile((1, 8))


def _row_min(x, out):
    out = x + ntl.min(x, axis=1)  # noqa: F841


def _row_max(x, out):
    out = x + ntl.max(x, axis=1)  # noqa: F841


def _row_exp_sum(x, out):
    out = x + ntl.sum(ntl.exp(x), axis=1)  # noqa: F841


def _nan_add(x, out):
    out = x + float("nan")  # noqa: F841


def _carry_arrangement(x, y, out):
    return x.tile((4,)), y.tile((4, 4)), out.tile((4, 4))


def _branch_carry(x, y, out):
    acc = x

    for i in range(2):
        if i > 0:
            acc = acc + 1
        else:
            acc = acc + 2

    out = y + acc  # noqa: F841


def _multiple_carries(x, y, out):
    a = x
    b = x * 2

    for i in range(2):
        if i > 0:
            a = a + 1
            b = b + 10
        else:
            a = a + 2
            b = b + 20

    out = y + a + b  # noqa: F841


def _sequential_carries(x, y, out):
    acc = x

    for i in range(2):
        acc = acc + 1

    for j in range(2):
        acc = acc + 2

    out = y + acc  # noqa: F841


def _initializer_arrangement(x, y, side, flag, out):
    return x.tile((4,)), y.tile((4, 4)), side.tile((4, 4)), flag, out.tile((4, 4))


def _effectful_initializer(x, y, side, flag, out):
    acc = x

    if flag:
        side = side + 1
        acc = x + 1
    else:
        side = side + 2
        acc = x + 2

    for i in range(2):
        acc = acc + 3

    out = y + acc  # noqa: F841


def _meta_arrangement(x, out, width=Symbol("ROW_WIDTH", constexpr=True)):
    return x.tile((1, width)), out.tile((1, width))


def _row_sum(x, out):
    out = x + ntl.sum(x, axis=1)[:, None]  # noqa: F841


def _transpose_arrangement(x, out):
    return x.tile((4, 4)), out.tile((4,))


def _transpose_sum(x, out):
    out = ntl.sum(x + x.T, axis=1)  # noqa: F841


@pytest.mark.parametrize("backend", ("cuda", "triton", "tilelang"))
@pytest.mark.parametrize("device", get_available_devices())
def test_scalar_row_lowering_runtime(device, backend):
    for application, x, expected in (
        (_row_min, torch.tensor([[1.0, 2.0, 3.0]], device=device), [[2.0, 3.0, 4.0]]),
        (
            _row_max,
            torch.tensor([[-1.0, -2.0, -3.0]], device=device),
            [[-2.0, -3.0, -4.0]],
        ),
        (_row_exp_sum, torch.zeros((1, 3), device=device), [[3.0, 3.0, 3.0]]),
        (_nan_add, torch.zeros((1, 3), device=device), [[float("nan")] * 3]),
    ):
        kernel = make(
            _row_arrangement,
            application,
            (
                Tensor(shape=(1, 3), dtype="float32"),
                Tensor(shape=(1, 3), dtype="float32"),
            ),
            backend=backend,
            max_num_configs=1,
        )
        out = torch.empty_like(x)
        kernel(x, out)
        torch.testing.assert_close(
            out, torch.tensor(expected, device=device), equal_nan=True
        )

    x = torch.arange(4, dtype=torch.float32, device=device)
    y = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
    out = torch.empty_like(y)

    for application, scale, offset in (
        (_branch_carry, 1, 3),
        (_multiple_carries, 3, 33),
        (_sequential_carries, 1, 6),
    ):
        kernel = make(
            _carry_arrangement,
            application,
            (
                Tensor(shape=(4,), dtype="float32"),
                Tensor(shape=(4, 4), dtype="float32"),
                Tensor(shape=(4, 4), dtype="float32"),
            ),
            backend=backend,
            max_num_configs=1,
        )
        kernel(x, y, out)
        torch.testing.assert_close(out, y + x * scale + offset)

    kernel = make(
        _initializer_arrangement,
        _effectful_initializer,
        (
            Tensor(shape=(4,), dtype="float32"),
            Tensor(shape=(4, 4), dtype="float32"),
            Tensor(shape=(4, 4), dtype="float32"),
            Tensor(0, dtype="bool"),
            Tensor(shape=(4, 4), dtype="float32"),
        ),
        backend=backend,
        max_num_configs=1,
    )

    for flag in (True, False):
        side = torch.zeros_like(y)
        increment = 1 if flag else 2
        kernel(x, y, side, flag, out)
        torch.testing.assert_close(side, torch.full_like(side, increment))
        torch.testing.assert_close(out, y + x + increment + 6)

    x = torch.cat((torch.zeros(256, device=device), torch.ones(256, device=device)))[
        None, :
    ]
    out = torch.empty_like(x)
    kernel = make(
        _meta_arrangement,
        _row_sum,
        (
            Tensor(shape=(1, 512), dtype="float32"),
            Tensor(shape=(1, 512), dtype="float32"),
        ),
        backend=backend,
        max_num_configs=1,
    )
    kernel(x, out)
    torch.testing.assert_close(out, torch.cat((x[:, :256], x[:, 256:] + 256), dim=1))

    if backend == "cuda":
        kernel = make(
            _transpose_arrangement,
            _transpose_sum,
            (
                Tensor(shape=(2, 4), dtype="float32", other=0),
                Tensor(shape=(2,), dtype="float32"),
            ),
            backend=backend,
            max_num_configs=1,
        )
        x = torch.arange(1, 9, device=device, dtype=torch.float32).reshape(2, 4)
        out = torch.empty(2, device=device)
        kernel(x, out)
        torch.testing.assert_close(out, torch.tensor([16.0, 34.0], device=device))
