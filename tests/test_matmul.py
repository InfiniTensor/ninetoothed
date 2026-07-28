import pytest
import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor
from tests.utils import get_available_devices

BLOCK_SIZE_M = Symbol("BLOCK_SIZE_M", meta=True)
BLOCK_SIZE_N = Symbol("BLOCK_SIZE_N", meta=True)
BLOCK_SIZE_K = Symbol("BLOCK_SIZE_K", meta=True)

_FP16_RTOL = 0.001
_FP16_ATOL = 0.001


def arrangement(
    lhs,
    rhs,
    output,
    BLOCK_SIZE_M=BLOCK_SIZE_M,
    BLOCK_SIZE_N=BLOCK_SIZE_N,
    BLOCK_SIZE_K=BLOCK_SIZE_K,
):
    output_tiled = output.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))

    lhs_tiled = (
        lhs.tile((BLOCK_SIZE_M, BLOCK_SIZE_K))
        .tile((1, -1))
        .expand((-1, output_tiled.shape[1]))
    )
    lhs_tiled.dtype = lhs_tiled.dtype.squeeze(0)

    rhs_tiled = (
        rhs.tile((BLOCK_SIZE_K, BLOCK_SIZE_N))
        .tile((-1, 1))
        .expand((output_tiled.shape[0], -1))
    )
    rhs_tiled.dtype = rhs_tiled.dtype.squeeze(1)

    return lhs_tiled, rhs_tiled, output_tiled


def application(lhs, rhs, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)

    for k in range(lhs.shape[0]):
        accumulator += ntl.dot(lhs[k], rhs[k])

    output = accumulator.to(ntl.float16)


def matmul(lhs, rhs):
    output = torch.empty(
        (lhs.shape[0], rhs.shape[1]), device=lhs.device, dtype=torch.float16
    )

    matmul_kernel = ninetoothed.make(
        arrangement, application, (Tensor(2), Tensor(2), Tensor(2))
    )

    matmul_kernel(lhs, rhs, output)

    return output


def _device_dtype_config(devices, *, fp16_rtol, fp16_atol):
    config = []

    for device in devices:
        config.append(pytest.param(device, torch.float16, fp16_rtol, fp16_atol))

        if hasattr(torch, "float8_e5m2"):
            config.append(
                pytest.param(
                    device,
                    torch.float8_e5m2,
                    1e-5,
                    0.125,
                    marks=pytest.mark.requires_capability(
                        f"tests.capabilities.float8:float8_e5m2_{device}"
                    ),
                )
            )

    return tuple(config)


@pytest.mark.parametrize(
    "device, dtype, rtol, atol",
    _device_dtype_config(
        get_available_devices(),
        fp16_rtol=_FP16_RTOL,
        fp16_atol=_FP16_ATOL,
    ),
)
@pytest.mark.parametrize("k", (512,))
@pytest.mark.parametrize("n", (512,))
@pytest.mark.parametrize("m", (512,))
def test(m, n, k, dtype, device, rtol, atol):
    randn_dtype = dtype if dtype != torch.float8_e5m2 else torch.float16

    input = torch.randn((m, k), dtype=randn_dtype, device=device)
    other = torch.randn((k, n), dtype=randn_dtype, device=device)

    if dtype == torch.float8_e5m2:
        input = input.to(dtype)
        other = other.T.to(dtype)

        output = matmul(input, other)
        expected = torch.matmul(input.to(torch.float16), other.to(torch.float16))
    else:
        output = matmul(input, other)
        expected = torch.matmul(input, other)

    assert torch.allclose(output, expected, rtol=rtol, atol=atol)
