import pytest
import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.compiler import load_built_artifact, make
from tests.utils import get_available_devices


def _arrangement(x, out):
    arranged = []

    for tensor in (x, out):
        tensor = tensor.tile((1, 32)).tile((1, -1))
        tensor.dtype = tensor.dtype.squeeze((0,))
        tensor.dtype.dtype = tensor.dtype.dtype.squeeze((0,))
        arranged.append(tensor)

    return tuple(arranged)


def _accumulate_scalar_reductions(x, out):
    maximum = ntl.cast(float("-inf"), ntl.float32)
    total = ntl.cast(0, ntl.float32)

    for i in range(x.shape[0]):
        block = ntl.cast(x[i], ntl.float32)
        maximum = ntl.maximum(maximum, ntl.max(block))
        total = total + ntl.sum(block)

    for i in range(x.shape[0]):
        out[i] = x[i] + total - maximum


def _accumulate_extrema(x, out):
    dtype = x.dtype.dtype
    maximum = ntl.cast(float("-inf"), dtype)
    minimum = ntl.cast(float("inf"), dtype)

    for i in range(x.shape[0]):
        maximum = ntl.cast(ntl.maximum(maximum, ntl.max(x[i])), dtype)
        minimum = ntl.cast(ntl.minimum(minimum, ntl.min(x[i])), dtype)

    for i in range(x.shape[0]):
        out[i] = ntl.cast(x[i], ntl.float32) + minimum - maximum


@pytest.mark.parametrize(
    "application", (_accumulate_scalar_reductions, _accumulate_extrema)
)
@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("width", (17, 65))
@pytest.mark.parametrize("aot", (False, True))
def test_tiled_scalar_reductions_preserve_each_loop_result(
    application, device, dtype, width, aot, tmp_path
):
    shape = (3, width)
    tensor_dtype = getattr(ninetoothed, str(dtype).removeprefix("torch."))
    tensors = (
        Tensor(shape=shape, dtype=tensor_dtype, other=0),
        Tensor(shape=shape, dtype=tensor_dtype),
    )
    kernel = make(
        _arrangement,
        application,
        tensors,
        backend="triton",
        caller=device if aot else "torch",
        output_dir=tmp_path if aot else None,
        max_num_configs=1,
    )
    x = (torch.rand((3, width * 2), device=device, dtype=dtype) + 1)[:, ::2]
    out = torch.empty_like(x)

    if application is _accumulate_extrema:
        expected = x.float() + x.float().min(dim=1, keepdim=True).values
    else:
        expected = x.float() + x.float().sum(dim=1, keepdim=True)

    expected -= x.float().max(dim=1, keepdim=True).values
    launchers = (
        (kernel, load_built_artifact(kernel._built_artifact)) if aot else (kernel,)
    )

    for launch in launchers:
        launch(x, out)
        torch.testing.assert_close(out, expected.to(dtype))
