import pytest
import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor
from tests.utils import get_available_devices


def _arrangement(x, out):
    arranged = []

    for tensor in (x, out):
        tensor = tensor.tile((1, 32)).tile((1, -1))
        tensor.dtype = tensor.dtype.squeeze((0,))
        tensor.dtype.dtype = tensor.dtype.dtype.squeeze((0,))
        arranged.append(tensor)

    return tuple(arranged)


def _exp(x, dtype):
    exp_dtype = dtype if dtype != ntl.float16 else ntl.float32

    return ntl.cast(ntl.exp(ntl.cast(x, exp_dtype)), dtype)


def _application(x, out):
    dtype = out.dtype.dtype
    prev_max = ntl.cast(float("-inf"), dtype)
    denominator = ntl.cast(0, dtype)

    for i in range(x.shape[0]):
        x_i = ntl.cast(x[i], dtype)
        curr_max = ntl.cast(ntl.maximum(prev_max, ntl.max(x_i)), dtype)
        numerator = _exp(x_i - curr_max, dtype)
        correction = _exp(prev_max - curr_max, dtype)
        denominator = denominator * correction + ntl.sum(numerator)
        prev_max = curr_max

    for i in range(x.shape[0]):
        numerator = _exp(x[i] - prev_max, dtype)
        out[i] = numerator / denominator


def _premake(width=None, dtype=None):
    tensors = (
        Tensor(
            shape=(3, width),
            dtype=dtype,
            other=float("-inf"),
            shape_options={"constexpr": True},
        ),
        Tensor(shape=(3, width), dtype=dtype),
    )

    return _arrangement, _application, tensors


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", ("jit", "aot"))
@pytest.mark.parametrize("width", (17, 257))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol", "sum_atol"),
    (
        (torch.float16, 1e-2, 1e-3, 2e-3),
        (torch.bfloat16, 1e-2, 1e-2, 1e-2),
        (torch.float32, 1e-5, 3e-5, 1e-5),
    ),
)
def test_online_softmax_with_local_dtype_helpers(
    tmp_path, device, mode, width, dtype, rtol, atol, sum_atol
):
    if mode == "jit":
        kernel = ninetoothed.make(*_premake(), backend="triton", max_num_configs=1)
    else:
        kernel = ninetoothed.build(
            _premake,
            (
                (
                    (),
                    {"width": width, "dtype": getattr(ninetoothed, str(dtype)[6:])},
                    {},
                ),
            ),
            meta_parameters=("width", "dtype"),
            backend="triton",
            caller=device,
            kernel_name="online_softmax",
            output_dir=tmp_path,
        )

    x = torch.randn((3, width * 2), dtype=dtype, device=device)[:, ::2]
    out = torch.empty((3, width), dtype=dtype, device=device)
    kernel(x, out)
    expected = torch.softmax(x.float(), dim=-1).to(dtype)
    # Match the consuming Softmax API's low-precision tolerances, and also
    # verify normalization so small probabilities cannot hide invalid outputs.
    torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(
        out.float().sum(dim=-1),
        torch.ones(3, device=device),
        rtol=0,
        atol=sum_atol,
    )
