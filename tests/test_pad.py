import torch
import torch.nn.functional as F
import pytest
import ninetoothed as nt
from typing import Dict, Tuple
import xxhash
import os

def analyze_pad_config(x, pad, mode):
    assert mode == "constant", "Only support constant padding"
    ndim = x.ndim
    old_shape = list(x.shape)
    new_shape = list(x.shape)
    slice_l = []
    slice_r = []
    for i in range(ndim):
        left = pad[2 * (ndim - 1 - i)]
        right = pad[2 * (ndim - 1 - i) + 1]
        new_shape[i] += left + right

        start_s = max(0, -left)
        end_s = min(old_shape[i], old_shape[i] + right)

        start_d = max(0, left)
        end_d = new_shape[i] - max(0, right)

        slice_l.append(slice(start_d, end_d))
        slice_r.append(slice(start_s, end_s))
    slice_l = tuple(slice_l)
    slice_r = tuple(slice_r)
    return new_shape, slice_l, slice_r

def fake_pad(x, pad, mode="constant", value=0):
    # torch implementation
    new_shape, slice_l, slice_r = analyze_pad_config(x, pad, mode)
    y = torch.full(new_shape, value, device=x.device, dtype=x.dtype)
    y[slice_l] = x[slice_r]
    return y

def arrangement(
    lhs,
    rhs,
    slice_l,
    slice_r,
):
    return lhs[slice_l], rhs[slice_r]


def application(lhs, rhs):
    lhs = rhs


kernel = {}

def pad_kernel(x, pad, mode="constant", value=0):
    # analyze pad config
    new_shape, slice_l, slice_r = analyze_pad_config(x, pad, mode)
    ndim = x.ndim
    y = torch.full(new_shape, value, device=x.device, dtype=x.dtype)

    # create kernel
    kernel_config = (ndim, slice_l, slice_r)
    kernel_hash = xxhash.xxh32(str(kernel_config)).intdigest()
    if kernel.get(kernel_hash, None) is None:
        t_kernel = nt.make(
            arrangement,
            application,
            (
                nt.Tensor(ndim, shape_options={"upper_bound": 128}),
                nt.Tensor(ndim, shape_options={"upper_bound": 128}),
                slice_l,
                slice_r,
            ),
        )
        kernel[kernel_hash] = t_kernel
    # apply padding
    kernel[kernel_hash](y, x)

    return y

def nt_pad(x, pad, mode="constant", value=0):
    if os.getenv("DEBUG", "0") == "1":
        print("fake pad!")
        return fake_pad(x, pad, mode, value)
    return pad_kernel(x, pad, mode, value)

@pytest.mark.parametrize(
    "mode",
    ["constant"],
)
@pytest.mark.parametrize(
    "value",
    [0, 1, -1],
)
@pytest.mark.parametrize(
    "shape, pad",
    [
        ((2, 3), (1, 2, 0, 1)),
        ((2, 3), (1, 1, 1, 2)),
        ((2, 3, 4), (1, 3, 1, 0, 0, 0)),
        ((2, 3), (-1, -1, 0, 0)),
    ],
)
def test_pad_basic(shape, pad, mode, value):
    x = torch.randn(shape, device="cuda", dtype=torch.float32)
    y_ref = F.pad(x, pad, mode, value)
    y = nt_pad(x, pad, mode, value)
    print(y)
    print(y_ref)
    assert torch.allclose(y, y_ref)
