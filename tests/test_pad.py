import torch
import torch.nn.functional as F
import pytest
import ninetoothed as nt
from typing import Dict, Tuple
import xxhash


# Only Support non-negative pad_config
def fake_pad(x, pad_config):
    ndim = x.ndim
    old_shape = list(x.shape)
    new_shape = list(x.shape)
    slice_l = []
    slice_r = []
    for i in range(ndim):
        left = pad_config[2 * (ndim - 1 - i)]
        right = pad_config[2 * (ndim - 1 - i) + 1]
        new_shape[i] += left + right

        start_s = max(0, -left)
        end_s = min(old_shape[i], old_shape[i] + right)

        start_d = max(0, left)
        end_d = new_shape[i] - max(0, right)

        slice_l.append(slice(start_d, end_d))
        slice_r.append(slice(start_s, end_s))
    slice_l = tuple(slice_l)
    slice_r = tuple(slice_r)
    y = torch.zeros(new_shape, device=x.device, dtype=x.dtype)

    y = torch.zeros(new_shape, device=x.device, dtype=x.dtype)
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


def pad_kernel(x, pad_config):
    # analyze pad_config
    ndim = x.ndim
    old_shape = list(x.shape)
    new_shape = list(x.shape)
    slice_l = []
    slice_r = []
    for i in range(ndim):
        left = pad_config[2 * (ndim - 1 - i)]
        right = pad_config[2 * (ndim - 1 - i) + 1]
        new_shape[i] += left + right

        start_s = max(0, -left)
        end_s = min(old_shape[i], old_shape[i] + right)

        start_d = max(0, left)
        end_d = new_shape[i] - max(0, right)

        slice_l.append(slice(start_d, end_d))
        slice_r.append(slice(start_s, end_s))
    slice_l = tuple(slice_l)
    slice_r = tuple(slice_r)
    y = torch.zeros(new_shape, device=x.device, dtype=x.dtype)

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


def nt_pad(x, pad_config):
    return pad_kernel(x, pad_config)
    # return fake_pad(x, pad_config)


@pytest.mark.parametrize(
    "shape, pad_config",
    [
        ((2, 3), (1, 2, 0, 1)),
        ((2, 3), (1, 1, 1, 2)),
        ((2, 3, 4), (1, 3, 1, 0, 0, 0)),
        ((2, 3), (-1, -1, 0, 0)),
    ],
)
def test_pad_basic(shape, pad_config):
    x = torch.randn(shape, device="cuda", dtype=torch.float32)
    y_ref = F.pad(x, pad_config)
    y = nt_pad(x, pad_config)
    print(y)
    print(y_ref)
    assert torch.allclose(y, y_ref)
