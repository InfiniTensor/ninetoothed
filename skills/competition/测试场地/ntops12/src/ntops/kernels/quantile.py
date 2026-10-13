import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(input, q, output, dim, block_size=None):
    def _arrange_input_or_output(tensor, dim=0):
        ndim = tensor.ndim

        if dim < 0:
            dim += ndim

        non_target_dims = tuple(i for i in range(ndim) if i != dim)

        arranged = tensor.permute(non_target_dims + (dim,))

        block_shape = tuple(1 for _ in non_target_dims) + (-1,)
        non_target_dim_indices = tuple(range(len(non_target_dims)))

        arranged = arranged.tile(block_shape)
        arranged.dtype = arranged.dtype.squeeze(non_target_dim_indices)

        return arranged

    if dim is None:
        input_arranged = input.flatten()
        input_arranged = input_arranged.tile((-1,))
    else:
        input_arranged = _arrange_input_or_output(input, dim=dim)

    output_arranged = _arrange_input_or_output(output)

    q_arranged = q.tile((-1,))
    q_arranged = q_arranged.squeeze(0)

    for _ in range(output_arranged.ndim):
        q_arranged = q_arranged.unsqueeze(0)

    q_arranged = q_arranged.expand(output_arranged.shape)

    return input_arranged, q_arranged, output_arranged


def linear_application(input, q, output):
    dim_size = input.shape[0]
    pos = ntl.cast(q * (dim_size - 1), ntl.float32)
    i = ntl.cast(ntl.floor(pos), ntl.int32)
    j = ntl.cast(ntl.ceil(pos), ntl.int32)
    frac = pos - i

    sorted = ntl.sort(input)
    lower_value = ntl.gather(sorted, i, 0)
    higher_value = ntl.gather(sorted, j, 0)

    output = lower_value + frac * (higher_value - lower_value)  # noqa: F841


def lower_application(input, q, output):
    dim_size = input.shape[0]
    pos = ntl.cast(q * (dim_size - 1), ntl.float32)
    i = ntl.cast(ntl.floor(pos), ntl.int32)

    sorted = ntl.sort(input)
    lower_value = ntl.gather(sorted, i, 0)

    output = lower_value  # noqa: F841


def higher_application(input, q, output):
    dim_size = input.shape[0]
    pos = ntl.cast(q * (dim_size - 1), ntl.float32)
    j = ntl.cast(ntl.ceil(pos), ntl.int32)

    sorted = ntl.sort(input)
    higher_value = ntl.gather(sorted, j, 0)

    output = higher_value  # noqa: F841


def nearest_application(input, q, output):
    dim_size = input.shape[0]
    pos = ntl.cast(q * (dim_size - 1), ntl.float32)

    # Rounding mode for float to int conversion is always towards zero,
    # we have to manually implement `rtne` (round to nearest, ties to even).
    i = ntl.cast(ntl.floor(pos), ntl.int32)
    frac = ntl.cast(pos - i, ntl.float32)
    i = ntl.where(frac > 0.5, ntl.minimum(i + 1, dim_size - 1), i)
    i = ntl.where((frac == 0.5) & (i % 2 == 1), ntl.minimum(i + 1, dim_size - 1), i)

    sorted = ntl.sort(input)
    output = ntl.gather(sorted, i, 0)  # noqa: F841


def midpoint_application(input, q, output):
    dim_size = input.shape[0]

    pos = ntl.cast(q * (dim_size - 1), ntl.float32)
    i = ntl.cast(ntl.floor(pos), ntl.int32)
    j = ntl.cast(ntl.ceil(pos), ntl.int32)

    sorted = ntl.sort(input)
    lower_value = ntl.gather(sorted, i, 0)
    higher_value = ntl.gather(sorted, j, 0)

    output = (higher_value + lower_value) / 2  # noqa: F841


def premake(in_ndim, out_ndim, dim, interpolation, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, dim=dim, block_size=block_size)

    input_shape_options = (
        {"constexpr": True}
        if dim is None
        else (None if i != dim else {"constexpr": True} for i in range(in_ndim))
    )
    q_shape_options = {"constexpr": True}
    output_shape_options = ({"constexpr": True},) + (None,) * (out_ndim - 1)

    tensors = (
        Tensor(
            in_ndim, dtype=dtype, other=float("inf"), shape_options=input_shape_options
        ),
        Tensor(1, dtype=dtype, shape_options=q_shape_options),
        Tensor(out_ndim, dtype=dtype, shape_options=output_shape_options),
    )

    if interpolation == "linear":
        application = linear_application
    elif interpolation == "lower":
        application = lower_application
    elif interpolation == "higher":
        application = higher_application
    elif interpolation == "nearest":
        application = nearest_application
    elif interpolation == "midpoint":
        application = midpoint_application
    else:
        raise ValueError(f"Unsupported interpolation method: {interpolation}")

    return arrangement_, application, tensors
