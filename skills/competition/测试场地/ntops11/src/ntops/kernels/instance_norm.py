import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.reduction import arrangement as reduction_arrangement


def arrangement(
    input,
    mean,
    var,
    running_mean,
    running_var,
    weight,
    bias,
    eps,
    output,
    num_normalized_elements,
    use_input_stats,
    dims,
    block_size=None,
):
    if block_size is None:
        block_size = ninetoothed.block_size()

    def _arrange_channel_tensor(tensor):
        arranged = tensor.tile((1,))
        arranged.dtype = arranged.dtype.squeeze(0)
        arranged = arranged.unsqueeze(0)
        arranged = arranged.expand((input.shape[0], -1))

        return arranged

    def _arrange_mean_or_var(tensor):
        arranged = tensor.tile((1, 1))
        arranged.dtype = arranged.dtype.squeeze((0, 1))

        return arranged

    input_arranged, output_arranged = reduction_arrangement(
        input, output, dim=dims, block_size=block_size
    )
    mean_arranged = _arrange_mean_or_var(mean)
    var_arranged = _arrange_mean_or_var(var)
    running_mean_arranged = _arrange_channel_tensor(running_mean)
    running_var_arranged = _arrange_channel_tensor(running_var)
    weight_arranged = _arrange_channel_tensor(weight)
    bias_arranged = _arrange_channel_tensor(bias)
    eps_arranged = eps
    num_normalized_elements_arranged = num_normalized_elements

    if use_input_stats:
        return (
            input_arranged,
            mean_arranged,
            var_arranged,
            weight_arranged,
            bias_arranged,
            eps_arranged,
            output_arranged,
            num_normalized_elements_arranged,
        )
    else:
        return (
            input_arranged,
            running_mean_arranged,
            running_var_arranged,
            weight_arranged,
            bias_arranged,
            eps_arranged,
            output_arranged,
        )


def application_using_input_stats(
    input,
    mean,
    var,
    weight,
    bias,
    eps,
    output,
    num_normalized_elements,
):
    _mean = ntl.zeros(input.dtype.shape, dtype=ntl.float32)

    for i in range(input.shape[0]):
        _mean += ntl.cast(input[i], ntl.float32)

    mean = ntl.sum(_mean, 0) / num_normalized_elements

    _var = ntl.zeros(input.dtype.shape, dtype=ntl.float32)

    for i in range(input.shape[0]):
        diff = ntl.cast(input[i], ntl.float32) - mean
        diff = ntl.where(input[i].offsets(-1) < input.source.shape[-1], diff, 0)
        _var += diff * diff

    var = ntl.sum(_var, 0) / num_normalized_elements

    application_with_mean_var(input, mean, var, weight, bias, eps, output)


def application_with_mean_var(
    input,
    mean,
    var,
    weight,
    bias,
    eps,
    output,
):
    std = ntl.sqrt(var + eps)

    for i in range(input.shape[0]):
        output[i] = (ntl.cast(input[i], ntl.float32) - mean) / std * weight + bias


def premake(
    ndim,
    use_input_stats,
    num_normalized_elements,
    dtype=None,
    block_size=None,
):
    dims = tuple(reversed(range(2, ndim)))

    arrangement_ = functools.partial(
        arrangement,
        use_input_stats=use_input_stats,
        dims=dims,
        block_size=block_size,
    )

    input = Tensor(ndim, other=0, dtype=dtype)
    mean, var = (Tensor(2, dtype=dtype) for _ in range(2))
    running_mean, running_var, weight, bias = (Tensor(1, dtype=dtype) for _ in range(4))
    eps = Tensor(0, dtype=ninetoothed.float64)
    output = Tensor(ndim, dtype=dtype)
    num_normalized_elements = Tensor(0, constexpr=True, value=num_normalized_elements)

    if use_input_stats:
        application = application_using_input_stats
    else:
        application = application_with_mean_var

    tensors = (
        input,
        mean,
        var,
        running_mean,
        running_var,
        weight,
        bias,
        eps,
        output,
        num_normalized_elements,
    )

    return arrangement_, application, tensors
