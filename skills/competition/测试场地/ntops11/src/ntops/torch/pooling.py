import math


def _calculate_output_size(
    input_size, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False
):
    if stride is None:
        stride = kernel_size

    int_ = math.ceil if ceil_mode else math.floor

    result = int_(
        (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    )

    if ceil_mode and (result - 1) * stride >= input_size + padding:
        result -= 1

    return result
