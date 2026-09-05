import ninetoothed
from ninetoothed import Symbol


def arrangement(
    input,
    output,
    kernel_size_h=None,
    kernel_size_w=None,
    stride_h=None,
    stride_w=None,
    padding_h=None,
    padding_w=None,
    dilation_h=None,
    dilation_w=None,
    ceil_mode=None,
    block_size=None,
):
    if kernel_size_h is None:
        kernel_size_h = Symbol("kernel_size_h", constexpr=True, upper_bound=16)

    if kernel_size_w is None:
        kernel_size_w = Symbol("kernel_size_w", constexpr=True, upper_bound=16)

    if stride_h is None:
        stride_h = Symbol("stride_h", constexpr=True)

    if stride_w is None:
        stride_w = Symbol("stride_w", constexpr=True)

    if padding_h is None:
        padding_h = Symbol("padding_h", constexpr=True)

    if padding_w is None:
        padding_w = Symbol("padding_w", constexpr=True)

    if dilation_h is None:
        dilation_h = Symbol("dilation_h", constexpr=True)

    if dilation_w is None:
        dilation_w = Symbol("dilation_w", constexpr=True)

    if ceil_mode is None:
        ceil_mode = False

    if block_size is None:
        block_size = ninetoothed.block_size()

    input_arranged = input.pad(
        ((0, 0), (0, 0), (padding_h, padding_h), (padding_w, padding_w))
    )
    input_arranged = input_arranged.tile(
        (1, 1, kernel_size_h, kernel_size_w),
        strides=(-1, -1, stride_h, stride_w),
        dilation=(1, 1, dilation_h, dilation_w),
        floor_mode=not ceil_mode,
    )
    input_arranged = input_arranged.ravel()
    input_arranged = input_arranged.flatten(end_dim=4).flatten(start_dim=1)
    input_arranged = input_arranged.tile((block_size, -1))

    output_arranged = output.tile((1, 1, 1, 1))
    output_arranged = output_arranged.ravel()
    output_arranged = output_arranged.flatten(end_dim=4).flatten(start_dim=1)
    output_arranged = output_arranged.tile((block_size, -1))
    output_arranged.dtype = output_arranged.dtype.squeeze(1)

    return input_arranged, output_arranged
