import ninetoothed
from ninetoothed import Tensor

# 组合模式：复用 matmul 的 arrangement 和 application（im2col + matmul）
from examples.matmul.kernel import arrangement as mm_arrangement
from examples.matmul.kernel import application as mm_application


def arrangement(input, filter, output):
    # im2col: 将 input 的滑动窗口重排为矩阵
    input_arranged = input.tile((1, *filter.shape[1:]), strides=(-1, -1, 1, 1))
    input_arranged = input_arranged.squeeze(1)
    input_arranged.dtype = input_arranged.dtype.squeeze(0)
    input_arranged = input_arranged.ravel()
    input_arranged = input_arranged.flatten(end_dim=3).flatten(start_dim=1)

    # filter: (K, C, R, S) → (R*S*C, K)
    filter_arranged = filter.flatten(start_dim=1)
    filter_arranged = filter_arranged.permute((1, 0))

    # output: (N, K, P, Q) → (N*P*Q, K)
    output_arranged = output.permute((0, 2, 3, 1)).flatten(end_dim=3)

    return mm_arrangement(input_arranged, filter_arranged, output_arranged)


shape_options = {"constexpr": True}
tensors = tuple(Tensor(4, shape_options=shape_options) for _ in range(3))

kernel = ninetoothed.make(arrangement, mm_application, tensors)
