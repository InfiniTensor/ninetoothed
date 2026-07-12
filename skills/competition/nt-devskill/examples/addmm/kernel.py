import ninetoothed
from ninetoothed import Tensor

# 组合模式：复用 matmul 的 arrangement 和 application
# 注意：必须用 from...import 而非 import...as，
# 因为 __init__.py 中的 kernel 变量会遮蔽 kernel.py 模块
from examples.matmul.kernel import arrangement as mm_arrangement
from examples.matmul.kernel import application as mm_application


def arrangement(input, mat1, mat2, beta, alpha, output):
    # 第一次调用：获取 bias 的 arranged 视图
    _, _, input_arranged = mm_arrangement(mat1, mat2, input)

    # 第二次调用：获取 matmul 结果的 arranged 视图
    mat1_arranged, mat2_arranged, output_arranged = mm_arrangement(mat1, mat2, output)

    return input_arranged, mat1_arranged, mat2_arranged, beta, alpha, output_arranged


def application(input, mat1, mat2, beta, alpha, output):
    # 先执行 matmul：output = mat1 @ mat2
    mm_application(mat1, mat2, output)
    # 再融合 bias：output = beta * input + alpha * output
    output = beta * input + alpha * output


tensors = (Tensor(2), Tensor(2), Tensor(2), Tensor(0), Tensor(0), Tensor(2))

kernel = ninetoothed.make(arrangement, application, tensors)
