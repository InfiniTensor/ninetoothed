"""
RMS Normalization 算子 —— 融合归约 + 逐元素 + 广播。

演示如何在单个 kernel 中融合多个操作：squaring → sum → divide → rsqrt → rescale × w。
参考 ninetoothed-examples/ops/ninetoothed/kernels/fused_rms_norm.py。

功能: y = x * rsqrt(mean(x^2) + eps) * w
      (沿最后一维归约，逐元素缩放)
"""
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(x, w, eps, y, BLOCK_SIZE=BLOCK_SIZE):
    """
    x, w, y: 2D 张量 (M, N)，tile 为 (1, BLOCK_SIZE)
    eps: 标量 Tensor(0)，直接透传
    """
    def arrange(tensor):
        return tensor.tile((1, BLOCK_SIZE))

    return arrange(x), arrange(w), eps, arrange(y)


def application(x, w, eps, y):
    # fp16 → fp32 确保精度
    x_fp32 = ntl.cast(x, ntl.float32)

    # 融合: square → sum → mean → rsqrt → rescale × w
    # ntl.sum 对 tile 内所有元素归约（axis=None）
    y = x_fp32 * ntl.rsqrt(
        ntl.sum(x_fp32 * x_fp32) / x.shape[-1] + eps
    ) * w  # noqa: F841


tensors = (Tensor(2), Tensor(2), Tensor(0), Tensor(2))


def create_rms_norm_kernel():
    """RMS Norm kernel 工厂函数

    调用示例:
        import torch
        x = torch.randn(1024, 512, dtype=torch.float16, device='cuda')
        w = torch.randn(512, dtype=torch.float16, device='cuda')  # 1D weight
        eps = 1e-5
        y = torch.empty_like(x)
        # kernel 要求 w 为 2D，需先 expand_as(x)
        kernel(x, w.expand_as(x), eps, y, BLOCK_SIZE=x.shape[-1])
        # 等价于 y = x * rsqrt(mean(x^2) + eps) * w
    """
    return ninetoothed.make(arrangement, application, tensors)
