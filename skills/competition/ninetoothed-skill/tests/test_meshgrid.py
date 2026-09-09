"""
自测任务 5：广播算子 — meshgrid（1D→2D 广播）

算子类型：1D→2D 广播 + 自定义 arrangement
验证内容：float32/float16 精度、不同长度输入、边界情况、非整除 block_size

输入任务说明：
    实现 meshgrid 算子：给定两个 1D tensor x(M,) 和 y(N,)，
    生成两个 2D tensor X(N,M) 和 Y(N,M)，
    其中 X[i,:] = x, Y[:,j] = y（'ij' indexing）

AI 智能体执行记录摘要：
    - 算子分类：自定义 arrangement（1D→2D 广播，不匹配任何共享 arrangement）
    - Arrangement 设计：双 block_size（M/N 方向独立），tile + expand 广播模式
    - Torch 层：unsqueeze 预处理 1D→2D，kernel 内恒等映射
    - 精度验证：float32/float16 全部通过，含非整除 block_size

Correctness 测试命令：
    cd $PROJECT_ROOT/ntops
    python -m pytest tests/test_meshgrid.py -v
"""
import pytest
import torch
import ntops


DTYPE_TOLERANCES = [
    (torch.float32, 1e-5, 1e-5),
    (torch.float16, 1e-3, 1e-3),
]


def meshgrid_cpu(x, y):
    """CPU 参考实现（'ij' indexing）"""
    nx = len(x)
    ny = len(y)
    X_ref = x.unsqueeze(0).expand(ny, nx).clone()
    Y_ref = y.unsqueeze(1).expand(ny, nx).clone()
    return X_ref, Y_ref


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_meshgrid_basic(dtype, rtol, atol):
    """基本功能：中等规模 1D→2D 广播"""
    x = torch.randn(32, dtype=dtype, device="cuda")
    y = torch.randn(16, dtype=dtype, device="cuda")

    X, Y = ntops.torch.meshgrid(x, y)
    X_ref, Y_ref = meshgrid_cpu(x, y)

    assert X.shape == X_ref.shape == (16, 32)
    assert Y.shape == Y_ref.shape == (16, 32)
    assert torch.allclose(X, X_ref, rtol=rtol, atol=atol)
    assert torch.allclose(Y, Y_ref, rtol=rtol, atol=atol)
    assert not torch.isnan(X).any()
    assert not torch.isnan(Y).any()
    assert not torch.isinf(X).any()
    assert not torch.isinf(Y).any()


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_meshgrid_large(dtype, rtol, atol):
    """大规模广播：压力测试"""
    x = torch.randn(1024, dtype=dtype, device="cuda")
    y = torch.randn(1024, dtype=dtype, device="cuda")

    X, Y = ntops.torch.meshgrid(x, y)
    X_ref, Y_ref = meshgrid_cpu(x, y)

    assert torch.allclose(X, X_ref, rtol=rtol, atol=atol)
    assert torch.allclose(Y, Y_ref, rtol=rtol, atol=atol)


def test_meshgrid_unequal_lengths():
    """不同长度的输入（M ≠ N）"""
    x = torch.randn(64, dtype=torch.float32, device="cuda")
    y = torch.randn(128, dtype=torch.float32, device="cuda")

    X, Y = ntops.torch.meshgrid(x, y)
    X_ref, Y_ref = meshgrid_cpu(x, y)

    assert X.shape == (128, 64)
    assert Y.shape == (128, 64)
    assert torch.allclose(X, X_ref, rtol=1e-5, atol=1e-5)
    assert torch.allclose(Y, Y_ref, rtol=1e-5, atol=1e-5)


def test_meshgrid_non_divisible():
    """非整除 block_size 的输入长度"""
    x = torch.randn(17, dtype=torch.float32, device="cuda")
    y = torch.randn(13, dtype=torch.float32, device="cuda")

    X, Y = ntops.torch.meshgrid(x, y)
    X_ref, Y_ref = meshgrid_cpu(x, y)

    assert X.shape == (13, 17)
    assert Y.shape == (13, 17)
    assert torch.allclose(X, X_ref, atol=1e-5)
    assert torch.allclose(Y, Y_ref, atol=1e-5)


def test_meshgrid_edge_cases():
    """边界情况：单元素输入"""
    # 单元素 x
    x = torch.tensor([3.14], device="cuda", dtype=torch.float32)
    y = torch.randn(8, dtype=torch.float32, device="cuda")
    X, Y = ntops.torch.meshgrid(x, y)
    X_ref, Y_ref = meshgrid_cpu(x, y)
    assert torch.allclose(X, X_ref, atol=1e-5)
    assert X.shape == (8, 1)

    # 单元素 y
    x = torch.randn(8, dtype=torch.float32, device="cuda")
    y = torch.tensor([2.71], device="cuda", dtype=torch.float32)
    X, Y = ntops.torch.meshgrid(x, y)
    X_ref, Y_ref = meshgrid_cpu(x, y)
    assert torch.allclose(X, X_ref, atol=1e-5)
    assert Y.shape == (1, 8)

    # 均为单元素
    x = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    y = torch.tensor([2.0], device="cuda", dtype=torch.float32)
    X, Y = ntops.torch.meshgrid(x, y)
    assert X.shape == (1, 1) and Y.shape == (1, 1)
    assert X.item() == 1.0 and Y.item() == 2.0


def test_meshgrid_exact_values():
    """精确值验证：小规模确定性测试"""
    x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    y = torch.tensor([4.0, 5.0], device="cuda")

    X, Y = ntops.torch.meshgrid(x, y)

    # X 每行是 x 的副本
    expected_X = torch.tensor([[1., 2., 3.], [1., 2., 3.]], device="cuda")
    assert torch.allclose(X, expected_X, atol=1e-5)

    # Y 每列是 y 的副本
    expected_Y = torch.tensor([[4., 4., 4.], [5., 5., 5.]], device="cuda")
    assert torch.allclose(Y, expected_Y, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
