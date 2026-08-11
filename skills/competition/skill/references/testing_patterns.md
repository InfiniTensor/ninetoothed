# Correctness Test 编写规范

## 测试覆盖维度

每个 correctness test **必须**覆盖以下维度：

### 1. dtype

| 测试 | 说明 |
|------|------|
| float16 (fp16) | 半精度基础测试 |
| float32 (fp32) | 单精度基础测试 |
| bfloat16 (bf16) | BF16 精度测试 |

> 如果算子不支持某些 dtype，在 test plan 中明确标注。

### 2. shape

| 测试 | 说明 |
|------|------|
| 最小 shape | 如 (1,), (1, 1) |
| 典型 shape | 如 (1024,), (256, 768) |
| 大 shape | 如 (131072,) 测试稳定性 |
| 非均匀 shape | 如 (3, 7, 11) 质数维度 |

### 3. broadcast

| 测试 | 说明 |
|------|------|
| scalar broadcast | `x + 2.0` |
| vector broadcast | `x + y` where y shape=(1,) |
| matrix broadcast | `x + y` where y shape=(1, N) |
| 3D broadcast | `x + y` where y shape=(1, 1, N) |
| 全广播（不同 ndim） | `x.shape=(3,1,5), y.shape=(1,4,1)` |

### 4. stride / contiguity

| 测试 | 说明 |
|------|------|
| contiguous | 标准连续布局 |
| transposed | `.T` 转置 |
| sliced | `x[::2]` 等间距切片 |
| view | `.view()` 变形 |
| expanded | `.expand()` 扩展 |
| permuted | `.permute()` 重排 |
| non-contiguous | `torch.empty(N, M).t()` |

## ndim 参数化测试

对支持任意维度输入（ndim）的算子，使用参数化测试：

```python
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_with_different_ndim(ndim):
    """测试不同维度的 element-wise 算子"""
    if ndim == 1:
        x = torch.randn(4096, device="cuda")
    elif ndim == 2:
        x = torch.randn(64, 64, device="cuda")
    elif ndim == 3:
        x = torch.randn(16, 16, 16, device="cuda")

    kernel = make_relu(ndim=ndim)
    out = torch.empty_like(x)
    kernel(x, out, BLOCK_SIZE=1024)
    expected = torch.relu(x)
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5)
```

## 非连续张量测试

对 element-wise 算子，必须测试非连续张量场景：

```python
def test_non_contiguous():
    """测试 2D 转置（非连续）张量"""
    x = torch.randn(128, 256, device="cuda").t()  # shape (256, 128)，非连续
    out = torch.empty_like(x)
    kernel = make_relu(ndim=2)  # 使用 ndim=2 支持非连续
    kernel(x, out, BLOCK_SIZE=256)
    expected = torch.relu(x)
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5), "非连续张量测试失败"
```

## 测试断言标准

```python
# 绝对误差容限
atol = 1e-3 if dtype == torch.float16 else 1e-5
rtol = 1e-3 if dtype == torch.float16 else 1e-5
torch.allclose(output, expected, atol=atol, rtol=rtol)
```

## 测试文件组织

```
tests/
├── test_broadcast_add.py       # Task 1
├── test_softmax.py             # Task 2
├── test_non_contiguous.py      # Task 3
├── test_benchmark.py           # Task 4
└── conftest.py                 # 共享 fixture 和辅助函数
```

## conftest.py 推荐内容

```python
import torch
import pytest

@pytest.fixture
def dtype_fixture(request):
    return request.param if hasattr(request, 'param') else torch.float32

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
