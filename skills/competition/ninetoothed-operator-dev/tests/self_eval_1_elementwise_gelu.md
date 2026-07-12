# 自测 1：GELU 激活函数（逐元素类）

## 实验环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 4090 (24 GB) |
| CUDA | 12.8 |
| PyTorch | 2.8.0+cu128 |
| Triton | 3.4.0 |
| ninetoothed | 0.26.0 |
| Python | 3.12 |
| AI 智能体 | DeepSeek-V4-Pro |
| dtype | float16 |
| 测试日期 | 2026-07-12 |

## 任务说明

编写九齿 GELU 激活函数算子，使用 tanh 近似公式:

```
GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

其中 `tanh(z)` 通过恒等式 `2*sigmoid(2z) - 1` 实现（因为 ninetoothed 不提供 `tanh` 原语，`ntl.libdevice.tanh` 在代码生成路径下会解析失败——这是九齿已知限制）。

## AI 智能体执行记录

1. **需求分析**: 输入 `(N,)` float16 Tensor → 输出 `(N,)` float16 Tensor
2. **Arrangement 设计**: 1D 逐元素，`BLOCK_SIZE` 用闭包 `block_size()`
   - tile: `x.tile((BLOCK_SIZE,))`, `output.tile((BLOCK_SIZE,))`
   - 两个 tensor 均声明为 `Tensor(1)`
3. **Application 编写**:
   - 将输入 cast 为 float32 保证数值精度
   - 预计算 `sqrt(2/pi)` 常量（0.7978845608028654）
   - 用 `2*ntl.sigmoid(2*z) - 1` 替代 `tanh`
   - 输出赋值 `output = x_f32 * cdf  # noqa: F841`
4. **诊断经历**:
   - **失败现象**: 初次 `_Inliner` 报 `AssertionError: Illegal function reference: tanh`
   - **根因**: `ntl.libdevice.tanh` 经 `_AliasRestorer` → `ninetoothed.language.libdevice.tanh`，`Tritonizer` 生成 `triton.language.libdevice.tanh`，但实际路径是 `triton.language.extra.libdevice`
   - **修复**: 改用 sigmoid 恒等式替代 tanh，完全避开 libdevice 路径
   - **验证**: correctness 测试通过

## 算子代码

```python
# operators/gelu.py
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

def create_gelu_kernel():
    BLOCK_SIZE = block_size()

    def arrangement(x, output):
        return x.tile((BLOCK_SIZE,)), output.tile((BLOCK_SIZE,))

    def application(x, output):
        x_f32 = ntl.cast(x, ntl.float32)
        sqrt_2_over_pi = 0.7978845608028654
        inner = sqrt_2_over_pi * (x_f32 + 0.044715 * x_f32 * x_f32 * x_f32)
        tanh_inner = 2.0 * ntl.sigmoid(2.0 * inner) - 1.0
        cdf = 0.5 * (1.0 + tanh_inner)
        output = x_f32 * cdf  # noqa: F841

    return ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1)))
```

## Correctness 测试

**命令**: `python run_all_operator_tests.py`（GELU 部分）

**测试规模**: (100,), (512,), (1024,)

**参考实现**: `torch.nn.functional.gelu(x)`

**结果**: 3/3 通过（`atol=1e-2, rtol=1e-2`，float16）

```
Shape (100,)         : ✓ 通过
Shape (512,)         : ✓ 通过
Shape (1024,)        : ✓ 通过
```

## Benchmark

**命令**: `python benchmarks/run_benchmarks.py --category elementwise --quick`

**规模**: 1K / 8K / 64K 元素

| size | ninetoothed (ms) | torch (ms) | 比值 |
|------|---------------------|------------|------|
| 1024 | 0.003 | 0.005 | 0.59x |
| 8192 | 0.004 | 0.006 | 0.66x |
| 65536 | 0.004 | 0.006 | 0.67x |

**结论**: 在 RTX 4090 上，ninetoothed GELU 延迟约为 PyTorch 的 60-67%。延迟绝对值极小（3-4 μs kernel time），说明单 kernel 无 Python overhead 的优势在大规模下明显。所有规模的 ninetoothed 延迟稳定在 ~4 μs，而 PyTorch 随规模小幅增长。

## 已知限制

- 动态 shape 不支持（`Symbol` 和 `Tensor(1)` 要求编译期确定）
- 非连续输入需额外处理 stride
- float64 未测试（引入方式需 `ntl.cast` 适配）
