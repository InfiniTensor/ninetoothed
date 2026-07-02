# logspace 算子开发报告

> linspace + libdevice.pow。与 linspace 共享 dummy input + program_id 模式。

## 1. 算子信息

| 项目 | 内容 |
|------|------|
| 算子名称 | `logspace` |
| 分类 | element_wise + libdevice.pow（dummy input 模式） |
| 实现方式 | linspace 计算指数 → `libdevice.pow(10.0, exp_val)` |
| 基线 | `torch.logspace` |
| 生成文件 | `kernels/logspace.py`, `torch/logspace.py` |

## 2. 精度验证

| 测试 | 结果 |
|------|:--:|
| logspace(0,2,5) vs torch.logspace | PASSED |

## 3. 性能评估

| 规模 | ntops (ms) | PyTorch (ms) | 比率 | 判定 |
|------|-----------|-------------|------|:--:|
| 256 | 0.0750 | 0.0166 | 4.53x | SLOW |
| 4,096 | 0.0765 | 0.0173 | 4.43x | SLOW |
| 1,048,576 | 0.0804 | 0.0201 | 3.99x | SLOW |

**性能分析**：与 linspace 同模式，~4x kernel launch overhead。libdevice.pow 在 float32 下精度略有损失（末位 ~0.00001 误差）。

## 4. 合计

- **迭代次数**：1
- **精度**：PASSED
- **性能**：~4x（kernel launch overhead）
