# logit 算子开发报告

> element_wise + ntl.log。标准 unary 模式，float16→float32 精度提升。

## 1. 算子信息

| 项目 | 内容 |
|------|------|
| 算子名称 | `logit` |
| 分类 | element_wise unary |
| 实现方式 | `ntl.log(ntl.cast(x, float32) / (1.0 - x))` |
| 基线 | `torch.logit` |
| 生成文件 | `kernels/logit.py`, `torch/logit.py` |

## 2. 精度验证

| 测试 | 结果 |
|------|:--:|
| logit([0.1,0.5,0.9]) vs torch.logit | PASSED |

## 3. 性能评估

| 规模 | ntops (ms) | PyTorch (ms) | 比率 | 判定 |
|------|-----------|-------------|------|:--:|
| 256×256 | 0.0656 | 0.0147 | 4.46x | SLOW |
| 1024×1024 | 0.0609 | 0.0309 | 1.97x | SLOW |
| 4096×4096 | 0.4477 | 0.4489 | 1.00x | **OK** |

## 4. 合计

- **迭代次数**：1
- **精度**：PASSED
- **性能**：1.00x @ 4096² ✅
