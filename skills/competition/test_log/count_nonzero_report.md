# count_nonzero 算子开发报告

> element_wise mask + torch.sum。和 trapezoid/kl_div 同模式。

## 1. 算子信息

| 项目 | 内容 |
|------|------|
| 算子名称 | `count_nonzero` |
| 分类 | element_wise + reduction（torch.sum） |
| 实现方式 | kernel: `ntl.where(ntl.abs(x) > eps, 1, 0)` + `torch.sum` |
| 基线 | `torch.abs(x) > eps).sum()` |
| 生成文件 | `kernels/count_nonzero.py`, `torch/count_nonzero.py` |

## 2. 精度验证

| 测试 | 结果 |
|------|:--:|
| count([1,0,-2,0,3,1e-13]) = 3 | PASSED |

## 3. 性能评估

| 规模 | ntops (ms) | PyTorch (ms) | 比率 | 判定 |
|------|-----------|-------------|------|:--:|
| 256 | 0.0926 | 0.0854 | 1.08x | **OK** |
| 4,096 | 0.0985 | 0.0818 | 1.21x | **OK** |
| 65,536 | 0.0938 | 0.0920 | 1.02x | **OK** |
| 1,048,576 | 0.1065 | 0.1148 | 0.93x | **OK** |

## 4. 合计

- **迭代次数**：1
- **精度**：PASSED
- **性能**：0.93-1.21x ✅
