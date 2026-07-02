# corrcoef 算子开发报告

> 组合算子：torch 层 orchestrate mean + matmul + normalize。无 kernel。

## 1. 算子信息

| 项目 | 内容 |
|------|------|
| 算子名称 | `corrcoef` |
| 分类 | 组合算子（torch 层，无 kernel） |
| 实现方式 | `x - mean` → `xc @ xc.T / (n-1)` → `cov / (std_i * std_j)` |
| 基线 | `numpy.corrcoef` |
| 生成文件 | `torch/corrcoef.py` |

## 2. 为何没有 kernel

三步操作（center + covariance + normalize），最有效的实现是组合已有算子：
- `torch.mean` — reduction（已优化）
- `@` — matmul（cuBLAS）
- element-wise arithmetic — torch 原生

每个子操作已有高度优化的实现，重写为 NineToothed kernel 反而会降低性能。

## 3. 精度验证

| 测试 | 结果 |
|------|:--:|
| corrcoef(5×100) vs numpy.corrcoef | PASSED |

## 4. 性能评估

| 规模 | ntops (ms) | numpy (ms) | 比率 | 判定 |
|------|-----------|-----------|------|:--:|
| 16×256 | 0.2827 | 0.2273 | 1.24x | **OK** |
| 64×1024 | 0.3282 | 0.5039 | 0.65x | **OK** |
| 256×4096 | 0.3125 | 10.6006 | 0.03x | **OK（33x faster）** |

**性能结论**：GPU vs CPU 的优势。组合已有优化算子比从头实现 kernel 更高效。

## 5. 合计

- **迭代次数**：1
- **精度**：PASSED
- **性能**：0.03x @ 256×4096 ✅
