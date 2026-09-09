# flatten 算子开发报告

> identity kernel。PyTorch 的 flatten 是零拷贝 view，kernel 拷贝有先天劣势。

## 1. 算子信息

| 项目 | 内容 |
|------|------|
| 算子名称 | `flatten` |
| 分类 | element_wise identity |
| 基线 | `torch.flatten` |
| 生成文件 | `kernels/flatten.py`, `torch/flatten.py` |

## 2. 精度验证

| 测试 | 结果 |
|------|:--:|
| flatten(4,3,2) start_dim=1 | PASSED |

## 3. 性能评估

| 规模 | ntops (ms) | PyTorch (ms) | 比率 | 判定 |
|------|-----------|-------------|------|:--:|
| 256×256 | 0.0819 | 0.0018 | 45.42x | **SLOW** |
| 1024×1024 | 0.0824 | 0.0018 | 44.98x | **SLOW** |
| 4096×4096 | 0.4370 | 0.0019 | 224.62x | **SLOW** |

**性能分析**：PyTorch 的 flatten 是 O(1) 的 view 操作（只改 metadata）。identity kernel 做 O(N) 的显存拷贝，先天劣势无法消除。对纯 view 类算子，kernel 方案在性能上不可能匹敌 PyTorch。

## 4. 合计

- **迭代次数**：1
- **精度**：PASSED
- **性能**：❌ 45-225x（view vs copy 的先天差距）
