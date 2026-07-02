# unbind 算子开发报告

> 循环 identity kernel。每切片一次 kernel launch，大维度下 launch 数量爆炸。

## 1. 算子信息

| 项目 | 内容 |
|------|------|
| 算子名称 | `unbind` |
| 分类 | element_wise identity（每切片一次 launch） |
| 基线 | `torch.unbind` |
| 生成文件 | `kernels/unbind.py`, `torch/unbind.py` |

## 2. 精度验证

| 测试 | 结果 |
|------|:--:|
| unbind(3,4) dim=0 | PASSED |

## 3. 性能评估

| 规模 | 切片数 | ntops (ms) | PyTorch (ms) | 比率 | 判定 |
|------|:--:|-----------|-------------|------|:--:|
| 1024×4096 | 1024 | 73.27 | 0.72 | 101.56x | **SLOW** |

**性能分析**：1024 次 kernel launch 导致灾难性开销。PyTorch 的 unbind 是 O(1) view。此算子是"全 kernel"策略在 view 类操作上的极端反例。

## 4. 合计

- **迭代次数**：1
- **精度**：PASSED
- **性能**：❌ 101x（1024 次 launch 的灾难）
