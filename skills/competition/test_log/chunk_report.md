# chunk 算子开发报告

> 循环 identity kernel。PyTorch 的 chunk 是零拷贝 view，kernel 拷贝 + 多次 launch 开销大。

## 1. 算子信息

| 项目 | 内容 |
|------|------|
| 算子名称 | `chunk` |
| 分类 | element_wise identity（每 chunk 一次 launch） |
| 基线 | `torch.chunk` |
| 生成文件 | `kernels/chunk.py`, `torch/chunk.py` |

## 2. 精度验证

| 测试 | 结果 |
|------|:--:|
| chunk(6,4) chunks=3 dim=0 | PASSED |

## 3. 性能评估

| 规模 | 分块 | ntops (ms) | PyTorch (ms) | 比率 | 判定 |
|------|:--:|-----------|-------------|------|:--:|
| 4096×1024 | 4 | 0.3153 | 0.0055 | 57.33x | **SLOW** |

**性能分析**：PyTorch 的 chunk 是 O(1) view。kernel 方案需 chunks 次 launch + O(N) 拷贝。先天劣势。

## 4. 合计

- **迭代次数**：1
- **精度**：PASSED
- **性能**：❌ 57x（view vs copy + 多次 launch）
