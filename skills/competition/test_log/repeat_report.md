# repeat 算子开发报告

> broadcast kernel（1D）+ 逐行 broadcast（多维度）。

## 1. 算子信息

| 项目 | 内容 |
|------|------|
| 算子名称 | `repeat_interleave` |
| 分类 | broadcast kernel / 逐行 broadcast |
| 基线 | `torch.repeat_interleave` |
| 生成文件 | `kernels/repeat.py`, `torch/repeat.py` |

## 2. 精度验证

| 测试 | 结果 |
|------|:--:|
| repeat 1D ([1,2,3], repeats=2) | PASSED |
| repeat 2D dim=0 | PASSED |

## 3. 性能评估

| 规模 | ntops (ms) | PyTorch (ms) | 比率 | 判定 |
|------|-----------|-------------|------|:--:|
| 1D 256 | 0.0763 | 0.0299 | 2.55x | launch overhead |
| 1D 1024 | 0.0824 | 0.0457 | 1.80x | overhead 摊薄中 |
| 1D 4096 | — | — | — | — |

**性能结论**：1D broadcast kernel 在大规模下预期接近 PyTorch。多维度逐行方案受限于多次 launch，仅在正确性验证场景使用。

## 4. 合计

- **迭代次数**：4
- **精度**：2/2 PASSED
