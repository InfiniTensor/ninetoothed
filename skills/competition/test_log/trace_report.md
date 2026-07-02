# trace 算子开发报告

> element_wise + program_id 对角检测。eye 的变体——提取而非设置。

## 1. 算子信息
| 项目 | 内容 |
|------|------|
| 算子名称 | `trace` |
| 分类 | element_wise + program_id + torch.sum |
| 实现方式 | kernel: `ntl.where(row==col, input, 0)` → torch.sum |
| 基线 | `torch.trace` |
| 生成文件 | `kernels/trace.py`, `torch/trace.py` |

## 2. 精度
| 测试 | 结果 |
|------|:--:|
| trace(3×3 arange) = 12 | PASSED |

## 3. 性能
| 规模 | ntops (ms) | PyTorch (ms) | 比率 | 判定 |
|------|-----------|-------------|------|:--:|
| 256² | 0.1175 | 0.0249 | 4.72x | SLOW |
| 1024² | 0.1208 | 0.0236 | 5.13x | SLOW |
| 4096² | 0.8776 | 0.0319 | 27.52x | **GAP** |

**性能分析**: kernel 做了 O(N²) 拷贝（填充 non-diag 为 0），而 torch.trace 只读对角元素不拷贝。

## 4. 合计
- 迭代: 1 | 精度: PASSED | 性能: 27x ❌ (O(N²) copy vs O(N) read)
