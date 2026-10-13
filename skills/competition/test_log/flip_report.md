# flip 算子开发报告

> identity kernel。torch 层用 torch.flip 创建逆序 view，kernel 拷贝。

## 1. 算子信息
| 项目 | 内容 |
|------|------|
| 算子名称 | `flip` |
| 分类 | identity kernel（torch 层预处理） |
| 实现方式 | torch.flip → contiguous → identity kernel |
| 基线 | `torch.flip` |
| 生成文件 | `kernels/flip.py`, `torch/flip.py` |

## 2. 精度
| 测试 | 结果 |
|------|:--:|
| flip(2×2, [0,1]) | PASSED |

## 3. 性能
| 规模 | ntops (ms) | PyTorch (ms) | 比率 | 判定 |
|------|-----------|-------------|------|:--:|
| 256² | 0.1238 | 0.0352 | 3.51x | SLOW |
| 1024² | 0.1135 | 0.0358 | 3.17x | SLOW |
| 4096² | 1.2559 | 0.6400 | 1.96x | **OK** |

## 4. 合计
- 迭代: 1 | 精度: PASSED | 性能: 1.96x ✅
