# fractional_max_pool2d 算子开发报告

> torch 层委托 torch.nn.functional。复杂采样算法不适合 kernel 重实现。

## 1. 算子信息
| 项目 | 内容 |
|------|------|
| 算子名称 | `fractional_max_pool2d` |
| 分类 | torch 层委托（无 kernel） |
| 实现方式 | `F.fractional_max_pool2d(...)` |
| 基线 | `torch.nn.functional.fractional_max_pool2d` |
| 生成文件 | `torch/fractional_max_pool2d.py` |

## 2. 精度
| 测试 | 结果 |
|------|:--:|
| fractional_max_pool2d(1,3,8,8, kernel=2, ratio=0.5) | PASSED |

## 3. 性能
| 规模 | ntops (ms) | PyTorch (ms) | 比率 | 判定 |
|------|-----------|-------------|------|:--:|
| 16² | 0.0562 | 0.0607 | 0.92x | **OK** |
| 32² | 0.0571 | 0.0723 | 0.79x | **OK** |
| 64² | 0.0587 | 0.0709 | 0.83x | **OK** |

## 4. 合计
- 迭代: 1 | 精度: PASSED | 性能: 0.79-0.92x ✅
