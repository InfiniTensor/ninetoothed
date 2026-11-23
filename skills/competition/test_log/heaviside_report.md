# heaviside 算子开发报告

> element_wise + ntl.where 三段条件。nan_to_num 同模式。

## 1. 算子信息
| 项目 | 内容 |
|------|------|
| 算子名称 | `heaviside` |
| 分类 | element_wise binary |
| 实现方式 | `ntl.where(x>0,1,values)` → `ntl.where(x<0,0,result)` |
| 基线 | `torch.heaviside` |
| 生成文件 | `kernels/heaviside.py`, `torch/heaviside.py` |

## 2. 精度
| 测试 | 结果 |
|------|:--:|
| heaviside([-1,0,1,2],0.5) | PASSED |

## 3. 性能
| 规模 | ntops (ms) | PyTorch (ms) | 比率 | 判定 |
|------|-----------|-------------|------|:--:|
| 256² | 0.0917 | 0.0159 | 5.78x | launch overhead |
| 1024² | 0.1008 | 0.0572 | 1.76x | OK |
| 4096² | 0.9748 | 0.8522 | 1.14x | **OK** |

## 4. 合计
- 迭代: 1 | 精度: PASSED | 性能: 1.14x ✅
