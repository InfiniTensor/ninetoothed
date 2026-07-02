# cartesian_prod 算子开发报告

> N 路笛卡尔积。torch 层用 meshgrid + stack 组合实现，无需 kernel。

## 1. 算子信息

| 项目 | 内容 |
|------|------|
| 算子名称 | `cartesian_prod` |
| 分类 | 组合算子（torch 层，无 kernel） |
| 实现方式 | `torch.meshgrid(*tensors, indexing='ij')` + `torch.stack` + `reshape` |
| 基线 | `torch.cartesian_prod` |
| 生成文件 | `torch/cartesian_prod.py` |

## 2. 为何没有 kernel

1. 可变输入数量（premake 的 Tensor 数量固定）
2. 输出大小是各输入长度的乘积，组合爆炸
3. meshgrid + stack 已经是高效实现（view 操作 + 一次 stack）

## 3. 精度验证

| 测试 | 结果 |
|------|:--:|
| cartesian_prod([1,2],[3,4],[5,6]) vs torch | PASSED |

## 4. 性能评估

| 规模 | 输出行数 | ntops (ms) | PyTorch (ms) | 比率 | 判定 |
|------|--:|-----------|-------------|------|:--:|
| 3×10 | 1,000 | 0.0535 | 0.1049 | 0.51x | **OK（更快）** |
| 5×10 | 100,000 | 0.1222 | 0.1435 | 0.85x | **OK** |
| 7×10 | 10,000,000 | 9.7480 | 14.1139 | 0.69x | **OK** |

**性能结论**：meshgrid（view）+ stack 比 torch.cartesian_prod 的直接实现更高效，全部规模反超。

## 5. 合计

- **迭代次数**：1
- **精度**：PASSED
- **性能**：0.51-0.85x ✅（反超 PyTorch）
