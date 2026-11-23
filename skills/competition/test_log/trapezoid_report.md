# trapezoid 算子开发报告

> element_wise 分段计算 + torch.sum 规约。torch 层做切片，kernel 计算梯形项。

## 1. 算子信息

| 项目 | 内容 |
|------|------|
| 算子名称 | `trapezoid` |
| 分类 | element_wise 分段计算 + reduction（torch.sum） |
| 实现方式 | torch 层切片 `x[:-1], x[1:]` → kernel 计算 `(y[i]+y[i+1])*(x[i+1]-x[i])*0.5` → `torch.sum` |
| 基线 | `torch.trapezoid(y, x)` |
| 生成文件 | `kernels/trapezoid.py`, `torch/trapezoid.py` |

## 2. 精度验证

| 测试 | 结果 |
|------|:--:|
| trapezoid(linspace(0,π), sin) vs torch | PASSED |

## 3. 性能评估

| 规模 | ntops (ms) | PyTorch (ms) | 比率 | 判定 |
|------|-----------|-------------|------|:--:|
| 256 | 0.1198 | 0.0975 | 1.23x | **OK** |
| 4,096 | 0.1320 | 0.0975 | 1.35x | **OK** |
| 65,536 | 0.1241 | 0.0984 | 1.26x | **OK** |

**六项策略**：内存访问 ✅ / 算子融合 ✅（kernel 内完成减法+加法+乘法）/ 其他 N/A

## 4. 设计要点

- 邻接元素访问（`x[i]` 和 `x[i+1]`）通过 torch 层切片实现，kernel 只做 element_wise 运算
- 最终 sum 使用 `torch.sum`（纯 reduction，无需 kernel）

## 5. 合计

- **迭代次数**：1
- **精度**：PASSED
- **性能**：1.2-1.3x ✅
