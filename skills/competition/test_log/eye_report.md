# eye 算子开发报告

> program_id 对角检测。首个验证 `ntl.program_id` + `ntl.arange` 可用性的算子。

## 1. 算子信息

| 项目 | 内容 |
|------|------|
| 算子名称 | `eye` |
| 分类 | 自定义（flatten+tile + 对角计算） |
| 核心技术 | `ntl.program_id` + `ntl.arange` → 全局索引 → `ntl.where(row==col,1,0)` |
| 基线 | `torch.eye` |
| 生成文件 | `kernels/eye.py`, `torch/eye.py` |

## 2. 精度验证

| 测试 | 结果 |
|------|:--:|
| eye(4) | PASSED |
| eye(3, 5) | PASSED |

## 3. 性能评估

| 规模 | ntops (ms) | PyTorch (ms) | 比率 | 判定 |
|------|-----------|-------------|------|:--:|
| 4096×4096 | 0.2139 | 0.2167 | 0.99x | **OK** |

**六项策略**：内存访问 ✅ / 算子融合 N/A（创建算子）/ 循环展开 N/A / 同步 N/A / 精度 N/A / 计算重组 ✅（单次 where 判断已最优）

## 4. 迭代历史

| 迭代 | 问题 | 修复 |
|:--:|------|------|
| 1 | `torch.eye` keyword args `None` | `**kwargs` 处理 |
| 2 | — | 纯 NineToothed kernel 实现 |

## 5. 合计

- **迭代次数**：2
- **精度**：2/2 PASSED
- **性能**：0.99x PyTorch ✅
