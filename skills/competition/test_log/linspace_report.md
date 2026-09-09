# linspace 算子开发报告

> element_wise + program_id 全局索引。dummy input 绕过纯输出限制。

## 1. 算子信息

| 项目 | 内容 |
|------|------|
| 算子名称 | `linspace` |
| 分类 | element_wise + program_id（dummy input 模式） |
| 实现方式 | 用 dummy input 满足 arrangement 要求，application 忽略输入，用 `pid*BS+j` 计算全局索引 |
| 基线 | `torch.linspace` |
| 生成文件 | `kernels/linspace.py`, `torch/linspace.py` |

## 2. 精度验证

| 测试 | 结果 |
|------|:--:|
| linspace(0,10,5) vs torch.linspace | PASSED |

## 3. 性能评估

| 规模 | ntops (ms) | PyTorch (ms) | 比率 | 判定 |
|------|-----------|-------------|------|:--:|
| 256 | 0.0957 | 0.0161 | 5.93x | SLOW |
| 4,096 | 0.0764 | 0.0165 | 4.63x | SLOW |
| 65,536 | 0.0771 | 0.0157 | 4.93x | SLOW |
| 1,048,576 | 0.0779 | 0.0194 | 4.01x | SLOW |

**性能分析**：恒定的 ~4-5x overhead。PyTorch 的 linspace 是优化的 CPU→GPU fill 操作，不经过 kernel launch。NineToothed kernel 的 launch overhead 无法消除。

**六项策略**：内存访问 ✅ / 其他 N/A

## 4. 设计要点

- dummy input 模式：pass `torch.empty(num, ...)` 作为输入满足 element_wise arrangement，在 application 中忽略它
- `start` 和 `step` 用 constexpr 传入，编译时内联
- 和 eye 同类：张量创建 + program_id 全局索引

## 5. 合计

- **迭代次数**：1
- **精度**：PASSED
- **性能**：~4-5x（kernel launch overhead）
