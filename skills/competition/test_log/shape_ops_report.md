# eye, flatten, chunk, unbind, repeat 开发报告（kernel 版本）

> 全部使用 NineToothed kernel 实现。eye 用 program_id 计算对角线，
> chunk/unbind 用 identity kernel，repeat 用 broadcast kernel。

## 1. 算子实现

| 算子 | Kernel 类型 | 核心技术 |
|------|------------|----------|
| `eye` | 自定义（flatten+tile+对角计算） | `ntl.program_id` + `ntl.arange` 计算全局索引，`ntl.where` 判断对角线 |
| `flatten` | element_wise identity | identity 拷贝 |
| `chunk` | element_wise identity（每 chunk 一次） | 循环调用 identity kernel |
| `unbind` | element_wise identity（每切片一次） | 循环调用 identity kernel |
| `repeat` | broadcast（1D）/ 逐行 broadcast（多维度） | meshgrid 式 tile+expand |

## 2. 精度验证

| 测试 | 结果 |
|------|:--:|
| eye(4) | PASSED |
| eye(3, 5) | PASSED |
| flatten(4,3,2) → (4,6) | PASSED |
| chunk(6,4) → 3×(2,4) | PASSED |
| unbind(3,4) → 3×(4,) | PASSED |
| repeat 1D | PASSED |
| repeat 2D dim=0 | PASSED |

**7/7 PASSED**

## 3. 关键技术点

### 3.1 eye 的 program_id 方案

eye 是首个使用 `ntl.program_id(0)` 和 `ntl.arange` 的 kernel。通过在 application 中计算全局索引（`pid * block_size + arange`），然后用 `ntl.where(row == col, 1, 0)` 判断对角线。

验证了 `ntl.program_id` 和 `ntl.full` 在 application 中可用——这是之前 skill 中未记录的能力。

### 3.2 repeat 的逐行方案

多维度 repeat 采用"转置→flat→逐行 1D kernel→reshape"方案。每行单独调用 broadcast kernel，保证正确性但增加了 kernel launch 次数。

**架构约束**：repeat 的 2D tile arrangement 只能处理 2D tensor。对于 3D+ tensor，Grid 计算无法正确映射。这源于 NineToothed 的 tile 机制——每个 tile 维度对应一个 program_id，tile 外的维度没有对应的 program_id。

## 4. 合计

- **迭代次数**：eye(2) + flatten(1) + chunk(2) + unbind(2) + repeat(4) = 11
- **精度**：7/7 PASSED
- **新发现**：`ntl.program_id` 可用；2D tile 不支持 3D+ tensor
