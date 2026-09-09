# rad2deg 算子开发报告

> 按照 `ninetoothed-skill` 六阶段工作流完成。AI 智能体从 CPU 参考实现出发，完成分析、生成、编译、验证、优化的完整闭环。

## 1. 算子信息

| 项目 | 内容 |
|------|------|
| 算子名称 | `rad2deg` |
| 分类 | Element-wise Unary（模式 1） |
| CPU 参考 | `rad * 180.0 / PI` |
| 共享 arrangement | `ntops.kernels.element_wise` |
| 标量参数 | constexpr: `180.0 / math.pi` |
| 生成文件 | `kernels/rad2deg.py`, `torch/rad2deg.py` |
| 文件夹联 | 更新 `kernels/__init__.py` + `torch/__init__.py`（import + `__all__`） |

## 2. 精度验证

**基线**：`torch.rad2deg`

| 测试 | dtype | 规模 | 结果 |
|------|-------|------|:--:|
| basic | float32 | 1024 | PASSED |
| basic | float16 | 1024 | PASSED |
| large | float32 | 4096×1024 | PASSED |
| large | float16 | 4096×1024 | PASSED |
| 3d | float32 | 8×64×128 | PASSED |
| 3d | float16 | 8×64×128 | PASSED |
| zeros | float32 | 256 | PASSED |
| negatives | float32 | 5 | PASSED |
| transposed | float32 | 512×512 | PASSED |

**四项必检**：allclose ✅ / NaN ✅ / Inf ✅ / 精确值 (0.0, ±π) ✅

## 3. 性能评估

**Baseline**: `torch.rad2deg`

| 规模 | dtype | ntops (ms) | PyTorch (ms) | 比率 | 判定 |
|------|-------|-----------|-------------|------|:--:|
| 256×256 | float32 | 0.0662 | 0.0184 | 3.59x | launch overhead |
| 1024×1024 | float32 | 0.0660 | 0.0299 | 2.21x | overhead 摊薄中 |
| 4096×4096 | float32 | 0.4567 | 0.4414 | 1.03x | **OK** |
| 4096×4096 | float16 | 0.2224 | 1.1409 | 0.19x | **ntops 更快** |

**六项策略评估**：

| 优先级 | 策略 | 评估 |
|:--:|------|------|
| 1 | 内存访问模式优化 | ✅ coalesced access（element-wise tile） |
| 2 | 算子融合 | 不适用 — 单算子无融合空间 |
| 3 | 循环展开 | 不适用 — application 无循环 |
| 4 | 减少同步开销 | 不适用 — 单 kernel launch |
| 5 | 精度策略调整 | ✅ constexpr multiplier 自动适配 dtype |
| 6 | 计算重组 | 不适用 — `input * constant` 已是最简 |

**性能结论**：
- 大规模下与 PyTorch 持平（1.03x），符合 ≥0.85x 目标
- float16 下 ntops 显著快于 PyTorch（0.19x，即 ntops 快 5.3x），原因是 ntops 原生支持 float16 且 constexpr 自动适配类型
- 小规模下开销来自 kernel launch latency，属于 NineToothed/Triton 的固有特征，非算子设计问题

## 4. 边界情况

- ✅ 零值输入
- ✅ 负值输入（-π, -π/2）
- ✅ 非连续输入（转置张量）
- ✅ float16 精度
- ✅ 3D 输入

## 5. 迭代历史

| 迭代 | 阶段 | 结果 | 备注 |
|:--:|------|:--:|------|
| 1 | 生成 + 编译 | PASSED | 首次编译即通过，constexpr 正确嵌入 |

## 6. Generated Source 检查

生成源码路径：`~/.ninetoothed/{hash}.py`

检查要点：
1. **tile 映射** ✅ — `tl.load` / `tl.store` 使用 block_size 偏移，element-wise 简单映射
2. **数据类型** ✅ — constexpr multiplier 被内联为具体数值，无 float64 残留
3. **内存访问** ✅ — 单次 load + 单次 store，无冗余
4. **constexpr** ✅ — `180.0 / math.pi` 被编译时内联
5. **block_size** ✅ — auto-tune 选择合理值

## 7. 不支持场景

- 动态 shape：无影响（element-wise 适配任意 ndim）
- 特定 dtype 不兼容：float64 可支持但未测试

## 8. 合计

- **总迭代次数**：1（首次通过）
- **静态验证清单**：15 项全部通过
- **精度验证**：9/9 PASSED
- **性能目标**：≥0.85x PyTorch @ 大规模 ✅
