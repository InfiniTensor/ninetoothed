# deg2rad 算子开发报告

> 按照 `ninetoothed-skill` 六阶段工作流完成。E2E 测试：验证 skill 对新算子的完整开发闭环。

## 1. 算子信息

| 项目 | 内容 |
|------|------|
| 算子名称 | `deg2rad` |
| 分类 | Element-wise（模式 1） |
| CPU 参考 | `x * (math.pi / 180.0)` |
| 共享 arrangement | `ntops.kernels.element_wise` |
| 关键 DSL 操作 | `output = input * constexpr_multiplier` |
| 基线 | `x * (math.pi / 180.0)`（PyTorch 直写） |
| 生成文件 | `kernels/deg2rad.py`, `torch/deg2rad.py` |

## 2. 精度验证

**基线**：`x * (math.pi / 180.0)`

| 测试 | dtype | 规模 | 结果 |
|------|-------|------|:--:|
| basic | float32 | 1024 | PASSED |
| basic | float16 | 1024 | PASSED |
| large | float32 | 4096×4096 | PASSED |
| large | float16 | 4096×4096 | PASSED |
| exact values (0°, 90°, 180°, 360°) | float32 | 4 | PASSED |
| edge cases (zeros, negatives) | float32 | 256/Neg | PASSED |
| 3D | float32 | 8×64×128 | PASSED |

**四项必检**：allclose ✅ / NaN ✅ / Inf ✅ / 精确值 ✅

## 3. 性能评估

**Baseline**: `x * (math.pi / 180.0)`

| 规模 | dtype | ntops (ms) | PyTorch (ms) | 比率 | 判定 |
|------|-------|-----------|-------------|------|:--:|
| 256×256 | float32 | 0.069 | 0.021 | 3.34x | launch overhead |
| 1024×1024 | float32 | 0.066 | 0.042 | 1.58x | overhead 摊薄中 |
| 4096×4096 | float32 | 0.440 | 0.450 | **0.98x** | **OK（略快于 PyTorch）** |
| 1024×1024 | float16 | 0.069 | 0.029 | 2.42x | overhead 摊薄中 |
| 4096×4096 | float16 | 0.237 | 0.235 | **1.01x** | **OK（持平）** |

**六项策略评估**：

| 优先级 | 策略 | 评估 |
|:--:|------|------|
| 1 | 内存访问模式优化 | ✅ coalesced access（element-wise tile） |
| 2 | 算子融合 | 不适用 — 单算子无融合空间 |
| 3 | 循环展开 | 不适用 — application 无循环 |
| 4 | 减少同步开销 | 不适用 — 单 kernel launch |
| 5 | 精度策略调整 | ✅ float16 下 constexpr 乘法精度正确 |
| 6 | 计算重组 | ✅ `input * constexpr` 已是最优表达 |

**性能结论**：大规模下与 PyTorch 持平甚至略快（0.98x float32, 1.01x float16），符合 ≥0.85x 目标。

## 4. 边界情况

- ✅ 零值（`deg2rad(0) = 0`）
- ✅ 负值（-180° → -π）
- ✅ 精确角度值（0°, 90°, 180°, 360°）
- ✅ float16 精度
- ✅ 3D 输入

## 5. 不支持场景

- 与 `element_wise` arrangement 共享约束：不同 ndim 的广播输入需在 torch 层统一 ndim

## 6. Generated Source 检查

生成源码路径：`~/.ninetoothed/{hash}.py`

检查结果：
1. tile 映射 ✅ — `triton.language.load` / `triton.language.store` 正确映射
2. 数据类型 ✅ — 无 float64 残留
3. 内存访问 ✅ — 单次 load + 单次 store
4. constexpr ✅ — `ninetoothed_constexpr` 作为编译时常量正确传入
5. block_size ✅ — auto-tune 搜索 32–1024，num_warps=8，num_stages=3

## 7. 迭代历史

| 迭代 | 阶段 | 结果 | 备注 |
|:--:|------|:--:|------|
| 1 | 生成 + 编译 | PASSED | 首次编译即通过 |

## 8. 合计

- **总迭代次数**：1（首次通过）
- **静态验证清单**：15/15 全部通过
- **精度验证**：7/7 PASSED
- **性能目标**：≥0.85x PyTorch @ 大规模 ✅
- **Skill E2E 测试**：六阶段工作流完整闭环验证通过
