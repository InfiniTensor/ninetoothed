# nextafter 算子开发报告

> 展示 libdevice 优先检查策略的价值。从 C 语言位操作实现出发，
> 发现 libdevice 已有 `nextafter`，但因 double 精度问题需要手动位操作。

## 1. 算子信息

| 项目 | 内容 |
|------|------|
| 算子名称 | `nextafter` |
| 分类 | Element-wise Binary（模式 1） |
| CPU 参考 | IEEE 754 位操作（union type punning） |
| 共享 arrangement | `ntops.kernels.element_wise` |
| 核心挑战 | float32 位操作 + subnormal 处理 + `-0.0` sign 检测 |
| 关键 DSL 操作 | `libdevice.float_as_int`, `libdevice.int_as_float`, `libdevice.signbit`, `ntl.where` |
| 基线 | `torch.nextafter` |
| 生成文件 | `kernels/nextafter.py`, `torch/nextafter.py` |

## 2. 精度验证

**基线**：`torch.nextafter`

| 测试 | dtype | 规模 | 结果 |
|------|-------|------|:--:|
| f32 basic | float32 | 1024 | PASSED |
| f32 large | float32 | 4096×4096 | PASSED |
| x == y | float32 | 4 | PASSED |
| subnormal | float32 | 2 | PASSED |
| 3D | float32 | 8×64×128 | PASSED |
| transposed | float32 | 512×512 | PASSED |

精确匹配验证（`torch.equal`）全部通过。包括 tricky 边界：
- `-0.0 → -1.0`：需要 `libdevice.signbit` 而非 `input >= 0`
- `0.0 → +1.0`：subnormal 1.4e-45 正确

## 3. 性能评估

**Baseline**: `torch.nextafter`

| 规模 | ntops (ms) | PyTorch (ms) | 比率 | 判定 |
|------|-----------|-------------|------|:--:|
| 256×256 | 0.0722 | 0.0163 | 4.44x | launch overhead |
| 1024×1024 | 0.1181 | 0.0525 | 2.25x | overhead 摊薄中 |
| 4096×4096 | 0.6483 | 0.7041 | 0.92x | **OK（略快于 PyTorch）** |

**性能结论**：大规模下与 PyTorch 持平甚至略快（0.92x）。

## 4. 迭代历史

| 迭代 | 方案 | 现象 | 根因 | 修复 | 结果 |
|:--:|------|------|------|------|:--:|
| 1 | `libdevice.nextafter()` 直接调用 | subnormal 错误（`x=0→y=1` 返回 1.17e-38 而非 1.40e-45） | libdevice 函数默认 double 精度，float32→double→nextafter→float32 截断丢 subnormal | — | ❌ |
| 2 | 手动 `float_as_int` + 位运算 | `-0.0` 失败 | `input >= 0` 对 `-0.0` 返回 True，导致步进方向错误 | `libdevice.signbit(input) != 0` 替代 `input >= 0` | — |
| 3 | `signbit` + 位运算 | — | — | — | **6/6 PASSED** |

## 5. 边界情况

- ✅ subnormal（`0.0 → 1.0` 返回 min subnormal 1.4e-45）
- ✅ 负零（`-0.0 → -1.0` 正确处理符号位）
- ✅ `x == y`（直接返回 y）
- ✅ 非连续输入（转置）

## 6. 本次验证对 skill 的贡献

- **Stage 1 新增第 4 步**：检查 libdevice 是否有现成实现
- **ntl_api.md 扩展**：libdevice 从 1 行 → 分类函数表 + 使用方式 + dtype 注意事项

## 7. 合计

- **总迭代次数**：3（libdevice 精度问题 → signbit → 成功）
- **精度验证**：6/6 PASSED
- **性能目标**：0.92x @ 4096² ✅
- **libdevice 直接调用尝试了** ✅（Stage 1 第 4 步验证有效）
