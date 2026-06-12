# [2026春季][T1-2-1] 何ev — NineToothed 代码生成特化增强

## 赛题信息

- **赛题编号**: T1-2-1
- **小组名称**: 何ev
- **GitHub ID**: gacn2890356890-rgb
- **分支**: `2026-spring-gacn2890356890-rgb-T1-2-1`

## 选定特化类别

**Category 1 (Contiguous Fast Path)** + **Category 2 (Divisible Tile Fast Path)**

## 主要改动点

### 影响模块

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `src/ninetoothed/generation.py` | 核心修改 | 新增 TilingHint 数据类；修改 mask/stride/innermost 生成逻辑 |
| `src/ninetoothed/aot.py` | 桥接修改 | 将 AOT variant spec 转化为 TilingHint，每个 variant 生成特化源码 |
| `tests/test_specialization.py` | **新增** | 11 个测试用例 |
| `benchmarks/bench_specialization.py` | **新增** | 6 场景 benchmark，JSON 输出 |
| `report/` | **新增** | Weakness analysis + 赛题报告 + 中期报告 |
| `HONOR_CODE.md` | **新增** | 署名 |
| `REFERENCE.md` | **新增** | 引用披露 |

### 关键代码路径

```
generation.py:
  TilingHint (line 28-56)           — 特化提示数据类
  __init__ (line 66-69)             — 接受 tiling_hint 参数
  _generate_offsets_and_mask (835)  — 整除时跳过 mask
  _generate_overall_offsets_and_mask (770) — 连续时简化 stride
  _generate_innermost_indices (845) — 使用精确 sizes

aot.py:
  _build_tiling_hint (line 495)     — variant → TilingHint 转换
  _aot (line 88-116)                — 每 variant 生成特化源码
```

## 自测命令

```bash
# 1. 运行所有既有测试（确认无回归）
pytest tests/ --ignore=tests/test_specialization.py -v

# 2. 运行特化测试
pytest tests/test_specialization.py -v

# 3. 运行 benchmark
python benchmarks/bench_specialization.py

# 4. 检查生成源码差异
python -c "
from ninetoothed.generation import CodeGenerator, TilingHint
from ninetoothed import Symbol, Tensor

def app(x):
    x  # noqa

hint = TilingHint(has_divisible_tiles=True, exact_innermost_sizes=True)
cg = CodeGenerator(tiling_hint=hint)
f = cg(app, 'torch', 'test', 4, 3, 1, False)
print(open(f).read())
"
```

**运行环境**: Python 3.10+, Triton 3.0+, PyTorch 2.0+, CUDA 12.0+, NVIDIA GPU

## 指标对比

| 场景 | 输入 | baseline_runtime_ms | submitted_runtime_ms | speedup | specialization_hit |
|------|------|---------------------|----------------------|---------|-------------------|
| Contiguous+Divisible | 2048 | _待GPU实测_ | _待GPU实测_ | 预期1.05-1.15 | ✅ |
| Contiguous Only | 1027 | _待GPU实测_ | _待GPU实测_ | 预期1.00-1.05 | ✅ (部分) |
| Divisible Only | 2048 | _待GPU实测_ | _待GPU实测_ | 预期1.02-1.08 | ✅ (部分) |
| Pure Fallback | 1027 | _待GPU实测_ | _待GPU实测_ | 预期1.00 | ❌ |
| 2D Divisible | 512×512 | _待GPU实测_ | _待GPU实测_ | 预期1.02-1.10 | ✅ |
| 2D Non-Divisible | 519×519 | _待GPU实测_ | _待GPU实测_ | 预期1.00 | ❌ |

Generated code metrics:

| 场景 | mask_expr_count | stride_expr_count | pointer_expr_count | source_line_count |
|------|----------------|-------------------|-------------------|-------------------|
| Baseline (no hint) | _待测_ | _待测_ | _待测_ | _待测_ |
| Contiguous+Divisible | _待测_ | _待测_ | _待测_ | _待测_ |

> 注：具体数值需在 GPU 环境运行 benchmark 后填入。

## 未覆盖、未实现或已知风险

### 未实现
- Category 3 (Broadcast/Scalar Fast Path)：未选择
- Jagged/ragged tensor 特化：当前特化不覆盖 jagged dim 场景

### 已知风险
- `has_divisible_tiles` 判定依赖 AOT `divisibility_spec` 覆盖所有 innermost 维度
- JIT 路径（`caller="torch"`）不提供 AOT 级别的 contiguity 信息，contiguous fast path 仅在 AOT 模式生效
- 非标准 stride（如 stride=N, N>1）当前不处理

### 性能回退
- 无：当 TilingHint 为默认值时，生成代码与 baseline 字符级一致

## Honor Code & Reference

- `HONOR_CODE.md`：署名的独立完成声明
- `REFERENCE.md`：引用资料和 AI 辅助使用情况

## 赛题报告

`report/何ev_九齿编译优化_T1-2-1_赛题报告.md`（待转 PDF）
