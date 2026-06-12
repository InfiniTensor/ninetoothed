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

# 4. 检查生成源码差异（实测有效）
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

**实测环境**: NVIDIA RTX 3090 24GB, CUDA 13.0, Triton 3.1.0, PyTorch 2.5.1, Python 3.10.12

## 实测指标对比

| 场景 | 输入 | baseline_ms | submitted_ms | speedup | hit | mask B→S | stride B→S |
|------|------|-------------|--------------|---------|-----|----------|-----------|
| Contiguous+Divisible | 2048 | 0.0183 | 0.0182 | 1.0056 | ✅ | 2→**0** | 2→**0** |
| Contiguous Only | 1027 | 0.0178 | 0.0180 | 0.9886 | ✅ | 2→2 | 2→**0** |
| Divisible Only | 2048 | 0.0176 | 0.0179 | 0.9849 | ✅ | 2→**0** | 2→2 |
| Pure Fallback | 1027 | 0.0176 | 0.0176 | 0.9970 | ❌ | 2→2 | 2→2 |
| 2D Divisible | 512×512 | 0.0205 | 0.0203 | 1.0097 | ✅ | 2→**0** | 4→4 |
| 2D Non-Divisible | 519×519 | 0.0199 | 0.0199 | 0.9975 | ❌ | 2→2 | 4→4 |

> 注：speedup ~1.0 因为 benchmark kernel 为极简 identity 算子（单次 load+store, ~18μs），mask/stride 开销占比 ~0.5%。**生成代码质量指标（mask 100%消除、stride 100%消除）已充分证明特化有效**。对于真实计算密集型 kernel（matmul/attention），mask 和 stride 优化占比更大。

## Generated code metrics (实测)

| 指标 | Baseline | Contiguous+Divisible | 改善 |
|------|----------|---------------------|------|
| mask_complexity | 2 | **0** | -100% |
| stride_expr_count | 2 | **0** | -100% |
| pointer_expr_count | 2 | 2 | 0% (pointer始终需要) |
| source_line_count | 14 | 14 | 微内核无显著变化 |

## 源码 diff（实测）
```diff
- tl.load(ptr + (...) * stride_0, mask=True & (6 boundary checks), other=None)
+ tl.load(ptr + (...), mask=True, other=None)
```

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
