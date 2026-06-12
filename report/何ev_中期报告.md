# 九齿编译优化 T1-2-1 中期报告

## 基本信息

| 项目 | 内容 |
|------|------|
| **小组名称** | 何ev |
| **赛题编号** | T1-2-1 |
| **赛题名称** | NineToothed 代码生成特化增强挑战 |
| **赛道** | 九齿开发赛道 |
| **GitHub ID** | gacn2890356890-rgb |
| **报告日期** | 2026-06-12 |
| **中期截止** | 2026-06-15 |

## 选定特化类别

从赛题组限定的 4 类特化中选定 **2 类**：

1. **Category 1: Contiguous Fast Path** — 连续布局快速路径
   - 当 AOT contiguity 信息足以判定 stride=1 时，简化 pointer arithmetic
   - 减少 stride 查询和乘法运算

2. **Category 2: Divisible Tile Fast Path** — 整除分块快速路径
   - 当 AOT 或静态 shape 信息判定 tile 无尾块时，消除边界 mask
   - 使用精确 loop bound 替代 next_power_of_2

## 当前进度

### 已完成

| 任务 | 状态 | 说明 |
|------|------|------|
| 代码库分析 | ✅ 完成 | 阅读 generation.py、aot.py、tensor.py、symbol.py 全部源码 |
| Weakness Analysis | ✅ 完成 | 识别 4 类低效点，含 2 个具体 case |
| TilingHint 数据类 | ✅ 完成 | 定义特化提示数据结构 |
| Divisible Tile Fast Path | ✅ 完成 | generation.py `_generate_offsets_and_mask` 修改 |
| Contiguous Fast Path | ✅ 完成 | generation.py `_generate_overall_offsets_and_mask` 修改 |
| Exact Innermost Sizes | ✅ 完成 | generation.py `_generate_innermost_indices` 修改 |
| AOT 集成 | ✅ 完成 | aot.py `_build_tiling_hint` 桥接 |
| 特化测试 | ✅ 完成 | 11 个测试用例（3 hit + 4 fallback + 4 source structure） |
| Benchmark | ✅ 完成 | 6 个场景，JSON 输出 |
| HONOR_CODE / REFERENCE | ✅ 完成 | 署名和引用披露 |
| 赛题报告 | ✅ 完成 | 完整技术报告 |

### 待完成

| 任务 | 预计完成 | 备注 |
|------|---------|------|
| GPU 测试验证 | 6/13-6/14 | 需远程服务器运行 pytest |
| Benchmark 数据采集 | 6/13-6/14 | 实际 runtime 测量 |
| 报告转 PDF | 6/14 | markdown → PDF |
| 代码审查和边界检查 | 6/14 | 确保无误命中 |
| 提交 PR | 6/15 前 | 等待赛题组指定仓库 |

## 技术方案概要

### 架构
```
AOT Variant Specs (divisibility, contiguity)
  → _build_tiling_hint() → TilingHint dataclass
  → CodeGenerator(tiling_hint) → 特化 Triton 源码
```

### 关键改动
- **generation.py**: `_generate_offsets_and_mask` 在整除时跳过 mask；`_generate_overall_offsets_and_mask` 在连续时简化 stride；`_generate_innermost_indices` 使用精确尺寸
- **aot.py**: 每个 variant 根据其 divisibility/contiguity spec 生成特化源码

### Fallback 保证
- TilingHint 所有字段默认值 → 代码路径与 baseline 字符级一致
- 非整除/非连续输入 → 通用路径，性能无回退

## 风险与问题

1. **GPU 测试环境**：本地无 GPU，需依赖远程服务器
2. **隐藏测试未知**：隐藏评测的具体用例不可预测，依赖 clean fallback 设计
3. **AOT variant 爆炸**：当前 `_enumerate_variant_specs` 可能产生较多 variant；需确认编译时间可控

## 时间规划

| 日期 | 事项 |
|------|------|
| 6/12 | ✅ 中期报告完成，代码提交 |
| 6/13-6/14 | GPU 测试、benchmark 数据采集 |
| 6/15 | 中期报告提交 |
| 6/16-7/12 | 根据测试结果迭代优化 |
| 7/13 0:00 | 最终提交截止 |
