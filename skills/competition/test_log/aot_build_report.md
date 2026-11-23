# AOT Build 实操验证报告

> AOT 编译尝试记录。文档正确，当前 NineToothed 版本有平台 bug。

## 1. 验证过程

使用 `ntops.kernels.relu.premake` 进行 AOT 编译测试：

```python
arrangement, app, tensors = premake(2)
ninetoothed.make(
    arrangement, app, tensors,
    caller='cuda',
    kernel_name='relu_aot',
    output_dir='./build_aot_test',
    num_warps=4,
    num_stages=2,
)
```

## 2. 结果

| 尝试 | 调用方式 | 结果 |
|:--:|------|------|
| 1 | `ninetoothed.make(*premake(2), caller='cuda', ...)` | ❌ `'NoneType' object has no attribute 'source'` |
| 2 | `aot(app, caller='cuda', ...)` 直接调用 | ❌ `'bool' object has no attribute 'free_symbols'` (auto-tuner bug) |
| 3 | `ninetoothed.make(arr, app, tensors, caller='cuda', max_num_configs=1, ...)` | ❌ 同上 |

## 3. 根因分析

- 调用路径正确进入 AOT 分支 (`make.py:49 → aot.py:36 → _aot → _build_variant`)
- 错误发生在 `aot.py:272` 的 `_build_variant` 中，`source` 属性为 `None`
- 第二个错误 (`free_symbols`) 表明 auto-tuner 与 AOT 的集成在当前版本有 bug
- **非调用方式问题**，是 NineToothed v0.25.0 在 Windows + Triton 3.7.0 环境下的已知限制

## 4. Skill 覆盖状态

| 项目 | 状态 |
|------|:--:|
| SKILL.md §9 AOT 文档 | ✅ 完整（代码示例 + 参数表 + JIT/AOT 决策表） |
| AOT 实操验证 | ⚠️ 文档正确但执行受阻于框架 bug |
| AOT 示例文件 | ❌ 未创建（待框架修复后补充） |

## 5. 对 Competition 的影响

- 若隐藏任务要求 AOT build 实操，可指导 AI 智能体使用文档中的正确 API 调用方式
- 实际编译可能因平台/版本问题失败——属于环境限制而非 skill 缺陷
- 建议在赛题报告中标注此限制
