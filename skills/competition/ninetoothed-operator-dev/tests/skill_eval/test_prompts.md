# Skill 消费端评测 — 测试 Prompts

以下是 3 个测试场景的精确 prompt。按顺序在新对话中执行。

---

## 测试 1：从零编写新算子

**Prompt 文本**（直接粘贴给搭载了本 skill 的 AI agent）：

```
请使用 NineToothed DSL 实现一个 LeakyReLU 算子。

要求：
- LeakyReLU 公式：f(x) = x if x > 0 else alpha * x，alpha=0.01
- 文件放在 operators/leaky_relu.py
- 遵循 operators/ 目录下现有的文件结构（参考 relu.py、gelu.py）
- 编写 correctness 测试，对比 torch.nn.functional.leaky_relu(x, 0.01)
- 将新的 create_leaky_relu_kernel 导出到 operators/__init__.py

请先阅读 SKILL.md 和 operators/relu.py、operators/gelu.py 了解模式，然后生成。
```

**验证清单**（作为评测者，检查 agent 输出是否符合以下标准）：

- [ ] arrangement 使用 `x.tile((BLOCK_SIZE,))` 和 `output.tile((BLOCK_SIZE,))`（逐元素模式）
- [ ] Symbol 通过模块级定义 + 参数默认值传入（或闭包 block_size()）
- [ ] application 中实现 LeakyReLU 逻辑（如 `ntl.where(x > 0, x, 0.01 * x)`）
- [ ] 赋值行末尾有 `# noqa: F841`
- [ ] 测试文件包含多个 shape 的测试（如 (100,) (512,) (1024,)）
- [ ] 测试使用 `torch.nn.functional.leaky_relu(x, 0.01)` 作为 expected
- [ ] __init__.py 中添加了 `create_leaky_relu_kernel` 的导入和导出
- [ ] 在 GPU 上运行测试通过（如适用）

参考实现：`tests/skill_eval/leaky_relu_reference.py`

---

## 测试 2：从零编写 2D 算子（skill 无直接范例）

**设计意图**：验证 skill 能否让 agent 泛化到「skill 中没有对应 2D 逐元素范例」的场景。skill 中所有逐元素算子（relu、sigmoid、gelu）都是 1D，agent 必须从 SKILL.md 的 tile 决策表自行推导 2D 策略。

**Prompt 文本**：

```
请使用 NineToothed DSL 实现一个 HardSwish 2D 算子。

HardSwish 公式：f(x) = x * clamp(x + 3, 0, 6) / 6

要求：
- 输入输出均为 2D 张量
- 文件放在 operators/hardswish_2d.py
- 遵循 operators/ 目录下的文件结构
- 编写 correctness 测试，对比 torch.nn.functional.hardswish(x)
- 将新的 create_hardswish_2d_kernel 导出到 operators/__init__.py

提示：skill 中没有 2D 逐元素算子的直接范例，请参考 SKILL.md 中的 tile 决策表自行推导。
```

**验证清单**：

- [ ] tile 形状使用 `(BLOCK_SIZE_M, BLOCK_SIZE_N)` — 2D 逐元素模式
- [ ] 两个 BLOCK_SIZE 均为模块级 `Symbol(constexpr=True)`，通过参数默认值传入
- [ ] application 实现 HardSwish：`ntl.minimum(ntl.maximum(x+3, 0), 6) * x / 6`
- [ ] 赋值行有 `# noqa: F841`
- [ ] `Tensor(2)` 而非 `Tensor(1)`（2D 张量）
- [ ] 测试使用多 shape（如 (32,64) (128,256) (512,512)）
- [ ] 测试对比 `torch.nn.functional.hardswish(x)`
- [ ] `__init__.py` 正确注册
- [ ] GPU 上测试通过（如适用）

参考实现：`tests/skill_eval/hardswish_2d_reference.py`

---

## 测试 3：诊断故意引入的 bug

**前置准备**：将 `tests/skill_eval/buggy_softmax.py` 复制替换 `operators/softmax.py`

```bash
cp tests/skill_eval/buggy_softmax.py operators/softmax.py
```

**Prompt 文本**：

```
我运行了 operators/softmax.py 的 correctness 测试，但失败了。以下是错误日志：

  Shape (512, 256)       : ✗ 失败
    最大差异: 1.235e+00
  Shape (512, 256)       : ✗ 和不为1
  Shape (1024, 512)      : ✗ 失败
    最大差异: 9.876e-01
  Shape (1024, 512)      : ✗ 和不为1

请帮我诊断问题。重点分析 arrangement 函数中的 tile 形状是否正确。
```

**验证清单**：

- [ ] agent 识别出 tile 形状 `(BLOCK_SIZE, BLOCK_SIZE)` 对于归约算子不正确
- [ ] agent 正确给出修复：改为 `(1, BLOCK_SIZE)`
- [ ] agent 解释了原因（归约维度用 BLOCK_SIZE，非归约维度用 1）
- [ ] agent 引用或体现了 SKILL.md 中 §2.2.3 的 tile 语义规则

**恢复**：测试完成后恢复正确版本

```bash
git checkout HEAD -- operators/softmax.py
```

---

## 测试 4：Benchmark 分析

**Prompt 文本**：

```
请帮我分析 operators/gelu.py 的 benchmark 结果。我需要了解：

1. 当前 BLOCK_SIZE 设计是否合理
2. 如果有更优的 BLOCK_SIZE，给出调优建议
3. 与 torch.nn.functional.gelu 的差距是否在预期范围内

如果没有可用的 benchmark 数据，请基于 SKILL.md 第 6 章的框架，给出：
- 如何运行 benchmark 的命令
- 关键指标（MS/Memory/Time）的含义
- 性能回退分析的步骤
```

**验证清单**：

- [ ] agent 引用了 SKILL.md 中性能验证章节
- [ ] agent 提及 `benchmarks/bench.py` 或 `tests/bench_light.py`
- [ ] agent 给出 BLOCK_SIZE 调优建议（如尝试 128/256/512/1024）
- [ ] agent 说明了与 PyTorch baseline 对比的方法

---

## 评测记录表

**核心原则**：只有写出多维算子和 skill 中没有直接范例的算子，skill 才算真正可用。

| 编号 | 测试 | 关键维度 | 通过条件 | 结果 |
|------|------|---------|---------|------|
| E1.1 | LeakyReLU arrangement | 1D 逐元素（有范例参照） | `(BLOCK_SIZE,)` | ☐ |
| E1.2 | LeakyReLU application | 新算子逻辑 | `ntl.where` + noqa | ☐ |
| E1.3 | LeakyReLU 测试 | 测试规范 | 多 shape + torch 对比 | ☐ |
| E1.4 | LeakyReLU 注册 | 集成 | `__init__.py` 导出 | ☐ |
| **E2.1** | **HardSwish tile** | **2D 逐元素（无直接范例）** | **`(B_M, B_N)` + `Symbol`** | ☐ |
| **E2.2** | **HardSwish application** | **新算子 + 新维度** | **clamp/max/min + noqa** | ☐ |
| **E2.3** | **HardSwish 测试** | **2D 测试规范** | **多 shape 2D + `Tensor(2)`** | ☐ |
| **E2.4** | **HardSwish 注册** | **集成** | **`__init__.py` 导出** | ☐ |
| E3.1 | bug 定位 | tile 语义推理 | 识别 `(B,B)` 错误 | ☐ |
| E3.2 | 根因解释 | tile 知识 | 归约维度→B，其余→1 | ☐ |
| E3.3 | 修复方案 | 纠正能力 | 改为 `(1, BLOCK_SIZE)` | ☐ |
| E4.1 | 性能引用 | 文档引用 | SKILL.md §6 | ☐ |
| E4.2 | 分析框架 | 方法论 | 调优 + baseline | ☐ |

**通过标准**：E1 全通过 + E2 全通过（共 8 项）为及格。E2 是核心评测指标。E3 和 E4 为加分项。
