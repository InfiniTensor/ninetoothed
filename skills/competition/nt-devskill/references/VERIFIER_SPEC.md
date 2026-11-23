# 评分标准参考（Verifier Spec）

---

## 评分子项（每任务 0-10 分）

| 子分           | 满分 | 通过条件                                             | AI 应如何满足                                            |
| -------------- | ---- | ---------------------------------------------------- | -------------------------------------------------------- |
| **任务完成度** | 4    | 算子语义/shape/dtype/boundary 或诊断目标达成         | 正确性矩阵全 PASS = 4；部分 = 1-3；无 = 0                |
| **测试与验证** | 2    | 指定测试通过且闭环                                   | 矩阵全部 PASS + 命令和结果已记录 = 2                     |
| **性能意识**   | 1    | benchmark / 生成源码分析 / 合理优化论证              | benchmark.csv + Roofline 结论，或 inspect_generated 证据 |
| **最小补丁**   | 1    | 无无关重构、无批量格式化、无破坏性改动               | `git diff` 仅包含算子相关文件                            |
| **仓库风格**   | 1    | 命名/结构/错误处理/测试/文档风格与仓库一致           | ruff/black clean + 匹配现有算子风格                      |
| **流程合规**   | 1    | 可复现的记录、无密钥、无网络、无测试绕过、无隐藏答案 | 审计笔记完整、无 eval/exec                               |

---

## 确定性检查门（可自动化验证）

| 检查项       | 命令                                                     | 通过条件     |
| ------------ | -------------------------------------------------------- | ------------ |
| 正确性       | `python scripts/validate.py --op <name>`                 | 全部 PASS    |
| 非连续测试   | pytest 包含 transposed/strided 测试用例                  | 全部 PASS    |
| Benchmark    | benchmark 输出含 ≥3 个 shape + compute/memory-bound 结论 | 有数值证据   |
| 生成源码检查 | `python scripts/inspect_generated.py --op <name>`        | 有输出记录   |
| 合规性       | 无 `eval(`/`exec(`/网络调用                              | AST 检查通过 |

---

## 三关评分（Three-Gate Grading）

三个门必须全部通过，否则**任务完成度被限制为 1/4**：

| 关卡   | 名称       | 检查内容                             | 失败后果       |
| ------ | ---------- | ------------------------------------ | -------------- |
| Gate 1 | **编译**   | 代码可被 Python AST 解析，无语法错误 | completion ≤ 1 |
| Gate 2 | **正确性** | 正确性矩阵至少 1 项 PASS             | completion ≤ 1 |
| Gate 3 | **合法性** | 解决方案使用了九齿框架               | completion ≤ 1 |

### 合法的九齿使用模式（两种均合法）

```python
# 模式 1: ninetoothed.make（主流模式）
kernel = ninetoothed.make(arrangement, application, tensors)

# 模式 2: @ninetoothed.jit 装饰器（同样合法）
@ninetoothed.jit
def my_kernel(input_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
    ...
```

> **注意：** 两种模式在真实测试套件中均有使用（make: ~13/15 测试，@jit: ~2/15 测试）。评分器必须同时识别两种模式。

### 禁止的 Fallback 模式

以下模式表明 AI 使用了 PyTorch 底层操作而非编写真正的九齿 kernel，将导致合法性门失败：

```python
# 禁止：直接使用 aten 操作
torch.aten.add(x, y)
torch.ops.aten.mm(a, b)

# 禁止：绕过 Python 层使用 C++ 后端
torch._C._linalg_utils(...)
```

**检测方法：** `evaluation/rubric_scorer.py` 中的 `banned_fallback_hits()` 函数。

### 禁止的 Wrapper 绕过模式

以下模式表明 AI 放弃了九齿实现，在 wrapper 中用纯 PyTorch 完成了计算：

```python
# 禁止：wrapper 用纯 PyTorch 计算，kernel 变空壳
def channel_shuffle(input, groups):
    return input.view(B, G, C//G, H, W).transpose(1, 2).contiguous().view(B, C, H, W)
    # kernel 从未被调用，或只是 output = input

# 禁止：kernel 是占位符
def application(input, output):
    output = input  # noqa: F841  — 无实际计算
```

**检测方法：** `scripts/reward_hacking_guard.py` 中的：
- `wrapper_bypass_check()` — 检测 wrapper 是否有 `kernel()` 调用
- `kernel_noop_check()` — 检测 kernel 是否为 `output = input` 占位符

**正确做法：** 如果九齿无法实现某算子，应在 Audit Note 中**诚实报告限制**：
```
已知限制：channel_shuffle 的 reshape+transpose 模式无法用九齿 arrangement 高效表达，
当前使用 PyTorch 原生实现。九齿 kernel 未参与计算。
```

---

## 奖项门槛（规则 §4.5）

| 门槛     | 条件                                           |
| -------- | ---------------------------------------------- |
| 无负迁移 | with-skill 总分 ≥ no-skill 总分（每个任务）    |
| 性能验证 | ≥ 2 个任务有有效的性能验证（benchmark + 结论） |
| 最低总分 | 预缩放 ≥ 48/80                                 |
| 任务覆盖 | ≥ 5/8 任务的完成度 ≥ 3/4                       |
| 性能工作 | ≥ 2 个任务有有效的性能优化工作                 |

---

## AI 输出清单（Artifact Checklist）

每个算子完成后，必须产出以下 artifact：

```
✅ kernel.py          — 九齿 kernel（arrangement + application + make）
✅ torch wrapper      — PyTorch 接口（分配输出 + 调用 kernel）
✅ correctness test   — shape × dtype × layout 矩阵测试
✅ non-contiguous     — 转置/步幅输入测试
✅ benchmark result   — nt vs torch 性能对比（≥ 3 shape）
✅ optimization log   — 应用的优化技术和 before/after 数据
✅ audit note         — 文件变更 + 正确性 + 性能 + 已知限制
```

可选 artifact：
```
⬜ arrangement debug  — simulate_arrangement 输出（OOB 检查）
⬜ generated source   — inspect_generated.py 分析结果
⬜ arrangement PNG    — 可视化 tile 结构
```

---

## 禁止行为（违反会扣分）

| 禁止行为                            | 后果           |
| ----------------------------------- | -------------- |
| 放宽容差（atol/rtol）来让测试通过   | 任务完成度扣分 |
| 删除或跳过失败的测试                | 测试与验证 = 0 |
| 重构无关代码                        | 最小补丁 = 0   |
| CPU PyTorch vs GPU NineToothed 对比 | 正确性无效     |
| 单个 shape/dtype 就宣称正确         | 任务完成度 ≤ 2 |
| 归约操作不做 fp32 累加              | fp16 测试失败  |
| 正确性未通过就优化性能              | 本末倒置       |

---

## 评分示例

### 优秀输出（9/10）
```
任务完成度: 4/4 — 正确性矩阵 8/8 PASS（3 shapes × 2 dtypes + 非连续）
测试验证:   2/2 — 全部 PASS，命令和结果已记录
性能意识:   1/1 — benchmark 含 5 shapes，标注 memory-bound
最小补丁:   1/1 — 仅修改算子相关文件
仓库风格:   1/1 — ruff clean，命名与现有算子一致
流程合规:   1/1 — 审计笔记完整
```

### 及格输出（5/10）
```
任务完成度: 2/4 — 仅 1 shape × 1 dtype 通过，无非连续测试
测试验证:   1/2 — 部分 PASS，未记录命令
性能意识:   0/1 — 无 benchmark
最小补丁:   1/1 — OK
仓库风格:   1/1 — OK
流程合规:   0/1 — 审计笔记缺失
```
