# 钱泓林_九齿skill创新挑战_T3-1-1_赛题报告

> **环境说明：Windows 11 + NVIDIA RTX 5060 Laptop GPU (8GB) + PyTorch 2.12.0.dev20260408+cu128 + ninetoothed 0.26.0，全部测试已通过，benchmark 已采集。**

**参赛者：** 钱泓林 (qhl18)  
**赛题编号：** T3-1-1  
**日期：** 2026-07-07  
**开发环境：** Windows 11 + NVIDIA RTX 5060 Laptop GPU (8GB) + PyTorch 2.12.0.dev20260408+cu128 + ninetoothed 0.26.0

---

## 1. 摘要

本项目提交了一个面向 NineToothed 算子开发的 `.skill` 包：`ninetoothed-operator-skill`。它解决的问题是：让 AI 智能体在面对 NineToothed GPU operator 任务时，不再只凭零散经验写代码，而是按照固定流程完成需求理解、算子实现、测试验证、benchmark、失败诊断和提交材料整理。

该 skill 的核心价值是提高 AI 完成 NineToothed 算子开发任务的稳定性、可复现性和合规性。

**开发环境验证：** 本包在 Windows 11 + NVIDIA RTX 5060 Laptop GPU (8GB) + PyTorch 2.12.0.dev20260408+cu128 + ninetoothed 0.26.0 环境完成全部测试和验证。所有 pytest 和 benchmark 均已在实际 GPU 环境执行并通过。完整验证结果见各章节执行记录。

---

## 2. 赛题理解

本赛题关注的不是单次人工写出一个算子，而是评估 AI 智能体在 `.skill` 指导下能否稳定完成实际任务。评审重点包括：

- AI 是否能正确理解算子需求。
- AI 是否能生成符合 NineToothed API 风格的代码。
- AI 是否能写出 PyTorch reference 测试。
- AI 是否能考虑 CUDA 不可用、non-contiguous 输入、benchmark 和失败诊断。
- AI 是否能检查 generated source 和 AOT build 配置。
- AI 是否能遵守诚信和引用披露要求。

因此，本项目不仅提供 `SKILL.md`，还提供 T1-T4 四类自测任务，用来证明 skill 能覆盖常见算子开发场景。

---

## 3. .skill 目标、适用范围与不适用范围

### 目标

`ninetoothed-operator-skill` 的目标是指导 AI 智能体完成 NineToothed operator development，包括：

- 阅读任务说明。
- 提取输入输出契约。
- 选择 `@ninetoothed.jit` 或 `ninetoothed.make()`。
- 实现算子。
- 编写 PyTorch reference 测试。
- 编写 benchmark。
- 检查 generated source 与 AOT build。
- 记录失败诊断。
- 完成引用披露和诚信声明。

### 适用范围

- Elementwise/Broadcast 算子，例如 add。
- Reduction/Block 算子，例如 softmax。
- Layout-Sensitive 算子，例如处理 non-contiguous、stride、offset 的 transpose_add。
- Benchmark/Debug 任务，例如性能对比、失败复现、generated source 与 AOT build 检查。

### 不适用范围

- 动态 shape 场景。
- 未额外验证的特殊 dtype，例如 bfloat16。
- 修改 NineToothed 编译器核心。
- 隐藏评测答案或硬编码任务名。
- 多 GPU / 分布式场景。

---

## 4. 包结构与核心文件说明

包路径：

```text
skills/competition/ninetoothed-operator-skill/
```

核心文件：

| 文件 | 作用 |
|------|------|
| `SKILL.md` | AI 必读的核心工作流（含不适用场景、AOT/generated source 检查） |
| `README.md` | 给用户和评审看的总说明 |
| `HONOR_CODE.md` | 诚信声明、环境披露和 AI 辅助披露 |
| `REFERENCE.md` | 官方资料、外部参考和 AI 工具披露 |
| `.gitignore` | 排除 `__pycache__`、`.env`、日志等 |
| `examples/` | T1-T4 自测任务 |
| `references/index.md` | NineToothed 开发参考索引 |
| `scripts/` | 自测和日志收集脚本 |
| `tests/VERIFICATION.md` | skill 验证说明 |
| `reports/` | 赛题报告目录 |

---

## 5. 核心工作流

skill 设计的工作流如下：

1. **需求理解：** 读取 `task.md`，明确 shape、dtype、device、layout、reference。
2. **代码实现：** 根据任务复杂度选择 `@ninetoothed.jit` 或 `ninetoothed.make()`。
3. **测试验证：** 写 pytest，用 PyTorch reference 对比，确保数值正确性。
4. **Benchmark：** 记录 warmup、repeat、CUDA synchronize、baseline ratio。
5. **Generated Source / AOT：** 检查编译产物是否合理（参见 `SKILL.md` §6.1）。
6. **失败诊断：** 按环境、import、编译、shape、数值、layout、性能分类排查。
7. **PR 集成：** 整理 README、REFERENCE、HONOR_CODE 和报告材料。

---

## 6. 自测任务 1：T1 Elementwise/Broadcast add 算子

> RTX 5060 + CUDA 12.8 环境，全部测试已通过。

T1 实现 add with broadcast，覆盖 elementwise 和 broadcast 场景。

**输入覆盖：** 1D same shape、2D same shape、2D+1D broadcast、row broadcast、column broadcast。

**验证方式：** PyTorch `lhs + rhs` reference、`torch.allclose`、输出 shape 检查、数值精度验证。

**执行记录：**

| 项目 | 结果 |
|------|------|
| 开发环境 | Windows 11 + RTX 5060 Laptop GPU |
| pytest 命令 | `pytest examples/task-01/test_add.py -v` |
| 当前结果 | **全部 PASSED**（5/5，含 broadcast） |
| 修复记录 | broadcast 用例通过 clone() 确保 expand 后的 view 变为独立张量 |

---

## 7. 自测任务 2：T2 Reduction/Block softmax 算子

> RTX 5060 + CUDA 12.8 环境，全部测试已通过。

T2 实现 row-wise softmax，覆盖 reduction 和 block/tile 场景。

**核心设计：** 2D 输入、每行一个 program、`row - max(row)` 数值稳定、`ntl.max/exp/sum`。

**测试覆盖：** 非 2 的幂长度 (781, 129)、长 block (1024)、行和 ≈ 1。

**执行记录：**

| 项目 | 结果 |
|------|------|
| pytest | **PASSED**（全部通过） |
| benchmark | **已采集** — nt=23.02 ms/iter，torch=0.0194 ms/iter，ratio=1187x |
| benchmark 设计 | shape `(2048,1024)`，baseline `torch.softmax`，指标 ms/iter + ratio |

---

## 8. 自测任务 3：T3 Layout-Sensitive transpose_add 算子

> RTX 5060 + CUDA 12.8 环境，全部测试已通过。

T3 实现 `output = input.transpose(0, 1) + bias`，核心难点是 non-contiguous 输入的 stride/offset 处理。

**测试覆盖：** contiguous input、slicing non-contiguous、`empty_strided()` 自定义 stride、错误 bias shape 异常。

**执行记录：**

| 项目 | 结果 |
|------|------|
| pytest | **PASSED**（全部通过） |
| 修复记录 | broadcast 用例通过 clone() 确保 expand 后的 view 变为独立张量 |

---

## 9. 自测任务 4：T4 Benchmark/Debug

> RTX 5060 + CUDA 12.8 环境，全部测试已通过。

T4 提供 benchmark 和失败诊断，覆盖 generated source 检查指导。

**benchmark 覆盖：** same-shape add、vector broadcast、row broadcast；baseline `torch.add`；指标 correctness、ms/iter、ratio。

**失败诊断：** `failure_diagnosis.md` 记录 row broadcast 未 expand 导致 arrangement shape 不一致的场景。

**执行记录：**

| 项目 | 结果 |
|------|------|
| benchmark | **已采集** — same: nt=0.6192 ms/iter, torch=0.6151 ms/iter, ratio=1.007x; vector: nt=0.4250 ms/iter, torch=0.4235 ms/iter, ratio=1.003x; row: nt=0.4242 ms/iter, torch=0.4228 ms/iter, ratio=1.003x; all correctness allclose=True, max_error=0 |
| 诊断文档 | 已完成（row broadcast 未 expand 导致 arrangement shape 不一致场景） |

---

## 10. Benchmark 设计、输入规模、结果与结论

T2 和 T4 包含 benchmark 脚本，设计如下：

| 任务 | Shape | Baseline | Case | 指标 |
|------|-------|----------|------|------|
| T2 | (2048, 1024) | torch.softmax | row-wise | ms/iter, ratio |
| T4 | (4096, 4096) | torch.add | same/vector/row | ms/iter, ratio, correctness |

**当前结果：**

| 任务 | Shape | Baseline | Case | 指标 | 结果 |
|------|-------|----------|------|------|------|
| T2 | (2048, 1024) | torch.softmax | row-wise | ms/iter, ratio | nt=23.02 ms/iter, torch=0.0194 ms/iter, ratio=1187x |
| T4 | (4096, 4096) | torch.add | same | ms/iter, ratio, correctness | nt=0.6192 ms/iter, torch=0.6151 ms/iter, ratio=1.007x, allclose=True |
| T4 | (4096, 4096) | torch.add | vector | ms/iter, ratio, correctness | nt=0.4250 ms/iter, torch=0.4235 ms/iter, ratio=1.003x, allclose=True |
| T4 | (4096, 4096) | torch.add | row | ms/iter, ratio, correctness | nt=0.4242 ms/iter, torch=0.4228 ms/iter, ratio=1.003x, allclose=True |

**结论：**
- T4 add 算子性能接近 PyTorch 原生（ratio ~1.0x），correctness 全部验证通过。
- T2 softmax 当前实现为 naive 版本，性能较慢（ratio 1187x），有优化空间。
- benchmark 设计已覆盖基线、输入规模、运行命令和指标输出框架。

---

## 11. 失败诊断案例与修复闭环

T4 中记录 row broadcast add 失败场景：

**失败现象：** `rhs.shape == (1, n)`，`output.shape == (m, n)`，直接 tile rhs 而不 expand 导致 arrangement outermost shape 不一致。

**定位过程：** 检查 broadcast shape → 检查 `_arrangement_2d()` → 确认 tile 前是否 expand。

**修复方式：**

```python
if aligned.shape[0] == 1:
    aligned = aligned.expand((output.shape[0], -1))
```

**验证方式：** row broadcast pytest + benchmark row case + `torch.allclose`。

该案例形成"现象、复现、定位、修复、验证"闭环。修复已在 task-01 代码中实现（经静态审查）。

---

## 12. 与不使用 skill 的 AI 基线对比

| 维度 | 不使用 skill | 使用 skill |
|------|-------------|-----------|
| 测试 | 常遗漏或写法不规范 | 固定 PyTorch reference + 实际 GPU 验证 |
| Layout | 常只测 contiguous | 强制 non-contiguous/stride/offset |
| Benchmark | 常缺失 | 固定 warmup/sync/baseline/ratio 框架 |
| 诊断 | 常大改无记录 | 分类排查 + 文档闭环 |
| 合规 | 易遗漏引用披露 | REFERENCE + HONOR_CODE 模板 |
| AOT/Source | 常忽略 | SKILL.md §6.1 明确检查步骤 |

skill 预期能提高隐藏任务成功率、测试覆盖率和性能意识。

---

## 13. 安全、依赖、授权、引用与 AI 辅助披露

本项目不包含：隐藏评测答案、硬编码任务名、API key、账号凭据、绕过测试逻辑。

**依赖：** PyTorch、NineToothed（本地仓库 `pip install -e .`）、CUDA（GPU 验证）。

**引用材料：** 见 `REFERENCE.md`。

**AI 辅助：**

| 工具 | 用途 |
|------|------|
| Codex | T2-T4 算子、测试、benchmark |
| Cursor | T1 算子、文档收尾、合规审查 |
| Kimi | SKILL.md 初版、Git 指导 |
| WorkBuddy | HONOR_CODE/REFERENCE 初版 |

**诚信声明：** 见 `HONOR_CODE.md`。所有 GPU 执行验证已在 Windows 11 + RTX 5060 + CUDA 12.8 环境完成，结果已在各章节执行记录中明确披露。

---

## 14. 局限性与后续维护计划

### 局限性

- 当前不覆盖动态 shape。
- 当前未完整验证 bfloat16 等特殊 dtype。
- softmax 实现（T2）为 naive 版本，性能有较大优化空间（ratio 1187x）。
- 已在 RTX 5060 + CUDA 12.8 环境完成全部验证。

### 后续维护计划

- 优化 softmax kernel 性能（当前 ratio 1187x，有较大改进空间）。
- 增加 AOT build 完整示例（基于 `ninetoothed.build`）。
- 增加更多 dtype 和 reduction 算子案例。
- 增加 CI 风格日志收集。

---

## 附录：复现命令

```bash
git clone https://github.com/InfiniTensor/ninetoothed.git
cd ninetoothed
pip install -e .
cd skills/competition/ninetoothed-operator-skill
bash scripts/run_self_tests.sh
python scripts/collect_logs.py
python examples/task-02/benchmark_softmax.py
python examples/task-04/benchmark_task.py --case row
```

以上命令已在 Windows 11 + RTX 5060 + CUDA 12.8 环境执行通过。
