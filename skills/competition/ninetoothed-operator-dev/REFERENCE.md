# REFERENCE.md

本次赛题提交涉及的参考资源披露。

---

## 情况说明

本人独立完成本 skill 包的核心设计与开发（算子实现、测试代码、benchmark、SKILL.md 文档、参考资料和模板）。以下外部资源仅在开发期用于了解 NineToothed API 形态、官方代码风格和 Triton benchmark 惯例，未直接复制核心代码片段；所有算子实现和 skill 文档均为独立编写。

---

## 参考资源清单

### 1. NineToothed 官方示例仓库

| 项目 | 内容 |
|------|------|
| 资源名称 | ninetoothed-examples |
| 资源链接 | https://github.com/InfiniTensor/ninetoothed-examples |
| 参考内容 | `ops/ninetoothed/kernels/` 下的官方算子实现（silu.py、swiglu.py、softmax.py、add.py、fused_rms_norm.py、matmul.py、max_pool2d.py、bmm.py）与 `tests/` 下的测试代码，用于了解 arrange-and-apply 范式的基本代码结构和 tile 分块策略 |
| 修改与优化 | 算子为参照官方结构独立重写：softmax 改用 sigmoid 恒等式替代 libdevice.tanh（因 libdevice 路径在代码生成中不可用）、rms_norm 简化为单层 tiling、strided_add 为独立开发。所有注释、测试用例和 SKILL.md 均为独立编写 |
| 开源协议 | Apache 2.0 |

### 2. NineToothed Python API

| 项目 | 内容 |
|------|------|
| 资源名称 | ninetoothed |
| 资源链接 | https://github.com/InfiniTensor/ninetoothed |
| 参考内容 | `ninetoothed.Symbol`、`ninetoothed.Tensor`、`ninetoothed.make()`、`ninetoothed.block_size()`、`ninetoothed.language` 原语（max、sum、sigmoid、exp、cast 等）的 API 用法和参数含义。benchmark 部分参考 `bench.py` 中 triton.testing.Benchmark 的调用惯例 |
| 修改与优化 | 独立开发：SKILL.md 中的 tile 决策速查表、operator_patterns.md 的模式总结、bench.py 包装器（封装 triton.testing.Benchmark）、bench_light.py（torch.cuda.Event 直接计时）。API 文档 `references/nine_toothed_api.md` 为自行整理 |
| 开源协议 | Apache 2.0 |

### 3. Triton

| 项目 | 内容 |
|------|------|
| 资源名称 | Triton |
| 资源链接 | https://github.com/triton-lang/triton |
| 参考内容 | `triton.testing.Benchmark`、`triton.testing.perf_report`、`triton.testing.do_bench` 的用法和接口形态（仅用于 benchmark 框架，不涉及算子代码） |
| 修改与优化 | 独立封装：`benchmarks/bench.py` 中的 benchmark() 统一接口、`bench_light.py` 完全不依赖 triton.testing 的轻量实现 |
| 开源协议 | MIT |

### 4. PyTorch

| 项目 | 内容 |
|------|------|
| 资源名称 | PyTorch |
| 资源链接 | https://github.com/pytorch/pytorch |
| 参考内容 | `torch.nn.functional.gelu`、`torch.nn.functional.softmax`、`torch.nn.functional.sigmoid`、`torch.nn.functional.relu`、`torch.sum` 的数学定义和数值行为，用作 correctness 测试的参考实现 |
| 修改与优化 | 所有算子均为九齿独立实现，PyTorch 仅用于输出对齐验证 |
| 开源协议 | BSD-3 |

### 5. Agent Skills 开放规范

| 项目 | 内容 |
|------|------|
| 资源名称 | Agent Skills — Agent Skills Overview |
| 资源链接 | https://agentskills.com |
| 参考内容 | SKILL.md 的文件结构要求、frontmatter 格式、references/scripts/examples/tests 目录约定。九齿技术文档中 Arrange-and-Apply 范式和 Tile 机制的官方说明 |
| 修改与优化 | SKILL.md 内容全部独立编写：九齿算子开发全流程指导（需求分析→arrangement→application→correctness→benchmark→诊断）、tile 决策速查表、融合算子模式、性能验证章节。references/ 下 3 篇文档为自行整理，模板独立的 operator/test/benchmark 模板 |
| 开源协议 | — |

### 6. 大赛赛题文档（T3-1-1）

| 项目 | 内容 |
|------|------|
| 资源名称 | 2026 春季人工智能大赛 — 九齿 .skill 创新挑战赛道 赛题说明 |
| 资源链接 | 赛题组提供的赛题要求文档 |
| 参考内容 | 赛题目标、包结构要求、自测任务要求、评分标准、提交要求 |
| 修改与优化 | Skill 包结构和自测任务按赛题要求组织 |
| 开源协议 | — |

---

## 补充说明

- 开发过程中使用本地副本 `ntd_code/` 存放 ntd_code 中部分文件用于快速查阅 API；该目录已从最终提交中删除
- 所有 NineToothed 算子实现（`operators/` 下 9 个文件）、测试代码（`run_all_operator_tests.py`、`tests/` 下自测文档）、benchmark 代码（`benchmarks/`、`tests/bench_light.py`）、SKILL.md、references/ 文档和 assets/ 模板均为独立编写
- 以上列出的外部资源仅在理解 API 形态、代码风格和规范要求时提供参考，未直接复制受版权保护的代码片段
