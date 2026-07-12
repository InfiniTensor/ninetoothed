# 参考资料披露 (REFERENCE.md)

**赛题**: T1-2-1 九齿编译优化

---

## 参考资源 1: Helion (PyTorch)

| 项目 | 详情 |
|------|------|
| **名称** | Helion — PyTorch 团队开发的高层 Triton DSL |
| **链接** | https://github.com/pytorch/helion |
| **协议** | BSD-3-Clause (PyTorch 项目标准协议) |
| **参考内容** | 以下代码模块的设计思路和实现模式，未直接复制代码： |
| | `helion/_compiler/tile_strategy.py:3522-3576` — `NDTileStrategy._setup_mask()` + `known_multiple()` 可整除 tile mask 消除机制 |
| | `helion/_compiler/indexing_strategy.py:1748-1776` — `SubscriptIndexing.create()` + `_is_size_one()` size-1 维度 stride 跳过 |
| | `helion/_compiler/indexing_strategy.py:245-322` — `_PointerLoadContiguity` 类：contiguity 分析 + `_max_run()` + `tl.max_contiguous` 包装 |
| | `helion/_compiler/indexing_strategy.py:595-612` — `PointerIndexingStrategy.codegen_load()` 中的 `tl.max_contiguous` 整合模式 |
| | `helion/_compiler/cute/cute_mma.py:2263-2268` — `tcgen05_static_full_tma_fast_path` AOT 快速路径概念 |
| | `helion/_compiler/node_masking.py:99-167` — `remove_unnecessary_masking()` mask 去重 pass |
| | `helion/_compiler/compile_environment.py:1037-1040` — `known_multiple()` 编译期整除性检测 |
| | `helion/_compiler/device_ir.py` — Device IR 架构、reduction rolling |
| **修改与优化** | 所有参考均为设计层面（算法思路、架构模式），在 NineToothed (Python) 中独立实现。具体： |
| | - Helion 的 `known_multiple()` 是基于 FX Graph + CompileEnvironment 的符号环境，NineToothed 使用 AST 层面的 `_try_get_constant_int()` |
| | - Helion 的 `_is_size_one()` 依赖 `env.known_equal(size, 1)` 符号推理，NineToothed 使用 AST 节点模式匹配 (`Symbol(shape) == 1` 或 `Constant(0) * expr`) |
| | - Helion 的 `_PointerLoadContiguity` 有完整的 allowlist/blocklist 过滤 + `_max_run()` 求最大连续运行长度，NineToothed 仅覆盖基础 1D tile 场景 |
| | - Helion 的 AOT 使用 CuTe 后端 TMA 快速路径，NineToothed 使用 Triton 单后端，通过 per-variant CodeGenerator 调用实现 AOT 变体感知 |
| **其他说明** | 详细分析见赛题报告 Appendix C |

---

## 参考资源 2: NineToothed (InfiniTensor 基线)

| 项目 | 详情 |
|------|------|
| **名称** | NineToothed 基线代码 |
| **链接** | https://github.com/InfiniTensor/ninetoothed (commit a1b0694) |
| **协议** | Apache 2.0 |
| **参考内容** | 赛题指定基线版本。所有修改在此基线之上进行 |
| **修改与优化** | 详见赛题报告 §3 "修改的文件" |

---

## 参考资源 3: Triton 语言文档

| 项目 | 详情 |
|------|------|
| **名称** | Triton DSL 参考文档 |
| **链接** | https://triton-lang.org/ |
| **协议** | MIT |
| **参考内容** | `tl.max_contiguous` API 语义、`tl.load`/`tl.store` mask 参数处理、`tl.make_block_ptr` 用法 |
| **修改与优化** | 用于理解 Triton 语言构造的语义，确保生成的代码符合 Triton 规范 |

---

## 参考资源 4: OpenAI Triton 编译器

| 项目 | 详情 |
|------|------|
| **名称** | Triton |
| **链接** | https://github.com/triton-lang/triton |
| **协议** | MIT |
| **参考内容** | `triton.tools.compile`、`triton.runtime.JITFunction` API 接口 |
| **修改与优化** | NineToothed 本身的代码生成目标后端，非直接修改对象 |

---

## 参考资源 5: SymPy

| 项目 | 详情 |
|------|------|
| **名称** | SymPy — Python 符号数学库 |
| **链接** | https://github.com/sympy/sympy |
| **协议** | BSD-3-Clause |
| **参考内容** | `sympy.logic.simplify_logic`、`sympy.simplify` — 用于 `_generate_autotune` 中的不等式验证 |
| **修改与优化** | 现有功能，非新引入。v0.0.1 修复了 `inequalities.free_symbols` 在有缺失符号时的崩溃 (Appendix E) |

---

## 辅助工具声明

本赛题实现过程中使用了以下工具辅助开发和调试：

| 工具 | 用途 | 说明 |
|------|------|------|
| Python `ast` 模块 | AST 解析、遍历、变换 | 标准库，`CodeGenerator` 的核心依赖 |
| Python `inspect` 模块 | 源码提取和注解获取 | 标准库，`_get_tree()` 和 `_context` 的依赖 |
| `subprocess` (nvcc) | AOT CUDA kernel 编译 | 现有工具链，九齿 AOT 路径已使用 |
| pytest | 测试框架 | 现有框架，所有可见测试和隐藏测试的依赖 |
| matplotlib / jupyter | 可视化分析和教程 notebook | 仅用于教程和内部分析，非代码产物 |

---

## AI 辅助使用声明

本赛题提交的**全部代码和报告均为 AI（Kilo/Claude）生成**。参赛者（孙博楷）的角色为：

| 角色 | 具体工作 |
|------|---------|
| 思路设计 | 设计特化方案的整体方向、四类特化的边界条件、fallback 策略。分析 Helion 源码后提出可迁移到 NineToothed 的具体优化点。 |
| 代码审查 | 审查 AI 生成的所有代码修改的正确性、边界条件和安全性。逐一验证每个特化条件的触发和回退逻辑。 |
| 正确性验证 | 运行全量测试（196 例）、benchmark 验证、人工检查 generated source。 |
| 报告审查 | 审查 AI 生成的赛题报告内容的技术准确性，确保与代码实现一致。 |

AI 工具（Kilo/Claude）负责：

| 类型 | 详情 |
|------|------|
| 代码生成 | **全部代码修改**（`generation.py`、`aot.py`、`tensor.py`、`test_specialization.py`、`benchmark_specialization.py`）均由 AI 生成，人工审查后确认。 |
| 代码分析 | 阅读和分析 Helion 源码（`/data/helion`），生成 Helion 参考分析（赛题报告 Appendix C）。 |
| 报告撰写 | **赛题报告全文**由 AI 生成，包括结构设计、技术描述、表格和代码引用。 |
| 测试与调试 | AI 分析测试失败原因、提供修复建议并生成修复代码。 |
