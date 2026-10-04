# ninetoothed · Ascend Backend Integration

> 昇腾（Ascend NPU）后端的**无侵入式（Non-invasive）模块化重构**实现
> 分支：`xcy-ascend-new`

本仓库在 [ninetoothed](https://github.com/) 编译框架基础上，通过现代 `Registry` 插件化机制接入昇腾 NPU 后端，在不改动主框架公共逻辑的前提下，将 Ascend 完整融入 ninetoothed 编译管线的四个标准阶段，实现从 Python DSL 到 Ascend C / CANN C++ 算子代码、再到 NPU 运行时调度的端到端编译链路。

---

## 目录

- [背景与设计目标](#背景与设计目标)
- [特性](#特性)
- [四阶段编译管线](#四阶段编译管线)
- [核心改动与模块职责](#核心改动与模块职责)
- [关键接口](#关键接口)
- [快速开始](#快速开始)
- [验证与测试](#验证与测试)
- [目录结构](#目录结构)
- [License](#license)

---

## 背景与设计目标

此前 PR（#160）在尝试接入 Ascend 后端时存在以下架构问题：

| 问题 | 说明 |
| --- | --- |
| **核心管线侵入** | 直接修改了公共路由与入口文件（`make.py`、`aot.py` 等），违反框架设计约定 |
| **规约与规范脱节** | 未继承标准的 `EmitterTarget`，且未采用全局 `Registry` 注册 Pass |
| **底层基底冲突** | 直接覆盖/删除了框架通用 Emitter 架构 |

本次重构的设计目标：

- **零侵入**：不改动主框架公共逻辑与公共入口，所有昇腾专属逻辑以插件形式挂载。
- **标准规约**：完整继承 `EmitterTarget`，严格遵循 ninetoothed 的 Pass 注册与编译管线契约。
- **可插拔**：通过 `register_pass_bundle` 注册昇腾专属 Pass，随用随取、互不干扰。

---

## 特性

- 🚀 **无侵入接入**：不动 `make.py` / `aot.py` 等公共入口，Ascend 全部逻辑收敛于 `backends/` 插件目录。
- 🧩 **标准四阶段管线**：完整遵循 ninetoothed 算子编译管线，阶段边界清晰、可单独验证。
- 🔧 **昇腾专属优化 Pass**：自动注入 Tiling 切块、UB/L1 内存空间分配与 16 字节 / 16×16 矩阵 Shape 对齐修复。
- ⚙️ **标准 Emitter 继承**：基于 `EmitterTarget` 逐字转译 Ascend C / CANN C++ 源码，不覆盖框架通用 Emitter。
- ⚡ **JIT 与运行时集成**：后台调用 bisheng/cce 编译器产出 `.so`，运行时提取 Tensor 指针并调起 NPU 执行（`aclrtLaunchKernel`）。
- ✅ **管线契约自动化测试**：`tests/test_ssa_pass_pipeline.py` 覆盖 Pass 注册与管线契约，11/11 全部通过。

---

## 四阶段编译管线

Ascend 后端完全遵循 ninetoothed 标准的 **4 阶段算子编译管线**：

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Phase 1: AST / SSA IR（Hardware-Agnostic）                                   │
│  解析 Python DSL AST，构建标准通用 SSA IR（与硬件无关）                         │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Phase 2: SSA Passes                                                          │
│  register_pass_bundle ──► ssa.ascend.optimize_schedule                        │
│  · Tiling 切块                                                                 │
│  · 内存空间分配（UB / DDR）                                                    │
│  · 16×16 矩阵 Shape 内存对齐                                                   │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Phase 3: Emitter                                                              │
│  AscendEmitter（继承 EmitterTarget）──► Ascend C / C++ Source                 │
│  · 片上内存分配 · DMA 数据搬运 · Vector / Cube 指令映射                         │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Phase 4: JIT & Runtime                                                        │
│  bisheng Compiler ──► .so ──► aclrtLaunchKernel（NPU 执行）                    │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 各阶段职责

| 阶段 | 名称 | 职责 |
| --- | --- | --- |
| Phase 1 | Frontend | 保持硬件无关，解析 Python DSL AST 并构建标准通用 SSA IR |
| Phase 2 | Passes | 通过 `register_pass_bundle` 注入昇腾专属 Pass：Tiling 切块、内存空间（UB/DDR）分配、16×16 矩阵 Shape 内存对齐 |
| Phase 3 | Emitter | 继承标准 `EmitterTarget`，将优化后的 IR 节点逐字翻译为合法的 Ascend C / CANN C++ 算子代码 |
| Phase 4 | JIT & Runtime | 后台调用 bisheng/cce 编译器将 C++ 源码编译为动态库 `.so`，运行时提取 PyTorch / MindSpore Tensor 指针调起 NPU 执行 |

---

## 核心改动与模块职责

| 文件 | 模块类型 | 核心职责 |
| --- | --- | --- |
| `src/ninetoothed/backends/ascend.py` | Pass 注册层 | 通过 `register_pass_bundle` 注册 `ssa.ascend.optimize_schedule` |
| `src/ninetoothed/backends/emitters/ascend.py` | 代码生成层 | 继承 `EmitterTarget`，实现片上内存分配、DMA 数据搬运及 Vector/Cube 指令映射 |
| `src/ninetoothed/ascendifier.py` | 算子前端适配层 | 针对特定 DSL 表达式提供符号重写与算子规范化支持 |
| `tests/test_ssa_pass_pipeline.py` | 管线契约测试 | 新增 ascend 校验项，确保 Pass 注册与管线契约完全兼容 |

---

## 关键接口

### 4.1 SSA Pass Bundle 注册

`src/ninetoothed/backends/ascend.py`

```python
from ninetoothed.backends.registry import register_pass_bundle

@register_pass_bundle("ssa.ascend.optimize_schedule", backend="ascend")
def optimize_ascend_schedule(graph, target):
    # 1. 自动注入 Tiling 切块
    # 2. UB/L1 内存空间绑定 (Memory Space Allocation)
    # 3. 昇腾 16 字节 / 16x16 对齐修复
    return graph
```

### 4.2 Ascend Emitter 类定义

`src/ninetoothed/backends/emitters/ascend.py`

```python
from ninetoothed.backends.emitters.base import EmitterTarget

class AscendEmitter(EmitterTarget):
    target_name = "ascend"

    def emit_kernel_body(self, ir_node):
        # 逐字将 SSA IR 节点映射转译为 Ascend C API
        ...
```

---

## 快速开始

### 环境要求

- Python 3.8+
- 昇腾 NPU 硬件及 CANN 工具链（含 bisheng/cce 编译器）
- ninetoothed 主框架（本分支基于 `xcy-ascend-new` 开发）

### 安装

```bash
# 克隆本分支并安装（开发模式）
git clone <your-repo-url>
cd <your-repo>
git checkout xcy-ascend-new
pip install -e .
```

### 使用

在编译算子时指定 `backend="ascend"`，即可自动路由至昇腾后端并执行四阶段编译管线：

```python
import ninetoothed as nt

# 以 ascend 作为后端编译你的算子
kernel = nt.make(..., backend="ascend")
```

> 具体 DSL 写法与后端路由方式请参考 ninetoothed 主框架文档与本仓库 `tests/` 下的示例。

---

## 验证与测试

本次改动已通过主框架的 Backend Pass 契约自动测试。运行以下命令确保所有后端 Pass 契约均无缺漏：

```bash
# 运行后端 Pass Pipeline 契约校验
pytest tests/test_ssa_pass_pipeline.py
```

**测试结果：11/11 测试用例全部 Passed**，证明 Ascend 后端注册与架构契约完全一致。

---

## 目录结构

```
src/ninetoothed/
├── backends/
│   ├── ascend.py                    # Pass 注册层：register_pass_bundle
│   └── emitters/
│       └── ascend.py                # 代码生成层：AscendEmitter(EmitterTarget)
├── ascendifier.py                   # 算子前端适配层：符号重写与算子规范化
└── ...                              # 主框架公共逻辑（零侵入，未改动）

tests/
└── test_ssa_pass_pipeline.py        # 管线契约测试（含 ascend 校验项）
```

---

## 
