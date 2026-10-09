# NineToothed Ascend 后端适配与 AOT 支持说明

> 昇腾（Ascend NPU）后端的**无侵入式（Non-invasive）模块化实现**
> 分支：`xcy-ascend-new`

NineToothed 是一个基于 Triton 的 DSL/编译框架。本分支在 `ninetoothed` 编译框架基础上，通过现代 `Registry` 插件化机制接入昇腾 NPU 后端，在不改动主框架公共逻辑的前提下，将 Ascend 完整融入 ninetoothed 编译管线的各阶段，实现**从 Python DSL、SSA IR、代码生成到 NPU 执行、AOT 产物回载**的端到端编译链路。

---

## 目录

- [项目背景与设计目标](#1-项目背景与设计目标)
- [功能概述](#2-功能概述)
- [后端注册](#3-后端注册)
- [平台配置](#4-平台配置)
- [AscendEmitter 与代码生成](#5-ascendemitter-与代码生成)
- [JIT 执行流程](#6-jit-执行流程)
- [AOT 支持](#7-aot-支持)
- [测试与验证](#8-测试与验证)
- [已知限制](#9-已知限制)
- [文件变更说明](#10-文件变更说明)
- [快速开始](#11-快速开始)
- [目录结构](#12-目录结构)
- [总结](#13-总结)
- [License](#license)

---

## 1. 项目背景与设计目标

### 1.1 背景

此前 PR（#160）在尝试接入 Ascend 后端时存在以下架构问题：

| 问题 | 说明 |
| --- | --- |
| **核心管线侵入** | 直接修改了公共路由与入口文件（`make.py`、`aot.py` 等），违反框架设计约定 |
| **规约与规范脱节** | 未继承标准的 `EmitterTarget`，且未采用全局 `Registry` 注册 Pass |
| **底层基底冲突** | 直接覆盖/删除了框架通用 Emitter 架构 |

### 1.2 设计目标

本次重构的设计目标：

- **零侵入**：不改动主框架公共逻辑与公共入口，所有昇腾专属逻辑以插件形式挂载。
- **标准规约**：完整继承 `EmitterTarget`，严格遵循 ninetoothed 的 Pass 注册与编译管线契约。
- **可插拔**：通过 `register_pass_bundle` 注册昇腾专属 Pass，随用随取、互不干扰。

### 1.3 架构演进

实现经历两代演进：初版采用 bisheng/cce 编译器直出 Ascend C / CANN C++ 算子代码；第二版（当前版本）改为**复用通用 SSA/Triton emitter** 生成 Triton Python 源码，经 `Ascendifier` 转换后交由 `triton-ascend` 编译器生成 NPU 可执行产物，并在此基础上补齐 **AOT 构建与回载**能力，形成完整闭环。

---

## 2. 功能概述

NineToothed 现在支持 **Ascend NPU 后端**，能力覆盖：

- **Ascend 后端注册**：通过全局 `Registry` 注册 `Target.ASCEND` 与默认后端 `AscendBackend`，随 `backend="ascend"` 自动路由。
- **Ascend 平台识别**：识别 `ascend-910b3` / `ascend-910b4` 等平台，解析设备类型、compute architecture 与能力约束。
- **AscendEmitter 代码生成**：复用共享 SSA/Triton emitter，经 `Ascendifier` 转换生成 `.ascend_triton.py` 源文件与 JSON 元数据清单。
- **Ascend JIT 执行**：`triton-ascend` 编译 + CANN/NPU 运行时调度，完成 NPU 上的即时编译与执行。
- **Ascend AOT 构建与回载**：`aot_build()` 产出设备二进制等 AOT 产物，`load_built_artifact()` 支持跨进程回载，回载不触发重新编译。
- **NPU 数值正确性验证**：在真实 Ascend 910B4 硬件上完成 JIT 与 AOT 数值测试。

---

## 3. 后端注册

Ascend 后端通过插件化 `Registry` 机制注册，不侵入主框架公共入口：

- **目标枚举**：新增 `Target.ASCEND`，作为 Ascend 后端的标准目标标识。
- **默认后端注册表**：注册表加入 `AscendBackend`，默认后端集合现包含 **Triton、CUDA、TileLang、Ascend** 四类。
- **SSA Pass**：注册昇腾专属调度 Pass `ssa.ascend.optimize_schedule`，负责 Ascend 目标上的 schedule 优化（Tiling 切块、内存空间分配、Shape 对齐等）。
- **Capability 信息**：Ascend 后端声明自身能力（capability），包括支持的平台、compute architecture 以及受限能力项，供上层进行能力查询与降级判断。

Pass 注册示例：

```python
from ninetoothed.backends.registry import register_pass_bundle

@register_pass_bundle("ssa.ascend.optimize_schedule", backend="ascend")
def optimize_ascend_schedule(graph, target):
    # 1. 自动注入 Tiling 切块
    # 2. UB / L1 内存空间分配
    # 3. Shape 对齐修复
    return graph
```

---

## 4. 平台配置

### 4.1 支持的平台

| 平台标识 | 设备类型 | compute architecture | JIT | AOT |
| --- | --- | --- | --- | --- |
| `ascend-910b3` | `npu` | `ascend910b3` | ✅ | ✅ |
| `ascend-910b4` | `npu` | `ascend910b4` | ✅ | ✅ |

平台配置内容：

- **设备类型**：`npu`
- **执行能力**：同时支持 Ascend JIT 与 AOT
- **compute architecture**：`ascend910b3` / `ascend910b4`
- **能力限制**：对不支持或受限的能力显式声明，例如：
  - `math.pow`（部分场景不支持，需改写或降级）
  - 部分 FP8 能力受限

### 4.2 示例

```python
from ninetoothed.targets import resolve_target_context

context = resolve_target_context(
    "ascend",
    platform="ascend-910b4",
)
```

---

## 5. AscendEmitter 与代码生成

### 5.1 生成策略

`AscendEmitter` 复用通用 SSA/Triton emitter，避免重复实现基础代码生成逻辑：

1. 使用**共享 SSA emitter** 生成基础 Triton Python 源码；
2. 通过 **`Ascendifier`** 将通用 Triton 语法转换为 Ascend 专用语法；
3. 生成 `.ascend_triton.py` 源文件；
4. 生成 **JSON 元数据清单**。

产物保留的关键信息：

- **kernel entrypoint**：算子内核入口符号
- **launch ABI**：运行时 launch 所需的 ABI 约定（参数布局、绑定方式）
- **target metadata**：目标平台、compute architecture 等元数据

### 5.2 Ascendifier 支持的转换

| 转换项 | 说明 |
| --- | --- |
| `triton.language.float64` | 转换为 `float32`（Ascend 硬件对 fp64 支持有限） |
| `tl.load(..., other=None)` | 转换为 `other=0.0`（边界加载的默认填充值） |
| `tl.clamp(x, min, max)` | 转换为 `minimum(maximum(x, min), max)` |
| `triton.language.extra.libdevice` | 转换为 Ascend 对应的 libdevice 模块 |
| autotune key | 进行限制与规范化（仅保留 Ascend 支持的 key，并对取值规范化） |

---

## 6. JIT 执行流程

### 6.1 完整流程

```
Python DSL
  -> SSA Program
  -> Ascend target lowering
  -> AscendEmitter
  -> Ascend Triton Python source
  -> triton-ascend compiler
  -> CANN/NPU runtime
  -> Ascend NPU execution
```

### 6.2 运行时支持

Ascend 运行时（`compiler/runtime.py`）提供以下能力：

- **NPU tensor 参数校验**：校验入参为合法 NPU tensor
- **dtype 校验**：校验参数 dtype 与 kernel 期望一致
- **shape/stride 校验**：校验 shape 与 stride 布局
- **NPU device 类型识别**：识别当前 NPU 设备型号
- **runtime launch ABI 封装**：封装底层 launch ABI，屏蔽 CANN 细节
- **NPU stream 调度**：在 NPU stream 上调度内核执行

### 6.3 示例

```python
import torch
import ninetoothed

@ninetoothed.jit(
    backend="ascend",
    platform="ascend-910b4",
)
def add_kernel(x, y, out):
    out = x + y

x = torch.randn(1024, device="npu")
y = torch.randn_like(x)
out = torch.empty_like(x)

add_kernel(x, y, out)
torch.npu.synchronize()
```

---

## 7. AOT 支持

### 7.1 核心接口

- **`AscendMaterializer.aot_build()`**：在构建期将内核编译为 AOT 产物（设备二进制 + launcher + 运行时工具扩展 + 元数据）。
- **`AscendMaterializer.load_built_artifact()`**：从已构建产物直接加载，**禁止回载时重新编译**。

构建过程包括：

- **设备二进制生成**：`kernel.bin`
- **launcher 扩展生成**：`launcher.so`
- **NPU utility 扩展保存**：`npu_utils.so`
- **`bundle.json` 元数据清单**：记录内核、目标与校验信息

加载时的完整性与兼容性校验：

- **SHA256 文件完整性校验**：逐文件校验产物哈希，防止损坏/篡改
- **Python 版本校验**：构建与加载环境的 Python 主版本必须一致
- **Ascend 设备型号校验**：加载环境设备型号必须与构建目标匹配
- **跨进程加载**：AOT 产物可在不同进程间直接加载复用
- **禁止回载时重新编译**：回载只做加载与绑定，不触发任何编译动作

### 7.2 产物结构

```
<output>/
├── kernel.bin
├── launcher.so
├── npu_utils.so
├── launch.py
└── bundle.json
```

### 7.3 示例

```python
import ninetoothed

handle = ninetoothed.aot(
    application,
    backend="ascend",
    platform="ascend-910b4",
    output_dir="./build",
)

reloaded = ninetoothed.load_built_artifact(
    handle._built_artifact
)
```

---

## 8. 测试与验证

### 8.1 测试覆盖

已执行以下测试：

- Ascend 后端注册测试
- 平台配置测试
- SSA pass pipeline 测试
- AscendEmitter 源码生成测试
- Ascendifier AST 转换测试
- JIT NPU 数值测试
- AOT 构建测试
- AOT 回载测试
- 跨进程回载测试
- 产物损坏检测
- dtype 错误检测
- 大整数归约和尾部边界测试

### 8.2 真实硬件验证环境

- Ascend **910B4**
- **8 个 NPU 设备**
- PyTorch NPU 可用
- CANN **9.0**
- Python **3.10**
- triton-ascend **3.2.0**

### 8.3 测试结果

```
118 passed, 4 warnings
```

其中 AOT 专项测试：

```
4 passed
```

数值测试：

```
13 passed
```

---

## 9. 已知限制

- Ascend AOT 产物与**设备型号相关**
- AOT 产物与 **CANN 版本、Python 主版本**相关
- 不保证跨不同 Ascend 型号直接复用
- `triton-ascend 3.2.0` 与 CANN 9.0 存在一个**驱动枚举名称兼容问题**
- 当前测试使用了**临时依赖副本**修正该兼容问题
- 正式发布前应**固定兼容版本**，或将补丁提交到上游
- AOT 当前不支持自动跨设备迁移
- autotune 能力仍然比标准 Triton 后端有限

---

## 10. 文件变更说明

| 文件 | 模块类型 | 核心作用 |
| --- | --- | --- |
| `src/ninetoothed/backends/__init__.py` | 后端注册 | 初始化后端注册表，将 `AscendBackend` 纳入默认后端集合 |
| `src/ninetoothed/backends/ascend.py` | 后端定义 | 定义 `Target.ASCEND` 对应后端，注册 `ssa.ascend.optimize_schedule`，声明 capability |
| `src/ninetoothed/backends/emitters/ascend.py` | 代码生成 | `AscendEmitter`：复用共享 SSA emitter，生成 `.ascend_triton.py` 与元数据 |
| `src/ninetoothed/backends/materializers/ascend.py` | 产物构建 | `AscendMaterializer`：JIT 编译与产物构建（`kernel.bin` / `launcher.so` / `npu_utils.so` / `bundle.json`） |
| `src/ninetoothed/backends/materializers/ascend_aot.py` | AOT | AOT 构建与回载：`aot_build()` / `load_built_artifact()`、SHA256 校验、版本/型号校验、跨进程加载 |
| `src/ninetoothed/ascendifier.py` | 语法转换 | `Ascendifier`：AST 级 Triton→Ascend 转换（float64→float32、`other=0.0`、clamp、libdevice、autotune key） |
| `src/ninetoothed/compiler/runtime.py` | 运行时 | tensor/dtype/shape 校验、device 识别、launch ABI 封装、NPU stream 调度 |
| `src/ninetoothed/targets.py` | 目标/平台 | `Target.ASCEND` 与平台解析：`resolve_target_context`、compute architecture、能力限制 |
| `tests/test_ascend_backend.py` | 测试 | 后端注册、平台配置、SSA pass pipeline、Emitter 源码生成、Ascendifier AST 转换测试 |
| `tests/test_ascend_runtime.py` | 测试 | JIT NPU 数值测试、dtype 错误检测、大整数归约与尾部边界测试 |
| `tests/test_ascend_aot.py` | 测试 | AOT 构建、回载、跨进程回载、产物损坏检测 |

---

## 11. 快速开始

### 11.1 环境要求

- Python 3.10+（AOT 产物与 Python 主版本绑定）
- 昇腾 NPU 硬件与 CANN 工具链（验证环境：CANN 9.0）
- triton-ascend（验证环境：3.2.0）
- ninetoothed 主框架（本分支基于 `xcy-ascend-new` 开发）

### 11.2 安装

```bash
# 克隆本分支并安装（开发模式）
git clone <your-repo-url>
cd <your-repo>
git checkout xcy-ascend-new
pip install -e .
```

### 11.3 使用

指定 `backend="ascend"` 与 `platform` 即可路由至昇腾后端：

```python
import torch
import ninetoothed

@ninetoothed.jit(
    backend="ascend",
    platform="ascend-910b4",
)
def add_kernel(x, y, out):
    out = x + y

x = torch.randn(1024, device="npu")
y = torch.randn_like(x)
out = torch.empty_like(x)

add_kernel(x, y, out)
torch.npu.synchronize()
```

> 具体 DSL 写法与后端路由方式请参考 ninetoothed 主框架文档与本仓库 `tests/` 下的示例。

---

## 12. 目录结构

```
src/ninetoothed/
├── backends/
│   ├── __init__.py                    # 后端注册表：默认后端集合（Triton / CUDA / TileLang / Ascend）
│   ├── ascend.py                      # 后端定义：Target.ASCEND、SSA pass、capability
│   ├── emitters/
│   │   └── ascend.py                  # AscendEmitter：.ascend_triton.py 源码与元数据生成
│   └── materializers/
│       ├── ascend.py                  # AscendMaterializer：JIT 产物构建
│       └── ascend_aot.py              # AOT 构建与回载
├── ascendifier.py                     # Ascendifier：Triton → Ascend 语法转换
├── compiler/
│   └── runtime.py                     # Ascend 运行时：校验 / launch ABI / stream 调度
├── targets.py                         # Target.ASCEND 与平台解析
└── ...                                # 主框架公共逻辑（零侵入，未改动）

tests/
├── test_ascend_backend.py             # 后端注册 / 平台 / SSA pass / Emitter / Ascendifier 测试
├── test_ascend_runtime.py             # JIT NPU 数值与边界测试
└── test_ascend_aot.py                 # AOT 构建 / 回载 / 跨进程测试
```

---

## 13. 总结

NineToothed Ascend 后端已经完成从**后端注册、平台解析、SSA lowering、源码生成、NPU JIT 执行到 AOT 构建和跨进程回载**的完整闭环，并已在真实 **Ascend 910B4** 上通过数值正确性验证。

---

## License

遵循 ninetoothed 主框架的开源许可协议。
#（注：内容由AI生成）
