# NineToothed 算子开发 Skill

> 2026 春季启元 AI 大赛 — 九齿 .skill 创新挑战赛道（T3-1-1）

## 概述

本 Skill 指导 AI 智能体完成 NineToothed GPU 算子开发的完整闭环：从 CPU 参考实现出发，经过分析分类、代码生成、编译调优、精度验证、性能优化，到最终输出报告。

支持 9 种 Arrangement 模式（Element-wise、Reduction、Matmul、BMM、Addmm、Conv2D、Attention、Pooling、RoPE），覆盖 NineToothed 生态的主要算子类型。

## 目录结构

```
ninetoothed-skill/
├── SKILL.md                          # 主 Skill 文件（AI 智能体入口）
├── README.md                         # 本文件
├── references/                       # 参考资料与模式说明
│   ├── code_templates.md             # 9 种 Arrangement 模式的完整代码模板
│   ├── ntl_api.md                    # ntl（ninetoothed.language）API 参考
│   ├── tensor_guide.md               # Tensor 声明与元操作参考
│   └── pitfalls.md                   # 15 种常见陷阱及修复方法
├── scripts/                          # 可执行脚本
│   ├── run_tests.py                  # 运行所有自测算子验证
│   ├── run_benchmark.py              # 性能基准测试（ntops vs PyTorch）
│   └── verify_skill.py              # Skill 自身有效性验证
├── examples/                         # 完整开发示例 + benchmark 文档
│   ├── 01_elementwise_leaky_relu.md  # Element-wise + 标量参数（5 次迭代修复）
│   ├── 02_reduction_log_softmax.md   # Reduction + 数值稳定性
│   ├── 03_binary_add.md              # Binary 算子 + 运行时标量参数
│   ├── 04_broadcast_meshgrid.md      # 1D→2D 广播 + 自定义 arrangement
│   ├── 01_benchmark_leaky_relu.md    # 自测任务 1 性能对比详细记录
│   └── 02_benchmark_log_softmax.md   # 自测任务 2 性能对比详细记录
└── tests/                            # 6 个自测算子验证任务
    ├── test_leaky_relu.py            # 任务 1：Element-wise 算子（leaky_relu）
    ├── test_log_softmax.py           # 任务 2：Reduction 算子（log_softmax）
    ├── test_non_contiguous.py        # 任务 3：非连续输入/步长场景
    ├── test_benchmark.py             # 任务 4：性能对比与回退分析
    ├── test_meshgrid.py              # 任务 5：广播算子（meshgrid）
    └── test_deg2rad.py              # 任务 6：Element-wise 算子（deg2rad）
```

## 安装

### 方式 1：作为 Claude Code Skill 安装

将本目录复制到 Claude Code 的 skills 目录：

```bash
cp -r ninetoothed-skill/ ~/.claude/skills/ninetoothed-skill/
```

或创建软链接：

```bash
ln -s $(pwd)/ninetoothed-skill ~/.claude/skills/ninetoothed-skill
```

### 方式 2：直接提供 SKILL.md

将 `SKILL.md` 的内容作为系统提示或上下文提供给 AI 智能体。

## 依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| Python | >= 3.10 | 运行环境 |
| PyTorch | >= 2.0 | CPU 参考 + torch 层 |
| Triton | >= 3.0 | GPU kernel 编译后端 |
| NineToothed | 本地开发版 | DSL 框架 |
| ntops | 本地开发版 | 算子库 |

## 使用方法

### 1. 基本用法：提供 CPU 参考实现

向 AI 智能体提供一个 CPU 参考实现：

```python
import numpy as np

def leaky_relu(x, negative_slope=0.01):
    return np.where(x >= 0, x, negative_slope * x)
```

Skill 会自动完成：分析 → 分类 → 生成 → 编译 → 验证 → 优化 → 报告。

### 2. 运行自测验证

```bash
# 运行全部自测
python ninetoothed-skill/scripts/run_tests.py

# 运行性能基准测试
python ninetoothed-skill/scripts/run_benchmark.py
```

### 3. 环境配置

确保以下依赖已安装并可用：

```bash
# NineToothed 和 ntops 需要以开发模式安装
cd $PROJECT_ROOT/NineToothed && pip install -e .
cd $PROJECT_ROOT/ntops && pip install -e .
```

验证环境：
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import triton; print(f'Triton: {triton.__version__}')"
python -c "import ninetoothed; print('NineToothed: OK')"
python -c "import ntops; print('ntops: OK')"
```

## 自测算子任务

本 Skill 包含 6 个自测算子开发任务，覆盖赛题全部要求：

| # | 任务 | 类型 | 测试文件 | Benchmark |
|---|------|------|------|:---:|
| 1 | leaky_relu | Element-wise + 标量参数 | `tests/test_leaky_relu.py` | [链接](examples/01_benchmark_leaky_relu.md) |
| 2 | log_softmax | Reduction + 数值稳定性 | `tests/test_log_softmax.py` | [链接](examples/02_benchmark_log_softmax.md) |
| 3 | 非连续输入 | 转置/步幅/偏移 | `tests/test_non_contiguous.py` | — |
| 4 | 性能对比 | ntops vs PyTorch 基准测试 | `tests/test_benchmark.py` | — |
| 5 | meshgrid | 1D→2D 广播 + 自定义 arrangement | `tests/test_meshgrid.py` | — |
| 6 | deg2rad | Element-wise + constexpr 常量 | `tests/test_deg2rad.py` | — |

## 开发示例

| # | 示例 | 类型 | 文件 |
|---|------|------|------|
| 1 | leaky_relu | Element-wise Unary + constexpr 标量 | `examples/01_elementwise_leaky_relu.md` |
| 2 | log_softmax | Reduction + 数值稳定性 | `examples/02_reduction_log_softmax.md` |
| 3 | add | Element-wise Binary + 运行时标量 | `examples/03_binary_add.md` |
| 4 | meshgrid | 1D→2D 广播 + 自定义 arrangement | `examples/04_broadcast_meshgrid.md` |

## 验证方式

### Skill 自身验证

验证 .skill 包本身的完整性和语法正确性（无需 GPU）：

```bash
cd $PROJECT_ROOT
python ninetoothed-skill/scripts/verify_skill.py
```

### 算子精度验证
```bash
cd $PROJECT_ROOT/ntops
python -m pytest tests/ -v
```

### 性能验证
```bash
cd $PROJECT_ROOT
python ninetoothed-skill/scripts/run_benchmark.py
```

> **平台说明**：Windows 用户如遇到 OpenMP 冲突，需在命令前添加 `KMP_DUPLICATE_LIB_OK=TRUE`。
> Linux 环境通常不需要此设置。

## 已验证行为

- **非连续输入**：NineToothed 自动处理 stride 信息，无需 `.contiguous()` 转换
- **标量参数传递**：`Tensor(0, constexpr=True, value=...)` 是传递标量系数的推荐方式
- **闭包限制**：application 函数内禁止引用外部变量，所有参数必须通过函数参数或 constexpr 传入
- **float16 精度**：中间计算（exp、log）必须提升到 float32

## 设计特点

1. **闭环工作流**：分析 → 生成 → 编译 → 验证 → 优化 → 报告，六阶段不跳过
2. **精确错误恢复**：12 种错误类型的诊断-修复映射表
3. **实战验证**：所有模板和规则均经过实际算子开发迭代验证
4. **静态检查优先**：15 项静态验证清单在运行测试前捕获大部分问题
5. **终止保障**：3 种终止条件防止无限迭代
