---
name: ntops-copilot
description: >-
  NineToothed ntops operator copilot: one-command run_task, env doctor, ntops-native
  premake+element_wise scaffolds, formula cookbook, preflight, pytest and contest PR.
  Use for ntops, NineToothed, 九齿, 算子开发, premake, element_wise, CUDA pytest,
  2026-spring PR, InfiniCore.
---

# ntops-copilot

让 Agent **更快、更准** 完成 ntops 九齿算子开发。核心不是堆文档，而是：

1. **对齐 ntops 真实写法**（`premake` + `element_wise.arrangement`）
2. **一条命令开工**（`run_task.py`）
3. **任务卡驱动公式注入**（YAML `formula_hint` → 骨架里已写好 `application()`）
4. **闭环验证**（`verify_task` / `compare_ref` / `register_op`）
5. **公式速查 + 自检护栏**（少返工、少写成 Triton）

## 何时启用

`ntops`、`NineToothed`、`九齿`、`算子`、`premake`、`kernels/`、`pytest`、`2026-spring` PR。

## 30 秒上手（人类/Agent 都适用）

```bash
# 0) 环境自检（GPU 机先 source conda）
python scripts/doctor.py

# 1) 一条命令：读任务卡 -> 注入公式 -> 生成 kernel/torch -> preflight
python scripts/run_task.py --task silu --ntops-root /path/to/ntops --contest-id T1-1-X

# 2) 一键完工：验证 + 记 A/B（推荐演示/提交）
python scripts/run_task.py --task add --ntops-root /path/to/ntops --finish

# 3) 新算子：自动注册 + 全链路验证
python scripts/run_task.py --task gelu --ntops-root /path/to/ntops --force --register --verify

# 4) 交 PR 前严格自检
python scripts/preflight.py src/ntops/kernels/silu.py --kernel --strict
python scripts/compare_ref.py src/ntops/kernels/silu.py --ref src/ntops/kernels/silu.py
```

GPU 云（如 SeetaCloud）常见要先：
```bash
source /root/miniconda3/bin/activate base
```

## ntops 真实范式（必须优先）

ntops **不是** ninetoothed-examples 那种顶层 `ninetoothed.make(...)`。

标准结构：

```python
from ntops.kernels.element_wise import arrangement

def application(input, output):
    output = ...  # noqa: F841

def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))
    return arrangement_, application, tensors
```

torch 封装：

```python
kernel = _cached_make(ntops.kernels.<op>.premake, input.ndim)
kernel(input, output)
```

生成骨架时用：`--style ntops`（默认）。

## 包结构

| 目录 | 内容 |
|------|------|
| `SKILL.md` | 本文件 |
| `tasks/` | 任务卡 YAML |
| `examples/` | `add_walkthrough.md` |
| `references/` | 查阅索引 |
| `tests/` | 验证说明 |

双 Skill 边界见 `docs/DualSkillGuide.md`。

## 标准工作流

### Step 0 — doctor

```bash
python scripts/doctor.py
```

确认：`ninetoothed`、`torch`、`CUDA`（跑 pytest 需要）、`ntops` 是否就绪。

### Step 1 — 选任务

读 `tasks/task_<op>.yaml`，或让用户给算子名 + 公式。

### Step 2 — run_task（推荐）

```bash
python scripts/run_task.py --task <op> --ntops-root <ntops> --contest-id <赛题号>
```

自动生成：
- `src/ntops/kernels/<op>.py`
- `src/ntops/torch/<op>.py`（若给了 ntops-root）

### Step 3 — 只改 application()

查 `formulas.md`，对照 `reference_kernel` 链接。**不要改 premake 结构。**

### Step 4 — 注册 + 验证

```bash
python scripts/register_op.py --name <op> --ntops-root <ntops>
python scripts/verify_task.py --name <op> --ntops-root <ntops> --pytest
python scripts/preflight.py src/ntops/kernels/<op>.py --kernel --strict
```

有官方参考实现时，用 `compare_ref.py` 对比 `application()` 是否对齐。

### Step 5 — PR

- 分支：`2026-spring-hongwei-2026-<赛题号>`
- 标题：`[2026春季][赛题号] hongwei-2026`
- 模板：`templates/PR_DESCRIPTION.md`

## 硬性原则

1. **正确性 > 性能**
2. **禁止 Triton 交卷**：发现 `@triton.jit` 必须改 NineToothed
3. **禁止抄袭**：更新 `REFERENCE.md`
4. **preflight 不过禁止 PR**

## 常见错误

| 现象 | 处理 |
|------|------|
| `can't open file '/root/scripts/...'` | 先 `cd /root/work/skill`（脚本不在 `$HOME`） |
| `conv2d` 等全量 pytest 失败 | **不要** `pytest tests/` 全跑；用任务卡 `pytest_file` |
| 写了 examples 风格 `make` | 改用 `--style ntops` 重新 scaffold |
| pytest 全 skipped | 无 CUDA；换 GPU 机并 `doctor` 确认 |
| `python3 not found` | 用 conda：`source .../activate base && python` |
| application 公式错 | 查 `formulas.md` + `compare_ref.py` |

## v0.5 改进

| 能力 | 说明 |
|------|------|
| **精准 pytest** | 任务卡 `pytest_file`，避免 `-k add` 误匹配 |
| **`--finish`** | `run_task` 一步：scaffold → verify → record_run |
| **`list_tasks`** | 列出全部任务卡与对应测试文件 |

## v0.4 创新点（可演示）

| 能力 | 脚本 | 价值 |
|------|------|------|
| **任务卡→代码** | `run_task` + `formula_hint` | Agent 不必猜公式，少幻觉 |
| **语义对照** | `compare_ref.py` | AST 对比 `application()` 与官方实现 |
| **闭环验收** | `verify_task.py` | preflight + 注册 + pytest 一步完成 |
| **可测 A/B** | `record_run.py` + `eval_ab.py` | 跑完自动记 CSV，报告可复现 |

## 脚本

| 脚本 | 作用 |
|------|------|
| `run_task.py` | **一键任务流**（`--finish` / `--register` / `--verify`） |
| `list_tasks.py` | 列出任务卡与 pytest 文件 |
| `verify_task.py` | 注册检查 + preflight + 可选 pytest |
| `register_op.py` | 幂等写入 `__init__.py` |
| `compare_ref.py` | `application()` 与参考实现对齐检查 |
| `record_run.py` | 记录 A/B 跑数到 CSV |
| `doctor.py` | 环境/CUDA/依赖自检 |
| `scaffold_kernel.py` | 生成 kernel（`--formula` 注入） |
| `scaffold_torch.py` | 生成 torch 封装 |
| `preflight.py` | 拦截 Triton/结构/TODO（`--strict`） |
| `eval_ab.py` | A/B 评测汇总 |

## 资源

- `formulas.md` — 公式速查
- `reference.md` — API 与目录
- `examples.md` — silu 端到端
- `tasks/*.yaml` — 任务卡
