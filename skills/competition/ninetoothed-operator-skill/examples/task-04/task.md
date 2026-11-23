# Task 04: Benchmark / Debug

> **环境说明：Windows + NVIDIA RTX 5060 Laptop GPU (8GB) + PyTorch 2.12.0.dev20260408+cu128 + ninetoothed 0.26.0，benchmark 已采集，三种 case 均通过。**

为 Task 01 的 add 算子补充性能对比和失败诊断材料。

## 要求

- 文件：`benchmark_task.py`, `failure_diagnosis.md`
- benchmark 对比 `examples/task-01/add.py` 与 `torch.add`
- benchmark 覆盖 same-shape、vector broadcast、row broadcast 三种 case
- 输出正确性 sanity check、shape、dtype、元素数、ms/iter、性能比值
- 无 CUDA 时正常打印 skip 信息并退出
- 诊断文档需要模拟一个失败场景，记录现象、复现命令、定位过程、修复方式和验证方法
- 指导 AI 检查 generated source 与 AOT build（参见 `SKILL.md` §6.1 和 `docs/source/build.rst`）

## 验收命令

```bash
python skills/competition/ninetoothed-operator-skill/examples/task-04/benchmark_task.py --case same
python skills/competition/ninetoothed-operator-skill/examples/task-04/benchmark_task.py --case vector
python skills/competition/ninetoothed-operator-skill/examples/task-04/benchmark_task.py --case row
```

## AI 执行记录摘要

| 项目 | 内容 |
|------|------|
| 开发环境 | Windows 11 + NVIDIA RTX 5060 Laptop GPU (8GB) + PyTorch 2.12.0.dev20260408+cu128 |
| benchmark 脚本 | 已实现 same/vector/row 三种 case |
| 失败诊断 | `failure_diagnosis.md` 记录 row broadcast expand 失败场景 |
| 本地 benchmark 结果 | **已采集** — same 0.96x、vector 1.004x、row 1.015x，均正确 |

### Benchmark 设计

| 项目 | 值 |
|------|-----|
| 默认 shape | `(4096, 4096)` |
| baseline | `torch.add` |
| case | same、vector `(m,n)+(n,)`、row `(m,n)+(1,n)` |
| 指标 | correctness check、元素数、ms/iter、NineToothed/PyTorch ratio |
| warmup / repeat | 10 / 50 |

### Benchmark 命令与结果

```bash
# 命令
python skills/competition/ninetoothed-operator-skill/examples/task-04/benchmark_task.py --case same
python skills/competition/ninetoothed-operator-skill/examples/task-04/benchmark_task.py --case vector
python skills/competition/ninetoothed-operator-skill/examples/task-04/benchmark_task.py --case row

# 结果（RTX 5060 + CUDA 12.8）
# same:  correctness allclose=True, max_error=0; torch=0.6052 ms/iter, nt=0.5811 ms/iter, ratio=0.960x
# vector: correctness allclose=True, max_error=0; torch=0.3981 ms/iter, nt=0.3997 ms/iter, ratio=1.004x
# row:    correctness allclose=True, max_error=0; torch=0.3983 ms/iter, nt=0.4045 ms/iter, ratio=1.015x
```

### 失败诊断

详见 `failure_diagnosis.md`，包含 row broadcast 未 expand 导致的 arrangement shape 不一致场景，形成"现象 → 复现 → 定位 → 修复 → 验证"闭环。

## 文件清单

| 文件 | 用途 |
|------|------|
| `benchmark_task.py` | 性能对比脚本 |
| `failure_diagnosis.md` | 失败诊断文档 |
