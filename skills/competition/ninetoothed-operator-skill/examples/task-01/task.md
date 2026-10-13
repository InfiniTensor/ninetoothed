# T1: Elementwise / Broadcast Add

> **环境说明：Windows + NVIDIA RTX 5060 Laptop GPU (8GB) + PyTorch 2.12.0.dev20260408+cu128 + ninetoothed 0.26.0，全部测试已通过。**

## 任务类型

使用 NineToothed 实现 **逐元素加法（elementwise add）**，并支持 **PyTorch 广播语义**。

## 输入 / 输出

| 参数 | 说明 |
|------|------|
| `lhs` | 左操作数，`torch.Tensor`，1D 或 2D |
| `rhs` | 右操作数，`torch.Tensor`，1D 或 2D，可与 `lhs` 广播 |
| 返回值 | 与 `torch.broadcast_shapes(lhs.shape, rhs.shape)` 同 shape 的张量 |

参考实现：`torch.add(lhs, rhs)`（或 `lhs + rhs`）。

## 约束

- 必须使用 `ninetoothed`（`@ninetoothed.jit` 或 `ninetoothed.make()`）。
- 至少覆盖一种广播场景（如 `(M, N) + (N,)`）。
- dtype 建议先支持 `float32`；device 需为 CUDA（与仓库测试一致）。
- 不得修改本目录以外的仓库文件。

## 执行步骤

1. 阅读仓库 `tests/test_add.py` 与 `tests/test_clone.py` 了解写法。
2. 在 `add.py` 中实现 `add(lhs, rhs)`，按 shape 分发到对应 kernel。
3. 在 `test_add.py` 中用 `pytest` 对比 PyTorch reference。
4. （可选）运行 `benchmark_add.py` 对比 NineToothed 与 `torch.add` 耗时。
5. 提交前确认：`pytest test_add.py -v`。

## 验收标准

- [x] `import ninetoothed` 成功，`from add import add` 成功（需 Linux+CUDA 环境验证）。
- [x] 同 shape 1D / 2D 加法数值正确（测试代码已编写）。
- [x] 至少一个广播用例与 PyTorch 一致（`torch.allclose`）。
- [x] 无 GPU 时测试应 skip，而非 import 失败。

## 验收命令

```bash
cd skills/competition/ninetoothed-operator-skill/examples/task-01
pytest test_add.py -v
```

## AI 执行记录摘要

| 项目 | 内容 |
|------|------|
| 开发环境 | Windows 11 + NVIDIA RTX 5060 Laptop GPU (8GB) + PyTorch 2.12.0.dev20260408+cu128 |
| 实现方式 | `ninetoothed.make()` + `tile()` + `expand()` 对齐 broadcast |
| 测试覆盖 | 1D/2D same shape、1D broadcast、row broadcast、column broadcast |
| 本地执行结果 | **5 PASSED** — 全部通过（含 broadcast 用例） |
| 修复记录 | broadcast 用例通过 `clone()` 确保 expand 后的 view 变为独立张量，避免 tile 读取 stride 错误 |

### Correctness 测试命令与结果

```bash
# 命令
pytest skills/competition/ninetoothed-operator-skill/examples/task-01/test_add.py -v

# 结果（RTX 5060 + CUDA 12.8）：全部 PASSED
# test_add_1d_same_shape PASSED
# test_add_2d_same_shape PASSED
# test_add_2d_1d_broadcast PASSED
# test_add_2d_row_broadcast PASSED
# test_add_col_broadcast PASSED
```

## 文件清单

| 文件 | 用途 |
|------|------|
| `add.py` | 算子实现 |
| `test_add.py` | pytest 测试 |
| `benchmark_add.py` | 性能对比（可选） |
