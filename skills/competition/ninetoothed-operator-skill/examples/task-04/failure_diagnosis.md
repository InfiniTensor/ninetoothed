# Task 04 Failure Diagnosis

## 场景

调试对象：`examples/task-01/add.py`

模拟失败：在二维 broadcast add 中，`rhs` 是 shape `(1, n)` 的 row broadcast 输入。如果实现时直接对 `rhs.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))`，而没有先把第 0 维 expand 到输出的 `m`，kernel 的 outermost shape 会和 `output` 不一致，或只覆盖第一行数据。

## 复现命令

```bash
pytest skills/competition/ninetoothed-operator-skill/examples/task-01/test_add.py -k row_broadcast
```

也可以用 benchmark 先做正确性 sanity check：

```bash
python skills/competition/ninetoothed-operator-skill/examples/task-04/benchmark_task.py --case row
```

## 观察

- PyTorch reference：`expected = lhs + rhs`
- NineToothed 输出：可能只在部分行正确，或编译阶段报 arrangement shape 不匹配
- 关键输入信息：
  - `lhs.shape == (m, n)`
  - `rhs.shape == (1, n)`
  - `output.shape == (m, n)`

## 定位过程

1. 先确认 CUDA 可用，排除环境 skip。
2. 打印或检查 `torch.broadcast_shapes(lhs.shape, rhs.shape)`，确认输出应为 `(m, n)`。
3. 阅读 `_arrangement_2d()`，确认 `lhs`、`rhs`、`output` 在 `tile()` 前是否对齐到输出 shape。
4. 对 row broadcast 输入，`rhs.shape[0] == 1`，必须先执行 `rhs.expand((output.shape[0], -1))`。
5. 如果没有 expand，`rhs` 的 tile 外层行数与 `output` 不同，违反 NineToothed arrangement 的对齐要求。

## 修复

在 task-01 的 `_align_to_output()` 中先把低维输入补到 2D，再按输出 shape expand：

```python
if aligned.shape[0] == 1:
    aligned = aligned.expand((output.shape[0], -1))

if aligned.shape[1] == 1:
    aligned = aligned.expand((-1, output.shape[1]))
```

随后 `_arrangement_2d()` 对已经对齐的输入执行相同 block shape 的 `tile()`。

## 验证

修复后运行：

```bash
pytest skills/competition/ninetoothed-operator-skill/examples/task-01/test_add.py
python skills/competition/ninetoothed-operator-skill/examples/task-04/benchmark_task.py --case row
```

期望结果：

- pytest 全部通过或在无 CUDA 环境自动 skip
- benchmark 输出 `correctness: allclose=True`
- 性能结果包含 `torch.add`、`ninetoothed` 和 `ratio (nt/torch)`

## 经验

layout 和 broadcast 问题优先检查 arrangement，而不是先怀疑 application。NineToothed 会按 arranged tensor 的 outermost shape 分发 program，多输入算子的 outermost shape 必须一致。
