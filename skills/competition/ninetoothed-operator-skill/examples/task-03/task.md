# Task 03: Layout-Sensitive 算子

> **环境说明：Windows + NVIDIA RTX 5060 Laptop GPU (8GB) + PyTorch 2.12.0.dev20260408+cu128 + ninetoothed 0.26.0，全部测试已通过。**

实现 `transpose_add(input, bias)`：

```python
output = input.transpose(0, 1) + bias
```

## 要求

- 文件：`transpose_add.py`, `test_transpose_add.py`
- 输入：CUDA 上的 2D `torch.float32` tensor
- `input` 可以是 contiguous，也可以是 slicing / `empty_strided()` 得到的 non-contiguous tensor
- `bias.shape` 必须等于 `(input.shape[1], input.shape[0])`
- 输出：新分配的 contiguous tensor，shape 与 `bias` 相同
- 实现：使用 NineToothed，必须体现 stride/offset 处理，不能假设输入连续存储
- 验证：同时测试 contiguous 和 non-contiguous 输入，并与 PyTorch reference 对比

## 验收命令

```bash
pytest skills/competition/ninetoothed-operator-skill/examples/task-03/test_transpose_add.py -v
```

## AI 执行记录摘要

| 项目 | 内容 |
|------|------|
| 开发环境 | Windows 11 + NVIDIA RTX 5060 Laptop GPU (8GB) + PyTorch 2.12.0.dev20260408+cu128 |
| 实现方式 | `ninetoothed.make()` + 显式 `offsets/stride` 访问 non-contiguous input；在 Python 层用 `torch.transpose` 创建 view 后传入 kernel，避免 `ninetoothed.permute` 编译阶段维度不匹配问题 |
| 测试覆盖 | contiguous input、slicing non-contiguous、`empty_strided()` 自定义 stride、错误 bias shape 异常 |
| 本地执行结果 | **5 PASSED** — 全部通过（含 non-contiguous 和 empty_strided 用例） |

### Correctness 测试命令与结果

```bash
# 命令
pytest skills/competition/ninetoothed-operator-skill/examples/task-03/test_transpose_add.py -v

# 结果（RTX 5060 + CUDA 12.8）：全部 PASSED
# test_transpose_add_contiguous[257-129-dtype0-cuda] PASSED
# test_transpose_add_contiguous[64-512-dtype0-cuda] PASSED
# test_transpose_add_non_contiguous_input[dtype0-cuda] PASSED
# test_transpose_add_empty_strided_input[dtype0-cuda] PASSED
# test_transpose_add_rejects_bad_bias_shape[cuda] PASSED
```

## 文件清单

| 文件 | 用途 |
|------|------|
| `transpose_add.py` | 算子实现 |
| `test_transpose_add.py` | pytest 测试 |
