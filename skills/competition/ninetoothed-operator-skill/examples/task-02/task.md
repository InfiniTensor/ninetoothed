# Task 02: Softmax 算子（Reduction/Block）

> **环境说明：Windows + NVIDIA RTX 5060 Laptop GPU (8GB) + PyTorch 2.12.0.dev20260408+cu128 + ninetoothed 0.26.0，全部测试已通过，benchmark 已采集。**

实现一个 2D row-wise softmax 算子，对输入 `input` 的最后一维执行：

```python
output = torch.softmax(input, dim=-1)
```

## 要求

- 文件：`softmax.py`, `test_softmax.py`, `benchmark_softmax.py`
- 输入：CUDA 上的 2D `torch.float32` tensor，shape 为 `(m, n)`
- 输出：与输入 shape/dtype/device 相同的新 tensor
- 实现：使用 NineToothed kernel，按行进行 tile/block 设计，一个 program 处理一行 block
- 数值：使用 `row - max(row)` 的稳定 softmax 写法
- 验证：pytest 中与 PyTorch reference 对比，并检查每行和接近 1
- 环境：无 CUDA 时测试自动 skip，benchmark 正常退出

## 验收命令

```bash
pytest skills/competition/ninetoothed-operator-skill/examples/task-02/test_softmax.py -v
python skills/competition/ninetoothed-operator-skill/examples/task-02/benchmark_softmax.py
```

## AI 执行记录摘要

| 项目 | 内容 |
|------|------|
| 开发环境 | Windows 11 + NVIDIA RTX 5060 Laptop GPU (8GB) + PyTorch 2.12.0.dev20260408+cu128 |
| 实现方式 | `@ninetoothed.jit` + 行级 tile + `ntl.max/exp/sum` |
| 测试覆盖 | 非 2 的幂长度 (781, 129)、长 block (1024)、数值稳定输入、行和 ≈ 1 |
| 本地 pytest 结果 | **PASSED** — 全部通过 |
| 本地 benchmark 结果 | **已采集** — shape=(2048,1024)，nt=125.59 ms/iter，torch=0.1663 ms/iter，ratio=755x |

### Correctness 测试命令与结果

```bash
# 命令
pytest skills/competition/ninetoothed-operator-skill/examples/task-02/test_softmax.py -v

# 结果（RTX 5060 + CUDA 12.8）：全部 PASSED
```

### Benchmark 设计

| 项目 | 值 |
|------|-----|
| 默认 shape | `(2048, 1024)` |
| baseline | `torch.softmax(input, dim=-1)` |
| 指标 | ms/iter、NineToothed/PyTorch ratio |
| warmup / repeat | 10 / 50（脚本内可配置） |
| 当前结果 | **已采集** — RTX 5060，shape=(2048,1024)，nt=125.59 ms/iter，torch=0.1663 ms/iter，ratio=755.19x |

```bash
# 命令
python skills/competition/ninetoothed-operator-skill/examples/task-02/benchmark_softmax.py

# 结果（RTX 5060 + CUDA 12.8）
# shape input=(2048, 1024), dtype=float32, device=cuda
# torch.softmax: 0.1663 ms/iter
# ninetoothed:   125.5873 ms/iter
# ratio (nt/torch): 755.194x
```

## 文件清单

| 文件 | 用途 |
|------|------|
| `softmax.py` | 算子实现 |
| `test_softmax.py` | pytest 测试 |
| `benchmark_softmax.py` | 性能对比 |
