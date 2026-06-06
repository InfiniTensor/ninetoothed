# Reduction Softmax — 示例场景

## 目标

实现一个行级 online softmax kernel，作为 `reduction_2d` 模式的展示。

## 任务描述

实现 `ninetoothed_softmax` 函数，使用 online softmax 算法，沿最后一维做行归约。

## 关键点

1. **Online Softmax** — 维护 `m_i` 和 `d_i` 两个状态变量实现数值稳定性
2. **Mask** — `ntl.load` 用 `mask` 和 `other=float("-inf")` 处理边界，避免 `exp` 溢出
3. **BLOCK_SIZE** — 必须是输入最后一维的大小，以确保一次 load 完整行

## Online Softmax 算法

```
m_0 = max(0, x[0:BLOCK_SIZE])
d_0 = exp(x[0:BLOCK_SIZE] - m_0)
m_prev = m_0; d_prev = sum(d_0)
output[0:BLOCK_SIZE] = exp(x[0:BLOCK_SIZE] - m_0) / d_prev
```

## 运行

```bash
python examples/reduction_softmax/run.py
python examples/reduction_softmax/benchmark.py
```

## 预期结果

- 所有 correctness 测试通过
- 数值精度与 `torch.softmax` 一致（atol=1e-3 for fp16）

## Torture Test 提示

- small shape: `(1, 2)` — 极小的行
- large shape: `(4, 131072)` — 极大的行
- non-contiguous: `x[:, ::2]` sliced input
- 极端值: `x.fill_(1000.0)` — 测试数值稳定性
