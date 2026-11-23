# Non-Contiguous Stride Case — 示例场景

## 目标

测试和验证 ninetoothed 在 non-contiguous 输入下的行为，作为 "stride 正确处理" 的展示。

## 任务描述

ninotoothed 在 non-contiguous tensor 上是否能正确计算 stride 和偏移？本场景通过 add kernel 测试多种 strided 模式。

## 覆盖的 stride 变体

| 变体 | 创建方式 | 预期 |
|------|----------|------|
| contiguous | `torch.randn(M, N)` | ✅ 基准 |
| transposed | `.t()` | ✅ stride 交换 |
| sliced | `x[::2, :]` | ✅ offset 步进 |
| view | `.view()` | ✅ 连续视图 |
| expanded | `.expand()` | ✅ 广播扩展 |
| permuted | `.permute(1, 0)` | ✅ 维度重排 |
| as_strided | `.as_strided(...)` | ✅ 自定义 stride |

## 核心问题

对于 non-contiguous tensor，ninetoothed 的 underlying ptr 计算是否为：

```
ptr + row * stride_row + col * stride_col
```

而不是：

```
ptr + row * cols + col   // 仅用于 contiguous
```

## 运行

```bash
python examples/non_contiguous_stride_case/run.py
python examples/non_contiguous_stride_case/benchmark.py
```
