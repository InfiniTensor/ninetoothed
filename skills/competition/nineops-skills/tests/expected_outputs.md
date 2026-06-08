# 期望输出参考

以下为各示例在 NVIDIA GPU 上的典型正确性输出。Agent 应能验证输出与预期一致。

## elementwise_broadcast_add

**命令:** `python examples/elementwise_broadcast_add/run.py`

```
=== Elementwise Broadcast Add — 6 test cases ===
  ✔ contiguous (256,) + (256,) → OK
  ✔ scalar    (256,) + (1,)   → OK
  ✔ broadcast (4,8) + (8,)   → OK
  ✔ broadcast (4,8) + (4,1)   → OK
  ✔ transposed (32,16) contig → OK
  ✔ uneven    (1000,) + (1,)  → OK
All 6 tests passed!
```

## reduction_softmax

**命令:** `python examples/reduction_softmax/run.py`

```
=== Reduction Softmax — 8 test cases ===
  ✔ basic (4,1024)           → OK
  ✔ multi-row (8,4096)       → OK
  ✔ uneven cols (4,768)      → OK
  ✔ single row (1,2048)      → OK
  ✔ fp16 (4,1024)            → OK
  ✔ extreme (4,1024) large   → OK
  ✔ non-contiguous (4,1024)  → OK
  ✔ prime cols (4,1021)      → OK
All 8 tests passed!
```

## non_contiguous_stride_case

**命令:** `python examples/non_contiguous_stride_case/run.py`

```
=== Non-Contiguous Stride Tests (10 scenarios) ===
  ✔ contiguous (128,)           → OK
  ✔ transposed (32,64) contig   → OK
  ✔ sliced rows (16,64) view    → OK
  ✔ sliced cols (64,16) view    → OK
  ✔ sliced both (32,32) view    → OK
  ✔ view3d (8,8,8) -> (64,)     → OK
  ✔ expanded (1,128) -> (4,128) → OK
  ✔ permuted (4,32,8) dims      → OK
  ✔ as_strided (32,64)          → OK
  ✔ small 1D (15,)              → OK
All 10 tests passed!
```

## performance_regression_case

**命令:** `python examples/performance_regression_case/run.py`

```
=== Performance Regression — Matmul 2D 示例 ===
  ✅ BLOCK=16    — M=1024, N=1024, K=1024
  ✅ BLOCK=32    — M=1024, N=1024, K=1024
  ✅ BLOCK=64x64x32 — M=1024, N=1024, K=1024
  ✅ BLOCK=128x128x32 — M=1024, N=1024, K=1024
  ... (256x512x128 同理)
🎉 所有 matmul 变体正确性验证通过！
```

**Benchmark 预期趋势:**

```
shape      | BLOCK=16  | BLOCK=128  | 退化倍数
(512,512,512)| ~2.0ms   | ~0.3ms     | ~6x
(1024,...)   | ~12ms    | ~2.0ms     | ~6x
(2048,...)   | ~90ms    | ~15ms      | ~6x
```

> 具体数值因 GPU 型号不同会有较大差异，但 **退化倍数 >2x** 是可靠的诊断指标。

## 通用断言 (Agent Validator)

在任何示例上验证时，使用以下逻辑：

```python
import torch

def assert_correct(kernel_output: torch.Tensor, expected: torch.Tensor, atol=1e-3):
    """确认 kernel 输出与 PyTorch 基准一致。"""
    assert torch.allclose(kernel_output, expected, atol=atol, rtol=1e-3), \
        f"max diff={torch.max(torch.abs(kernel_output - expected)).item():.6f}"
```

如果该断言通过，则 agent 可确认实现正确。
