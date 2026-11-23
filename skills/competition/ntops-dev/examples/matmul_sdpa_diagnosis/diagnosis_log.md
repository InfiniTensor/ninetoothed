# Matmul / SDPA 诊断记录

环境：NVIDIA GeForce RTX 4090 D，Python 3.10.8，PyTorch 2.1.2+cu121，CUDA 12.1，Triton 3.0.0。

## SDPA 收集失败

```bash
cd ntops
python3 -m pytest \
  tests/test_scaled_dot_product_attention.py -q
```

```text
ModuleNotFoundError: No module named 'torch.nn.attention'
```

测试引用 `torch.nn.attention.bias.causal_lower_right`，当前 PyTorch 2.1.2 没有该模块路径。初赛材料将其标记为环境兼容失败；后续应使用支持该 API 的 PyTorch，或在任务允许时增加 reference fallback。

## Matmul 数值失败

```bash
python3 -m pytest tests/test_matmul.py -q
```

```text
1 failed, 7 passed in 28.79s
```

失败 shape：

```text
(1, 394, 724) @ (1, 724, 388), float16
```

`torch.allclose` 在 `rtol=0.01, atol=0.01` 下失败。该问题不是导入或 CUDA 可用性问题，需要先固定输入再判断 accumulation/precision 差异。

## 固定种子复现

```bash
cd <ntskill-repository>
timeout 180 python3 -u \
  skills/competition/ntops-dev/scripts/run_coverage_checks.py matmul
```

```text
seed=0
shape_a=(1, 394, 724)
shape_b=(1, 724, 388)
ok=False
max_abs=0.0625
max_rel=18800.0
max_ref=142.5
```

`max_abs=0.0625` 高于 `atol=0.01`，说明失败可以稳定复现。`max_rel` 受接近零的参考值放大，后续应检查误差分布和 accumulation dtype，在查清原因前不放宽容差。
