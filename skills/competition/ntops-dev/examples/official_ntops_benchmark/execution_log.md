# 官方 ntops 性能对比记录

环境：

- GPU：NVIDIA GeForce RTX 4090 D
- PyTorch：2.1.2+cu121
- CUDA：12.1
- 官方 ntops：`6bc90d5 Develop ResNet operators (#73)`
- warmup：20
- iters：100

Correctness 预检查：

```bash
cd <clean-ntops-repository>
python3 -m pytest \
  tests/test_gelu.py tests/test_softmax.py tests/test_addmm.py -q
```

结果：

```text
18 passed, 8 skipped in 22.78s
```

Benchmark 命令：

```bash
cd <clean-ntops-repository>
python3 \
  <ntskill-repository>/skills/competition/ntops-dev/scripts/run_official_ntops_benchmark.py \
  --warmup 20 --iters 100
```

结果：

| 算子 | dtype | shape | ntops ms | PyTorch ms | ntops / PyTorch |
| --- | --- | --- | ---: | ---: | ---: |
| gelu | fp16 | `(1048576,)` | 0.0457 | 0.0079 | 5.7976 |
| relu | fp16 | `(1048576,)` | 0.0451 | 0.0085 | 5.2794 |
| silu | fp16 | `(1048576,)` | 0.0793 | 0.0087 | 9.0681 |
| add | fp16 | `(1048576,)` | 0.0521 | 0.0081 | 6.4273 |
| mul | fp16 | `(1048576,)` | 0.0520 | 0.0083 | 6.2449 |
| gelu | fp32 | `(1048576,)` | 0.0451 | 0.0079 | 5.7182 |
| relu | fp32 | `(1048576,)` | 0.0463 | 0.0085 | 5.4642 |
| silu | fp32 | `(1048576,)` | 0.0454 | 0.0084 | 5.4181 |
| add | fp32 | `(1048576,)` | 0.0518 | 0.0080 | 6.4540 |
| mul | fp32 | `(1048576,)` | 0.0500 | 0.0082 | 6.0875 |
| softmax | fp16 | `(1024, 1024)` | 0.0517 | 0.0112 | 4.6243 |
| addmm | fp16 | `(512, 512, 512)` | 0.1497 | 0.0179 | 8.3579 |

结论：

- 本次选取的官方 ntops 算子在上述输入和环境下均慢于 PyTorch eager。
- 该结果说明当前材料中的性能差距不只来自新增 `maximum`，也出现在若干官方已有算子样例中。
- 结论不能外推到 ntops 全部算子，也不代表已经完成性能优化。
- Proposal 中“性能达到 PyTorch/Triton 80%-90%”的计划目标需要在后续阶段调整为“先建立可复现 benchmark，再针对具体算子逐项优化”。
