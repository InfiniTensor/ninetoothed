# Softmax 执行记录

环境：NVIDIA GeForce RTX 4090 D，Python 3.10.8，PyTorch 2.1.2+cu121，CUDA 12.1，Triton 3.0.0。

## Correctness

```bash
cd ntops
python3 -m pytest tests/test_softmax.py -q
```

```text
........                                                                 [100%]
8 passed in 11.58s
```

## Benchmark

```bash
cd <ntskill-repository>
python3 \
  skills/competition/ntops-dev/scripts/run_operator_benchmark.py \
  softmax --shape 1024,1024 --dtype float16 --warmup 10 --iters 50
```

```text
ntops_ms=0.0536
torch_ms=0.0091
relative_to_torch=5.8902
```

## Generated source

```bash
python3 \
  skills/competition/ntops-dev/scripts/inspect_generated_source.py \
  softmax --shape 1024,1024 --dtype float16 --latest 1
```

关键统计：

```text
language_load=2
language_store=1
language_exp=3
language_arange=33
language_sum=1
language_max=1
autotune=True
```

当前检查方向是 reduction/mask 开销和两阶段表达式的重复生成；尚无复测证明这些因素已被优化。

## 边界用例

```text
single_last_dim_fp16 ok=True shape=(7, 1) max_abs=0.0
wide_fp16 ok=True shape=(2, 1024) max_abs=7.62939453125e-06
middle_dim_fp32 ok=True shape=(3, 5, 7) dim=1 max_abs=5.960464477539063e-08
large_values_fp32 ok=True shape=(4, 33) max_abs=9.313225746154785e-10
```

尚未覆盖空 tensor 和非连续 softmax 输入。Correctness 在上述范围内通过，性能未达到 Proposal 目标。
